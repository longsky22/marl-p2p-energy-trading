"""
env.py — 多微电网 P2P 交易的 Gymnasium 风格并行多智能体环境（阶段二）。

严格实现 model_spec.md：
  - SoC 转移 式(1)（eta=1）；储能边界 式(2)（上界 eps*E_max）；功率平衡 式(8)。
  - DA 式(4)；P2P 报价边界 式(9)(10)(11)；MRDAC/Greedy/VDA 出清（market.py）。
  - 奖励 式(15) = P^G(式12) + P^p2p(式13)。
  - 观测 式(14)：O_i = {m_t, E_i, h_i(窗口[t-1,t+6]×4), t}。

硬约束通过"构造式投影"保证：动作解码到合法区间，储能/电网吸收残差，
任何情况下功率平衡恒成立、SoC 与功率不越界（见 test_env.py 100% 断言）。

API（并行式，类 PettingZoo parallel）：
  obs, info = env.reset(seed=..., day_index=...)
  obs, rewards, terminated, truncated, info = env.step(actions)   # actions: [N,3] in [-1,1]
  obs[i] / rewards[i] 为智能体 i 的向量/标量；info["global_state"] 供集中式 critic。
"""
from __future__ import annotations
import os
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from market import clear

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def load_dataset(path=None):
    path = path or os.path.join(DATA_DIR, "profiles.npz")
    d = np.load(path)
    return {k: d[k] for k in d.files}


class P2PMicrogridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset=None, mechanism="mrdac", tau1=1, tau2=6,
                 seed=0, n_days_train=800, eval_days=None):
        super().__init__()
        self.data = dataset if dataset is not None else load_dataset()
        d = self.data
        self.N = int(d["L"].shape[1])
        self.T = int(d["L"].shape[2])
        self.n_days = int(d["L"].shape[0])
        self.mechanism = mechanism
        self.tau1, self.tau2 = tau1, tau2
        self.win = tau1 + tau2 + 1                     # 窗口步数 = 8
        # 参数
        self.L_MAX = d["L_MAX"]; self.G_MAX = d["G_MAX"]
        self.E_MAX = d["E_MAX"]; self.E_MIN = d["E_MIN"]
        self.T_ES_MAX = d["T_ES_MAX"]; self.T_ES_MIN = d["T_ES_MIN"]
        self.E_0 = d["E_0"]; self.P_FIT = float(d["P_FIT"])
        self.eta_chr = float(d["ETA_CHR"]); self.eta_dis = float(d["ETA_DIS"])
        self.kappa = d["KAPPA"]
        self.dt = 1.0
        # 归一化尺度
        self.sum_L = float(self.L_MAX.sum())
        self.pe_hi = 35.0
        # 训练/测试日划分
        self.n_days_train = min(n_days_train, self.n_days)
        self.eval_days = eval_days if eval_days is not None else list(range(self.n_days_train, self.n_days))
        self.rng = np.random.default_rng(seed)
        # 维度
        self.static_dim = 5                            # m_t(3)+E(1)+t(1)
        self.seq_dim = self.win * 4                    # 8×4=32
        self.obs_dim = self.static_dim + self.seq_dim  # 37
        self.act_dim = 3
        self.state_dim = self.N * self.obs_dim
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)
        self.agent_ids = list(range(self.N))
        self._day = 0
        self._t = 0
        self._E = self.E_0.copy()
        self._last_phys = None
        self.episode_metrics = self._empty_metrics()

    # ---------- 内部工具 ----------
    def _empty_metrics(self):
        return dict(reward=0.0, profit_G=0.0, profit_p2p=0.0, q_e=0.0, q_fit=0.0,
                    p2p_volume=0.0, soc_sum=0.0, soc_count=0, steps=0)

    def _day_arrays(self, day):
        d = self.data
        return (d["L"][day], d["G"][day], d["Lhat"][day], d["Ghat"][day],
                d["qda"][day], d["pe"][day], d["pda"][day])

    def _residual(self, day, t):
        L, G, _, _, qda, _, _ = self._day_arrays(day)
        return G[:, t] + qda[:, t] - L[:, t]           # r_i

    def _market_info(self, day, t):
        r = self._residual(day, t)
        surplus = float(np.maximum(0.0, r).sum()) / self.sum_L
        deficit = float(np.maximum(0.0, -r).sum()) / self.sum_L
        pe = float(self.data["pe"][day, t]) / self.pe_hi
        return np.array([surplus, deficit, pe], dtype=np.float32)

    def _obs_agent(self, i, day, t):
        L, G, Lhat, Ghat, qda, pe, pda = self._day_arrays(day)
        m = self._market_info(day, t)
        E_norm = self._E[i] / self.E_MAX[i]
        t_norm = t / (self.T - 1)
        static = np.concatenate([m, [E_norm, t_norm]]).astype(np.float32)
        # 时序窗口 [t-tau1, t+tau2]，边界 clamp
        seq = np.zeros((self.win, 4), dtype=np.float32)
        for k, z in enumerate(range(t - self.tau1, t + self.tau2 + 1)):
            zz = int(np.clip(z, 0, self.T - 1))
            seq[k, 0] = qda[i, zz] / self.L_MAX[i]
            seq[k, 1] = Lhat[i, zz] / self.L_MAX[i]
            seq[k, 2] = Ghat[i, zz] / self.G_MAX[i]
            seq[k, 3] = pe[zz] / self.pe_hi
        return np.concatenate([static, seq.reshape(-1)])

    def _build_obs(self, day, t):
        return {i: self._obs_agent(i, day, t) for i in range(self.N)}

    def global_state(self, obs=None):
        obs = obs if obs is not None else self._build_obs(self._day, self._t)
        return np.concatenate([obs[i] for i in range(self.N)]).astype(np.float32)

    # ---------- Gym API ----------
    def reset(self, *, seed=None, day_index=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if day_index is None:
            day_index = int(self.rng.integers(0, self.n_days_train))
        self._day = int(day_index)
        self._t = 0
        self._E = self.E_0.copy()
        self.episode_metrics = self._empty_metrics()
        obs = self._build_obs(self._day, self._t)
        info = {"global_state": self.global_state(obs), "day": self._day}
        return obs, info

    def _decode(self, actions, day, t):
        """把 [-1,1]^3 动作解码为 (p, q, eps)，并计算报量上限（式9-11）。"""
        L, G, Lhat, Ghat, qda, pe, pda = self._day_arrays(day)
        pe_t = float(pe[t])
        quotes, eps_arr, p_arr, q_arr, caps = [], np.zeros(self.N), np.zeros(self.N), np.zeros(self.N), []
        for i in range(self.N):
            a = np.clip(np.asarray(actions[i], dtype=np.float64), -1.0, 1.0)
            eps = (a[2] + 1.0) / 2.0                          # 式(11) [0,1]
            p = self.P_FIT + (a[0] + 1.0) / 2.0 * (pe_t - self.P_FIT)  # 式(9)
            r = G[i, t] + qda[i, t] - L[i, t]
            d_cap = min(self.T_ES_MAX[i] * self.dt, max(0.0, self._E[i] - self.E_MIN[i]) * self.eta_dis)
            c_cap = min(self.T_ES_MAX[i] * self.dt, max(0.0, eps * self.E_MAX[i] - self._E[i]) / self.eta_chr)
            q_s_bar = max(0.0, r) + d_cap                     # 最大可供给
            q_b_bar = max(0.0, -r) + c_cap                    # 最大可需求
            q = a[1] * (q_b_bar if a[1] >= 0 else q_s_bar)    # 式(10) [-q_s_bar, q_b_bar]
            quotes.append((i, p, q))
            eps_arr[i], p_arr[i], q_arr[i] = eps, p, q
            caps.append((q_s_bar, q_b_bar))
        return quotes, eps_arr, p_arr, q_arr, caps

    def step(self, actions):
        day, t = self._day, self._t
        L, G, Lhat, Ghat, qda, pe, pda = self._day_arrays(day)
        pe_t = float(pe[t])
        if isinstance(actions, dict):
            actions = [actions[i] for i in range(self.N)]
        quotes, eps_arr, p_arr, q_arr, caps = self._decode(actions, day, t)
        per_agent, trades = clear(quotes, self.mechanism, rng=self.rng)

        rewards, phys = {}, {}
        E_next = self._E.copy()
        for i in range(self.N):
            r = G[i, t] + qda[i, t] - L[i, t]
            qb, qs = per_agent[i]["qb"], per_agent[i]["qs"]
            r_post = r + qb - qs                              # P2P 后残差
            eps = eps_arr[i]
            if r_post > 0:                                    # 盈余→充电→上网
                charge = min(r_post, self.T_ES_MAX[i] * self.dt,
                             max(0.0, eps * self.E_MAX[i] - self._E[i]) / self.eta_chr)
                T_es = charge
                q_fit = r_post - charge
                q_e = 0.0
            elif r_post < 0:                                  # 缺额→放电→紧急购电
                discharge = min(-r_post, self.T_ES_MAX[i] * self.dt,
                                max(0.0, self._E[i] - self.E_MIN[i]) * self.eta_dis)
                T_es = -discharge
                q_e = -r_post - discharge
                q_fit = 0.0
            else:
                T_es, q_fit, q_e = 0.0, 0.0, 0.0
            E_before = self._E[i]
            E_after = float(np.clip(E_before + T_es * self.dt, self.E_MIN[i], self.E_MAX[i]))
            E_next[i] = E_after
            profit_G = self.P_FIT * q_fit - pe_t * q_e        # 式(12)
            profit_p2p = per_agent[i]["p2p_profit"]           # 式(13)
            reward = profit_G + profit_p2p                    # 式(15)
            rewards[i] = float(reward)
            phys[i] = dict(L=float(L[i, t]), G=float(G[i, t]), qda=float(qda[i, t]),
                           T_es=float(T_es), q_fit=float(q_fit), q_e=float(q_e),
                           qb=float(qb), qs=float(qs), E_before=float(E_before),
                           E_after=E_after, eps=float(eps), p=float(p_arr[i]),
                           q=float(q_arr[i]), q_s_bar=caps[i][0], q_b_bar=caps[i][1],
                           reward=float(reward), profit_G=float(profit_G),
                           profit_p2p=float(profit_p2p), pe=pe_t)
            m = self.episode_metrics
            m["reward"] += reward; m["profit_G"] += profit_G; m["profit_p2p"] += profit_p2p
            m["q_e"] += q_e; m["q_fit"] += q_fit; m["p2p_volume"] += qb
            m["soc_sum"] += E_after; m["soc_count"] += 1; m["steps"] += 1

        self._E = E_next
        self._last_phys = phys
        self._t += 1
        terminated = {i: False for i in range(self.N)}
        truncated = {i: self._t >= self.T for i in range(self.N)}
        if self._t >= self.T:
            obs = {i: np.zeros(self.obs_dim, dtype=np.float32) for i in range(self.N)}
        else:
            obs = self._build_obs(day, self._t)
        info = {"global_state": self.global_state(obs) if self._t < self.T else np.zeros(self.state_dim, np.float32),
                "phys": phys, "trades": trades, "t": self._t,
                "episode_metrics": dict(self.episode_metrics)}
        return obs, rewards, terminated, truncated, info

    # ---------- 约束自检（供 test_env.py） ----------
    def check_constraints(self, phys, tol=1e-6):
        """返回 (ok, violations)。校验功率平衡/SoC/功率/报价边界。"""
        viol = []
        for i, ph in phys.items():
            lhs = ph["L"] + ph["T_es"] + ph["q_fit"] + ph["qs"]
            rhs = ph["G"] + ph["qda"] + ph["qb"] + ph["q_e"]
            if abs(lhs - rhs) > tol:
                viol.append(f"MG{i} power_balance {lhs:.4f}!={rhs:.4f}")
            if not (self.E_MIN[i] - tol <= ph["E_after"] <= self.E_MAX[i] + tol):
                viol.append(f"MG{i} SoC_bound {ph['E_after']}")
            if not (self.T_ES_MIN[i] - tol <= ph["T_es"] <= self.T_ES_MAX[i] + tol):
                viol.append(f"MG{i} power_bound {ph['T_es']}")
            if ph["T_es"] > tol:  # 充电须满足 eps 上限
                if ph["E_after"] > ph["eps"] * self.E_MAX[i] + tol:
                    viol.append(f"MG{i} eps_ceiling {ph['E_after']}>{ph['eps']*self.E_MAX[i]}")
            if not (self.P_FIT - tol <= ph["p"] <= ph["pe"] + tol):
                viol.append(f"MG{i} price_bound {ph['p']}")
            if not (-ph["q_s_bar"] - tol <= ph["q"] <= ph["q_b_bar"] + tol):
                viol.append(f"MG{i} qty_bound {ph['q']}")
            if not (-tol <= ph["eps"] <= 1 + tol):
                viol.append(f"MG{i} eps_bound {ph['eps']}")
            if ph["q_fit"] < -tol or ph["q_e"] < -tol or ph["qb"] < -tol or ph["qs"] < -tol:
                viol.append(f"MG{i} nonneg_flow")
            if abs((ph["E_after"] - ph["E_before"]) - ph["T_es"] * self.dt) > tol:
                viol.append(f"MG{i} soc_dynamics")
        return (len(viol) == 0), viol

    def set_mechanism(self, mechanism):
        self.mechanism = mechanism