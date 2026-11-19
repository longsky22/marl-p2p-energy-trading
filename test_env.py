"""
test_env.py — 环境单元测试（阶段二）。

在 Conda 环境 ies_marl 内运行 100 个 Episode 的随机策略测试（Random Rollout），
打印当前 Conda 环境路径，断言：无 NaN、无越界、所有物理约束 100% 满足。
覆盖 mrdac / greedy / vda 三种出清机制。

运行: conda run -n ies_marl python test_env.py
"""
from __future__ import annotations
import os
import sys
import numpy as np

from env import P2PMicrogridEnv
import make_data


def ensure_data():
    if not os.path.join(make_data.DATA_DIR, "profiles.npz"):
        pass
    if not os.path.exists(os.path.join(make_data.DATA_DIR, "profiles.npz")):
        print("[data] profiles.npz not found -> generating ...")
        make_data.main()


def random_rollout(mechanism, n_episodes=100, seed=123):
    env = P2PMicrogridEnv(mechanism=mechanism, seed=seed)
    rng = np.random.default_rng(seed)
    total_steps = 0
    total_viol = 0
    nan_count = 0
    reward_sum = 0.0
    min_soc, max_soc = np.inf, -np.inf
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        # reset obs 有限性
        for i in range(env.N):
            if not np.all(np.isfinite(obs[i])):
                nan_count += 1
        done = False
        while not done:
            acts = rng.uniform(-1.0, 1.0, size=(env.N, env.act_dim)).astype(np.float32)
            obs, rew, term, trunc, info = env.step(acts)
            total_steps += 1
            # 奖励/观测有限性
            for i in range(env.N):
                if not np.isfinite(rew[i]):
                    nan_count += 1
                if not np.all(np.isfinite(obs[i])):
                    nan_count += 1
                reward_sum += rew[i]
            # 物理约束
            ok, viol = env.check_constraints(info["phys"])
            if not ok:
                total_viol += len(viol)
                if total_viol <= 5:
                    print("  VIOLATION:", viol)
            for i in range(env.N):
                e = info["phys"][i]["E_after"]
                min_soc = min(min_soc, e); max_soc = max(max_soc, e)
            done = all(trunc.values())
    return dict(mechanism=mechanism, episodes=n_episodes, steps=total_steps,
                violations=total_viol, nan=nan_count, reward_sum=reward_sum,
                min_soc=min_soc, max_soc=max_soc)


def main():
    print("=" * 70)
    print("CONDA ENV PYTHON EXE :", sys.executable)
    print("PYTHON VERSION       :", sys.version.split()[0])
    print("ENV NAME             :", os.environ.get("CONDA_DEFAULT_ENV", "(via conda run)"))
    print("=" * 70)
    ensure_data()

    all_ok = True
    results = []
    for mech in ["mrdac", "greedy", "vda"]:
        r = random_rollout(mech, n_episodes=100, seed=123)
        results.append(r)
        ok = (r["violations"] == 0 and r["nan"] == 0)
        all_ok = all_ok and ok
        print(f"[{mech:6s}] episodes={r['episodes']} steps={r['steps']} "
              f"violations={r['violations']} nan={r['nan']} "
              f"SoC_range=[{r['min_soc']:.3f},{r['max_soc']:.3f}] "
              f"sum_reward={r['reward_sum']:.2f} -> {'PASS' if ok else 'FAIL'}")

    # 额外断言：SoC 全局不越界
    env = P2PMicrogridEnv()
    for r in results:
        assert r["min_soc"] >= env.E_MIN.min() - 1e-6, "SoC below global min"
        assert r["max_soc"] <= env.E_MAX.max() + 1e-6, "SoC above global max"

    print("=" * 70)
    if all_ok:
        print("ALL TESTS PASSED: 100 episodes x 3 mechanisms, "
              "no NaN, no out-of-bounds, 100% physical constraints satisfied.")
    else:
        print("TESTS FAILED — see violations above.")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()