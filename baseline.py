"""
baseline.py — 基准调度与参考线（阶段二）。

论文基准为 MARL 变体 + 出清机制对比（无 RBC/MILP）。本脚本提供非学习参考线：
  - rule   : 规则竞价（预测盈余→卖出、缺额→买入，报价取 FIT 与紧急电价中点，eps=1）
  - no_p2p : 不参与 P2P（q=0），仅靠储能+主网平衡（P2P 价值下界）
在 {mrdac, greedy, vda} 三种出清机制下评估，输出日运行成本/利润与关键指标，
作为阶段三 MARL 的对照参考线，并复现 Table II 的机制对比结构。

运行: conda run -n ies_marl python baseline.py
输出: results/baseline_metrics.csv, results/baseline_typical_trace.csv
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from env import P2PMicrogridEnv

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def rule_policy(obs, env):
    """规则竞价：用当前窗口预测估计盈亏，满容量报价，中点价，eps=1。"""
    N = env.N
    acts = np.zeros((N, 3), dtype=np.float32)
    ti = env.tau1                                   # 当前 t 在窗口中的索引
    for i in range(N):
        o = obs[i]
        seq = o[env.static_dim:].reshape(env.win, 4)
        Lhat = seq[ti, 1] * env.L_MAX[i]
        Ghat = seq[ti, 2] * env.G_MAX[i]
        qda = seq[ti, 0] * env.L_MAX[i]
        est_r = Ghat + qda - Lhat                   # 预测净盈余
        acts[i, 0] = 0.0                            # 价格中点 a0=0
        acts[i, 1] = 1.0 if est_r < 0 else (-1.0 if est_r > 0 else 0.0)
        acts[i, 2] = 1.0                            # eps=1
    return acts


def no_p2p_policy(obs, env):
    acts = np.zeros((env.N, 3), dtype=np.float32)
    acts[:, 2] = 1.0                                # eps=1, q=0
    return acts


POLICIES = {"rule": rule_policy, "no_p2p": no_p2p_policy}


def evaluate_policy(env, policy_fn, days, mechanism, collect_trace_day=None):
    env.set_mechanism(mechanism)
    agg = dict(total_profit=0.0, emergency=0.0, fit=0.0, p2p_volume=0.0,
               soc_sum=0.0, soc_count=0, n_days=0)
    trace_rows = []
    for day in days:
        obs, info = env.reset(day_index=day)
        done = False
        while not done:
            acts = policy_fn(obs, env)
            obs, rew, term, trunc, info = env.step(acts)
            done = all(trunc.values())
            if collect_trace_day is not None and day == collect_trace_day:
                t = info["t"] - 1
                for i in range(env.N):
                    ph = info["phys"][i]
                    trace_rows.append(dict(day=day, hour=t, mg=i + 1, **ph))
        m = env.episode_metrics
        agg["total_profit"] += m["reward"]
        agg["emergency"] += m["q_e"]
        agg["fit"] += m["q_fit"]
        agg["p2p_volume"] += m["p2p_volume"]
        agg["soc_sum"] += m["soc_sum"]
        agg["soc_count"] += m["soc_count"]
        agg["n_days"] += 1
    nd = max(1, agg["n_days"])
    metrics = dict(
        total_profit=agg["total_profit"] / nd,                 # 社区日合计利润
        profit_per_mg=agg["total_profit"] / nd / env.N,        # 单微电网日均
        emergency_kwh=agg["emergency"] / nd,
        fit_kwh=agg["fit"] / nd,
        p2p_kwh=agg["p2p_volume"] / nd,
        avg_soc=agg["soc_sum"] / max(1, agg["soc_count"]),
    )
    return metrics, pd.DataFrame(trace_rows)


def run_baselines(n_eval=200, seed=0):
    os.makedirs(RESULTS, exist_ok=True)
    env = P2PMicrogridEnv(seed=seed)
    days = env.eval_days[:n_eval]
    rows = []
    best_trace = None
    for pname, pfn in POLICIES.items():
        for mech in ["mrdac", "greedy", "vda"]:
            metrics, trace = evaluate_policy(env, pfn, days, mech,
                                             collect_trace_day=days[0])
            rows.append(dict(policy=pname, mechanism=mech, **metrics))
            if pname == "rule" and mech == "mrdac":
                best_trace = trace
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "baseline_metrics.csv"), index=False)
    if best_trace is not None:
        best_trace.to_csv(os.path.join(RESULTS, "baseline_typical_trace.csv"), index=False)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\n=== Baseline metrics (avg over %d eval days) ===" % len(days))
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nSaved: results/baseline_metrics.csv")
    return df


if __name__ == "__main__":
    run_baselines()