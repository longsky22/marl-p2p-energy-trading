"""
plot_results.py — 科研级可视化与复现对比表（阶段三）。

图1: 训练收敛曲线（Episode Reward / 日运行成本，滑动平均，多算法对比 + baseline 参考线）。
图2: 典型日 24h 源-荷-储功率平衡堆叠柱状图 + 储能 SOC 与分时电价双轴曲线。
输出: results/*.png, *.pdf（高分辨率），results/comparison_table.csv。

运行: conda run -n ies_marl python plot_results.py
依赖: logs/*.csv, checkpoints/*_best.pt, results/baseline_metrics.csv
"""
from __future__ import annotations
import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import P2PMicrogridEnv
from train_marl import MMAPPO
import baseline as bl

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
LOG_DIR = os.path.join(HERE, "logs")
CKPT_DIR = os.path.join(HERE, "checkpoints")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3})


def _smooth(x, w=20):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def load_logs():
    logs = {}
    for f in glob.glob(os.path.join(LOG_DIR, "*.csv")):
        name = os.path.basename(f)
        if name.endswith("_final.csv"):
            continue
        df = pd.read_csv(f)
        if "episode" in df.columns and "reward" in df.columns:
            logs[name[:-4]] = df
    return logs


def fig1_convergence(logs, baseline_df, outprefix="fig1_convergence"):
    # 收敛对比仅保留 MRDAC 下 seed0 的各算法（对齐论文 Fig.2，单种子主结果）
    logs = {k: v for k, v in logs.items() if "_mrdac_seed0" in k}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = plt.cm.tab10.colors
    # 左：Episode Reward（社区合计）滑动平均
    ax = axes[0]
    w = 20
    for k, (tag, df) in enumerate(sorted(logs.items())):
        rew = df["reward"].values
        epv = df["episode"].values
        if len(rew) >= w:
            ys = _smooth(rew, w); xs = epv[w - 1:]
        else:
            ys = rew; xs = epv
        ax.plot(xs, ys, label=tag, color=colors[k % 10], lw=1.6)
    ax.set_xlabel("Episode"); ax.set_ylabel("Episode reward (community sum)")
    ax.set_title("(a) Training reward (moving avg, w=20)")
    ax.legend(fontsize=8)
    # 右：评估日运行成本（=-profit）+ baseline 参考线
    ax = axes[1]
    for k, (tag, df) in enumerate(sorted(logs.items())):
        if "eval_total_profit" in df.columns:
            d = df.dropna(subset=["eval_total_profit"])
            ax.plot(d["episode"], -d["eval_total_profit"], "o-", ms=3,
                    label=tag + " (eval)", color=colors[k % 10], lw=1.4)
    if baseline_df is not None:
        for _, r in baseline_df.iterrows():
            lbl = f"baseline {r['policy']}/{r['mechanism']}"
            ax.axhline(-r["total_profit"], ls="--", lw=1.0, alpha=0.7, label=lbl)
    ax.set_xlabel("Episode"); ax.set_ylabel("Daily operating cost ($, = -profit)")
    ax.set_title("(b) Eval daily cost vs. baselines")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, outprefix + ".png"))
    fig.savefig(os.path.join(RESULTS, outprefix + ".pdf"))
    plt.close(fig)


def collect_typical_trace(agent, env, day, mechanism):
    env.set_mechanism(mechanism)
    obs, info = env.reset(day_index=day)
    hidden = agent._zero_hidden()
    rows = []
    done = False
    while not done:
        acts, _, _, hidden = agent.act(obs, hidden, deterministic=True)
        obs, rew, term, trunc, info = env.step(acts)
        done = all(trunc.values())
        t = info["t"] - 1
        for i in range(env.N):
            ph = info["phys"][i]
            rows.append(dict(hour=t, mg=i + 1, **ph))
    return pd.DataFrame(rows)


def fig2_dispatch(trace, env, outprefix="fig2_typical_day_dispatch"):
    """每个微电网一行：源-荷-储堆叠柱 + SOC/电价双轴。"""
    N = env.N
    fig, axes = plt.subplots(N, 1, figsize=(12, 3.0 * N), sharex=True)
    if N == 1:
        axes = [axes]
    hours = np.arange(env.T)
    for i in range(N):
        ax = axes[i]
        sub = trace[trace["mg"] == i + 1].sort_values("hour")
        G = sub["G"].values; qda = sub["qda"].values; qb = sub["qb"].values
        qe = sub["q_e"].values; L = sub["L"].values; Tes = sub["T_es"].values
        qfit = sub["q_fit"].values; qs = sub["qs"].values
        # 供给侧堆叠（正）：光伏、日前购电、P2P买入、紧急购电、储能放电(-T_es>0)
        dis = np.maximum(0.0, -Tes)
        ax.bar(hours, G, label="PV", color="#f4a300")
        ax.bar(hours, qda, bottom=G, label="DA purchase", color="#4c72b0")
        ax.bar(hours, qb, bottom=G + qda, label="P2P buy", color="#55a868")
        ax.bar(hours, dis, bottom=G + qda + qb, label="ES discharge", color="#c44e52")
        ax.bar(hours, qe, bottom=G + qda + qb + dis, label="Emergency", color="#8172b2")
        # 消耗侧（负）：负荷、储能充电、上网、P2P卖出
        ch = np.maximum(0.0, Tes)
        ax.bar(hours, -L, label="Load", color="#333333", alpha=0.85)
        ax.bar(hours, -ch, bottom=-L, label="ES charge", color="#c44e52", alpha=0.5)
        ax.bar(hours, -qfit, bottom=-L - ch, label="Feed-in", color="#937860")
        ax.bar(hours, -qs, bottom=-L - ch - qfit, label="P2P sell", color="#55a868", alpha=0.5)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(f"MG{i+1} power (kWh)")
        # 双轴：SOC 与电价
        ax2 = ax.twinx()
        ax2.plot(hours, sub["E_after"].values, "o-", color="#1f77b4", ms=3, label="SoC")
        ax2.plot(hours, sub["pe"].values, "s--", color="#d62728", ms=3, label="Emergency price")
        ax2.set_ylabel("SoC (kWh) / price ($/kWh)")
        ax2.grid(False)
        if i == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, ncol=4, loc="upper left")
        ax.set_title(f"Microgrid {i+1}: source-load-storage balance (typical day)", fontsize=10)
    axes[-1].set_xlabel("Hour")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, outprefix + ".png"))
    fig.savefig(os.path.join(RESULTS, outprefix + ".pdf"))
    plt.close(fig)


def steady_metrics_from_logs():
    """从训练日志计算稳态（末段1/3评估均值）指标，作为稳健对比判据。"""
    out = {}
    for f in sorted(glob.glob(os.path.join(LOG_DIR, "*_seed*.csv"))):
        tag = os.path.basename(f)[:-4]
        if tag.endswith("_final"):
            continue
        df = pd.read_csv(f)
        if "eval_total_profit" not in df.columns:
            continue
        ev = df.dropna(subset=["eval_total_profit"])
        if len(ev) == 0:
            continue
        late = ev.tail(max(3, len(ev) // 3))
        mech = next((t for t in tag.split("_") if t in ("mrdac", "greedy", "vda")), "mrdac")
        out[tag] = dict(total_profit=late["eval_total_profit"].mean(),
                        profit_per_mg=late["eval_total_profit"].mean() / 4.0,
                        emergency_kwh=late["eval_emergency_kwh"].mean(),
                        fit_kwh=late["eval_fit_kwh"].mean(),
                        p2p_kwh=late["eval_p2p_kwh"].mean(),
                        avg_soc=late["eval_avg_soc"].mean(),
                        mechanism=mech)
    return out


def fig3_mechanism(agent_finals, outprefix="fig3_mechanism_comparison"):
    """MMAPPO 在 MRDAC/VDA/Greedy 下的日性能对比（对齐论文 Table II）。"""
    mechs = ["mrdac", "vda", "greedy"]
    data = {}
    for tag, fin in agent_finals.items():
        if tag.startswith("mmappo_"):
            m = fin.get("mechanism", "mrdac")
            if m in mechs:
                data[m] = fin
    present = [m for m in mechs if m in data]
    if len(present) < 2:
        print("Not enough mechanism checkpoints for fig3; skip.")
        return
    metrics = [("total_profit", "Total profit ($/day)"),
               ("emergency_kwh", "Emergency purchase (kWh)"),
               ("fit_kwh", "Total feed-in (kWh)"),
               ("p2p_kwh", "P2P trade volume (kWh)"),
               ("avg_soc", "Average SoC (kWh)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 4.2))
    colors = {"mrdac": "#55a868", "vda": "#4c72b0", "greedy": "#c44e52"}
    labels = {"mrdac": "MRDAC", "vda": "VDA", "greedy": "Greedy"}
    for ax, (key, title) in zip(axes, metrics):
        vals = [data[m][key] for m in present]
        bars = ax.bar([labels[m] for m in present], vals,
                      color=[colors[m] for m in present])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    fig.suptitle("MMAPPO under different market-clearing mechanisms (reproduction of Table II)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, outprefix + ".png"))
    fig.savefig(os.path.join(RESULTS, outprefix + ".pdf"))
    plt.close(fig)


def table2_markdown(agent_finals):
    """生成论文 Table II 风格的 Markdown 对比表。"""
    mechs = ["mrdac", "vda", "greedy"]
    data = {}
    for tag, fin in agent_finals.items():
        if tag.startswith("mmappo_"):
            m = fin.get("mechanism", "mrdac")
            if m in mechs:
                data[m] = fin
    if not data:
        return None
    rows = [("Total Profit ($)", "total_profit"),
           ("Emergency Purchase (kWh)", "emergency_kwh"),
           ("Total FIT (kWh)", "fit_kwh"),
           ("P2P Trade Volume (kWh)", "p2p_kwh"),
           ("Average SoC (kWh)", "avg_soc")]
    header = "| Metric | " + " | ".join(m.upper() for m in mechs if m in data) + " |"
    sep = "|---|" + "---|" * len([m for m in mechs if m in data])
    lines = [header, sep]
    for name, key in rows:
        cells = " | ".join(f"{data[m][key]:.2f}" for m in mechs if m in data)
        lines.append(f"| {name} | {cells} |")
    md = "\n".join(lines)
    with open(os.path.join(RESULTS, "table2_reproduction.md"), "w", encoding="utf-8") as f:
        f.write("# Table II reproduction (MMAPPO under different mechanisms)\n\n" + md + "\n")
    return md


def comparison_table(logs, baseline_df, agent_finals):
    rows = []
    if baseline_df is not None:
        for _, r in baseline_df.iterrows():
            rows.append(dict(method=f"baseline:{r['policy']}", mechanism=r["mechanism"],
                             total_profit=r["total_profit"], profit_per_mg=r["profit_per_mg"],
                             emergency_kwh=r["emergency_kwh"], fit_kwh=r["fit_kwh"],
                             p2p_kwh=r["p2p_kwh"], avg_soc=r["avg_soc"]))
    for tag, fin in agent_finals.items():
        rows.append(dict(method=tag, mechanism=fin.get("mechanism", "mrdac"),
                         total_profit=fin["total_profit"], profit_per_mg=fin["profit_per_mg"],
                         emergency_kwh=fin["emergency_kwh"], fit_kwh=fin["fit_kwh"],
                         p2p_kwh=fin["p2p_kwh"], avg_soc=fin["avg_soc"]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "comparison_table.csv"), index=False)
    return df


def main():
    os.makedirs(RESULTS, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=None, help="typical day index for fig2")
    args = ap.parse_args()

    logs = load_logs()
    bpath = os.path.join(RESULTS, "baseline_metrics.csv")
    baseline_df = pd.read_csv(bpath) if os.path.exists(bpath) else None

    # 加载各算法 best checkpoint 做最终评估与典型日 trace
    env = P2PMicrogridEnv()
    # 表格/机制对比使用稳态（末段均值）指标，避免瞬时峰值噪声；主结果用 seed0
    agent_finals = {k: v for k, v in steady_metrics_from_logs().items() if k.endswith("_seed0")}
    # 图2 调度图使用 MMAPPO×MRDAC 的具体策略 checkpoint（last=稳态策略）
    best_agent = None
    for cand in ["mmappo_mrdac_seed0_last.pt", "mmappo_mrdac_seed0_best.pt"]:
        p = os.path.join(CKPT_DIR, cand)
        if os.path.exists(p):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            agent = MMAPPO(env, algo=ck.get("algo", "mmappo"))
            agent.load(p)
            best_agent = (agent, ck.get("mechanism", "mrdac"))
            break

    fig1_convergence(logs, baseline_df)
    print("Saved fig1_convergence.png/pdf")

    if best_agent is not None:
        agent, mech = best_agent
        day = args.day if args.day is not None else env.eval_days[0]
        trace = collect_typical_trace(agent, env, day, mech)
        trace.to_csv(os.path.join(RESULTS, "marl_typical_trace.csv"), index=False)
        fig2_dispatch(trace, env)
        print(f"Saved fig2_typical_day_dispatch.png/pdf (day={day}, mechanism={mech})")
    else:
        print("No MMAPPO checkpoint found; skip fig2.")

    fig3_mechanism(agent_finals)
    print("Saved fig3_mechanism_comparison.png/pdf")
    md = table2_markdown(agent_finals)
    if md:
        print("\n=== Table II reproduction (MMAPPO x mechanisms) ===\n" + md)

    df = comparison_table(logs, baseline_df, agent_finals)
    pd.set_option("display.width", 180); pd.set_option("display.max_columns", 20)
    print("\n=== Reproduction comparison table (MARL vs. Baseline) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nSaved results/comparison_table.csv")


if __name__ == "__main__":
    main()