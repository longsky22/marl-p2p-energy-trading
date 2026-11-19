"""
summarize_results.py — 汇总各训练 run 的评估指标（稳态/last/best）。

对 logs/*_seed*.csv 计算：
  late_mean : 训练末段 1/3 评估点的平均日利润（稳健稳态指标）
  last      : 最后一次评估日利润
  best      : 训练期最佳评估日利润
输出 results/training_summary.csv，供报告主对比表使用（以 late_mean 为稳态判据）。

运行: conda run -n ies_marl python summarize_results.py
"""
from __future__ import annotations
import os
import glob
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "logs")
RES = os.path.join(HERE, "results")


def main():
    os.makedirs(RES, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(LOG, "*_seed*.csv"))):
        name = os.path.basename(f)[:-4]
        if name.endswith("_final"):
            continue
        df = pd.read_csv(f)
        if "eval_total_profit" not in df.columns:
            continue
        ev = df.dropna(subset=["eval_total_profit"])
        if len(ev) == 0:
            continue
        k = max(3, len(ev) // 3)
        late = ev.tail(k)
        rows.append(dict(
            run=name,
            n_eval=len(ev),
            last_episode=int(ev["episode"].iloc[-1]),
            late_mean_profit=late["eval_total_profit"].mean(),
            last_profit=ev["eval_total_profit"].iloc[-1],
            best_profit=ev["eval_total_profit"].max(),
            late_mean_emg=late["eval_emergency_kwh"].mean(),
            late_mean_p2p=late["eval_p2p_kwh"].mean(),
            late_mean_fit=late["eval_fit_kwh"].mean(),
            late_mean_soc=late["eval_avg_soc"].mean(),
        ))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "training_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nSaved results/training_summary.csv")


if __name__ == "__main__":
    main()