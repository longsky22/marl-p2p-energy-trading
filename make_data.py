"""
make_data.py — 生成复现所需的合成数据集（阶段二）。

论文[16]的澳大利亚住宅负荷/光伏原始数据未随论文附带，按用户授权"逻辑生成"。
本脚本按 Table I 的 L_max / G_max 缩放，构造 4 个微电网的典型日负荷/光伏曲线、
时变紧急电价 p_e^G ∈ [15,35]、日前电价 p_da^G，以及含噪预测 (L_hat, G_hat)。

输出（data/ 目录）:
  - profiles.npz : 多日数组，供 env.py 训练/测试采样
  - typical_day.csv : 一个典型日的可读表格（供绘图/核对）
  - meta.json : 参数与生成说明

所有随机性由 seed 控制，可复现。
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

# ---- Table I（精确） ----
N = 4
T = 24
L_MAX = np.array([25.0, 6.0, 40.0, 5.0])    # kWh
G_MAX = np.array([5.0, 7.0, 10.0, 15.0])    # kWh
E_MAX = np.array([8.0, 15.0, 15.0, 30.0])   # kWh
T_ES_MAX = np.array([4.0, 5.0, 8.0, 10.0])  # kW
T_ES_MIN = -T_ES_MAX                        # kW
E_0 = np.array([0.0, 2.0, 0.0, 20.0])       # kWh
E_MIN = np.zeros(N)                         # 假设 A3
ETA_CHR = 1.0
ETA_DIS = 1.0
P_FIT = 2.0                                 # $/kWh 固定
P_EMG_LO, P_EMG_HI = 15.0, 35.0             # $/kWh 时变范围
KAPPA = np.ones(N)                          # 假设 A2: 日前采购因子=1


def _load_shape(kind: int) -> np.ndarray:
    """归一化典型日负荷形状（峰值=1）。不同微电网代表不同区域类型。"""
    h = np.arange(T)
    if kind == 0:      # 住宅：早晚双峰
        s = 0.35 + 0.45 * np.exp(-((h - 7.5) ** 2) / 6) + 0.75 * np.exp(-((h - 19.5) ** 2) / 8)
    elif kind == 1:    # 商业：白天单峰
        s = 0.25 + 0.85 * np.exp(-((h - 13.0) ** 2) / 18)
    elif kind == 2:    # 工业：白天高位平坦
        s = 0.45 + 0.55 / (1 + np.exp(-(h - 7))) * (1 - 1 / (1 + np.exp(-(h - 18))))
    else:              # 小型住宅：晚峰
        s = 0.30 + 0.35 * np.exp(-((h - 8.0) ** 2) / 7) + 0.80 * np.exp(-((h - 20.0) ** 2) / 6)
    s = np.clip(s, 0.05, None)
    return s / s.max()


def _pv_shape() -> np.ndarray:
    """归一化光伏出力形状（正午峰值=1，夜间=0）。"""
    h = np.arange(T)
    s = np.exp(-((h - 12.5) ** 2) / 7.0)
    s[h < 6] = 0.0
    s[h > 18] = 0.0
    return s / s.max()


def _emergency_price_shape() -> np.ndarray:
    """归一化紧急电价形状 [0,1]，映射到 [15,35]：夜间低、早晚峰高。"""
    h = np.arange(T)
    s = 0.30 + 0.40 * np.exp(-((h - 8.0) ** 2) / 5) + 0.70 * np.exp(-((h - 19.0) ** 2) / 6)
    s = np.clip(s, 0.0, 1.0)
    return s / s.max()


def generate(n_days: int = 1000, seed: int = 2026, forecast_noise: float = 0.08,
             day_var: float = 0.10, hour_var: float = 0.05):
    """生成 n_days 天的实际/预测曲线与电价。

    forecast_noise: 预测相对误差标准差（假设 A12，占额定比例）。
    day_var: 日间整体幅度波动；hour_var: 逐小时波动。
    """
    rng = np.random.default_rng(seed)
    load_shapes = np.stack([_load_shape(i) for i in range(N)])   # [N,24]
    pv_shape = _pv_shape()                                       # [24]
    pe_shape = _emergency_price_shape()                          # [24]

    L = np.zeros((n_days, N, T))
    G = np.zeros((n_days, N, T))
    Lhat = np.zeros((n_days, N, T))
    Ghat = np.zeros((n_days, N, T))
    pe = np.zeros((n_days, T))
    pda = np.zeros((n_days, T))

    for d in range(n_days):
        # 紧急电价：基础形状 + 日间小幅波动，限幅 [15,35]
        pe_day = P_EMG_LO + (P_EMG_HI - P_EMG_LO) * pe_shape
        pe_day = pe_day * (1 + rng.normal(0, 0.04, T))
        pe[d] = np.clip(pe_day, P_EMG_LO, P_EMG_HI)
        # 日前电价：位于 FIT 与紧急电价之间（仅记录，不影响奖励）
        pda[d] = P_FIT + 0.5 * (pe[d] - P_FIT)

        for i in range(N):
            day_f = 1 + rng.normal(0, day_var)
            hour_f = 1 + rng.normal(0, hour_var, T)
            base_L = L_MAX[i] * load_shapes[i] * day_f * hour_f
            base_G = G_MAX[i] * pv_shape * (1 + rng.normal(0, day_var)) * (1 + rng.normal(0, hour_var, T))
            L[d, i] = np.clip(base_L, 0.0, None)
            G[d, i] = np.clip(base_G, 0.0, None)
            # 含噪预测（乘性高斯噪声），非负
            Lhat[d, i] = np.clip(L[d, i] * (1 + rng.normal(0, forecast_noise, T)), 0.0, None)
            Ghat[d, i] = np.clip(G[d, i] * (1 + rng.normal(0, forecast_noise, T)), 0.0, None)

    qda = np.maximum(0.0, KAPPA[None, :, None] * (Lhat - Ghat))   # 式(4)

    return dict(L=L, G=G, Lhat=Lhat, Ghat=Ghat, pe=pe, pda=pda, qda=qda,
                L_MAX=L_MAX, G_MAX=G_MAX, E_MAX=E_MAX, T_ES_MAX=T_ES_MAX,
                T_ES_MIN=T_ES_MIN, E_0=E_0, E_MIN=E_MIN,
                ETA_CHR=ETA_CHR, ETA_DIS=ETA_DIS, P_FIT=P_FIT, KAPPA=KAPPA)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    data = generate(n_days=1000, seed=2026)
    np.savez_compressed(os.path.join(DATA_DIR, "profiles.npz"), **data)

    # 典型日 CSV（第 0 天）
    d0 = 0
    rows = []
    for t in range(T):
        row = {"hour": t, "p_emg": data["pe"][d0, t], "p_da": data["pda"][d0, t]}
        for i in range(N):
            row[f"L_{i+1}"] = data["L"][d0, i, t]
            row[f"G_{i+1}"] = data["G"][d0, i, t]
            row[f"Lhat_{i+1}"] = data["Lhat"][d0, i, t]
            row[f"Ghat_{i+1}"] = data["Ghat"][d0, i, t]
            row[f"qda_{i+1}"] = data["qda"][d0, i, t]
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "typical_day.csv"), index=False)

    meta = dict(N=N, T=T, dt=1.0, n_days=1000, seed=2026,
                L_MAX=L_MAX.tolist(), G_MAX=G_MAX.tolist(), E_MAX=E_MAX.tolist(),
                T_ES_MAX=T_ES_MAX.tolist(), T_ES_MIN=T_ES_MIN.tolist(),
                E_0=E_0.tolist(), E_MIN=E_MIN.tolist(),
                eta_chr=ETA_CHR, eta_dis=ETA_DIS, p_fit=P_FIT,
                p_emg_range=[P_EMG_LO, P_EMG_HI], kappa=KAPPA.tolist(),
                tau1=1, tau2=6, forecast_noise=0.08,
                note="Synthetic data logically generated per Table I scaling; "
                     "original Australian dataset [16] not bundled with the paper.")
    with open(os.path.join(DATA_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("Saved data to", DATA_DIR)
    print("profiles.npz keys:", list(data.keys()))
    print("L shape", data["L"].shape, "pe range", data["pe"].min(), data["pe"].max())


if __name__ == "__main__":
    main()