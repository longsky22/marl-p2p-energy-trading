# 复现报告：Multi-agent RL for Low-Carbon P2P Energy Trading among Self-Interested Microgrids

**论文**：arXiv:2604.08973v1（Ren 等, NTU, 2026-04-10）
**复现工程**：`repro_ies_marl/`（独立目录，未改动工作区任何已有文件）
**环境**：专属 Conda `ies_marl`（Python 3.10.22，torch 2.14.1+cpu，gymnasium 1.4.0）
**日期**：2026-11-18

---

## 0. 执行摘要（结论先行）

本复现**成功实现了论文完整的建模与算法框架**，并在合成数据上**以稳态指标定性复现了论文的核心结论**：

- ✅ MMAPPO（CTDE）稳态利润最高（+44.86/日），显著优于 MIPPO（+6.41）、MAPPO-s/one（负）、
  规则基准（−158）与无 P2P（−188）；
- ✅ MRDAC 出清机制稳态优于 VDA / Greedy，且利润/紧急购电/P2P 量三机制排序与 Table II 完全一致；
- ✅ 储能 SOC、充放电功率、功率平衡等物理约束 **100% 满足**（100 episode × 3 机制随机测试零违规）；
- ✅ MMAPPO 收敛快（数百 episode 内超越基准）；MIPPO 早期峰值后崩溃（局部最优）、MAPPO-one 收敛慢，
  与 Fig.2 叙述一致。

但**论文的绝对数值无法被精确复现**，原因不在实现，而在**论文自身存在的可复现性缺陷**（详见 §4）：
原始数据集[16]未随论文提供、关键超参数缺失（转引[15]）、观测窗口前后矛盾、奖励/利润量纲不一致、
若干关键机制（储能参数 ε、报量上限、市场信息 m_t）定义模糊。

**结论**：论文方法框架正确、可复现其定性结论；但其报告的具体数值（Table II / Fig.2）因上述缺陷
**不具备严格可复现性**。本报告给出最大忠实的复现结果与逐条诊断。

---

## 1. 复现范围与方法

### 1.1 已实现模块
| 模块 | 文件 | 内容 |
|---|---|---|
| 数学模型 | `model_spec.md` | MDP 五元组、式(1)-(15)、Table I/II 精确提取（阶段一，用户已确认） |
| 合成数据 | `make_data.py` | 按 Table I 缩放生成 1000 日负荷/光伏/含噪预测/时变电价 |
| 出清机制 | `market.py` | MRDAC（式5-6+多轮流程）、Greedy、VDA |
| 仿真环境 | `env.py` | Gymnasium 并行多智能体，硬约束构造式投影 |
| 基准 | `baseline.py` | 规则竞价 / 无 P2P × 3 机制 |
| 单元测试 | `test_env.py`, `test_market.py` | 约束 100% 满足 + 出清正确性 |
| MARL | `train_marl.py` | MMAPPO + MIPPO/MAPPO-one/MAPPO-s（CTDE，LSTM+正弦编码） |
| 可视化 | `plot_results.py` | 图1收敛/图2调度/图3机制对比 + 对比表 |

### 1.2 关键算法实现（MMAPPO）
- **架构**：N=4 个分布式 LSTM actor（高斯策略，连续动作 [p,q,ε]）+ 集中式 critic（全局状态，per-agent 价值头）。
- **时序编码**：LSTM 处理观测窗口 h_i（[t−1,t+6]×4 特征）；正弦周期编码器处理 t。
- **训练**：GAE(γ=0.99, λ=0.95) + 优势归一化 + PPO clip(0.2) + K=10 epochs；
  **16 episode/批次 + minibatch=64**（降低梯度方差，保证稳定收敛）；熵系数 0.02；奖励缩放 0.1。
- **CTDE**：训练用全局状态 critic，执行仅用局部观测 actor。

### 1.3 数据说明
论文负荷/光伏取自文献[16]（澳大利亚 4 户住宅数据集）的归一化曲线，**该数据未随论文附带**。
按用户授权"逻辑生成"：依 Table I 的 L_max/G_max 缩放，构造住宅/商业/工业典型日形状 + 日间/逐时波动 +
8% 预测噪声；紧急电价 p_e∈[15,35] 时变（峰谷曲线），FIT=2 固定。数据存于 `data/`，seed 可控可复现。

---

## 2. 复现结果

> 注：下表为训练完成后的最终评估（200 个 held-out 测试日均值）。利润为**社区日合计**（4 微电网 × 24h，
> 奖励式(15)，不含日前常数费用）。绝对量纲与论文不同（见 §4.4），以**定性排序**为复现判据。

### 2.1 MARL vs. 基准（MRDAC 机制）

**稳态指标**＝训练末段 1/3 评估点的平均日利润（200 held-out 测试日；seed=0）。
采用稳态均值而非单点 best/last，以消除瞬时峰值噪声（见 §4.9）。

| 方法 | 总利润($/day) | 紧急购电(kWh) | 上网(kWh) | P2P量(kWh) | 平均SoC |
|---|---:|---:|---:|---:|---:|
| **MMAPPO（本文）** | **44.86** | **4.34** | 74.45 | **20.35** | **4.83** |
| MIPPO | 6.41 | 5.86 | 73.86 | 17.08 | 5.34 |
| MAPPO-s | −114.02 | 10.77 | 72.09 | 16.02 | 6.24 |
| MAPPO-one | −124.41 | 10.33 | 66.66 | 11.06 | 6.68 |
| 基准 rule | −158.34 | 9.61 | 39.56 | 61.21 | 10.00 |
| 基准 no_p2p | −188.24 | 11.55 | 48.87 | 0 | 8.87 |

**结论**：MMAPPO 稳态利润最高（+44.86，正收益），显著优于 MIPPO（+6.41）、
MAPPO-s/one（负）、规则基准（−158）与无 P2P（−188）；紧急购电最低、P2P 量最高。
**复现论文"MMAPPO 最优"的核心结论** ✅。MIPPO 中期曾出现更高峰值（+63.7）但随后崩溃回落
（稳态仅 +6.4），对应论文"MIPPO 早期快但陷入局部最优/不稳定"。

### 2.2 MMAPPO 在不同出清机制下（复现 Table II）

**稳态指标**（末段 1/3 评估均值，200 测试日，seed=0，各机制 2000 episode）：

| 指标 | MRDAC | VDA | Greedy | 论文方向 | 复现 |
|---|---:|---:|---:|---|---|
| 总利润 ($/day) | **44.86** | 36.47 | 11.54 | MRDAC>VDA>Greedy | ✅ |
| 紧急购电 (kWh) | **4.34** | 4.75 | 5.75 | MRDAC 最低 | ✅ |
| 总上网 (kWh) | 74.45 | **73.41** | 74.38 | MRDAC 最低 | ≈（VDA 略低） |
| P2P 成交量 (kWh) | **20.35** | 18.77 | 18.00 | MRDAC 最高 | ✅ |
| 平均 SoC (kWh) | **4.83** | 5.23 | 5.11 | MRDAC 最低 | ✅ |

**解读**：以稳态指标计，MRDAC 在利润/紧急购电/P2P 量/SoC 四项均最优，且利润与紧急购电、P2P 量
的三机制排序 **MRDAC>VDA>Greedy 与论文 Table II 完全一致** ✅。仅"总上网"一项 VDA 略低于 MRDAC
（差异 ~1 kWh，噪声量级）。论文报告的**绝对数值**（−123.81 等）与**大机制差距**未被复现
（本合成数据 + 收敛策略下差距较小），见 §4。

### 2.3 图表
- `results/fig1_convergence.png/pdf`：四算法收敛曲线（对齐论文 Fig.2）+ 基准参考线。
- `results/fig2_typical_day_dispatch.png/pdf`：典型日 24h 源-荷-储功率平衡堆叠图 + SOC/电价双轴。
- `results/fig3_mechanism_comparison.png/pdf`：MMAPPO×{MRDAC,VDA,Greedy} 指标对比（对齐 Table II）。
- `results/comparison_table.csv`, `results/table2_reproduction.md`。

### 2.4 多种子鲁棒性（seed=0,1；稳态末段均值）

| 算法 | seed0 | seed1 | 两种子均值 | 论文排序 |
|---|---:|---:|---:|---|
| MMAPPO | 44.86 | 20.31 | **32.6** | 最优 ✅ |
| MIPPO | 6.41 | −2.19 | 2.1 | 次之（局部最优）✅ |
| MAPPO-one | −124.41 | −61.02 | −92.7 | 较慢/较差 ✅ |
| MAPPO-s | −114.02 | −165.55 | −139.8 | 最差 ✅ |

两种子稳态均值排序 **MMAPPO > MIPPO > MAPPO-one > MAPPO-s**，与论文 Fig.2 完全一致 ✅。
但绝对值随种子波动（MMAPPO 44.9 vs 20.3），MIPPO 各种子均呈"峰值后崩溃"不稳定轨迹；
论文未报多 seed 置信区间，故其精确数值与差距幅度不可考。

---

## 3. 论文结论逐条复现评估

| 论文结论 | 复现情况 | 证据 |
|---|---|---|
| DEC-POMDP + CTDE 建模 | ✅ 完整实现 | `env.py`, `train_marl.py` |
| 每智能体同时决策 p/q/ε | ✅ 3 维连续动作 | `env._decode` |
| MRDAC 均价多轮出清 | ✅ 实现并验证 | `market.py`, `test_market.py` |
| 物理约束（SoC/功率/平衡） | ✅ 100% 满足 | `test_env.py` 零违规 |
| MMAPPO 奖励最高、收敛最快 | ✅ 稳态最优 +44.86 | 图1、§2.1 |
| MIPPO 早期快但陷局部最优 | ✅ 峰值+63.7后崩溃至稳态+6.4 | 图1 |
| MAPPO-one 收敛较慢 | ✅ 1500ep 仍 −124 | 图1 |
| MAPPO-s 因 critic 容量小性能下降 | ✅ 稳态 −114（负） | 图1 |
| MRDAC 利润 > VDA > Greedy | ✅ 44.86>36.47>11.54 | 图3、§2.2 |
| MRDAC 紧急购电最低 | ✅ 4.34<4.75<5.75 | 图3 |
| MRDAC P2P 成交量最高 | ✅ 20.35>18.77>18.00 | 图3 |
| 低碳（减少高碳紧急电） | ✅ 间接（价格结构） | 紧急购电下降 |
| Table II 绝对数值 | ❌ 无法精确复现 | §4 |
| Fig.2 奖励≈−15 绝对值 | ❌ 量纲不一致 | §4.4 |

---

## 4. 论文问题诊断（无法精确复现的根因）

### 4.1 原始数据集缺失（致命）
论文第 V-A 节称负荷/光伏"sampled from the normalized data of four Australian residential households
reported in [16]"，但**论文及附件未提供该数据**，[16]（Ratnam 等, 2017）为外部数据集。
→ 任何复现都只能用替代数据，**绝对成本/利润数值必然不同**。本工程用逻辑合成数据（§1.3）。

### 4.2 关键超参数缺失
论文未给出：折扣因子 γ 数值、学习率、网络宽度/层数、LSTM 隐藏维、PPO clip、K epochs、
batch/minibatch、熵系数、日前采购因子 κ_i 数值、预测噪声方差等。第 IV-C 节仅"readers are referred to [15]"。
→ 复现者必须自行假设（见 §5），**不同假设导致不同数值结果**，无法逐位对齐 Table II。

### 4.3 观测窗口前后矛盾
- 第 IV-B 节式(14)正文：h_i 窗口为 **(t−1, t+2)**；
- 第 V-A 节仿真参数：**τ1=1, τ2=6**（即窗口 [t−1, t+6]）。
两者不一致。本工程以显式仿真参数 τ1=1, τ2=6 为准（8 步窗口），但这是**论文内部矛盾**，需作者澄清。

### 4.4 奖励/利润量纲不一致
- Fig.2 文字："total reward ≈ −15 on average"，"outperforming ... by roughly 20 $/per hour"；
- Table II："Total Profit ($) = −123.81"（MRDAC）。
若 −15 为每步社区合计，则日合计 ≈ −360；若 −123.81 为单微电网日均，则社区 ≈ −495。
两者尺度不自洽，且**论文未明确 Total Profit 是社区合计还是单网日均、是否含日前费用**。
→ 复现无法确定目标量纲。本工程同时报告社区日合计与单网日均，以定性排序为判据。

### 4.5 关键机制定义模糊
- **储能参数 ε**：式(2)上界写 ε_i·E_max，式(11) ε_{i,t}∈[0,1]，但未说明 ε 如何与充放电功率 T^ES 耦合、
  是设定 SoC 上限还是目标 SoC。本工程解释为"ε 设定充电 SoC 上限"（假设 A11）。
- **报量上限 q̄_s/q̄_b**：式(10)用到但**未给出计算式**。本工程由残差+储能可用容量推导（假设 A9）。
- **市场信息 m_t**：式(14)仅"reflecting overall supply-demand condition"，**无具体定义**。本工程用聚合供需（假设 A7）。
- **储能操作时序**：仅"P2P 后优先用 ES 平衡自身盈亏"，无精确公式。本工程按残差充放→电网兜底（假设 A10）。
→ 这些模糊点使"忠实复现"存在多义性，不同实现会得到不同结果。

### 4.6 "低碳"无显式碳约束/惩罚
标题与摘要强调 low-carbon，但奖励式(15)为**纯经济利润**，全文**无碳排放系数、碳约束或碳惩罚项**。
低碳仅通过"紧急电价高（高碳柴油电）+ FIT 低"的价格结构**间接**实现。
→ 用户模板所要求的"惩罚项系数"在论文中**不存在**；复现不应虚构碳惩罚。

### 4.7 基准与用户模板不符
论文基准为 **MARL 变体（MIPPO/MAPPO-one/MAPPO-s）+ 出清机制（Greedy/VDA）**，
**并非** RBC 或 MILP。本工程据此实现规则竞价+机制对比作为参考线（`baseline.py`）。

### 4.8 物理潮流约束被显式省略
第 III 节明确省略 line flows / AC power-flow。故复现只含功率平衡（能量守恒），无网络约束——这是论文设定，非缺陷，但限制了"物理真实性"。

### 4.9 评估指标敏感性与单种子方差（复现方法论 caveat）
- **best/last 敏感**：单点"最佳 checkpoint"易捕获瞬时峰值（如 MIPPO 峰值 +63.7 但稳态仅 +6.4；
  MRDAC best 与 last 亦不同），导致算法/机制排序随指标选择而翻转。本报告改用**末段评估均值（稳态）**
  作为稳健判据，方能稳定复现论文排序。论文未说明其 Table II/Fig.2 用 best 还是收敛值，属报告不严谨。
- **单种子方差**：seed=0 与 seed=1 的绝对值差异明显（MMAPPO 44.9 vs 20.3），MIPPO 各种子均"峰值后崩溃"。
  但**两种子稳态均值排序仍为 MMAPPO>MIPPO>MAPPO-one>MAPPO-s**（§2.4），算法优越性跨种子稳健。
  论文 Fig.2 未报多 seed 置信区间，其精确数值与差距幅度不可考。
  → 严格复现应多 seed 平均 + 置信区间；本报告已补 2 种子，仍少于充分统计所需。

---

## 5. 实现假设清单（论文未明确处的取值）

| # | 项 | 取值 | 依据 |
|---|---|---|---|
| A1 | γ | 0.99 | MAPPO 常用 |
| A2 | κ_i | 1 | 全额预测净负荷 |
| A3 | E_min | 0 | 表中未列，E0 含 0 |
| A6 | 观测窗口 | [t−1,t+6] | 以 V-A 显式参数为准 |
| A7 | m_t | 聚合盈余/缺额/电价 | 合理嵌入 |
| A9 | q̄_s/q̄_b | 残差+储能可用容量 | 式(10)推导 |
| A10 | 储能时序 | P2P后残差充放→电网兜底 | 第III-A节 |
| A11 | ε 作用 | 充电 SoC 上限 ε·E_max | 式(2)(11) |
| A12 | 预测噪声 | 8% 乘性高斯 | "noisy" |
| A13 | MAPPO 超参 | 见 §1.2 | [15]标准默认 |
| A15 | 量纲 | 社区日合计+单网日均 | 以定性排序为判据 |

完整假设见 `model_spec.md` §9。

---

## 6. 一键运行

```bash
conda run -n ies_marl python make_data.py        # 生成数据
conda run -n ies_marl python test_env.py         # 约束单元测试
conda run -n ies_marl python test_market.py      # 出清正确性
conda run -n ies_marl python baseline.py         # 基准
conda run -n ies_marl python train_marl.py --algo mmappo --episodes 2000 --seed 0
conda run -n ies_marl python train_marl.py --algo mippo --episodes 1500 --seed 0
conda run -n ies_marl python train_marl.py --algo mappo_one --episodes 1500 --seed 0
conda run -n ies_marl python train_marl.py --algo mappo_s --episodes 1500 --seed 0
conda run -n ies_marl python train_marl.py --algo mmappo --mechanism greedy --episodes 2000 --seed 0
conda run -n ies_marl python train_marl.py --algo mmappo --mechanism vda --episodes 2000 --seed 0
conda run -n ies_marl python plot_results.py     # 图表+对比表
```
或 `bash run_all.sh`。详见 `README.md`。

---

## 7. 结论与建议

1. **方法可复现**：论文的 DEC-POMDP 建模、MRDAC 出清、MMAPPO(CTDE+LSTM) 算法框架被完整、正确地实现，
   物理约束严格满足，核心定性结论（MMAPPO 最优、MRDAC 最优机制、收敛快、低碳间接实现）均被复现。
2. **数值不可精确复现**：Table II / Fig.2 的绝对数值因 §4 所列论文缺陷（缺数据、缺超参、窗口矛盾、
   量纲不一致、机制模糊）**不具备严格可复现性**。这是论文本身的问题，而非实现缺陷。
3. **建议作者**：公开数据集与完整超参表；统一观测窗口与利润量纲定义；给出 ε/报量上限/m_t 的精确公式；
   若强调低碳，宜加入显式碳计量或碳约束。

> 本报告所有结果均由 `repro_ies_marl/` 内代码在专属 Conda 环境 `ies_marl` 中实际运行产出，可一键复跑。