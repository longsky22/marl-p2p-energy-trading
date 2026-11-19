# 复现：Multi-agent RL for Low-Carbon P2P Energy Trading among Self-Interested Microgrids

本目录为论文 **arXiv:2604.08973v1**（Ren 等, NTU, 2026）的独立复现工程。
论文核心：4 个自利微电网在日内 P2P 电力市场中同时决策**报价 p / 报量 q / 储能参数 ε**，
建模为 **DEC-POMDP**，用 **MMAPPO**（MAPPO + LSTM + 正弦周期编码，CTDE）求解，
市场出清采用 **MRDAC**（多轮双向拍卖）。

> 论文原文（arXiv，请自行下载）：https://arxiv.org/abs/2604.08973 ；数学模型与符号定义见 [`model_spec.md`](model_spec.md)（阶段一产出，已用户确认）。
> 本工程**不修改工作区任何已有源文件**，所有产出均在 `repro_ies_marl/` 内。

---

## 1. 运行环境铁律：专属 Conda 环境 `ies_marl`

所有 Python 命令**必须**在专属 Conda 环境内运行（严禁污染全局环境）。

```bash
# 查看现有环境
conda env list

# 创建专属环境（已创建可跳过）
conda create -n ies_marl python=3.10 -y

# 安装依赖（CPU 版 torch 即可，本问题网络极小）
conda run -n ies_marl python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
conda run -n ies_marl python -m pip install gymnasium numpy pandas matplotlib scipy

# 验证解释器路径（应指向 .../envs/ies_marl/python.exe）
conda run -n ies_marl python -c "import sys; print(sys.executable)"
```

环境锁定文件：[`environment.yml`](environment.yml)、[`requirements.txt`](requirements.txt)。
从锁定文件重建：

```bash
conda env create -f environment.yml      # 或
conda create -n ies_marl python=3.10 -y && conda run -n ies_marl python -m pip install -r requirements.txt
```

> 下文所有命令均以 `conda run -n ies_marl python ...` 形式给出，确保使用专属环境。
> 也可先 `conda activate ies_marl` 再 `which python` 验证后直接 `python ...`。

---

## 2. 一键运行（完整复现流程）

```bash
# 阶段二：生成数据 -> 单元测试 -> 基准
conda run -n ies_marl python make_data.py
conda run -n ies_marl python test_env.py
conda run -n ies_marl python baseline.py

# 阶段三：训练 MMAPPO（主算法）+ 基准算法，绘图与对比表
conda run -n ies_marl python train_marl.py --algo mmappo --mechanism mrdac --episodes 2000 --seed 0
conda run -n ies_marl python train_marl.py --algo mippo  --mechanism mrdac --episodes 2000 --seed 0
conda run -n ies_marl python plot_results.py
```

提供便捷脚本 [`run_all.sh`](run_all.sh)（Git Bash）一键串行执行上述步骤。

---

## 3. 目录结构

```
repro_ies_marl/
├── model_spec.md          # 阶段一：数学模型提取（已确认）
├── make_data.py           # 合成数据生成（Table I 缩放，含噪预测，时变电价）
├── market.py              # MRDAC / Greedy / VDA 三种出清机制
├── env.py                 # Gymnasium 风格并行多智能体环境（硬约束投影）
├── baseline.py            # 规则竞价 / 无P2P 基准 + 机制对比参考线
├── test_env.py            # 100 episode 随机 rollout 单元测试
├── test_market.py         # 出清机制正确性测试
├── train_marl.py          # MMAPPO + MIPPO/MAPPO-one/MAPPO-s（CTDE）
├── plot_results.py        # 图1收敛 / 图2调度 / 图3机制对比 + 对比表
├── summarize_results.py   # 多种子稳态指标汇总
├── REPRODUCTION_REPORT.md # 复现报告（结果 + 论文问题诊断）
├── data/                  # profiles.npz, typical_day.csv, meta.json
├── logs/                  # 训练 CSV 日志
├── checkpoints/           # 最优/最终模型权重
├── results/               # 高分辨率 PNG/PDF 图表 + 对比表 CSV
├── environment.yml        # Conda 环境锁定
└── requirements.txt       # pip 依赖锁定
```

---

## 4. 关键实现要点（对齐 model_spec.md）

- **物理硬约束 100% 满足**：动作解码到合法区间（式9-11），储能/主网吸收残差，
  功率平衡（式8）恒成立，SoC（式1-2）与功率不越界。`test_env.py` 对 100 episode × 3 机制断言。
- **奖励**：`R_i = p_fit·q_fit − p_emg·q_e + P2P利润`（式12/13/15），纯经济利润，无显式碳惩罚。
- **观测**：`O_i = {m_t, E_i, h_i(窗口[t−1,t+6]×4), t}`（式14），LSTM 处理时序窗口，正弦编码处理 t。
- **CTDE**：N 个分布式 LSTM actor + 集中式 critic（全局状态，per-agent 价值头）。
- **数据**：论文未附原始数据集[16]，按用户授权逻辑生成合成典型曲线（`data/`），按 Table I 缩放。

## 5. 复现判据

- 定性：MRDAC 总利润 > VDA > Greedy；紧急购电最低；P2P 成交量最高（对齐 Table II）。
- 收敛：MMAPPO 优于 MIPPO/MAPPO-s（对齐 Fig.2）。
- 绝对数值因缺原始数据与全部超参，不强求逐位吻合（见 model_spec.md §9 A15）。

## 6. 主要超参数（默认）

| 参数 | 值 | 参数 | 值 |
|---|---|---|---|
| γ | 0.99 | GAE λ | 0.95 |
| PPO clip | 0.2 | K epochs | 10 |
| actor lr | 3e-4 | critic lr | 5e-4 |
| 熵系数 | 0.01 | 价值损失系数 | 0.5 |
| LSTM 隐藏 | 64 | MLP 隐藏 | 64 |
| critic 隐藏 | 128 | 梯度裁剪 | 0.5 |
| episodes | 2000 | 评估日数 | 200 |

## 7. 论文引用

本仓库仅收录论文的**引用信息**，不附带论文 PDF。原文请从 arXiv 获取：
**arXiv:2604.08973v1 [cs.MA]**，2026-04-10。

```bibtex
@misc{ren2026marl,
  title         = {Multi-agent Reinforcement Learning for Low-Carbon P2P Energy
                   Trading among Self-Interested Microgrids},
  author        = {Junhao Ren and Honglin Gao and Lan Zhao and Qiyu Kang and
                   Gaoxi Xiao and Yajuan Sun},
  year          = {2026},
  month         = apr,
  eprint        = {2604.08973},
  archivePrefix = {arXiv},
  primaryClass  = {cs.MA},
  url           = {https://arxiv.org/abs/2604.08973}
}
```

> 作者单位：Ren / Gao / Zhao / Xiao — 南洋理工大学（NTU）电子与电气工程学院；
> Kang — 中国科学技术大学信息科学技术学院；Sun — 新加坡科技研究局（A*STAR）先进制造技术研究所。