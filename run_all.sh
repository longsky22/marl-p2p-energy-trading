#!/usr/bin/env bash
# 一键复现脚本（Git Bash）。所有 Python 命令均在专属 Conda 环境 ies_marl 内运行。
set -e
cd "$(dirname "$0")"
RUN="conda run -n ies_marl python"

echo "==== [0] verify conda env ===="
conda run -n ies_marl python -c "import sys; print('PYTHON:', sys.executable)"

echo "==== [1] generate data ===="
$RUN make_data.py

echo "==== [2] unit tests ===="
$RUN test_env.py
$RUN test_market.py

echo "==== [3] baselines ===="
$RUN baseline.py

echo "==== [4] train 4 algorithms under MRDAC (Fig.2) ===="
$RUN train_marl.py --algo mmappo    --mechanism mrdac --episodes 3000 --seed 1
$RUN train_marl.py --algo mippo     --mechanism mrdac --episodes 3000 --seed 1
$RUN train_marl.py --algo mappo_one --mechanism mrdac --episodes 3000 --seed 1
$RUN train_marl.py --algo mappo_s   --mechanism mrdac --episodes 3000 --seed 1

echo "==== [5] train MMAPPO under Greedy/VDA (Table II) ===="
$RUN train_marl.py --algo mmappo --mechanism greedy --episodes 2000 --seed 0
$RUN train_marl.py --algo mmappo --mechanism vda    --episodes 2000 --seed 0

echo "==== [6] plots + comparison tables ===="
$RUN plot_results.py

echo "==== DONE. See results/ and REPRODUCTION_REPORT.md ===="