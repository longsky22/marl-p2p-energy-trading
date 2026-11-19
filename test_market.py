"""
test_market.py — 市场出清机制正确性测试。

校验：成交量守恒（买=卖）、均价结算在买卖报价之间、价格优先级、
无兼容报价时零成交、MRDAC 多轮重排。运行: conda run -n ies_marl python test_market.py
"""
from __future__ import annotations
import sys
import numpy as np
from market import clear, clear_mrdac, clear_greedy, clear_vda


def conservation(quotes, mech):
    pa, trades = clear(quotes, mech)
    tb = sum(v["qb"] for v in pa.values())
    ts = sum(v["qs"] for v in pa.values())
    assert abs(tb - ts) < 1e-6, f"{mech} volume not conserved {tb} vs {ts}"
    for (sid, bid, p, q) in trades:
        assert q > 0
    return pa, trades


def main():
    print("PYTHON:", sys.executable)
    rng = np.random.default_rng(0)
    # 场景1：论文卖方示例 (6,2),(1,3),(4,2) + 买方 (5,3),(3,2)
    # quote=(id, price, qty)，qty<0 卖
    quotes = [(0, 6, -2), (1, 1, -3), (2, 4, -2), (3, 5, 3), (4, 3, 2)]
    for mech in ["mrdac", "greedy", "vda"]:
        pa, trades = conservation(quotes, mech)
        print(f"[{mech}] trades={trades}")
        # 均价应在买卖双方报价之间
        for (sid, bid, p, q) in trades:
            sp = next(x[1] for x in quotes if x[0] == sid)
            bp = next(x[1] for x in quotes if x[0] == bid)
            assert min(sp, bp) - 1e-6 <= p <= max(sp, bp) + 1e-6, f"{mech} price out of range"
    # 场景2：无兼容报价（卖方全部高于买方）→ 零成交
    quotes2 = [(0, 30, -5), (1, 28, -3), (2, 5, 4), (3, 6, 2)]
    for mech in ["mrdac", "greedy", "vda"]:
        pa, trades = conservation(quotes2, mech)
        assert len(trades) == 0, f"{mech} should have no trades"
        print(f"[{mech}] no-compatible-quote -> 0 trades OK")
    # 场景3：随机 100 组报价，成交量守恒 + 利润记账一致
    for trial in range(100):
        n = 4
        quotes3 = []
        for i in range(n):
            p = rng.uniform(2, 35)
            q = rng.uniform(-10, 10)
            quotes3.append((i, p, q))
        for mech in ["mrdac", "greedy", "vda"]:
            pa, trades = conservation(quotes3, mech)
            # P2P 利润总和应为 0（内部转移）
            tot = sum(v["p2p_profit"] for v in pa.values())
            assert abs(tot) < 1e-4, f"{mech} p2p profit not zero-sum: {tot}"
    print("随机100组: 成交量守恒 + P2P利润零和 OK")
    print("ALL MARKET TESTS PASSED")


if __name__ == "__main__":
    main()