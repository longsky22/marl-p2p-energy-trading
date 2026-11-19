"""
market.py — P2P 市场出清机制（供 env.py 与 baseline.py 共用）。

实现三种机制：
  - mrdac : 多轮双向拍卖出清（本文，式(5)(6) + 第 III-C 节流程）
  - greedy: 单轮贪心双向拍卖（基准）
  - vda   : 统一出清价双向拍卖（基准，文献[17]风格）

报价约定: quote = (agent_id, price, qty)，qty>0 表示买方，qty<0 表示卖方。
返回: (per_agent, trades)
  per_agent[i] = {"qb": 买入量, "qs": 卖出量, "p2p_profit": P2P 利润}
  trades = [(seller_id, buyer_id, price, qty), ...]
"""
from __future__ import annotations
from collections import deque
import numpy as np

EPS = 1e-9


def _split(quotes):
    buyers, sellers = [], []
    for (aid, p, q) in quotes:
        if q > EPS:
            buyers.append([aid, float(p), float(q)])
        elif q < -EPS:
            sellers.append([aid, float(p), float(-q)])
    buyers.sort(key=lambda x: -x[1])   # 买方价高优先（降序）
    sellers.sort(key=lambda x: x[1])   # 卖方价低优先（升序）
    return buyers, sellers


def _init_result(quotes):
    return {aid: {"qb": 0.0, "qs": 0.0, "p2p_profit": 0.0} for (aid, _, _) in quotes}


def _record(per_agent, trades, sid, bid, price, qty):
    trades.append((sid, bid, price, qty))
    per_agent[sid]["qs"] += qty
    per_agent[sid]["p2p_profit"] += price * qty   # 卖方收入
    per_agent[bid]["qb"] += qty
    per_agent[bid]["p2p_profit"] -= price * qty   # 买方成本


def clear_mrdac(quotes, rng=None, max_iter=1000):
    """多轮双向拍卖：部分成交者剩余配额移到队尾，多轮撮合；均价结算。"""
    buyers, sellers = _split(quotes)
    per_agent = _init_result(quotes)
    trades = []
    bq, sq = deque(buyers), deque(sellers)
    it = 0
    while bq and sq and it < max_iter:
        it += 1
        s, b = sq[0], bq[0]
        max_buy = max(x[1] for x in bq)
        min_sell = min(x[1] for x in sq)
        # 移除条件：卖方价高于所有买方 / 买方价低于所有卖方
        if s[1] > max_buy + EPS:
            sq.popleft(); continue
        if b[1] < min_sell - EPS:
            bq.popleft(); continue
        if s[1] <= b[1] + EPS:
            qty = min(s[2], b[2])
            price = 0.5 * (s[1] + b[1])          # 式(5) 均价
            _record(per_agent, trades, s[0], b[0], price, qty)  # 式(6) min 量
            s[2] -= qty; b[2] -= qty
            if s[2] <= EPS:
                sq.popleft()
            else:
                sq.rotate(-1)                    # 剩余移到队尾
            if b[2] <= EPS:
                bq.popleft()
            else:
                bq.rotate(-1)
        else:
            # 理论上被上面移除条件覆盖；保险起见随机移动一方
            if rng is not None and rng.random() < 0.5:
                sq.rotate(-1)
            else:
                bq.rotate(-1)
    return per_agent, trades


def clear_greedy(quotes, rng=None, max_iter=1000):
    """单轮贪心：按优先级顺序成交，剩余留在原位继续与下一对手成交；均价结算。"""
    buyers, sellers = _split(quotes)
    per_agent = _init_result(quotes)
    trades = []
    si = bi = 0
    it = 0
    while si < len(sellers) and bi < len(buyers) and it < max_iter:
        it += 1
        s, b = sellers[si], buyers[bi]
        if s[1] > b[1] + EPS:
            break                                # 价格不兼容，贪心终止
        qty = min(s[2], b[2])
        price = 0.5 * (s[1] + b[1])
        _record(per_agent, trades, s[0], b[0], price, qty)
        s[2] -= qty; b[2] -= qty
        if s[2] <= EPS:
            si += 1
        if b[2] <= EPS:
            bi += 1
    return per_agent, trades


def clear_vda(quotes, rng=None, max_iter=1000):
    """统一出清价双向拍卖：先求边际成交量 Q* 与统一价 p*，再按优先级分配，全部按 p* 结算。"""
    buyers, sellers = _split(quotes)
    per_agent = _init_result(quotes)
    trades = []
    # 复制用于求边际
    s_copy = [list(x) for x in sellers]
    b_copy = [list(x) for x in buyers]
    si = bi = 0
    Q = 0.0
    marg_s = marg_b = None
    it = 0
    while si < len(s_copy) and bi < len(b_copy) and it < max_iter:
        it += 1
        s, b = s_copy[si], b_copy[bi]
        if s[1] > b[1] + EPS:
            break
        qty = min(s[2], b[2])
        Q += qty
        marg_s, marg_b = s[1], b[1]
        s[2] -= qty; b[2] -= qty
        if s[2] <= EPS:
            si += 1
        if b[2] <= EPS:
            bi += 1
    if Q <= EPS or marg_s is None:
        return per_agent, trades
    p_star = 0.5 * (marg_s + marg_b)             # 统一出清价
    # 按优先级分配 Q*（卖方升序、买方降序），全部按 p_star 结算
    rem_s = Q
    sell_alloc = {}
    for (aid, p, q) in sellers:
        a = min(q, rem_s)
        if a > EPS:
            sell_alloc[aid] = a
            rem_s -= a
    rem_b = Q
    buy_alloc = {}
    for (aid, p, q) in buyers:
        a = min(q, rem_b)
        if a > EPS:
            buy_alloc[aid] = a
            rem_b -= a
    # 撮合分配（仅用于记账，价格统一）
    s_items = deque([[aid, q] for aid, q in sell_alloc.items()])
    b_items = deque([[aid, q] for aid, q in buy_alloc.items()])
    while s_items and b_items:
        s, b = s_items[0], b_items[0]
        qty = min(s[1], b[1])
        _record(per_agent, trades, s[0], b[0], p_star, qty)
        s[1] -= qty; b[1] -= qty
        if s[1] <= EPS:
            s_items.popleft()
        if b[1] <= EPS:
            b_items.popleft()
    return per_agent, trades


MECHANISMS = {"mrdac": clear_mrdac, "greedy": clear_greedy, "vda": clear_vda}


def clear(quotes, mechanism="mrdac", rng=None):
    fn = MECHANISMS[mechanism]
    return fn(quotes, rng=rng)