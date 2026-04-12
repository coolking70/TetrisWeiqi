#!/usr/bin/env python3
"""
规则合理性分析器

从大量自对弈数据中提取关键指标，诊断规则平衡性问题。

核心思路：
  好的规则 → AI学到多样化策略 → 数据分布健康
  坏的规则 → AI发现退化策略 → 数据出现异常信号

用法:
  python analyze_rules.py                           # 用启发式AI快速分析
  python analyze_rules.py --games 1000              # 更大样本
  python analyze_rules.py --model checkpoints/best.pt  # 用训练好的模型
  python analyze_rules.py --compare                 # 对比不同规则变体
"""

import sys
import json
import argparse
import random
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from tetris_weiqi import (TetrisWeiqi, SimpleAI, PIECE_NAMES, parse_bool_flag,
                          parse_piece_distribution, parse_terminal_mode,
                          parse_end_condition_mode, parse_no_legal_move_mode,
                          parse_resolution_mode, parse_dead_zone_activation_mode,
                          parse_non_negative_int,
                          EMPTY, P1, P2, DEAD1, DEAD2)


# ============================================================
# 数据收集器：每局记录详细事件
# ============================================================
@dataclass
class GameStats:
    """单局统计"""
    winner: int = 0
    total_moves: int = 0
    p1_pieces_final: int = 0
    p2_pieces_final: int = 0
    p1_score_final: float = 0.0
    p2_score_final: float = 0.0

    # 围杀相关
    p1_captures: int = 0        # P1总围杀子数
    p2_captures: int = 0
    p1_capture_events: int = 0  # P1围杀次数
    p2_capture_events: int = 0
    capture_first_move: int = 0 # 首次围杀发生在第几步

    # 死区相关
    p1_dead_zones_final: int = 0  # 终局P1制造的死区数
    p2_dead_zones_final: int = 0
    p1_dead_zone_fills: int = 0   # P1填入对方死区的次数
    p2_dead_zone_fills: int = 0

    # 消行相关
    total_line_clears: int = 0
    line_clear_events: int = 0

    # 方块使用
    piece_usage: Dict[str, int] = field(default_factory=lambda: Counter())
    piece_win_contrib: Dict[str, List] = field(default_factory=lambda: defaultdict(list))

    # 自杀禁手触发
    suicide_blocks: int = 0     # 因自杀禁手被阻止的次数

    # 跳过
    skips: int = 0

    # 节奏
    lead_changes: int = 0       # 领先交替次数
    max_lead: int = 0           # 最大领先幅度


def play_and_collect(game: TetrisWeiqi, ai1: SimpleAI, ai2: SimpleAI) -> GameStats:
    """执行一局对弈并收集详细统计"""
    stats = GameStats()
    prev_leader = 0
    capture_happened = False

    while not game.game_over:
        player = game.current_player
        ai = ai1 if player == P1 else ai2

        # 记录落子前状态
        board_before = [r[:] for r in game.board]

        # 统计可用合法着法中有多少被自杀禁手阻止
        piece = game.pieces[player]
        if piece:
            cells_r0 = game.rotate_cells(piece['cells'], 0)
            blocked = 0
            for rot in range(4):
                cells = game.rotate_cells(piece['cells'], rot)
                for r in range(game.size):
                    for c in range(game.size):
                        if game.can_place(cells, r, c, player) and not game.is_legal_move(cells, r, c, player):
                            blocked += 1
            stats.suicide_blocks += blocked

        move = ai.choose_move(game)
        if move is None:
            game.do_skip()
            stats.skips += 1
            continue

        # 检查是否落在死区上（占领转化）
        for dr, dc in move['cells']:
            r, c = move['row'] + dr, move['col'] + dc
            cell = board_before[r][c]
            if player == P1 and cell == DEAD2:
                stats.p1_dead_zone_fills += 1
            elif player == P2 and cell == DEAD1:
                stats.p2_dead_zone_fills += 1

        result = game.do_move(move['rot'], move['row'], move['col'])
        stats.total_moves += 1

        # 方块使用统计
        stats.piece_usage[piece['name']] += 1

        # 围杀统计
        if result.get('captured', 0) > 0:
            cap = result['captured']
            if player == P1:
                stats.p1_captures += cap
                stats.p1_capture_events += 1
            else:
                stats.p2_captures += cap
                stats.p2_capture_events += 1
            if not capture_happened:
                stats.capture_first_move = stats.total_moves
                capture_happened = True

        # 消行统计
        if result.get('lines_cleared', 0) > 0:
            stats.total_line_clears += result['lines_cleared']
            stats.line_clear_events += 1

        # 领先交替
        s1 = game.count_pieces(P1)
        s2 = game.count_pieces(P2)
        leader = 1 if s1 > s2 else (2 if s2 > s1 else 0)
        if leader != 0 and prev_leader != 0 and leader != prev_leader:
            stats.lead_changes += 1
        prev_leader = leader
        stats.max_lead = max(stats.max_lead, abs(s1 - s2))

    # 终局统计
    stats.winner = game.winner
    stats.p1_pieces_final = game.count_pieces(P1)
    stats.p2_pieces_final = game.count_pieces(P2)
    stats.p1_score_final = game.final_score(P1)
    stats.p2_score_final = game.final_score(P2)
    stats.p1_dead_zones_final = game.count_dead_zones(P1)
    stats.p2_dead_zones_final = game.count_dead_zones(P2)

    return stats


# ============================================================
# 分析引擎
# ============================================================
def run_analysis(num_games: int = 500, board_size: int = 10,
                 ai_level: int = 2, seed: int = 0,
                 dead_zone_fills_line: bool = True,
                 score_dead_zone_weight: float = 0.0,
                 piece_distribution: str = 'bag7',
                 terminal_mode: str = 'pieces_only',
                 allow_voluntary_skip: bool = False,
                 end_condition_mode: str = 'double_forced_pass',
                 no_legal_move_mode: str = 'reroll_once_then_pass',
                 resolution_mode: str = 'capture_then_clear_recheck',
                 dead_zone_activation_mode: str = 'immediate',
                 no_legal_move_rerolls: int = 1) -> List[GameStats]:
    """运行大量对局并收集统计"""
    all_stats = []
    t0 = time.time()

    for i in range(num_games):
        game = TetrisWeiqi(board_size, seed=seed + i,
                           dead_zone_fills_line=dead_zone_fills_line,
                           score_dead_zone_weight=score_dead_zone_weight,
                           piece_distribution=piece_distribution,
                           terminal_mode=terminal_mode,
                           allow_voluntary_skip=allow_voluntary_skip,
                           end_condition_mode=end_condition_mode,
                           no_legal_move_mode=no_legal_move_mode,
                           resolution_mode=resolution_mode,
                           dead_zone_activation_mode=dead_zone_activation_mode,
                           no_legal_move_rerolls=no_legal_move_rerolls)
        ai1 = SimpleAI(ai_level)
        ai2 = SimpleAI(ai_level)
        stats = play_and_collect(game, ai1, ai2)
        all_stats.append(stats)

        if (i + 1) % 100 == 0 or (i + 1) == num_games:
            elapsed = time.time() - t0
            print(f'\r  对弈进度: {i+1}/{num_games} ({elapsed:.1f}s)', end='', flush=True)

    print()
    return all_stats


def diagnose(all_stats: List[GameStats], board_size: int = 10):
    """分析数据并输出诊断报告"""
    n = len(all_stats)
    total_cells = board_size * board_size

    print()
    print('=' * 62)
    print('  俄罗斯方块棋 规则合理性诊断报告')
    print('=' * 62)
    print(f'  样本: {n} 局 | 棋盘: {board_size}x{board_size}')
    print()

    # ---- 1. 先后手平衡 ----
    p1w = sum(1 for s in all_stats if s.winner == P1)
    p2w = sum(1 for s in all_stats if s.winner == P2)
    draws = sum(1 for s in all_stats if s.winner == 0)
    p1_rate = p1w / n

    print('┌────────────────────────────────────────────────────────────┐')
    print('│ 1. 先后手平衡                                             │')
    print('├────────────────────────────────────────────────────────────┤')
    print(f'│  P1(先手) 胜率: {p1_rate:.1%}  ({p1w}胜 {p2w}负 {draws}平)')

    if abs(p1_rate - 0.5) < 0.05:
        print(f'│  ✅ 平衡性良好 (偏差 {abs(p1_rate-0.5):.1%})')
    elif abs(p1_rate - 0.5) < 0.10:
        side = '先手' if p1_rate > 0.5 else '后手'
        print(f'│  ⚠️  轻微{side}优势 (偏差 {abs(p1_rate-0.5):.1%})')
        print(f'│  → 建议: 可考虑贴目或后手补偿机制')
    else:
        side = '先手' if p1_rate > 0.5 else '后手'
        print(f'│  ❌ 严重{side}优势 (偏差 {abs(p1_rate-0.5):.1%})')
        print(f'│  → 建议: 需要引入先后手平衡机制 (贴目/禁手/补偿)')

    # ---- 2. 围杀机制诊断 ----
    games_with_capture = sum(1 for s in all_stats
                             if s.p1_capture_events + s.p2_capture_events > 0)
    capture_rate = games_with_capture / n
    avg_captures = sum(s.p1_captures + s.p2_captures for s in all_stats) / n
    avg_cap_events = sum(s.p1_capture_events + s.p2_capture_events for s in all_stats) / n

    # 围杀方的胜率
    capture_wins = 0
    capture_total = 0
    for s in all_stats:
        p1_cap = s.p1_capture_events
        p2_cap = s.p2_capture_events
        if p1_cap > p2_cap:
            capture_total += 1
            if s.winner == P1:
                capture_wins += 1
        elif p2_cap > p1_cap:
            capture_total += 1
            if s.winner == P2:
                capture_wins += 1
    cap_win_rate = capture_wins / capture_total if capture_total > 0 else 0

    print('├────────────────────────────────────────────────────────────┤')
    print('│ 2. 围杀机制                                               │')
    print('├────────────────────────────────────────────────────────────┤')
    print(f'│  出现围杀的对局: {capture_rate:.1%} ({games_with_capture}/{n})')
    print(f'│  场均围杀子数: {avg_captures:.1f} | 场均围杀次数: {avg_cap_events:.1f}')
    print(f'│  围杀更多方的胜率: {cap_win_rate:.1%} ({capture_total} 局有差异)')

    if capture_rate < 0.1:
        print(f'│  ❌ 围杀几乎不发生 — 机制形同虚设')
        print(f'│  → 建议: 降低围杀难度或增加围杀收益')
    elif capture_rate < 0.3:
        print(f'│  ⚠️  围杀偏少 — 可能投入产出比不足')
        print(f'│  → 建议: 考虑围杀奖励(额外行动/方块选择权)')
    else:
        print(f'│  ✅ 围杀频率健康')

    if cap_win_rate < 0.45 and capture_total > 20:
        print(f'│  ❌ 围杀多不代表赢 — 围杀投入产出比过低')
        print(f'│  → 关键问题：你的"围杀→占领→转化"循环代价太高')
    elif cap_win_rate < 0.55 and capture_total > 20:
        print(f'│  ⚠️  围杀优势不显著 — 转化效率有提升空间')
    elif capture_total > 20:
        print(f'│  ✅ 围杀策略有效，能转化为胜势')

    # ---- 3. 死区转化诊断 ----
    avg_dz = sum(s.p1_dead_zones_final + s.p2_dead_zones_final for s in all_stats) / n
    avg_fills = sum(s.p1_dead_zone_fills + s.p2_dead_zone_fills for s in all_stats) / n
    total_dz_created = sum(s.p1_captures + s.p2_captures for s in all_stats)
    total_fills = sum(s.p1_dead_zone_fills + s.p2_dead_zone_fills for s in all_stats)
    fill_rate = total_fills / total_dz_created if total_dz_created > 0 else 0

    print('├────────────────────────────────────────────────────────────┤')
    print('│ 3. 死区占领转化（核心循环诊断）                           │')
    print('├────────────────────────────────────────────────────────────┤')
    print(f'│  终局残余死区: {avg_dz:.1f}/局')
    print(f'│  死区填入次数: {avg_fills:.1f}/局')
    print(f'│  转化率(填入/产生): {fill_rate:.1%}')

    if total_dz_created == 0:
        print(f'│  (无围杀数据，跳过分析)')
    elif fill_rate < 0.1:
        print(f'│  ❌ AI几乎不填充死区 — 占领转化循环不成立')
        print(f'│  → 围杀后的死区被浪费了，AI认为填充不值得')
        print(f'│  → 建议: (A)让死区也算分 (B)降低填充成本 (C)增加填充奖励')
    elif fill_rate < 0.3:
        print(f'│  ⚠️  死区利用率偏低 — AI偶尔转化')
        print(f'│  → 建议: 可微调平衡，让占领更有吸引力')
    else:
        print(f'│  ✅ 死区转化积极，设计意图实现')

    # ---- 4. 消行机制 ----
    avg_clears = sum(s.total_line_clears for s in all_stats) / n
    clear_games = sum(1 for s in all_stats if s.line_clear_events > 0)
    clear_rate = clear_games / n

    print('├────────────────────────────────────────────────────────────┤')
    print('│ 4. 消行机制                                               │')
    print('├────────────────────────────────────────────────────────────┤')
    print(f'│  出现消行的对局: {clear_rate:.1%}')
    print(f'│  场均消行数: {avg_clears:.1f}')

    if clear_rate < 0.05:
        print(f'│  ⚠️  消行极少发生 — 10x10棋盘上填满整行/列很难')
        print(f'│  → 这在预期之内，但消行作为反制手段可能偏弱')
    elif clear_rate > 0.5:
        print(f'│  ✅ 消行频繁，俄罗斯方块机制活跃')
    else:
        print(f'│  ✅ 消行频率适中')

    # ---- 5. 对局节奏 ----
    avg_moves = sum(s.total_moves for s in all_stats) / n
    avg_lead_changes = sum(s.lead_changes for s in all_stats) / n
    avg_max_lead = sum(s.max_lead for s in all_stats) / n
    avg_margin = sum(abs(s.p1_score_final - s.p2_score_final) for s in all_stats) / n

    print('├────────────────────────────────────────────────────────────┤')
    print('│ 5. 对局节奏                                               │')
    print('├────────────────────────────────────────────────────────────┤')
    print(f'│  平均步数: {avg_moves:.1f}')
    print(f'│  平均领先交替: {avg_lead_changes:.1f} 次')
    print(f'│  平均最大领先: {avg_max_lead:.1f} 子')
    print(f'│  平均胜负差: {avg_margin:.1f} 子')

    if avg_moves < 20:
        print(f'│  ❌ 对局过短 — 可能存在速胜策略')
    elif avg_moves > 80:
        print(f'│  ⚠️  对局偏长 — 可能存在僵局')
    else:
        print(f'│  ✅ 对局长度适中')

    if avg_lead_changes < 1:
        print(f'│  ❌ 几乎无翻盘 — 先取得优势的一方稳赢，缺乏悬念')
        print(f'│  → 建议: 增加翻盘机制（消行威力/围杀收益）')
    elif avg_lead_changes < 3:
        print(f'│  ⚠️  翻盘较少 — 优势方较易滚雪球')
    else:
        print(f'│  ✅ 领先多次交替，对局有悬念')

    # ---- 6. 自杀禁手 ----
    avg_suicide = sum(s.suicide_blocks for s in all_stats) / n
    print('├────────────────────────────────────────────────────────────┤')
    print('│ 6. 自杀禁手                                               │')
    print('├────────────────────────────────────────────────────────────┤')
    print(f'│  场均触发: {avg_suicide:.1f} 次')
    if avg_suicide < 0.5:
        print(f'│  → 极少触发 — 规则存在但几乎不影响对局')
    elif avg_suicide > 10:
        print(f'│  → 频繁触发 — 可能过度限制了落子选择')
    else:
        print(f'│  ✅ 适度触发，保护玩家不犯低级错误')

    # ---- 7. 方块平衡 ----
    total_usage = Counter()
    for s in all_stats:
        total_usage.update(s.piece_usage)
    total_pieces = sum(total_usage.values())

    print('├────────────────────────────────────────────────────────────┤')
    print('│ 7. 方块出现频率（应接近均匀 14.3%）                       │')
    print('├────────────────────────────────────────────────────────────┤')
    for name in PIECE_NAMES:
        pct = total_usage[name] / total_pieces * 100 if total_pieces > 0 else 0
        bar = '█' * int(pct / 2)
        dev = abs(pct - 100/7)
        flag = '  ⚠️' if dev > 3 else ''
        print(f'│  {name}: {pct:5.1f}% {bar}{flag}')

    # ---- 8. 棋盘利用率 ----
    avg_fill = sum(s.p1_pieces_final + s.p2_pieces_final for s in all_stats) / n
    fill_pct = avg_fill / total_cells

    print('├────────────────────────────────────────────────────────────┤')
    print('│ 8. 棋盘利用率                                             │')
    print('├────────────────────────────────────────────────────────────┤')
    print(f'│  终局平均棋面方块: {avg_fill:.1f}/{total_cells} ({fill_pct:.1%})')
    if fill_pct > 0.85:
        print(f'│  → 棋盘几乎填满才结束 — 对局可能偏长偏无聊')
    elif fill_pct < 0.3:
        print(f'│  → 棋盘大量空白 — 对局可能过早结束')
    else:
        print(f'│  ✅ 利用率适中')

    # ---- 综合诊断 ----
    print('├────────────────────────────────────────────────────────────┤')
    print('│ 综合诊断                                                   │')
    print('├────────────────────────────────────────────────────────────┤')

    issues = []
    if abs(p1_rate - 0.5) > 0.10:
        issues.append('先后手不平衡')
    if capture_rate < 0.1:
        issues.append('围杀机制失效')
    if cap_win_rate < 0.45 and capture_total > 20:
        issues.append('围杀投入产出比过低')
    if fill_rate < 0.1 and total_dz_created > 0:
        issues.append('死区转化循环不成立')
    if avg_lead_changes < 1:
        issues.append('缺乏翻盘机制')
    if avg_moves < 20:
        issues.append('对局过短')

    if not issues:
        print('│  ✅ 未发现严重平衡性问题！')
        print('│  当前规则集在启发式AI层面表现健康。')
        print('│  建议用AlphaZero训练后再做深层分析。')
    else:
        print(f'│  发现 {len(issues)} 个问题:')
        for i, issue in enumerate(issues):
            print(f'│    {i+1}. {issue}')
    print('└────────────────────────────────────────────────────────────┘')

    return {
        'p1_winrate': p1_rate,
        'capture_rate': capture_rate,
        'capture_winrate': cap_win_rate,
        'dead_zone_fill_rate': fill_rate,
        'avg_moves': avg_moves,
        'avg_lead_changes': avg_lead_changes,
        'avg_line_clears': avg_clears,
        'board_fill_pct': fill_pct,
        'issues': issues,
    }


# ============================================================
# 规则变体对比
# ============================================================
def compare_dead_zone_line(num_games: int = 500, board_size: int = 10,
                           ai_level: int = 2, seed: int = 0):
    """A/B对比：死区是否参与消行判定"""
    results = {}
    for label, flag in [('A: 死区参与消行 (dead_zone_fills_line=True)', True),
                        ('B: 死区不参与消行 (dead_zone_fills_line=False)', False)]:
        print(f'\n{"="*62}')
        print(f'  {label}')
        print(f'{"="*62}')
        stats = run_analysis(num_games, board_size, ai_level, seed,
                             dead_zone_fills_line=flag)
        result = diagnose(stats, board_size)
        results[flag] = result

    # ---- 对比摘要 ----
    a, b = results[True], results[False]
    print()
    print('=' * 62)
    print('  A/B 对比摘要: 死区参与消行 vs 不参与')
    print('=' * 62)
    metrics = [
        ('先手胜率',       'p1_winrate',        '{:.1%}'),
        ('围杀出现率',     'capture_rate',      '{:.1%}'),
        ('围杀方胜率',     'capture_winrate',   '{:.1%}'),
        ('死区转化率',     'dead_zone_fill_rate','{:.1%}'),
        ('场均步数',       'avg_moves',         '{:.1f}'),
        ('领先交替次数',   'avg_lead_changes',  '{:.1f}'),
        ('场均消行数',     'avg_line_clears',   '{:.1f}'),
        ('棋盘利用率',     'board_fill_pct',    '{:.1%}'),
    ]
    print(f'  {"指标":<16} {"参与(A)":>10} {"不参与(B)":>10} {"差异":>10}')
    print(f'  {"─"*16} {"─"*10} {"─"*10} {"─"*10}')
    for label, key, fmt in metrics:
        va, vb = a[key], b[key]
        diff = vb - va
        sign = '+' if diff > 0 else ''
        print(f'  {label:<16} {fmt.format(va):>10} {fmt.format(vb):>10} {sign}{fmt.format(diff):>9}')

    print()
    print(f'  A 问题数: {len(a["issues"])}  |  B 问题数: {len(b["issues"])}')
    if a['issues']:
        print(f'  A 问题: {", ".join(a["issues"])}')
    if b['issues']:
        print(f'  B 问题: {", ".join(b["issues"])}')
    print()

    return results


def compare_variants(num_games: int = 200, ai_level: int = 2):
    """对比不同规则变体的效果"""
    print('\n比较规则变体...\n')

    variants = {
        '当前规则(仅数方块)': {'score_territory': False, 'dead_zone_one_sided': True},
        '变体A(方块+领地)': {'score_territory': True, 'dead_zone_one_sided': True},
        '变体B(死区双方封锁)': {'score_territory': False, 'dead_zone_one_sided': False},
    }

    for name, cfg in variants.items():
        print(f'--- {name} ---')
        stats = run_analysis(num_games, ai_level=ai_level)
        result = diagnose(stats)
        print()


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser(description='俄罗斯方块棋 规则合理性分析')
    p.add_argument('--games', type=int, default=500, help='分析对局数 (默认500)')
    p.add_argument('--size', type=int, default=10, help='棋盘大小')
    p.add_argument('--ai-level', type=int, default=2, help='AI难度')
    p.add_argument('--seed', type=int, default=0, help='随机种子')
    p.add_argument('--compare', action='store_true', help='对比规则变体')
    p.add_argument('--dead-zone-ab', action='store_true',
                   help='A/B对比死区是否参与消行判定')
    p.add_argument('--dead-zone-fills-line', type=parse_bool_flag, default=True,
                   help='死区是否参与消行判定: true/false (默认 true)')
    p.add_argument('--score-dead-zone-weight', type=float, default=0.0,
                   help='终局计分时死区权重，默认 0.0')
    p.add_argument('--piece-distribution', type=parse_piece_distribution, default='bag7',
                   help='方块发牌模式: uniform / bag7 / bag7_independent (默认 bag7)')
    p.add_argument('--terminal-mode', type=parse_terminal_mode, default='pieces_only',
                   help='终局判定: pieces_only / pieces_then_deadzones / area_like')
    p.add_argument('--allow-voluntary-skip', type=parse_bool_flag, default=False,
                   help='是否允许玩家在仍有合法着法时主动 skip: true/false (默认 false)')
    p.add_argument('--end-condition-mode', type=parse_end_condition_mode,
                   default='double_forced_pass',
                   help='终局触发: double_forced_pass 或 single_forced_pass')
    p.add_argument('--no-legal-move-mode', type=parse_no_legal_move_mode,
                   default='reroll_once_then_pass',
                   help='无合法着法处理: pass_and_redraw 或 reroll_once_then_pass (默认 reroll_once_then_pass)')
    p.add_argument('--resolution-mode', type=parse_resolution_mode,
                   default='capture_then_clear_recheck',
                   help='结算顺序: capture_then_clear_recheck / clear_then_capture / capture_then_clear_once')
    p.add_argument('--dead-zone-activation-mode', type=parse_dead_zone_activation_mode,
                   default='immediate',
                   help='死区转化生效时序: immediate 或 next_turn')
    p.add_argument('--no-legal-move-rerolls', type=parse_non_negative_int, default=1,
                   help='无合法着法时最多额外重抽几次，仅在 reroll_once_then_pass 模式下生效')
    p.add_argument('--json', type=str, default=None, help='输出JSON报告')
    args = p.parse_args()

    if args.dead_zone_ab:
        compare_dead_zone_line(args.games, args.size, args.ai_level, args.seed)
    elif args.compare:
        compare_variants(args.games, args.ai_level)
    else:
        stats = run_analysis(
            args.games, args.size, args.ai_level, args.seed,
            dead_zone_fills_line=args.dead_zone_fills_line,
            score_dead_zone_weight=args.score_dead_zone_weight,
            piece_distribution=args.piece_distribution,
            terminal_mode=args.terminal_mode,
            allow_voluntary_skip=args.allow_voluntary_skip,
            end_condition_mode=args.end_condition_mode,
            no_legal_move_mode=args.no_legal_move_mode,
            resolution_mode=args.resolution_mode,
            dead_zone_activation_mode=args.dead_zone_activation_mode,
            no_legal_move_rerolls=args.no_legal_move_rerolls
        )
        result = diagnose(stats, args.size)
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f'\n报告已保存到 {args.json}')


if __name__ == '__main__':
    main()
