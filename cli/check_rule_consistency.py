#!/usr/bin/env python3
"""
规则一致性校验脚本

用途：
1. 用当前优化后的引擎生成随机局面
2. 使用保守参考实现复算合法着法与落子结果
3. 确认优化没有改变规则语义

运行示例：
  python cli/check_rule_consistency.py
  python cli/check_rule_consistency.py --states 200 --move-checks 4 --seed 123
"""

import argparse
import random
from typing import Dict, List, Optional, Set, Tuple

from tetris_weiqi import (
    TetrisWeiqi, PIECE_NAMES, P1, P2, EMPTY, DEAD1, DEAD2,
    parse_bool_flag, parse_piece_distribution, parse_terminal_mode,
    parse_end_condition_mode, parse_no_legal_move_mode, parse_resolution_mode,
    parse_dead_zone_activation_mode, parse_non_negative_int,
)


class ReferenceTetrisWeiqi(TetrisWeiqi):
    """保守参考实现：始终使用整盘扫描逻辑，便于和优化版对照。"""

    def is_legal_move(self, cells, row: int, col: int, player: int) -> bool:
        if not self.can_place(cells, row, col, player):
            return False
        snapshot = [r[:] for r in self.board]
        converted_cells = set()
        for dr, dc in cells:
            rr, cc = row + dr, col + dc
            if self.board[rr][cc] in (DEAD1, DEAD2):
                converted_cells.add((rr, cc))
            self.board[rr][cc] = player
        self._resolve_placement_effects(
            player,
            allow_self_capture=False,
            placed_positions=None,
            converted_cells=converted_cells,
            previous_board=snapshot,
        )
        inactive = converted_cells if self.dead_zone_activation_mode == 'next_turn' else None
        self_dead = self._has_dead_groups(player, inactive, snapshot)
        self.board = snapshot
        return not self_dead

    def _capture_groups_of(self, target: int,
                           candidates=None,
                           inactive_conversions: Optional[Set[Tuple[int, int]]] = None,
                           previous_board: Optional[List[List[int]]] = None) -> int:
        dead_mark = DEAD1 if target == P1 else DEAD2
        size = self.size
        board = self.board
        has_inactive = inactive_conversions is not None and previous_board is not None
        visited = bytearray(size * size)
        total = 0
        for r in range(size):
            for c in range(size):
                k = r * size + c
                cell = previous_board[r][c] if has_inactive and (r, c) in inactive_conversions else board[r][c]
                if visited[k] or cell != target:
                    continue
                g = self._get_group(r, c, visited, inactive_conversions, previous_board)
                if g and not g['has_liberty']:
                    for gr, gc in g['group']:
                        board[gr][gc] = dead_mark
                    total += len(g['group'])
        return total

    def _check_line_clears(self,
                           candidates=None,
                           inactive_conversions: Optional[Set[Tuple[int, int]]] = None,
                           previous_board: Optional[List[List[int]]] = None) -> int:
        size = self.size
        board = self.board
        has_inactive = inactive_conversions is not None and previous_board is not None
        cleared = 0
        for r in range(size):
            full = True
            for c in range(size):
                cell = previous_board[r][c] if has_inactive and (r, c) in inactive_conversions else board[r][c]
                if not self._cell_fills_line(cell):
                    full = False
                    break
            if full:
                for c in range(size):
                    board[r][c] = EMPTY
                cleared += 1
        for c in range(size):
            full = True
            for r in range(size):
                cell = previous_board[r][c] if has_inactive and (r, c) in inactive_conversions else board[r][c]
                if not self._cell_fills_line(cell):
                    full = False
                    break
            if full:
                for r in range(size):
                    board[r][c] = EMPTY
                cleared += 1
        return cleared

    def _resolve_placement_effects(self, player: int, allow_self_capture: bool = True,
                                   placed_positions=None,
                                   converted_cells: Optional[Set[Tuple[int, int]]] = None,
                                   previous_board: Optional[List[List[int]]] = None) -> Dict:
        opponent = P2 if player == P1 else P1
        captured = 0
        lines_cleared = 0
        inactive = None
        prior = None
        if self.dead_zone_activation_mode == 'next_turn' and converted_cells and previous_board is not None:
            inactive = converted_cells
            prior = previous_board

        if self.resolution_mode == 'clear_then_capture':
            lines_cleared = self._check_line_clears(None, inactive, prior)
            captured += self._capture_groups_of(opponent, None, inactive, prior)
            if allow_self_capture:
                self._capture_groups_of(player, None, inactive, prior)
            return {'captured': captured, 'lines_cleared': lines_cleared}

        captured += self._capture_groups_of(opponent, None, inactive, prior)
        if allow_self_capture:
            self._capture_groups_of(player, None, inactive, prior)
        lines_cleared = self._check_line_clears(None, inactive, prior)

        if self.resolution_mode == 'capture_then_clear_recheck' and lines_cleared > 0:
            captured += self._capture_groups_of(opponent, None, inactive, prior)
            if allow_self_capture:
                self._capture_groups_of(player, None, inactive, prior)

        return {'captured': captured, 'lines_cleared': lines_cleared}


def clone_into_reference(game: TetrisWeiqi) -> ReferenceTetrisWeiqi:
    ref = ReferenceTetrisWeiqi(
        size=game.size,
        dead_zone_fills_line=game.dead_zone_fills_line,
        score_dead_zone_weight=game.score_dead_zone_weight,
        piece_distribution=game.piece_distribution,
        terminal_mode=game.terminal_mode,
        allow_voluntary_skip=game.allow_voluntary_skip,
        end_condition_mode=game.end_condition_mode,
        no_legal_move_mode=game.no_legal_move_mode,
        resolution_mode=game.resolution_mode,
        dead_zone_activation_mode=game.dead_zone_activation_mode,
        no_legal_move_rerolls=game.no_legal_move_rerolls,
    )
    ref.board = [row[:] for row in game.board]
    ref.current_player = game.current_player
    ref.pieces = {
        P1: dict(game.pieces[P1]) if game.pieces[P1] else None,
        P2: dict(game.pieces[P2]) if game.pieces[P2] else None,
    }
    ref._piece_bag = game._piece_bag[:]
    ref._player_piece_bags = {
        P1: game._player_piece_bags[P1][:],
        P2: game._player_piece_bags[P2][:],
    }
    ref.skip_count = game.skip_count
    ref.move_number = game.move_number
    ref.game_over = game.game_over
    ref.winner = game.winner
    ref.history = [dict(item) for item in game.history]
    ref.rng.setstate(game.rng.getstate())
    return ref


def move_signature(move: Dict) -> Tuple[int, int, int]:
    return (move['rot'], move['row'], move['col'])


def result_signature(game: TetrisWeiqi, result: Dict) -> Dict:
    return {
        'result': dict(result),
        'board': [row[:] for row in game.board],
        'current_player': game.current_player,
        'pieces': {
            P1: game.pieces[P1]['name'] if game.pieces[P1] else None,
            P2: game.pieces[P2]['name'] if game.pieces[P2] else None,
        },
        'skip_count': game.skip_count,
        'move_number': game.move_number,
        'game_over': game.game_over,
        'winner': game.winner,
    }


def random_playout_state(rng: random.Random, args) -> TetrisWeiqi:
    game = TetrisWeiqi(
        size=args.size,
        seed=rng.randrange(1 << 30),
        dead_zone_fills_line=args.dead_zone_fills_line,
        score_dead_zone_weight=args.score_dead_zone_weight,
        piece_distribution=args.piece_distribution,
        terminal_mode=args.terminal_mode,
        allow_voluntary_skip=args.allow_voluntary_skip,
        end_condition_mode=args.end_condition_mode,
        no_legal_move_mode=args.no_legal_move_mode,
        resolution_mode=args.resolution_mode,
        dead_zone_activation_mode=args.dead_zone_activation_mode,
        no_legal_move_rerolls=args.no_legal_move_rerolls,
    )

    steps = rng.randint(0, args.max_moves)
    for _ in range(steps):
        if game.game_over:
            break
        legal_moves = game.get_legal_moves(game.current_player)
        if legal_moves:
            move = rng.choice(legal_moves)
            result = game.do_move(move['rot'], move['row'], move['col'])
        else:
            result = game.do_skip()
        if 'error' in result:
            raise RuntimeError(f'随机对局出现错误: {result}')
    return game


def compare_state(game: TetrisWeiqi, rng: random.Random, move_checks: int):
    ref = clone_into_reference(game)
    fast_moves = sorted(move_signature(m) for m in game.get_legal_moves(game.current_player))
    ref_moves = sorted(move_signature(m) for m in ref.get_legal_moves(ref.current_player))
    if fast_moves != ref_moves:
        raise AssertionError(
            '合法着法不一致\n'
            f'fast={fast_moves}\n'
            f'ref={ref_moves}\n'
            f'current_player={game.current_player}\n'
            f'piece={game.pieces[game.current_player]}'
        )

    if not fast_moves:
        return

    sample_moves = fast_moves[:]
    rng.shuffle(sample_moves)
    sample_moves = sample_moves[:move_checks]
    for rot, row, col in sample_moves:
        fast_game = clone_into_reference(game)
        ref_game = clone_into_reference(game)
        fast_game.__class__ = TetrisWeiqi
        fast_result = fast_game.do_move(rot, row, col)
        ref_result = ref_game.do_move(rot, row, col)
        if result_signature(fast_game, fast_result) != result_signature(ref_game, ref_result):
            raise AssertionError(
                '落子结果不一致\n'
                f'move={(rot, row, col)}\n'
                f'fast={result_signature(fast_game, fast_result)}\n'
                f'ref={result_signature(ref_game, ref_result)}'
            )


def build_parser():
    p = argparse.ArgumentParser(description='校验优化版规则实现与参考实现的一致性')
    p.add_argument('--seed', type=int, default=1234, help='随机种子')
    p.add_argument('--states', type=int, default=100, help='随机局面数量')
    p.add_argument('--max-moves', type=int, default=40, help='每个随机局面的最大铺垫步数')
    p.add_argument('--move-checks', type=int, default=3, help='每个局面抽样验证几个合法落子')

    p.add_argument('--size', type=int, default=10)
    p.add_argument('--dead-zone-fills-line', type=parse_bool_flag, default=True)
    p.add_argument('--score-dead-zone-weight', type=float, default=0.0)
    p.add_argument('--piece-distribution', type=parse_piece_distribution, default='bag7')
    p.add_argument('--terminal-mode', type=parse_terminal_mode, default='pieces_only')
    p.add_argument('--allow-voluntary-skip', type=parse_bool_flag, default=False)
    p.add_argument('--end-condition-mode', type=parse_end_condition_mode, default='double_forced_pass')
    p.add_argument('--no-legal-move-mode', type=parse_no_legal_move_mode, default='reroll_once_then_pass')
    p.add_argument('--resolution-mode', type=parse_resolution_mode, default='capture_then_clear_recheck')
    p.add_argument('--dead-zone-activation-mode', type=parse_dead_zone_activation_mode, default='immediate')
    p.add_argument('--no-legal-move-rerolls', type=parse_non_negative_int, default=1)
    return p


def main():
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    checked = 0

    for idx in range(args.states):
        game = random_playout_state(rng, args)
        compare_state(game, rng, args.move_checks)
        checked += 1
        if (idx + 1) % 20 == 0 or idx + 1 == args.states:
            print(f'[OK] checked_states={idx + 1}')

    print(f'[PASS] 所有 {checked} 个随机局面均与参考实现一致')


if __name__ == '__main__':
    main()
