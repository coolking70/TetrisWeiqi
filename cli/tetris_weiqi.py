#!/usr/bin/env python3
"""
俄罗斯方块棋 CLI 版本
Tetris Weiqi - Command Line Interface

用途：人机对弈、AI agent 测试、训练数据生成
支持三种模式：
  1. 交互模式（人类玩家）
  2. pipe 模式（AI agent 通过 stdin/stdout 通信）
  3. self-play 模式（两个 AI 自动对弈，输出棋谱）

用法：
  python tetris_weiqi.py                    # 交互模式，人 vs AI
  python tetris_weiqi.py --mode pvp         # 交互模式，人 vs 人
  python tetris_weiqi.py --mode pipe        # pipe模式，AI agent 输入输出
  python tetris_weiqi.py --mode selfplay    # 自动对弈，输出棋谱
  python tetris_weiqi.py --size 13          # 指定棋盘大小
  python tetris_weiqi.py --seed 42          # 固定随机种子（可复现）
"""

import sys
import json
import random
import argparse
import copy
from typing import List, Tuple, Optional, Dict, Set

# ============================================================
# Constants
# ============================================================
EMPTY = 0
P1 = 1
P2 = 2
DEAD1 = 3  # 死区：封锁P1（由P2围杀产生）
DEAD2 = 4  # 死区：封锁P2（由P1围杀产生）

PIECE_SHAPES = {
    'I': [(0,0),(0,1),(0,2),(0,3)],
    'O': [(0,0),(0,1),(1,0),(1,1)],
    'T': [(0,0),(0,1),(0,2),(1,1)],
    'S': [(0,1),(0,2),(1,0),(1,1)],
    'Z': [(0,0),(0,1),(1,1),(1,2)],
    'L': [(0,0),(1,0),(2,0),(2,1)],
    'J': [(0,1),(1,1),(2,0),(2,1)],
}
PIECE_NAMES = list(PIECE_SHAPES.keys())

CELL_CHARS = {
    EMPTY: '.',
    P1:    'X',
    P2:    'O',
    DEAD1: '#',  # 封锁P1
    DEAD2: '%',  # 封锁P2
}

NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _rotate_cells(cells: List[Tuple[int, int]], rot: int) -> List[Tuple[int, int]]:
    c = list(cells)
    for _ in range(rot % 4):
        c = [(col, -row) for row, col in c]
    min_r = min(r for r, _ in c)
    min_c = min(c_ for _, c_ in c)
    return [(r - min_r, col - min_c) for r, col in c]


def _piece_bounds(cells):
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    return max_r + 1, max_c + 1


def _build_piece_rotation_cache():
    cache = {}
    for name, base_cells in PIECE_SHAPES.items():
        seen = set()
        rotations = []
        for rot in range(4):
            cells = tuple(_rotate_cells(base_cells, rot))
            if cells in seen:
                continue
            seen.add(cells)
            height, width = _piece_bounds(cells)
            rotations.append({
                'rot': rot,
                'cells': cells,
                'height': height,
                'width': width,
            })
        cache[name] = rotations
    return cache


PIECE_ROTATIONS = _build_piece_rotation_cache()


def parse_bool_flag(value: str) -> bool:
    value = value.strip().lower()
    if value in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if value in ('0', 'false', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'无效布尔值: {value}')


def parse_piece_distribution(value: str) -> str:
    value = value.strip().lower()
    if value in ('uniform', 'bag7', 'bag7_independent'):
        return value
    raise argparse.ArgumentTypeError(f'无效发牌模式: {value} (应为 uniform / bag7 / bag7_independent)')


def parse_terminal_mode(value: str) -> str:
    value = value.strip().lower()
    if value in ('pieces_only', 'pieces_then_deadzones', 'area_like'):
        return value
    raise argparse.ArgumentTypeError(
        f'无效终局判定模式: {value} (应为 pieces_only / pieces_then_deadzones / area_like)'
    )


def parse_end_condition_mode(value: str) -> str:
    value = value.strip().lower()
    if value in ('double_forced_pass', 'single_forced_pass'):
        return value
    raise argparse.ArgumentTypeError(
        f'无效终局触发模式: {value} (应为 double_forced_pass 或 single_forced_pass)'
    )


def parse_no_legal_move_mode(value: str) -> str:
    value = value.strip().lower()
    if value in ('pass_and_redraw', 'reroll_once_then_pass'):
        return value
    raise argparse.ArgumentTypeError(
        f'无效无着法处理模式: {value} (应为 pass_and_redraw 或 reroll_once_then_pass)'
    )


def parse_resolution_mode(value: str) -> str:
    value = value.strip().lower()
    if value in ('capture_then_clear_recheck', 'clear_then_capture', 'capture_then_clear_once'):
        return value
    raise argparse.ArgumentTypeError(
        '无效结算顺序模式: '
        f'{value} (应为 capture_then_clear_recheck / clear_then_capture / capture_then_clear_once)'
    )


def parse_dead_zone_activation_mode(value: str) -> str:
    value = value.strip().lower()
    if value in ('immediate', 'next_turn'):
        return value
    raise argparse.ArgumentTypeError(
        f'无效死区转化生效模式: {value} (应为 immediate 或 next_turn)'
    )


def parse_non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'无效非负整数: {value}') from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f'无效非负整数: {value}')
    return parsed


# ============================================================
# Game Engine
# ============================================================
class TetrisWeiqi:
    """完整游戏引擎，与 index.html 规则一致"""

    def __init__(self, size: int = 10, seed: Optional[int] = None,
                 dead_zone_fills_line: bool = True,
                 score_dead_zone_weight: float = 0.0,
                 piece_distribution: str = 'bag7',
                 terminal_mode: str = 'pieces_only',
                 allow_voluntary_skip: bool = False,
                 end_condition_mode: str = 'double_forced_pass',
                 no_legal_move_mode: str = 'reroll_once_then_pass',
                 resolution_mode: str = 'capture_then_clear_recheck',
                 dead_zone_activation_mode: str = 'immediate',
                 no_legal_move_rerolls: int = 1,
                 local_search: bool = True,
                 komi: float = 0.0):
        self.size = size
        self.dead_zone_fills_line = dead_zone_fills_line  # 死区是否参与消行判定
        self.score_dead_zone_weight = score_dead_zone_weight
        self.piece_distribution = piece_distribution
        self.terminal_mode = terminal_mode
        self.allow_voluntary_skip = allow_voluntary_skip
        self.end_condition_mode = end_condition_mode
        self.no_legal_move_mode = no_legal_move_mode
        self.resolution_mode = resolution_mode
        self.dead_zone_activation_mode = dead_zone_activation_mode
        self.no_legal_move_rerolls = no_legal_move_rerolls
        self.local_search = local_search
        self.komi = komi  # 贴目: 正值补偿后手(P2)，建议 0.5 的倍数以避免平局
        self.rng = random.Random(seed)
        self._piece_bag = []
        self._player_piece_bags = {P1: [], P2: []}
        self.reset()

    def reset(self):
        self.board = [[EMPTY] * self.size for _ in range(self.size)]
        self.current_player = P1
        self.pieces = {P1: None, P2: None}
        self.skip_count = 0
        self.move_number = 0
        self.game_over = False
        self.winner = None  # None=ongoing, 0=draw, P1/P2=winner
        self.history = []   # 棋谱记录
        self._generate_piece(P1)
        self._generate_piece(P2)

    # --- Piece Logic ---

    def _generate_piece(self, player: int):
        if self.piece_distribution == 'bag7':
            if not self._piece_bag:
                self._piece_bag = PIECE_NAMES[:]
                self.rng.shuffle(self._piece_bag)
            name = self._piece_bag.pop()
        elif self.piece_distribution == 'bag7_independent':
            if not self._player_piece_bags[player]:
                self._player_piece_bags[player] = PIECE_NAMES[:]
                self.rng.shuffle(self._player_piece_bags[player])
            name = self._player_piece_bags[player].pop()
        else:
            name = self.rng.choice(PIECE_NAMES)
        self.pieces[player] = {
            'name': name,
            'cells': list(PIECE_SHAPES[name]),
        }

    @staticmethod
    def rotate_cells(cells: List[Tuple[int,int]], rot: int) -> List[Tuple[int,int]]:
        return _rotate_cells(cells, rot)

    @staticmethod
    def piece_bounds(cells):
        return _piece_bounds(cells)

    # --- Placement Validation ---

    def _cell_allowed(self, r: int, c: int, player: int) -> bool:
        return self.board[r][c] == EMPTY

    def can_place(self, cells, row: int, col: int, player: int) -> bool:
        size = self.size
        board = self.board
        for dr, dc in cells:
            r, c = row + dr, col + dc
            if r < 0 or r >= size or c < 0 or c >= size:
                return False
            if board[r][c] != EMPTY:
                return False
        return True

    def is_legal_move(self, cells, row: int, col: int, player: int) -> bool:
        if not self.can_place(cells, row, col, player):
            return False
        # 模拟完整结算顺序，再检查己方是否仍为自杀
        snapshot = [r[:] for r in self.board]
        placed_positions = [] if self.local_search else None
        for dr, dc in cells:
            rr, cc = row + dr, col + dc
            if placed_positions is not None:
                placed_positions.append((rr, cc))
            self.board[rr][cc] = player
        self._resolve_placement_effects(
            player,
            allow_self_capture=False,
            placed_positions=placed_positions,
        )
        if placed_positions is not None:
            self_dead = self._has_dead_group_from_cells(player, placed_positions)
        else:
            self_dead = self._has_dead_groups(player)
        self.board = snapshot
        return not self_dead

    def can_place_anywhere(self, player: int) -> bool:
        piece = self.pieces[player]
        if not piece:
            return False
        for rotation in PIECE_ROTATIONS[piece['name']]:
            cells = rotation['cells']
            max_row = self.size - rotation['height'] + 1
            max_col = self.size - rotation['width'] + 1
            for r in range(max_row):
                for c in range(max_col):
                    if self.is_legal_move(cells, r, c, player):
                        return True
        return False

    def get_legal_moves(self, player: int) -> List[Dict]:
        """返回所有合法着法列表 [{rot, row, col, cells}]"""
        moves = []
        piece = self.pieces[player]
        if not piece:
            return moves
        for rotation in PIECE_ROTATIONS[piece['name']]:
            cells = rotation['cells']
            max_row = self.size - rotation['height'] + 1
            max_col = self.size - rotation['width'] + 1
            for r in range(max_row):
                for c in range(max_col):
                    if self.is_legal_move(cells, r, c, player):
                        moves.append({'rot': rotation['rot'], 'row': r, 'col': c, 'cells': cells})
        return moves

    # --- Capture Logic ---

    def _cell_at(self, row: int, col: int,
                 inactive_conversions: Optional[Set[Tuple[int, int]]] = None,
                 previous_board: Optional[List[List[int]]] = None) -> int:
        if inactive_conversions and previous_board is not None and (row, col) in inactive_conversions:
            return previous_board[row][col]
        return self.board[row][col]

    def _get_group(self, row: int, col: int, visited,
                   inactive_conversions: Optional[Set[Tuple[int, int]]] = None,
                   previous_board: Optional[List[List[int]]] = None):
        size = self.size
        board = self.board
        has_inactive = inactive_conversions is not None and previous_board is not None
        owner = previous_board[row][col] if has_inactive and (row, col) in inactive_conversions else board[row][col]
        if owner not in (P1, P2):
            return None
        group = []
        has_liberty = False
        stack = [(row, col)]
        visited[row * size + col] = 1

        while stack:
            r, c = stack.pop()
            group.append((r, c))
            for dr, dc in NEIGHBORS:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= size or nc < 0 or nc >= size:
                    continue
                k = nr * size + nc
                if visited[k]:
                    continue
                cell = previous_board[nr][nc] if has_inactive and (nr, nc) in inactive_conversions else board[nr][nc]
                if cell == EMPTY:
                    has_liberty = True
                elif cell == owner:
                    visited[k] = 1
                    stack.append((nr, nc))
        return {'owner': owner, 'group': group, 'has_liberty': has_liberty}

    def _capture_groups_of(self, target: int,
                           candidates=None,
                           inactive_conversions: Optional[Set[Tuple[int, int]]] = None,
                           previous_board: Optional[List[List[int]]] = None) -> int:
        size = self.size
        board = self.board
        has_inactive = inactive_conversions is not None and previous_board is not None
        visited = bytearray(size * size)
        total = 0
        if candidates is None:
            for r in range(size):
                for c in range(size):
                    k = r * size + c
                    cell = previous_board[r][c] if has_inactive and (r, c) in inactive_conversions else board[r][c]
                    if visited[k] or cell != target:
                        continue
                    g = self._get_group(r, c, visited, inactive_conversions, previous_board)
                    if g and not g['has_liberty']:
                        for gr, gc in g['group']:
                            board[gr][gc] = EMPTY
                        total += len(g['group'])
        else:
            candidate_marks = bytearray(size * size)
            for r, c in candidates:
                if 0 <= r < size and 0 <= c < size:
                    candidate_marks[r * size + c] = 1
                for dr, dc in NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        candidate_marks[nr * size + nc] = 1
            for k, marked in enumerate(candidate_marks):
                if not marked or visited[k]:
                    continue
                r = k // size
                c = k % size
                cell = previous_board[r][c] if has_inactive and (r, c) in inactive_conversions else board[r][c]
                if cell != target:
                    continue
                g = self._get_group(r, c, visited, inactive_conversions, previous_board)
                if g and not g['has_liberty']:
                    for gr, gc in g['group']:
                        board[gr][gc] = EMPTY
                    total += len(g['group'])
        return total

    def _has_dead_groups(self, player: int,
                         inactive_conversions: Optional[Set[Tuple[int, int]]] = None,
                         previous_board: Optional[List[List[int]]] = None) -> bool:
        size = self.size
        board = self.board
        has_inactive = inactive_conversions is not None and previous_board is not None
        visited = bytearray(size * size)
        for r in range(size):
            for c in range(size):
                k = r * size + c
                cell = previous_board[r][c] if has_inactive and (r, c) in inactive_conversions else board[r][c]
                if visited[k] or cell != player:
                    continue
                g = self._get_group(r, c, visited, inactive_conversions, previous_board)
                if g and not g['has_liberty']:
                    return True
        return False

    def _has_dead_group_from_cells(self, player: int, cells,
                                   inactive_conversions: Optional[Set[Tuple[int, int]]] = None,
                                   previous_board: Optional[List[List[int]]] = None) -> bool:
        size = self.size
        board = self.board
        has_inactive = inactive_conversions is not None and previous_board is not None
        visited = bytearray(size * size)

        for r, c in cells:
            cell = previous_board[r][c] if has_inactive and (r, c) in inactive_conversions else board[r][c]
            if cell != player:
                continue
            k = r * size + c
            if visited[k]:
                continue
            g = self._get_group(r, c, visited, inactive_conversions, previous_board)
            if g and not g['has_liberty']:
                return True
        return False

    # --- Line Clear ---

    def _cell_fills_line(self, cell: int) -> bool:
        """判断一个格子是否算作"填满"用于消行判定"""
        return cell != EMPTY

    def _check_line_clears(self,
                           candidates=None,
                           inactive_conversions: Optional[Set[Tuple[int, int]]] = None,
                           previous_board: Optional[List[List[int]]] = None) -> int:
        size = self.size
        board = self.board
        has_inactive = inactive_conversions is not None and previous_board is not None
        cleared = 0
        if candidates is None:
            rows = range(size)
            cols = range(size)
        else:
            row_marks = bytearray(size)
            col_marks = bytearray(size)
            for r, c in candidates:
                if 0 <= r < size:
                    row_marks[r] = 1
                if 0 <= c < size:
                    col_marks[c] = 1
            rows = (idx for idx, marked in enumerate(row_marks) if marked)
            cols = (idx for idx, marked in enumerate(col_marks) if marked)

        for r in rows:
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
        for c in cols:
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

    # --- Scoring ---

    def count_pieces(self, player: int) -> int:
        return sum(1 for r in range(self.size) for c in range(self.size)
                   if self.board[r][c] == player)

    def count_dead_zones(self, player: int) -> int:
        dead_cell = DEAD2 if player == P1 else DEAD1
        return sum(1 for r in range(self.size) for c in range(self.size)
                   if self.board[r][c] == dead_cell)

    def bag_piece_counts(self, player: Optional[int] = None) -> Dict[str, int]:
        counts = {name: 0 for name in PIECE_NAMES}
        if self.piece_distribution == 'bag7_independent':
            bag = self._player_piece_bags[player if player is not None else self.current_player]
        else:
            bag = self._piece_bag
        for name in bag:
            counts[name] += 1
        return counts

    def final_score(self, player: int) -> float:
        return self.count_pieces(player) + self.score_dead_zone_weight * self.count_dead_zones(player)

    def terminal_tuple(self, player: int) -> Tuple[float, float]:
        pieces = float(self.count_pieces(player))
        dead_zones = float(self.count_dead_zones(player))
        if self.terminal_mode == 'pieces_then_deadzones':
            return (pieces, dead_zones)
        if self.terminal_mode == 'area_like':
            return (pieces + dead_zones, 0.0)
        return (pieces, 0.0)

    # --- Game Flow ---

    def _resolve_placement_effects(self, player: int, allow_self_capture: bool = True,
                                   placed_positions=None,
                                   converted_cells: Optional[Set[Tuple[int, int]]] = None,
                                   previous_board: Optional[List[List[int]]] = None) -> Dict:
        opponent = P2 if player == P1 else P1
        captured = 0
        lines_cleared = 0

        if self.resolution_mode == 'clear_then_capture':
            lines_cleared = self._check_line_clears(placed_positions)
            captured += self._capture_groups_of(opponent, placed_positions)
            if allow_self_capture:
                self._capture_groups_of(player, placed_positions)
            return {'captured': captured, 'lines_cleared': lines_cleared}

        captured += self._capture_groups_of(opponent, placed_positions)
        if allow_self_capture:
            self._capture_groups_of(player, placed_positions)
        lines_cleared = self._check_line_clears(placed_positions)

        if self.resolution_mode == 'capture_then_clear_recheck' and lines_cleared > 0:
            captured += self._capture_groups_of(opponent, None)
            if allow_self_capture:
                self._capture_groups_of(player, None)

        return {'captured': captured, 'lines_cleared': lines_cleared}

    def place_piece(self, cells, row: int, col: int, player: int) -> Dict:
        placed_positions = [] if self.local_search else None
        for dr, dc in cells:
            rr, cc = row + dr, col + dc
            if placed_positions is not None:
                placed_positions.append((rr, cc))
            self.board[rr][cc] = player
        return self._resolve_placement_effects(
            player,
            allow_self_capture=True,
            placed_positions=placed_positions,
        )

    def do_move(self, rot: int, row: int, col: int) -> Dict:
        """执行一步着法，返回结果"""
        if self.game_over:
            return {'error': 'game is over'}

        player = self.current_player
        piece = self.pieces[player]
        cells = self.rotate_cells(piece['cells'], rot)

        if not self.is_legal_move(cells, row, col, player):
            return {'error': 'illegal move'}

        result = self.place_piece(cells, row, col, player)
        self.skip_count = 0
        self.move_number += 1

        # 记录棋谱
        self.history.append({
            'move': self.move_number,
            'player': player,
            'piece': piece['name'],
            'rot': rot,
            'row': row,
            'col': col,
            'captured': result['captured'],
            'lines_cleared': result['lines_cleared'],
        })

        self._generate_piece(player)
        self._next_turn()

        return {
            'ok': True,
            'captured': result['captured'],
            'lines_cleared': result['lines_cleared'],
        }

    def do_skip(self) -> Dict:
        if self.game_over:
            return {'error': 'game is over'}
        if not self.allow_voluntary_skip and self.can_place_anywhere(self.current_player):
            return {'error': 'voluntary skip disabled'}
        return self._handle_forced_no_move(self.current_player, reason='skip')

    def _handle_forced_no_move(self, player: int, reason: str = 'forced_pass') -> Dict:
        self.skip_count += 1
        self._generate_piece(player)
        self.history.append({
            'move': self.move_number + 1,
            'player': player,
            'action': reason,
        })
        self.move_number += 1
        if self.end_condition_mode == 'single_forced_pass' or self.skip_count >= 2:
            self._end_game()
            return {'ok': True, 'game_over': True}
        self._next_turn()
        return {'ok': True}

    def _resolve_no_legal_move(self, player: int) -> bool:
        if self.can_place_anywhere(player):
            return True

        if self.no_legal_move_mode == 'reroll_once_then_pass':
            for reroll_idx in range(self.no_legal_move_rerolls):
                self._generate_piece(player)
                self.history.append({
                    'move': self.move_number,
                    'player': player,
                    'action': 'reroll_no_move',
                    'reroll_index': reroll_idx + 1,
                })
                if self.can_place_anywhere(player):
                    return True

        self.current_player = player
        self._handle_forced_no_move(player)
        return False

    def _next_turn(self):
        self.current_player = P2 if self.current_player == P1 else P1
        self._resolve_no_legal_move(self.current_player)

    def _end_game(self):
        self.game_over = True
        s1 = self.terminal_tuple(P1)
        s2 = self.terminal_tuple(P2)
        # 贴目: komi 加到 P2 (后手) 的第一分量上
        if self.komi != 0.0:
            s2 = (s2[0] + self.komi, s2[1])
        if s1 > s2:
            self.winner = P1
        elif s2 > s1:
            self.winner = P2
        else:
            self.winner = 0

    # --- State Export (for AI) ---

    def get_state(self) -> Dict:
        """导出完整游戏状态（JSON可序列化）"""
        player = self.current_player
        piece = self.pieces[player]
        return {
            'board': [row[:] for row in self.board],
            'size': self.size,
            'current_player': player,
            'piece': piece['name'] if piece else None,
            'move_number': self.move_number,
            'skip_count': self.skip_count,
            'game_over': self.game_over,
            'winner': self.winner,
            'scores': {
                'p1_pieces': self.count_pieces(P1),
                'p2_pieces': self.count_pieces(P2),
                'p1_dead_zones': self.count_dead_zones(P1),
                'p2_dead_zones': self.count_dead_zones(P2),
                'p1_final_score': self.final_score(P1),
                'p2_final_score': self.final_score(P2),
                'bag_counts': self.bag_piece_counts(),
                'piece_distribution': self.piece_distribution,
                'terminal_mode': self.terminal_mode,
                'allow_voluntary_skip': self.allow_voluntary_skip,
                'end_condition_mode': self.end_condition_mode,
                'no_legal_move_mode': self.no_legal_move_mode,
                'resolution_mode': self.resolution_mode,
                'dead_zone_activation_mode': self.dead_zone_activation_mode,
                'no_legal_move_rerolls': self.no_legal_move_rerolls,
                'p1_terminal_tuple': self.terminal_tuple(P1),
                'p2_terminal_tuple': self.terminal_tuple(P2),
            },
            'legal_move_count': len(self.get_legal_moves(player)) if not self.game_over else 0,
        }

    def board_to_str(self) -> str:
        """棋盘的文本表示"""
        lines = []
        # 列号
        header = '   ' + ' '.join(f'{c:X}' for c in range(self.size))
        lines.append(header)
        lines.append('   ' + '-' * (self.size * 2 - 1))
        for r in range(self.size):
            row_str = ' '.join(CELL_CHARS[self.board[r][c]] for c in range(self.size))
            lines.append(f'{r:2d}|{row_str}')
        return '\n'.join(lines)

    def piece_to_str(self, player: int, rot: int = 0) -> str:
        """方块的文本表示"""
        piece = self.pieces[player]
        if not piece:
            return '(none)'
        cells = self.rotate_cells(piece['cells'], rot)
        rows, cols = self.piece_bounds(cells)
        grid = [['.' for _ in range(cols)] for _ in range(rows)]
        ch = CELL_CHARS[player]
        for r, c in cells:
            grid[r][c] = ch
        return '\n'.join(' '.join(row) for row in grid)


# ============================================================
# Simple Heuristic AI (same logic as JS version)
# ============================================================
class SimpleAI:
    def __init__(self, level: int = 2):
        self.level = level

    def choose_move(self, game: TetrisWeiqi) -> Optional[Dict]:
        player = game.current_player
        moves = game.get_legal_moves(player)
        if not moves:
            return None

        best_score = float('-inf')
        best_moves = []

        for move in moves:
            snapshot = [r[:] for r in game.board]
            for dr, dc in move['cells']:
                game.board[move['row'] + dr][move['col'] + dc] = player
            captured = game._capture_groups_of(P2 if player == P1 else P1)
            game._capture_groups_of(player)
            lines = game._check_line_clears()
            if lines > 0:
                game._capture_groups_of(P2 if player == P1 else P1)

            score = self._evaluate(game, move, player, captured, lines)
            game.board = snapshot

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return game.rng.choice(best_moves)

    def _evaluate(self, game, move, player, captured, lines_cleared):
        score = captured * 15 + lines_cleared * 3
        center = game.size / 2
        avg_r = sum(move['row'] + dr for dr, _ in move['cells']) / 4
        avg_c = sum(move['col'] + dc for _, dc in move['cells']) / 4
        score -= (abs(avg_r - center) + abs(avg_c - center)) * 0.3

        opponent = P2 if player == P1 else P1
        own_adj = enemy_adj = 0
        for dr, dc in move['cells']:
            r, c = move['row'] + dr, move['col'] + dc
            for nr, nc in [(-1,0),(1,0),(0,-1),(0,1)]:
                ar, ac = r + nr, c + nc
                if 0 <= ar < game.size and 0 <= ac < game.size:
                    if game.board[ar][ac] == player:
                        own_adj += 1
                    elif game.board[ar][ac] == opponent:
                        enemy_adj += 1
        score += own_adj * 1.5

        if self.level >= 2:
            score += enemy_adj * 2
            score += game.count_pieces(player) * 0.1

        if self.level >= 3:
            visited = set()
            for r in range(game.size):
                for c in range(game.size):
                    k = r * game.size + c
                    if k in visited or game.board[r][c] != opponent:
                        continue
                    g = game._get_group(r, c, visited)
                    if g and g['liberties'] <= 2:
                        score += (3 - g['liberties']) * 5
            score += captured * 5

        score += game.rng.random() * 0.5
        return score


# ============================================================
# Interactive Mode
# ============================================================
def interactive_mode(game: TetrisWeiqi, ai_player: Optional[int] = P2, ai_level: int = 2):
    ai = SimpleAI(ai_level) if ai_player else None

    print('=' * 50)
    print('  俄罗斯方块棋 Tetris Weiqi CLI')
    print('=' * 50)
    print(f'棋盘: {game.size}x{game.size}')
    print(f'图例: X=玩家1  O=玩家2  #=死区(封P1)  %=死区(封P2)  .=空')
    print(f'输入: rot row col (如 "0 4 5") | skip=跳过 | quit=退出')
    print()

    while not game.game_over:
        player = game.current_player
        p_name = f'P{player}({"X" if player == P1 else "O"})'

        print(game.board_to_str())
        print(f'\n回合 {game.move_number + 1} | {p_name} | '
              f'P1:{game.count_pieces(P1)} vs P2:{game.count_pieces(P2)}')
        print(f'当前方块 [{game.pieces[player]["name"]}]:')
        print(game.piece_to_str(player))
        legal = game.get_legal_moves(player)
        print(f'合法着法数: {len(legal)}')

        if ai_player == player and ai:
            move = ai.choose_move(game)
            if move is None:
                print(f'{p_name} AI 无法落子，跳过')
                game.do_skip()
            else:
                print(f'{p_name} AI 落子: rot={move["rot"]} row={move["row"]} col={move["col"]}')
                result = game.do_move(move['rot'], move['row'], move['col'])
                if result.get('captured'):
                    print(f'  围杀 {result["captured"]} 子!')
                if result.get('lines_cleared'):
                    print(f'  消除 {result["lines_cleared"]} 行/列!')
            print()
            continue

        # 人类输入
        try:
            line = input(f'{p_name}> ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print('\n游戏中断')
            break

        if line in ('quit', 'q', 'exit'):
            print('退出游戏')
            break
        elif line in ('skip', 's'):
            result = game.do_skip()
            if result.get('game_over'):
                break
            print('已跳过')
        elif line in ('moves', 'm'):
            for i, mv in enumerate(legal):
                print(f'  {i}: rot={mv["rot"]} row={mv["row"]} col={mv["col"]}')
        elif line.startswith('r'):
            # 显示指定旋转的方块
            try:
                rot = int(line[1:].strip())
                print(game.piece_to_str(player, rot))
            except ValueError:
                print('用法: r0 / r1 / r2 / r3')
        else:
            parts = line.split()
            if len(parts) == 3:
                try:
                    rot, row, col = int(parts[0]), int(parts[1]), int(parts[2])
                    result = game.do_move(rot, row, col)
                    if 'error' in result:
                        print(f'  错误: {result["error"]}')
                    else:
                        if result.get('captured'):
                            print(f'  围杀 {result["captured"]} 子!')
                        if result.get('lines_cleared'):
                            print(f'  消除 {result["lines_cleared"]} 行/列!')
                except ValueError:
                    print('格式: rot row col (如 "0 4 5")')
            else:
                print('输入 "rot row col" 落子, "skip" 跳过, "moves" 查看合法着法, "r0"-"r3" 预览旋转')
        print()

    # 游戏结束
    print(game.board_to_str())
    print('\n' + '=' * 30)
    print('游戏结束!')
    s1, s2 = game.count_pieces(P1), game.count_pieces(P2)
    print(f'P1(X): {s1} 方块 | P2(O): {s2} 方块')
    if game.winner == P1:
        print('P1(X) 获胜!')
    elif game.winner == P2:
        print('P2(O) 获胜!')
    else:
        print('平局!')


# ============================================================
# Pipe Mode (for AI agents)
# ============================================================
def pipe_mode(game: TetrisWeiqi):
    """
    JSON 行协议：
    每回合输出一行 JSON 状态，从 stdin 读一行 JSON 指令。

    输出格式: {"board":[[...]], "current_player":1, "piece":"T", ...}
    输入格式: {"action":"move", "rot":0, "row":3, "col":4}
              {"action":"skip"}
              {"action":"quit"}

    AI agent 对接示例:
      proc = subprocess.Popen(['python', 'tetris_weiqi.py', '--mode', 'pipe'],
                               stdin=PIPE, stdout=PIPE, text=True)
      state = json.loads(proc.stdout.readline())
      proc.stdin.write(json.dumps({"action":"move","rot":0,"row":4,"col":5}) + '\\n')
      proc.stdin.flush()
    """
    while not game.game_over:
        state = game.get_state()
        print(json.dumps(state, separators=(',', ':')), flush=True)

        try:
            line = input()
        except EOFError:
            break

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({'error': 'invalid JSON'}), flush=True)
            continue

        action = cmd.get('action', 'move')
        if action == 'quit':
            break
        elif action == 'skip':
            result = game.do_skip()
            print(json.dumps(result, separators=(',', ':')), flush=True)
        elif action == 'move':
            rot = cmd.get('rot', 0)
            row = cmd.get('row', 0)
            col = cmd.get('col', 0)
            result = game.do_move(rot, row, col)
            print(json.dumps(result, separators=(',', ':')), flush=True)
        elif action == 'state':
            pass  # 下一轮循环会自动输出状态
        elif action == 'legal_moves':
            moves = game.get_legal_moves(game.current_player)
            out = [{'rot': m['rot'], 'row': m['row'], 'col': m['col']} for m in moves]
            print(json.dumps({'legal_moves': out}, separators=(',', ':')), flush=True)
        else:
            print(json.dumps({'error': f'unknown action: {action}'}), flush=True)

    # 最终状态
    state = game.get_state()
    print(json.dumps(state, separators=(',', ':')), flush=True)


# ============================================================
# Self-play Mode (for training data generation)
# ============================================================
def selfplay_mode(game: TetrisWeiqi, ai_level: int = 2, num_games: int = 1,
                  output_file: Optional[str] = None):
    ai1 = SimpleAI(ai_level)
    ai2 = SimpleAI(ai_level)
    results = {'p1_wins': 0, 'p2_wins': 0, 'draws': 0, 'games': []}

    for game_idx in range(num_games):
        game.reset()
        while not game.game_over:
            player = game.current_player
            ai = ai1 if player == P1 else ai2
            move = ai.choose_move(game)
            if move is None:
                game.do_skip()
            else:
                game.do_move(move['rot'], move['row'], move['col'])

        s1, s2 = game.count_pieces(P1), game.count_pieces(P2)
        record = {
            'game_id': game_idx,
            'moves': game.move_number,
            'p1_score': s1,
            'p2_score': s2,
            'winner': game.winner,
            'history': game.history,
        }
        results['games'].append(record)

        if game.winner == P1:
            results['p1_wins'] += 1
        elif game.winner == P2:
            results['p2_wins'] += 1
        else:
            results['draws'] += 1

        if num_games <= 10 or (game_idx + 1) % 100 == 0:
            print(f'Game {game_idx + 1}/{num_games}: '
                  f'P1={s1} P2={s2} Winner={"P1" if game.winner == P1 else "P2" if game.winner == P2 else "Draw"} '
                  f'Moves={game.move_number}',
                  file=sys.stderr)

    print(f'\n总计: P1胜{results["p1_wins"]} P2胜{results["p2_wins"]} '
          f'平局{results["draws"]}',
          file=sys.stderr)

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'棋谱已保存到 {output_file}', file=sys.stderr)
    else:
        print(json.dumps(results, indent=2))


# ============================================================
# Gym-style Environment (for RL training)
# ============================================================
class TetrisWeiqiEnv:
    """
    OpenAI Gym 风格的环境接口，供 RL 训练使用。

    observation: dict with 'board' (size x size x 5 one-hot), 'piece' (7-dim one-hot)
    action: (rot, row, col) tuple 或 int index
    reward: +1 win, -1 lose, 0 ongoing, +0.1 per capture
    """

    def __init__(self, size=10, opponent='heuristic', opponent_level=2, seed=None,
                 dead_zone_fills_line: bool = True, score_dead_zone_weight: float = 0.0,
                 piece_distribution: str = 'bag7', terminal_mode: str = 'pieces_only',
                 allow_voluntary_skip: bool = False,
                 end_condition_mode: str = 'double_forced_pass',
                 no_legal_move_mode: str = 'reroll_once_then_pass',
                 resolution_mode: str = 'capture_then_clear_recheck',
                 dead_zone_activation_mode: str = 'immediate',
                 no_legal_move_rerolls: int = 1):
        self.game = TetrisWeiqi(
            size, seed,
            dead_zone_fills_line=dead_zone_fills_line,
            score_dead_zone_weight=score_dead_zone_weight,
            piece_distribution=piece_distribution,
            terminal_mode=terminal_mode,
            allow_voluntary_skip=allow_voluntary_skip,
            end_condition_mode=end_condition_mode,
            no_legal_move_mode=no_legal_move_mode,
            resolution_mode=resolution_mode,
            dead_zone_activation_mode=dead_zone_activation_mode,
            no_legal_move_rerolls=no_legal_move_rerolls
        )
        self.opponent = SimpleAI(opponent_level) if opponent == 'heuristic' else None
        self.player = P1  # 训练的玩家始终为 P1

    def reset(self):
        self.game.reset()
        return self._get_obs()

    def step(self, action):
        """action = (rot, row, col) or 'skip'"""
        if action == 'skip':
            self.game.do_skip()
        else:
            rot, row, col = action
            result = self.game.do_move(rot, row, col)
            if 'error' in result:
                # 非法着法，给负奖励
                return self._get_obs(), -0.5, True, {'error': result['error']}

        # 对手回合
        if not self.game.game_over and self.opponent:
            self._opponent_move()

        obs = self._get_obs()
        done = self.game.game_over
        reward = 0
        if done:
            if self.game.winner == self.player:
                reward = 1.0
            elif self.game.winner == 0:
                reward = 0.0
            else:
                reward = -1.0

        return obs, reward, done, {}

    def _opponent_move(self):
        move = self.opponent.choose_move(self.game)
        if move is None:
            self.game.do_skip()
        else:
            self.game.do_move(move['rot'], move['row'], move['col'])

    def _get_obs(self):
        return self.game.get_state()

    def get_legal_actions(self):
        moves = self.game.get_legal_moves(self.player)
        return [(m['rot'], m['row'], m['col']) for m in moves]


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='俄罗斯方块棋 CLI')
    parser.add_argument('--mode', choices=['pvai', 'pvp', 'pipe', 'selfplay'],
                        default='pvai', help='游戏模式')
    parser.add_argument('--size', type=int, default=10, help='棋盘大小 (默认10)')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--ai-level', type=int, default=2, choices=[1,2,3],
                        help='AI难度 1-3 (默认2)')
    parser.add_argument('--games', type=int, default=1,
                        help='selfplay模式对局数')
    parser.add_argument('--output', type=str, default=None,
                        help='selfplay模式棋谱输出文件')
    parser.add_argument('--dead-zone-fills-line', type=parse_bool_flag, default=True,
                        help='死区是否参与消行判定: true/false (默认 true)')
    parser.add_argument('--score-dead-zone-weight', type=float, default=0.0,
                        help='终局计分时死区权重，默认 0.0')
    parser.add_argument('--piece-distribution', type=parse_piece_distribution, default='bag7',
                        help='方块发牌模式: uniform / bag7 / bag7_independent (默认 bag7)')
    parser.add_argument('--terminal-mode', type=parse_terminal_mode, default='pieces_only',
                        help='终局判定: pieces_only / pieces_then_deadzones / area_like')
    parser.add_argument('--allow-voluntary-skip', type=parse_bool_flag, default=False,
                        help='是否允许玩家在仍有合法着法时主动 skip: true/false (默认 false)')
    parser.add_argument('--end-condition-mode', type=parse_end_condition_mode,
                        default='double_forced_pass',
                        help='终局触发: double_forced_pass 或 single_forced_pass')
    parser.add_argument('--no-legal-move-mode', type=parse_no_legal_move_mode,
                        default='reroll_once_then_pass',
                        help='无合法着法处理: pass_and_redraw 或 reroll_once_then_pass (默认 reroll_once_then_pass)')
    parser.add_argument('--resolution-mode', type=parse_resolution_mode,
                        default='capture_then_clear_recheck',
                        help='结算顺序: capture_then_clear_recheck / clear_then_capture / capture_then_clear_once')
    parser.add_argument('--dead-zone-activation-mode', type=parse_dead_zone_activation_mode,
                        default='immediate',
                        help='死区转化生效时序: immediate 或 next_turn')
    parser.add_argument('--no-legal-move-rerolls', type=parse_non_negative_int, default=1,
                        help='无合法着法时最多额外重抽几次，仅在 reroll_once_then_pass 模式下生效')
    parser.add_argument('--komi', type=float, default=0.0,
                        help='贴目: 正值补偿后手(P2)，建议 0.5 的倍数以避免平局 (默认 0.0)')
    args = parser.parse_args()

    game = TetrisWeiqi(
        size=args.size,
        seed=args.seed,
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
        komi=args.komi
    )

    if args.mode == 'pvai':
        interactive_mode(game, ai_player=P2, ai_level=args.ai_level)
    elif args.mode == 'pvp':
        interactive_mode(game, ai_player=None)
    elif args.mode == 'pipe':
        pipe_mode(game)
    elif args.mode == 'selfplay':
        selfplay_mode(game, ai_level=args.ai_level,
                      num_games=args.games, output_file=args.output)


if __name__ == '__main__':
    main()
