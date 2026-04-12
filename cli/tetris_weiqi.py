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


# ============================================================
# Game Engine
# ============================================================
class TetrisWeiqi:
    """完整游戏引擎，与 index.html 规则一致"""

    def __init__(self, size: int = 10, seed: Optional[int] = None,
                 dead_zone_fills_line: bool = True):
        self.size = size
        self.dead_zone_fills_line = dead_zone_fills_line  # 死区是否参与消行判定
        self.rng = random.Random(seed)
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
        name = self.rng.choice(PIECE_NAMES)
        self.pieces[player] = {
            'name': name,
            'cells': list(PIECE_SHAPES[name]),
        }

    @staticmethod
    def rotate_cells(cells: List[Tuple[int,int]], rot: int) -> List[Tuple[int,int]]:
        c = list(cells)
        for _ in range(rot % 4):
            c = [(col, -row) for row, col in c]
        min_r = min(r for r, _ in c)
        min_c = min(c_ for _, c_ in c)
        return [(r - min_r, col - min_c) for r, col in c]

    @staticmethod
    def piece_bounds(cells):
        max_r = max(r for r, _ in cells)
        max_c = max(c for _, c in cells)
        return max_r + 1, max_c + 1

    # --- Placement Validation ---

    def _cell_allowed(self, r: int, c: int, player: int) -> bool:
        cell = self.board[r][c]
        if cell == EMPTY:
            return True
        if player == P1 and cell == DEAD2:
            return True
        if player == P2 and cell == DEAD1:
            return True
        return False

    def can_place(self, cells, row: int, col: int, player: int) -> bool:
        for dr, dc in cells:
            r, c = row + dr, col + dc
            if r < 0 or r >= self.size or c < 0 or c >= self.size:
                return False
            if not self._cell_allowed(r, c, player):
                return False
        return True

    def is_legal_move(self, cells, row: int, col: int, player: int) -> bool:
        if not self.can_place(cells, row, col, player):
            return False
        # 模拟落子 → 提对方 → 检查己方自杀
        snapshot = [r[:] for r in self.board]
        for dr, dc in cells:
            self.board[row + dr][col + dc] = player
        opponent = P2 if player == P1 else P1
        self._capture_groups_of(opponent)
        self_dead = self._has_dead_groups(player)
        self.board = snapshot
        return not self_dead

    def can_place_anywhere(self, player: int) -> bool:
        piece = self.pieces[player]
        if not piece:
            return False
        for rot in range(4):
            cells = self.rotate_cells(piece['cells'], rot)
            for r in range(self.size):
                for c in range(self.size):
                    if self.is_legal_move(cells, r, c, player):
                        return True
        return False

    def get_legal_moves(self, player: int) -> List[Dict]:
        """返回所有合法着法列表 [{rot, row, col, cells}]"""
        moves = []
        piece = self.pieces[player]
        if not piece:
            return moves
        seen = set()
        for rot in range(4):
            cells = self.rotate_cells(piece['cells'], rot)
            cells_key = tuple(sorted(cells))
            if cells_key in seen:
                continue
            seen.add(cells_key)
            for r in range(self.size):
                for c in range(self.size):
                    if self.is_legal_move(cells, r, c, player):
                        moves.append({'rot': rot, 'row': r, 'col': c, 'cells': cells})
        return moves

    # --- Capture Logic ---

    def _get_group(self, row: int, col: int, visited: Set[int]):
        owner = self.board[row][col]
        if owner not in (P1, P2):
            return None
        group = []
        liberties = set()
        stack = [(row, col)]
        visited.add(row * self.size + col)

        while stack:
            r, c = stack.pop()
            group.append((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size:
                    continue
                k = nr * self.size + nc
                if k in visited:
                    continue
                cell = self.board[nr][nc]
                if cell == EMPTY:
                    liberties.add(k)
                elif cell == owner:
                    visited.add(k)
                    stack.append((nr, nc))
        return {'owner': owner, 'group': group, 'liberties': len(liberties)}

    def _capture_groups_of(self, target: int) -> int:
        dead_mark = DEAD1 if target == P1 else DEAD2
        visited = set()
        total = 0
        for r in range(self.size):
            for c in range(self.size):
                k = r * self.size + c
                if k in visited or self.board[r][c] != target:
                    continue
                g = self._get_group(r, c, visited)
                if g and g['liberties'] == 0:
                    for gr, gc in g['group']:
                        self.board[gr][gc] = dead_mark
                    total += len(g['group'])
        return total

    def _has_dead_groups(self, player: int) -> bool:
        visited = set()
        for r in range(self.size):
            for c in range(self.size):
                k = r * self.size + c
                if k in visited or self.board[r][c] != player:
                    continue
                g = self._get_group(r, c, visited)
                if g and g['liberties'] == 0:
                    return True
        return False

    # --- Line Clear ---

    def _cell_fills_line(self, cell: int) -> bool:
        """判断一个格子是否算作"填满"用于消行判定"""
        if cell == EMPTY:
            return False
        if cell in (P1, P2):
            return True
        # 死区 (DEAD1, DEAD2): 取决于规则配置
        return self.dead_zone_fills_line

    def _check_line_clears(self) -> int:
        cleared = 0
        for r in range(self.size):
            if all(self._cell_fills_line(self.board[r][c]) for c in range(self.size)):
                for c in range(self.size):
                    self.board[r][c] = EMPTY
                cleared += 1
        for c in range(self.size):
            if all(self._cell_fills_line(self.board[r][c]) for r in range(self.size)):
                for r in range(self.size):
                    self.board[r][c] = EMPTY
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

    # --- Game Flow ---

    def place_piece(self, cells, row: int, col: int, player: int) -> Dict:
        for dr, dc in cells:
            self.board[row + dr][col + dc] = player
        captured = self._capture_groups_of(P2 if player == P1 else P1)
        self._capture_groups_of(player)  # 安全检查
        lines_cleared = self._check_line_clears()
        if lines_cleared > 0:
            self._capture_groups_of(P2 if player == P1 else P1)
            self._capture_groups_of(player)
        return {'captured': captured, 'lines_cleared': lines_cleared}

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
        self.skip_count += 1
        player = self.current_player
        self._generate_piece(player)
        self.history.append({
            'move': self.move_number + 1,
            'player': player,
            'action': 'skip',
        })
        self.move_number += 1
        if self.skip_count >= 2:
            self._end_game()
            return {'ok': True, 'game_over': True}
        self._next_turn()
        return {'ok': True}

    def _next_turn(self):
        self.current_player = P2 if self.current_player == P1 else P1
        if not self.can_place_anywhere(self.current_player):
            self._generate_piece(self.current_player)
            if not self.can_place_anywhere(self.current_player):
                self._end_game()

    def _end_game(self):
        self.game_over = True
        s1 = self.count_pieces(P1)
        s2 = self.count_pieces(P2)
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

    def __init__(self, size=10, opponent='heuristic', opponent_level=2, seed=None):
        self.game = TetrisWeiqi(size, seed)
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
    args = parser.parse_args()

    game = TetrisWeiqi(size=args.size, seed=args.seed)

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
