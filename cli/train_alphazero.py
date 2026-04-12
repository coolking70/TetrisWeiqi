#!/usr/bin/env python3
"""
俄罗斯方块棋 AlphaZero-Lite 训练

架构: ResNet(6层) + MCTS
训练循环: 自对弈生成数据 → 训练网络 → 评估 → 循环

用法:
  python train_alphazero.py                     # 默认训练
  python train_alphazero.py --iterations 50     # 50轮迭代
  python train_alphazero.py --resume ckpt.pt    # 从检查点恢复
  python train_alphazero.py --eval best.pt      # 评估模型
"""

import os
import sys
import time
import math
import random
import argparse
import json
import itertools
from collections import deque
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tetris_weiqi import (
    TetrisWeiqi, SimpleAI, PIECE_SHAPES, PIECE_NAMES,
    parse_bool_flag, parse_piece_distribution, parse_terminal_mode,
    parse_end_condition_mode, parse_no_legal_move_mode, parse_resolution_mode,
    parse_dead_zone_activation_mode, parse_non_negative_int
)
from tetris_weiqi import EMPTY, P1, P2, DEAD1, DEAD2

# ============================================================
# Device
# ============================================================
def get_device(device_arg: Optional[str] = None):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def configure_torch_runtime(device: torch.device):
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

def get_autocast_dtype(device: torch.device):
    if device.type == 'cuda':
        return torch.float16
    if device.type == 'mps':
        return torch.float16
    if device.type == 'cpu':
        return torch.bfloat16
    return None

def autocast_context(device: torch.device, enabled: bool):
    dtype = get_autocast_dtype(device)
    if not enabled or dtype is None:
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type=device.type, dtype=dtype)

def maybe_compile_model(model: nn.Module, enabled: bool):
    if not enabled:
        return model, False
    if not hasattr(torch, 'compile'):
        print('[Compile] torch.compile 不可用，跳过')
        return model, False
    try:
        dynamo = __import__('torch._dynamo', fromlist=['config'])
        dynamo.config.suppress_errors = True
        compiled = torch.compile(model)
        return compiled, True
    except Exception as exc:
        print(f'[Compile] 启用失败，回退到 eager: {exc}')
        return model, False

DEVICE = get_device()
configure_torch_runtime(DEVICE)
print(f'[Device] {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'[CUDA] {torch.cuda.get_device_name(0)} | torch.cuda={torch.version.cuda}')

# ============================================================
# Neural Network: ResNet Policy-Value Network
# ============================================================
BOARD_SIZE = 10
NUM_CHANNELS = 5   # EMPTY, P1, P2, DEAD1, DEAD2
PIECE_DIM = 7      # 7种方块
BAG_DIM = 7        # 7-bag 剩余计数
RULE_DIM = 1       # 当前是否为 bag7 模式
INPUT_CHANNELS = NUM_CHANNELS + PIECE_DIM + BAG_DIM + RULE_DIM + 2
# = 5 + 7 + 7 + 1 + 2 = 22 input channels

# Action space: 4 rotations × BOARD_SIZE × BOARD_SIZE
ACTION_SIZE = 4 * BOARD_SIZE * BOARD_SIZE  # = 400

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x

class PolicyValueNet(nn.Module):
    """
    输入: (batch, 14, 10, 10) 棋盘状态编码
    输出: policy (batch, 400) 走法概率, value (batch, 1) 胜率
    """
    def __init__(self, board_size=10, num_res_blocks=6, channels=128):
        super().__init__()
        self.board_size = board_size
        self.action_size = 4 * board_size * board_size

        # Input conv
        self.input_conv = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, self.action_size),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


# ============================================================
# State Encoding
# ============================================================
def encode_state(game: TetrisWeiqi, player: int) -> np.ndarray:
    """
    将游戏状态编码为神经网络输入张量 (22, size, size)

    Channels 0-4: board one-hot (EMPTY, P1, P2, DEAD1, DEAD2)
    Channels 5-11: current piece one-hot (I,O,T,S,Z,L,J) 全平面
    Channels 12-18: bag 剩余方块计数（归一化到 [0,1]）
    Channel 19: 当前发牌模式是否为 bag7
    Channel 20: current player plane (全1=P1, 全0=P2)
    Channel 21: opponent player plane (反过来)
    """
    size = game.size
    state = np.zeros((INPUT_CHANNELS, size, size), dtype=np.float32)

    # Board one-hot
    for r in range(size):
        for c in range(size):
            cell = game.board[r][c]
            # 从当前玩家视角编码：己方=P1通道，对方=P2通道
            if cell == player:
                state[1, r, c] = 1  # 己方
            elif cell == (P2 if player == P1 else P1):
                state[2, r, c] = 1  # 对方
            elif cell == DEAD1:
                state[3, r, c] = 1
            elif cell == DEAD2:
                state[4, r, c] = 1
            else:
                state[0, r, c] = 1  # EMPTY

    # Piece one-hot (全平面)
    piece = game.pieces[player]
    if piece:
        idx = PIECE_NAMES.index(piece['name'])
        state[5 + idx, :, :] = 1

    # Bag remaining counts. In bag7 mode these planes make the state Markov.
    bag_counts = game.bag_piece_counts(player)
    bag_norm = 7.0
    for idx, name in enumerate(PIECE_NAMES):
        state[12 + idx, :, :] = bag_counts[name] / bag_norm

    # Distinguish bag-based distributions from uniform, since zero bag counts mean different things.
    state[19, :, :] = 1 if game.piece_distribution in ('bag7', 'bag7_independent') else 0

    # Player indicator
    state[20, :, :] = 1 if player == P1 else 0
    state[21, :, :] = 0 if player == P1 else 1

    return state


def action_to_index(rot: int, row: int, col: int, size: int = BOARD_SIZE) -> int:
    return rot * size * size + row * size + col

def index_to_action(idx: int, size: int = BOARD_SIZE) -> Tuple[int, int, int]:
    rot = idx // (size * size)
    remainder = idx % (size * size)
    row = remainder // size
    col = remainder % size
    return rot, row, col


def apply_move_or_forced_skip(game: TetrisWeiqi, rot: int, row: int, col: int) -> Dict:
    """优先执行落子；若失败则在无合法步时被动 skip，否则回退到第一个合法着法。"""
    result = game.do_move(rot, row, col)
    if 'error' not in result:
        return result

    legal_moves = game.get_legal_moves(game.current_player)
    if legal_moves:
        fallback = legal_moves[0]
        return game.do_move(fallback['rot'], fallback['row'], fallback['col'])
    return game.do_skip()


def resolve_zero_policy_turn(game: TetrisWeiqi) -> Dict:
    """策略全零时避免无意义主动 skip。"""
    legal_moves = game.get_legal_moves(game.current_player)
    if legal_moves:
        fallback = legal_moves[0]
        return game.do_move(fallback['rot'], fallback['row'], fallback['col'])
    return game.do_skip()


# ============================================================
# MCTS
# ============================================================
class MCTSNode:
    __slots__ = ['parent', 'action', 'prior', 'visit_count', 'value_sum', 'children', 'is_expanded']

    def __init__(self, parent=None, action=None, prior=0.0):
        self.parent = parent
        self.action = action  # (rot, row, col)
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = []
        self.is_expanded = False

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct=1.5):
        if self.parent is None:
            return 0
        u = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value + u

    def best_child(self):
        return max(self.children, key=lambda c: c.ucb_score())

    def select_leaf(self):
        node = self
        while node.is_expanded and node.children:
            node = node.best_child()
        return node

    def backpropagate(self, value):
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # 对手视角翻转
            node = node.parent


class MCTS:
    def __init__(self, model: PolicyValueNet, num_simulations=50, c_puct=1.5,
                 temperature=1.0, device=DEVICE, use_amp=True,
                 inference_batch_size=1):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = device
        self.use_amp = use_amp and device.type in ('cuda', 'mps')
        self.inference_batch_size = max(1, inference_batch_size)

    @torch.no_grad()
    def _predict(self, game: TetrisWeiqi, player: int):
        state = encode_state(game, player)
        tensor = torch.from_numpy(state).unsqueeze(0).to(
            self.device, non_blocking=(self.device.type == 'cuda')
        )
        with autocast_context(self.device, self.use_amp):
            policy_logits, value = self.model(tensor)
        policy = F.softmax(policy_logits.squeeze(0), dim=0).cpu().numpy()
        value = value.item()
        return policy, value

    @torch.no_grad()
    def _predict_batch(self, games: List[TetrisWeiqi], players: List[int]):
        states = np.stack([encode_state(game, player) for game, player in zip(games, players)])
        tensor = torch.from_numpy(states).to(
            self.device, non_blocking=(self.device.type == 'cuda')
        )
        with autocast_context(self.device, self.use_amp):
            policy_logits, values = self.model(tensor)
        policies = F.softmax(policy_logits, dim=1).cpu().numpy()
        values = values.squeeze(1).detach().cpu().numpy()
        return policies, values

    def _expand_with_prediction(self, node: MCTSNode, game: TetrisWeiqi, policy, value):
        player = game.current_player
        legal_moves = game.get_legal_moves(player)

        if not legal_moves:
            node.is_expanded = True
            return -value  # 无子可下，大概率要输

        # 只保留合法着法的先验概率
        total_prior = 0
        for move in legal_moves:
            idx = action_to_index(move['rot'], move['row'], move['col'], game.size)
            total_prior += policy[idx]

        for move in legal_moves:
            idx = action_to_index(move['rot'], move['row'], move['col'], game.size)
            prior = policy[idx] / (total_prior + 1e-8)
            child = MCTSNode(parent=node, action=(move['rot'], move['row'], move['col']), prior=prior)
            node.children.append(child)

        node.is_expanded = True
        return value

    def _expand(self, node: MCTSNode, game: TetrisWeiqi):
        player = game.current_player
        policy, value = self._predict(game, player)
        return self._expand_with_prediction(node, game, policy, value)

    def _clone_game(self, game: TetrisWeiqi) -> TetrisWeiqi:
        sim_game = TetrisWeiqi(
            game.size,
            dead_zone_fills_line=game.dead_zone_fills_line,
            score_dead_zone_weight=game.score_dead_zone_weight,
            piece_distribution=game.piece_distribution,
            terminal_mode=game.terminal_mode,
            allow_voluntary_skip=game.allow_voluntary_skip,
            end_condition_mode=game.end_condition_mode,
            no_legal_move_mode=game.no_legal_move_mode,
            resolution_mode=game.resolution_mode,
            dead_zone_activation_mode=game.dead_zone_activation_mode,
            no_legal_move_rerolls=game.no_legal_move_rerolls
        )
        sim_game.board = [r[:] for r in game.board]
        sim_game.current_player = game.current_player
        sim_game.pieces = {
            P1: dict(game.pieces[P1]) if game.pieces[P1] else None,
            P2: dict(game.pieces[P2]) if game.pieces[P2] else None,
        }
        sim_game._piece_bag = game._piece_bag[:]
        sim_game._player_piece_bags = {
            P1: game._player_piece_bags[P1][:],
            P2: game._player_piece_bags[P2][:],
        }
        sim_game.skip_count = game.skip_count
        sim_game.move_number = game.move_number
        sim_game.game_over = game.game_over
        sim_game.winner = game.winner
        sim_game.rng = random.Random()
        return sim_game

    def _prepare_leaf_request(self, root: MCTSNode, game: TetrisWeiqi):
        node = root.select_leaf()
        sim_game = self._clone_game(game)

        path = []
        n = node
        while n.parent is not None:
            path.append(n.action)
            n = n.parent
        path.reverse()

        for action in path:
            if sim_game.game_over:
                break
            rot, row, col = action
            sim_game.do_move(rot, row, col)

        if sim_game.game_over:
            if sim_game.winner == game.current_player:
                value = 1.0
            elif sim_game.winner == 0:
                value = 0.0
            else:
                value = -1.0
            if len(path) % 2 == 1:
                value = -value
            return {
                'terminal': True,
                'node': node,
                'value': value,
            }

        return {
            'terminal': False,
            'node': node,
            'sim_game': sim_game,
        }

    def search(self, game: TetrisWeiqi) -> Tuple[np.ndarray, float]:
        """
        返回: (action_probs, root_value)
        action_probs: shape (ACTION_SIZE,) 归一化概率
        """
        root = MCTSNode()

        # Expand root
        value = self._expand(root, game)
        root.backpropagate(value)

        remaining = self.num_simulations - 1
        while remaining > 0:
            batch_cap = min(remaining, self.inference_batch_size)
            pending = []
            seen_nodes = set()
            attempts = 0

            while len(pending) < batch_cap and attempts < batch_cap * 4:
                attempts += 1
                request = self._prepare_leaf_request(root, game)
                node_key = id(request['node'])
                if node_key in seen_nodes:
                    continue
                seen_nodes.add(node_key)
                pending.append(request)

            if not pending:
                break

            expand_requests = []
            for request in pending:
                if request['terminal']:
                    request['node'].backpropagate(request['value'])
                else:
                    expand_requests.append((request['node'], request['sim_game']))

            if expand_requests:
                batch_games = [sim_game for _, sim_game in expand_requests]
                batch_players = [sim_game.current_player for _, sim_game in expand_requests]
                batch_policies, batch_values = self._predict_batch(batch_games, batch_players)

                for i, (node, sim_game) in enumerate(expand_requests):
                    value = self._expand_with_prediction(
                        node, sim_game, batch_policies[i], float(batch_values[i])
                    )
                    node.backpropagate(value)

            remaining -= len(pending)

        # 从根节点提取动作概率
        action_probs = np.zeros(ACTION_SIZE, dtype=np.float32)
        for child in root.children:
            idx = action_to_index(child.action[0], child.action[1], child.action[2], game.size)
            if self.temperature < 0.01:
                action_probs[idx] = child.visit_count
            else:
                action_probs[idx] = child.visit_count ** (1.0 / self.temperature)

        total = action_probs.sum()
        if total > 0:
            action_probs /= total

        return action_probs, root.q_value

    def search_many(self, games: List[TetrisWeiqi],
                    temperatures: Optional[List[float]] = None) -> List[Tuple[np.ndarray, float]]:
        if not games:
            return []

        roots = [MCTSNode() for _ in games]
        root_games = list(games)

        root_players = [game.current_player for game in root_games]
        root_policies, root_values = self._predict_batch(root_games, root_players)
        for i, root in enumerate(roots):
            value = self._expand_with_prediction(
                root, root_games[i], root_policies[i], float(root_values[i])
            )
            root.backpropagate(value)

        remaining = [self.num_simulations - 1 for _ in root_games]
        while any(r > 0 for r in remaining):
            pending = []
            for game_idx, root in enumerate(roots):
                if remaining[game_idx] <= 0:
                    continue
                request = self._prepare_leaf_request(root, root_games[game_idx])
                request['game_idx'] = game_idx
                pending.append(request)
                remaining[game_idx] -= 1
                if len(pending) >= self.inference_batch_size:
                    break

            if not pending:
                break

            expand_requests = []
            for request in pending:
                if request['terminal']:
                    request['node'].backpropagate(request['value'])
                else:
                    expand_requests.append(request)

            if expand_requests:
                batch_games = [req['sim_game'] for req in expand_requests]
                batch_players = [req['sim_game'].current_player for req in expand_requests]
                batch_policies, batch_values = self._predict_batch(batch_games, batch_players)
                for i, request in enumerate(expand_requests):
                    value = self._expand_with_prediction(
                        request['node'], request['sim_game'],
                        batch_policies[i], float(batch_values[i])
                    )
                    request['node'].backpropagate(value)

        results = []
        for i, root in enumerate(roots):
            temp = self.temperature if temperatures is None else temperatures[i]
            action_probs = np.zeros(ACTION_SIZE, dtype=np.float32)
            for child in root.children:
                idx = action_to_index(child.action[0], child.action[1], child.action[2], root_games[i].size)
                if temp < 0.01:
                    action_probs[idx] = child.visit_count
                else:
                    action_probs[idx] = child.visit_count ** (1.0 / temp)

            total = action_probs.sum()
            if total > 0:
                action_probs /= total
            results.append((action_probs, root.q_value))

        return results


# ============================================================
# Self-Play Data Generation
# ============================================================
def self_play_game(model: PolicyValueNet, num_simulations=50, temperature=1.0,
                   temp_threshold=20, device=DEVICE, use_amp=True,
                   inference_batch_size=1,
                   dead_zone_fills_line=True,
                   score_dead_zone_weight=0.0,
                   piece_distribution='bag7',
                   terminal_mode='pieces_only',
                   allow_voluntary_skip=False,
                   end_condition_mode='double_forced_pass',
                   no_legal_move_mode='reroll_once_then_pass',
                   resolution_mode='capture_then_clear_recheck',
                   dead_zone_activation_mode='immediate',
                   no_legal_move_rerolls=1) -> List[Tuple]:
    """
    用 MCTS + 神经网络进行一局自对弈，返回训练数据。

    返回: [(state, action_probs, winner_value), ...]
    """
    game = TetrisWeiqi(
        BOARD_SIZE,
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
    mcts = MCTS(
        model, num_simulations=num_simulations, device=device, use_amp=use_amp,
        inference_batch_size=inference_batch_size
    )
    data = []

    while not game.game_over:
        player = game.current_player
        state = encode_state(game, player)

        # 前期高温探索，后期低温贪心
        temp = temperature if game.move_number < temp_threshold else 0.1
        mcts.temperature = temp

        action_probs, _ = mcts.search(game)

        data.append((state, action_probs, player))

        # 按概率采样动作
        action_idx = np.random.choice(ACTION_SIZE, p=action_probs)
        rot, row, col = index_to_action(action_idx, game.size)

        apply_move_or_forced_skip(game, rot, row, col)

    # 标注胜负
    training_data = []
    for state, probs, player in data:
        if game.winner == player:
            value = 1.0
        elif game.winner == 0:
            value = 0.0
        else:
            value = -1.0
        training_data.append((state, probs, value))

    return training_data


def self_play_games_parallel(model: PolicyValueNet, num_games=1, num_simulations=50,
                             temperature=1.0, temp_threshold=20, device=DEVICE,
                             use_amp=True, inference_batch_size=16,
                             dead_zone_fills_line=True,
                             score_dead_zone_weight=0.0,
                             piece_distribution='bag7',
                             terminal_mode='pieces_only',
                             allow_voluntary_skip=False,
                             end_condition_mode='double_forced_pass',
                             no_legal_move_mode='reroll_once_then_pass',
                             resolution_mode='capture_then_clear_recheck',
                             dead_zone_activation_mode='immediate',
                             no_legal_move_rerolls=1) -> List[Tuple]:
    """
    并行推进多局自对弈，并将不同对局中的 MCTS 推理合并批量送入 GPU。
    """
    if num_games <= 0:
        return []

    mcts = MCTS(
        model, num_simulations=num_simulations, device=device, use_amp=use_amp,
        inference_batch_size=inference_batch_size
    )

    games = [
        TetrisWeiqi(
            BOARD_SIZE,
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
        for _ in range(num_games)
    ]
    histories = [[] for _ in range(num_games)]
    active_indices = list(range(num_games))

    while active_indices:
        active_games = [games[i] for i in active_indices]
        temperatures = [
            temperature if game.move_number < temp_threshold else 0.1
            for game in active_games
        ]
        batch_results = mcts.search_many(active_games, temperatures=temperatures)

        finished = []
        for local_idx, game_idx in enumerate(active_indices):
            game = games[game_idx]
            player = game.current_player
            state = encode_state(game, player)
            action_probs, _ = batch_results[local_idx]
            histories[game_idx].append((state, action_probs, player))

            if action_probs.sum() <= 0:
                resolve_zero_policy_turn(game)
            else:
                action_idx = np.random.choice(ACTION_SIZE, p=action_probs)
                rot, row, col = index_to_action(action_idx, game.size)
                apply_move_or_forced_skip(game, rot, row, col)

            if game.game_over:
                finished.append(game_idx)

        active_indices = [idx for idx in active_indices if idx not in finished]

    training_data = []
    for game_idx, game in enumerate(games):
        for state, probs, player in histories[game_idx]:
            if game.winner == player:
                value = 1.0
            elif game.winner == 0:
                value = 0.0
            else:
                value = -1.0
            training_data.append((state, probs, value))

    return training_data


# ============================================================
# Training
# ============================================================
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def push(self, data_list):
        self.buffer.extend(data_list)

    def sample(self, batch_size):
        batch = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        states = np.array([b[0] for b in batch])
        policies = np.array([b[1] for b in batch])
        values = np.array([b[2] for b in batch], dtype=np.float32)
        return states, policies, values

    def __len__(self):
        return len(self.buffer)


def train_step(model: PolicyValueNet, optimizer, states, target_policies,
               target_values, device=DEVICE, scaler=None, use_amp=True):
    non_blocking = device.type == 'cuda'
    states_t = torch.from_numpy(states).to(device, non_blocking=non_blocking)
    target_p = torch.from_numpy(target_policies).to(device, non_blocking=non_blocking)
    target_v = torch.from_numpy(target_values).unsqueeze(1).to(device, non_blocking=non_blocking)

    optimizer.zero_grad()
    with autocast_context(device, use_amp):
        policy_logits, value = model(states_t)
        log_policy = F.log_softmax(policy_logits, dim=1)

        # Policy loss: cross-entropy with MCTS probabilities
        policy_loss = -torch.mean(torch.sum(target_p * log_policy, dim=1))

        # Value loss: MSE
        value_loss = F.mse_loss(value, target_v)

        loss = policy_loss + value_loss

    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return {
        'loss': loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
    }


def _make_eval_summary(games, wins_as_p1: int, wins_as_p2: int, draws: int,
                       avg_moves: float, avg_margin: float,
                       extra: Optional[Dict] = None) -> Dict:
    total = max(1, games)
    summary = {
        'games': games,
        'wins_as_p1': wins_as_p1,
        'wins_as_p2': wins_as_p2,
        'wins_total': wins_as_p1 + wins_as_p2,
        'losses_total': total - draws - wins_as_p1 - wins_as_p2,
        'draws': draws,
        'winrate': (wins_as_p1 + wins_as_p2) / total,
        'draw_rate': draws / total,
        'avg_moves': avg_moves,
        'avg_margin': avg_margin,
    }
    if extra:
        summary.update(extra)
    return summary


def evaluate_vs_heuristic(model: PolicyValueNet, num_games=20,
                          num_simulations=30, ai_level=2, device=DEVICE,
                          use_amp=True, inference_batch_size=1,
                          dead_zone_fills_line=True,
                          score_dead_zone_weight=0.0,
                          piece_distribution='bag7',
                          terminal_mode='pieces_only',
                          allow_voluntary_skip=False,
                          end_condition_mode='double_forced_pass',
                          no_legal_move_mode='reroll_once_then_pass',
                          resolution_mode='capture_then_clear_recheck',
                          dead_zone_activation_mode='immediate',
                          no_legal_move_rerolls=1,
                          parallel_games=1,
                          seed_base=1000) -> Dict:
    """模型 vs 启发式AI，返回结构化评估结果。"""
    model.eval()
    mcts = MCTS(model, num_simulations=num_simulations, temperature=0.1,
                device=device, use_amp=use_amp,
                inference_batch_size=inference_batch_size)
    wins_as_p1 = 0
    wins_as_p2 = 0
    draws = 0
    total_moves = 0
    total_margin = 0.0
    jobs = []
    for i in range(num_games):
        jobs.append({
            'game': TetrisWeiqi(
                BOARD_SIZE, seed=seed_base + i,
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
            ),
            'heuristic': SimpleAI(ai_level),
            'model_player': P1 if i % 2 == 0 else P2,
        })

    active_indices = list(range(len(jobs)))
    parallel_games = max(1, parallel_games)

    while active_indices:
        progress_made = False

        model_turn_indices = []
        for idx in active_indices:
            job = jobs[idx]
            if not job['game'].game_over and job['game'].current_player == job['model_player']:
                model_turn_indices.append(idx)
                if len(model_turn_indices) >= parallel_games:
                    break

        if model_turn_indices:
            batch_games = [jobs[idx]['game'] for idx in model_turn_indices]
            batch_results = mcts.search_many(batch_games, temperatures=[0.1] * len(batch_games))
            for local_idx, job_idx in enumerate(model_turn_indices):
                game = jobs[job_idx]['game']
                action_probs, _ = batch_results[local_idx]
                if action_probs.sum() <= 0:
                    resolve_zero_policy_turn(game)
                else:
                    action_idx = np.argmax(action_probs)
                    rot, row, col = index_to_action(action_idx, game.size)
                    apply_move_or_forced_skip(game, rot, row, col)
            progress_made = True

        heuristic_turn_indices = []
        for idx in active_indices:
            job = jobs[idx]
            if not job['game'].game_over and job['game'].current_player != job['model_player']:
                heuristic_turn_indices.append(idx)

        for job_idx in heuristic_turn_indices:
            job = jobs[job_idx]
            move = job['heuristic'].choose_move(job['game'])
            if move is None:
                job['game'].do_skip()
            else:
                job['game'].do_move(move['rot'], move['row'], move['col'])
        if heuristic_turn_indices:
            progress_made = True

        if not progress_made:
            break

        active_indices = [idx for idx in active_indices if not jobs[idx]['game'].game_over]

    for job in jobs:
        game = job['game']
        model_player = job['model_player']
        total_moves += game.move_number
        total_margin += abs(game.count_pieces(P1) - game.count_pieces(P2))
        if game.winner == 0:
            draws += 1
        elif game.winner == model_player:
            if model_player == P1:
                wins_as_p1 += 1
            else:
                wins_as_p2 += 1

    model.train()
    return _make_eval_summary(
        games=num_games,
        wins_as_p1=wins_as_p1,
        wins_as_p2=wins_as_p2,
        draws=draws,
        avg_moves=total_moves / max(1, num_games),
        avg_margin=total_margin / max(1, num_games),
        extra={
            'opponent': f'heuristic_lv{ai_level}',
            'seed_base': seed_base,
            'num_simulations': num_simulations,
        }
    )


def evaluate_model_vs_model(model_a: PolicyValueNet, model_b: PolicyValueNet, num_games=20,
                            num_simulations=30, device=DEVICE, use_amp=True,
                            inference_batch_size=1, dead_zone_fills_line=True,
                            score_dead_zone_weight=0.0, piece_distribution='bag7',
                            terminal_mode='pieces_only', allow_voluntary_skip=False,
                            end_condition_mode='double_forced_pass',
                            no_legal_move_mode='reroll_once_then_pass',
                            resolution_mode='capture_then_clear_recheck',
                            dead_zone_activation_mode='immediate',
                            no_legal_move_rerolls=1,
                            parallel_games=1, seed_base=5000) -> Dict:
    """当前模型 A 对最佳模型 B 的 head-to-head 评估。"""
    model_a.eval()
    model_b.eval()
    mcts_a = MCTS(model_a, num_simulations=num_simulations, temperature=0.1,
                  device=device, use_amp=use_amp,
                  inference_batch_size=inference_batch_size)
    mcts_b = MCTS(model_b, num_simulations=num_simulations, temperature=0.1,
                  device=device, use_amp=use_amp,
                  inference_batch_size=inference_batch_size)

    wins_as_p1 = 0
    wins_as_p2 = 0
    draws = 0
    total_moves = 0
    total_margin = 0.0
    jobs = []
    for i in range(num_games):
        jobs.append({
            'game': TetrisWeiqi(
                BOARD_SIZE, seed=seed_base + i,
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
            ),
            'a_player': P1 if i % 2 == 0 else P2,
        })

    active_indices = list(range(len(jobs)))
    parallel_games = max(1, parallel_games)

    while active_indices:
        progress_made = False
        for agent_label, agent_mcts in [('a', mcts_a), ('b', mcts_b)]:
            turn_indices = []
            for idx in active_indices:
                job = jobs[idx]
                target_player = job['a_player'] if agent_label == 'a' else (P2 if job['a_player'] == P1 else P1)
                if not job['game'].game_over and job['game'].current_player == target_player:
                    turn_indices.append(idx)
                    if len(turn_indices) >= parallel_games:
                        break

            if not turn_indices:
                continue

            batch_games = [jobs[idx]['game'] for idx in turn_indices]
            batch_results = agent_mcts.search_many(batch_games, temperatures=[0.1] * len(batch_games))
            for local_idx, job_idx in enumerate(turn_indices):
                game = jobs[job_idx]['game']
                action_probs, _ = batch_results[local_idx]
                if action_probs.sum() <= 0:
                    resolve_zero_policy_turn(game)
                else:
                    action_idx = np.argmax(action_probs)
                    rot, row, col = index_to_action(action_idx, game.size)
                    apply_move_or_forced_skip(game, rot, row, col)
            progress_made = True

        if not progress_made:
            break
        active_indices = [idx for idx in active_indices if not jobs[idx]['game'].game_over]

    for job in jobs:
        game = job['game']
        a_player = job['a_player']
        total_moves += game.move_number
        total_margin += abs(game.count_pieces(P1) - game.count_pieces(P2))
        if game.winner == 0:
            draws += 1
        elif game.winner == a_player:
            if a_player == P1:
                wins_as_p1 += 1
            else:
                wins_as_p2 += 1

    model_a.train()
    model_b.train()
    return _make_eval_summary(
        games=num_games,
        wins_as_p1=wins_as_p1,
        wins_as_p2=wins_as_p2,
        draws=draws,
        avg_moves=total_moves / max(1, num_games),
        avg_margin=total_margin / max(1, num_games),
        extra={
            'opponent': 'best_model',
            'seed_base': seed_base,
            'num_simulations': num_simulations,
        }
    )


def append_eval_report(path: str, payload: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=False) + '\n')



def parse_int_list(spec: str) -> List[int]:
    values = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def run_selfplay_batch(model: PolicyValueNet, optimizer, replay_buffer: ReplayBuffer,
                       batch_games: int, num_simulations: int, batch_size: int,
                       device, use_amp: bool, scaler, inference_batch_size: int,
                       dead_zone_fills_line: bool, score_dead_zone_weight: float,
                       piece_distribution: str, terminal_mode: str,
                       allow_voluntary_skip: bool, end_condition_mode: str,
                       no_legal_move_mode: str, resolution_mode: str,
                       dead_zone_activation_mode: str,
                       no_legal_move_rerolls: int):
    t0 = time.time()
    model.eval()
    data = self_play_games_parallel(
        model,
        num_games=batch_games,
        num_simulations=num_simulations,
        temperature=1.0,
        temp_threshold=15,
        device=device,
        use_amp=use_amp,
        inference_batch_size=inference_batch_size,
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
    t_selfplay = time.time() - t0

    replay_buffer.push(data)

    model.train()
    train_batches = max(1, len(data) // batch_size)
    total_loss = 0.0
    t1 = time.time()
    for _ in range(train_batches):
        states, policies, values = replay_buffer.sample(batch_size)
        losses = train_step(
            model, optimizer, states, policies, values,
            device=device, scaler=scaler, use_amp=use_amp
        )
        total_loss += losses['loss']
    t_train = time.time() - t1

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    else:
        peak_mem_mb = 0.0

    avg_loss = total_loss / train_batches
    positions = len(data)
    total_time = t_selfplay + t_train
    return {
        'positions': positions,
        'selfplay_s': t_selfplay,
        'train_s': t_train,
        'total_s': total_time,
        'pos_per_s': positions / total_time if total_time > 0 else 0.0,
        'loss': avg_loss,
        'peak_mem_mb': peak_mem_mb,
    }


def benchmark(args):
    device = get_device(args.device)
    configure_torch_runtime(device)
    use_amp = args.amp and device.type in ('cuda', 'mps')

    parallel_games_list = parse_int_list(args.benchmark_parallel_games)
    infer_batch_list = parse_int_list(args.benchmark_inference_batches)
    train_batch_list = parse_int_list(args.benchmark_train_batches)

    combos = list(itertools.product(parallel_games_list, infer_batch_list, train_batch_list))
    if args.benchmark_max_cases > 0:
        combos = combos[:args.benchmark_max_cases]

    print(f'[Benchmark] device={device} amp={use_amp} sims={args.num_simulations} '
          f'games/iter={args.games_per_iter} dead_zone_fills_line={args.dead_zone_fills_line} '
          f'score_dead_zone_weight={args.score_dead_zone_weight} '
          f'piece_distribution={args.piece_distribution} '
          f'terminal_mode={args.terminal_mode} '
          f'allow_voluntary_skip={args.allow_voluntary_skip} '
          f'end_condition_mode={args.end_condition_mode} '
          f'no_legal_move_mode={args.no_legal_move_mode} '
          f'resolution_mode={args.resolution_mode} '
          f'dead_zone_activation_mode={args.dead_zone_activation_mode} '
          f'no_legal_move_rerolls={args.no_legal_move_rerolls}')
    print(f'[Benchmark] cases={len(combos)}')
    print()

    results = []
    for idx, (parallel_games, infer_batch, train_batch) in enumerate(combos, 1):
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        elif device.type == 'mps':
            torch.mps.empty_cache()

        model = PolicyValueNet(
            board_size=BOARD_SIZE,
            num_res_blocks=args.res_blocks,
            channels=args.channels
        ).to(device)
        model, compile_enabled = maybe_compile_model(model, args.compile)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        replay_buffer = ReplayBuffer(max(args.buffer_size, args.games_per_iter * 128))
        scaler = torch.amp.GradScaler(device='cuda', enabled=use_amp and device.type == 'cuda')

        print(f'[Case {idx}/{len(combos)}] parallel={parallel_games} '
              f'infer_batch={infer_batch} train_batch={train_batch} '
              f'compile={compile_enabled}')

        metrics = run_selfplay_batch(
            model=model,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            batch_games=args.games_per_iter if args.games_per_iter > 0 else parallel_games,
            num_simulations=args.num_simulations,
            batch_size=train_batch,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            inference_batch_size=infer_batch,
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
        metrics.update({
            'parallel_games': parallel_games,
            'inference_batch_size': infer_batch,
            'train_batch_size': train_batch,
            'compile': compile_enabled,
        })
        results.append(metrics)

        print(f'  positions={metrics["positions"]} total={metrics["total_s"]:.2f}s '
              f'pos/s={metrics["pos_per_s"]:.2f} selfplay={metrics["selfplay_s"]:.2f}s '
              f'train={metrics["train_s"]:.2f}s peak_mem={metrics["peak_mem_mb"]:.0f}MB')
        print()

    results.sort(key=lambda x: x['pos_per_s'], reverse=True)

    print('=== Benchmark Ranking ===')
    for rank, item in enumerate(results[:args.benchmark_top_k], 1):
        print(f'{rank:2d}. pos/s={item["pos_per_s"]:.2f} '
              f'parallel={item["parallel_games"]} '
              f'infer_batch={item["inference_batch_size"]} '
              f'train_batch={item["train_batch_size"]} '
              f'peak_mem={item["peak_mem_mb"]:.0f}MB '
              f'total={item["total_s"]:.2f}s')

    if args.benchmark_output:
        payload = {
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(device),
            'amp': use_amp,
            'num_simulations': args.num_simulations,
            'games_per_iter': args.games_per_iter,
            'dead_zone_fills_line': args.dead_zone_fills_line,
            'score_dead_zone_weight': args.score_dead_zone_weight,
            'piece_distribution': args.piece_distribution,
            'terminal_mode': args.terminal_mode,
            'allow_voluntary_skip': args.allow_voluntary_skip,
            'end_condition_mode': args.end_condition_mode,
            'no_legal_move_mode': args.no_legal_move_mode,
            'resolution_mode': args.resolution_mode,
            'dead_zone_activation_mode': args.dead_zone_activation_mode,
            'no_legal_move_rerolls': args.no_legal_move_rerolls,
            'results': results,
        }
        with open(args.benchmark_output, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print()
        print(f'[Benchmark] 已写入 {args.benchmark_output}')


def benchmark_rule_ab(args):
    device = get_device(args.device)
    configure_torch_runtime(device)
    use_amp = args.amp and device.type in ('cuda', 'mps')
    variants = [
        ('A', True),
        ('B', False),
    ]

    print(f'[RuleAB] device={device} amp={use_amp} sims={args.num_simulations} '
          f'games/iter={args.games_per_iter}')
    print('[RuleAB] 对比规则: dead_zone_fills_line = True vs False')
    print()

    results = []
    for label, flag in variants:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        model = PolicyValueNet(
            board_size=BOARD_SIZE,
            num_res_blocks=args.res_blocks,
            channels=args.channels
        ).to(device)
        model, compile_enabled = maybe_compile_model(model, args.compile)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        replay_buffer = ReplayBuffer(max(args.buffer_size, args.games_per_iter * 128))
        scaler = torch.amp.GradScaler(device='cuda', enabled=use_amp and device.type == 'cuda')

        print(f'[RuleAB {label}] dead_zone_fills_line={flag} compile={compile_enabled}')
        metrics = run_selfplay_batch(
            model=model,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            batch_games=args.games_per_iter,
            num_simulations=args.num_simulations,
            batch_size=args.batch_size,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            inference_batch_size=args.inference_batch_size,
            dead_zone_fills_line=flag,
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
        metrics.update({
            'label': label,
            'dead_zone_fills_line': flag,
            'compile': compile_enabled,
        })
        results.append(metrics)
        print(f'  positions={metrics["positions"]} total={metrics["total_s"]:.2f}s '
              f'pos/s={metrics["pos_per_s"]:.2f} peak_mem={metrics["peak_mem_mb"]:.0f}MB')
        print()

    print('=== Rule A/B Summary ===')
    for item in results:
        print(f'{item["label"]}. dead_zone_fills_line={item["dead_zone_fills_line"]} '
              f'pos/s={item["pos_per_s"]:.2f} positions={item["positions"]} '
              f'peak_mem={item["peak_mem_mb"]:.0f}MB total={item["total_s"]:.2f}s')

    if args.benchmark_output:
        payload = {
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(device),
            'amp': use_amp,
            'num_simulations': args.num_simulations,
            'games_per_iter': args.games_per_iter,
            'rule_ab': results,
        }
        with open(args.benchmark_output, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print()
        print(f'[RuleAB] 已写入 {args.benchmark_output}')


# ============================================================
# Main Training Loop
# ============================================================
def train(args):
    device = get_device(args.device)
    configure_torch_runtime(device)
    model = PolicyValueNet(
        board_size=BOARD_SIZE,
        num_res_blocks=args.res_blocks,
        channels=args.channels
    ).to(device)
    model, compile_enabled = maybe_compile_model(model, args.compile)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    replay_buffer = ReplayBuffer(args.buffer_size)
    use_amp = args.amp and device.type in ('cuda', 'mps')
    scaler = torch.amp.GradScaler(device='cuda', enabled=use_amp and device.type == 'cuda')
    eval_report_path = os.path.join(args.save_dir, args.eval_report_name)

    start_iter = 0
    best_winrate = 0.0
    best_model_state = None
    last_heuristic_eval = None
    last_head_to_head = None

    # 恢复检查点
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_iter = ckpt.get('iteration', 0)
        best_winrate = ckpt.get('best_winrate', 0)
        best_model_state = {
            key: value.detach().cpu().clone()
            for key, value in ckpt['model'].items()
        }
        print(f'[Resume] 从 {args.resume} 恢复, iteration={start_iter}')

    os.makedirs(args.save_dir, exist_ok=True)

    param_count = sum(p.numel() for p in model.parameters())
    print(f'[Model] ResNet-{args.res_blocks} x {args.channels}ch, '
          f'{param_count:,} parameters')
    print(f'[Runtime] device={device} amp={use_amp} tf32={device.type == "cuda"} '
          f'compile={compile_enabled} infer_batch={args.inference_batch_size} '
          f'parallel_games={args.selfplay_parallel_games} '
          f'dead_zone_fills_line={args.dead_zone_fills_line} '
          f'score_dead_zone_weight={args.score_dead_zone_weight} '
          f'piece_distribution={args.piece_distribution} '
          f'terminal_mode={args.terminal_mode} '
          f'allow_voluntary_skip={args.allow_voluntary_skip} '
          f'end_condition_mode={args.end_condition_mode} '
          f'no_legal_move_mode={args.no_legal_move_mode} '
          f'resolution_mode={args.resolution_mode} '
          f'dead_zone_activation_mode={args.dead_zone_activation_mode} '
          f'no_legal_move_rerolls={args.no_legal_move_rerolls}')
    print(f'[Config] iterations={args.iterations}, '
          f'games/iter={args.games_per_iter}, '
          f'simulations={args.num_simulations}, '
          f'batch={args.batch_size}, '
          f'train_steps={args.train_steps_per_iter if args.train_steps_per_iter > 0 else "auto"}, '
          f'min_train_batches={args.min_train_batches}, '
          f'eval_sims={args.eval_num_simulations}, '
          f'eval_parallel={args.eval_parallel_games}')
    print()

    for iteration in range(start_iter, args.iterations):
        t0 = time.time()
        model.eval()

        # --- Self-play ---
        all_data = []
        generated_games = 0
        while generated_games < args.games_per_iter:
            batch_games = min(args.selfplay_parallel_games, args.games_per_iter - generated_games)
            data = self_play_games_parallel(
                model,
                num_games=batch_games,
                num_simulations=args.num_simulations,
                temperature=1.0,
                temp_threshold=15,
                device=device,
                use_amp=use_amp,
                inference_batch_size=args.inference_batch_size,
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
            all_data.extend(data)
            generated_games += batch_games
            if generated_games % 5 == 0 or generated_games == args.games_per_iter:
                print(f'\r  [Iter {iteration+1}] Self-play: {generated_games}/{args.games_per_iter} '
                      f'({len(all_data)} positions)', end='', flush=True)

        replay_buffer.push(all_data)
        t_selfplay = time.time() - t0
        print(f'\r  [Iter {iteration+1}] Self-play: {args.games_per_iter} games, '
              f'{len(all_data)} positions, {t_selfplay:.1f}s')

        # --- Training ---
        model.train()
        total_loss = {'loss': 0, 'policy_loss': 0, 'value_loss': 0}
        auto_batches = max(1, len(all_data) // args.batch_size)
        if args.train_steps_per_iter > 0:
            num_batches = args.train_steps_per_iter
        else:
            num_batches = max(args.min_train_batches, auto_batches)

        for b in range(num_batches):
            states, policies, values = replay_buffer.sample(args.batch_size)
            losses = train_step(
                model, optimizer, states, policies, values,
                device=device, scaler=scaler, use_amp=use_amp
            )
            for k in total_loss:
                total_loss[k] += losses[k]

        for k in total_loss:
            total_loss[k] /= num_batches

        scheduler.step()
        t_train = time.time() - t0 - t_selfplay

        print(f'  [Iter {iteration+1}] Train: loss={total_loss["loss"]:.4f} '
              f'(policy={total_loss["policy_loss"]:.4f} value={total_loss["value_loss"]:.4f}) '
              f'{t_train:.1f}s')

        # --- Evaluation ---
        if (iteration + 1) % args.eval_every == 0:
            heuristic_eval = evaluate_vs_heuristic(
                model, num_games=args.eval_games,
                num_simulations=args.eval_num_simulations,
                ai_level=2, device=device, use_amp=use_amp,
                inference_batch_size=args.inference_batch_size,
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
                parallel_games=args.eval_parallel_games,
                seed_base=args.eval_seed_base + iteration * 1000
            )
            head_to_head = None
            if best_model_state is not None and args.eval_headtohead_games > 0:
                best_model = PolicyValueNet(
                    board_size=BOARD_SIZE,
                    num_res_blocks=args.res_blocks,
                    channels=args.channels
                ).to(device)
                best_model.load_state_dict(best_model_state)
                head_to_head = evaluate_model_vs_model(
                    model, best_model,
                    num_games=args.eval_headtohead_games,
                    num_simulations=args.eval_num_simulations,
                    device=device, use_amp=use_amp,
                    inference_batch_size=args.inference_batch_size,
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
                    parallel_games=args.eval_parallel_games,
                    seed_base=args.eval_seed_base + 500000 + iteration * 1000
                )
            t_eval = time.time() - t0 - t_selfplay - t_train
            print(f'  [Iter {iteration+1}] Eval: vs Heuristic AI winrate = {heuristic_eval["winrate"]:.1%} '
                  f'({heuristic_eval["wins_total"]}-{heuristic_eval["losses_total"]}-{heuristic_eval["draws"]}, '
                  f'best={best_winrate:.1%}) {t_eval:.1f}s')
            if head_to_head is not None:
                print(f'  [Iter {iteration+1}] Eval: vs BestModel winrate = {head_to_head["winrate"]:.1%} '
                      f'({head_to_head["wins_total"]}-{head_to_head["losses_total"]}-{head_to_head["draws"]})')

            last_heuristic_eval = heuristic_eval
            last_head_to_head = head_to_head
            append_eval_report(eval_report_path, {
                'iteration': iteration + 1,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'heuristic_eval': heuristic_eval,
                'head_to_head': head_to_head,
            })

            if heuristic_eval['winrate'] > best_winrate:
                best_winrate = heuristic_eval['winrate']
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                path = os.path.join(args.save_dir, 'best.pt')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iteration': iteration + 1,
                    'best_winrate': best_winrate,
                    'last_heuristic_eval': heuristic_eval,
                    'last_head_to_head': head_to_head,
                }, path)
                print(f'  [Save] New best model → {path}')

        # 定期保存检查点
        if (iteration + 1) % args.save_every == 0:
            path = os.path.join(args.save_dir, f'ckpt_iter{iteration+1}.pt')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration + 1,
                'best_winrate': best_winrate,
                'last_heuristic_eval': last_heuristic_eval,
                'last_head_to_head': last_head_to_head,
            }, path)
            print(f'  [Save] Checkpoint → {path}')

        total_time = time.time() - t0
        print(f'  [Iter {iteration+1}] Total: {total_time:.1f}s | '
              f'Buffer: {len(replay_buffer)} positions')
        print()

    print('训练完成!')
    print(f'最佳胜率: {best_winrate:.1%}')


def eval_model(args):
    device = get_device(args.device)
    configure_torch_runtime(device)
    model = PolicyValueNet(
        board_size=BOARD_SIZE,
        num_res_blocks=args.res_blocks,
        channels=args.channels
    ).to(device)
    model, compile_enabled = maybe_compile_model(model, args.compile)

    ckpt = torch.load(args.eval, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f'[Load] {args.eval}, iteration={ckpt.get("iteration","?")}')
    print(f'[Runtime] device={device} amp={args.amp and device.type == "cuda"} '
          f'compile={compile_enabled} infer_batch={args.inference_batch_size} '
          f'dead_zone_fills_line={args.dead_zone_fills_line} '
          f'score_dead_zone_weight={args.score_dead_zone_weight} '
          f'piece_distribution={args.piece_distribution} '
          f'terminal_mode={args.terminal_mode} '
          f'allow_voluntary_skip={args.allow_voluntary_skip} '
          f'end_condition_mode={args.end_condition_mode} '
          f'no_legal_move_mode={args.no_legal_move_mode} '
          f'resolution_mode={args.resolution_mode} '
          f'dead_zone_activation_mode={args.dead_zone_activation_mode} '
          f'no_legal_move_rerolls={args.no_legal_move_rerolls} '
          f'eval_sims={args.eval_num_simulations} '
          f'eval_parallel={args.eval_parallel_games}')

    for level in [1, 2, 3]:
        result = evaluate_vs_heuristic(
            model, num_games=args.eval_games,
            num_simulations=args.eval_num_simulations,
            ai_level=level, device=device, use_amp=args.amp and device.type in ('cuda', 'mps'),
            inference_batch_size=args.inference_batch_size,
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
            parallel_games=args.eval_parallel_games
        )
        print(f'  vs Heuristic Lv{level}: {result["winrate"]:.1%} '
              f'({result["wins_total"]}-{result["losses_total"]}-{result["draws"]}, '
              f'{args.eval_games} games)')


# ============================================================
# Export to ONNX (for browser deployment)
# ============================================================
def export_onnx(args):
    model = PolicyValueNet(
        board_size=BOARD_SIZE,
        num_res_blocks=args.res_blocks,
        channels=args.channels
    ).to('cpu')

    ckpt = torch.load(args.export, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    dummy = torch.randn(1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    path = args.export.replace('.pt', '.onnx')
    torch.onnx.export(
        model, dummy, path,
        input_names=['state'],
        output_names=['policy', 'value'],
        dynamic_axes={'state': {0: 'batch'}, 'policy': {0: 'batch'}, 'value': {0: 'batch'}},
        opset_version=17,
    )
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f'[Export] {path} ({size_mb:.1f} MB)')


# ============================================================
# Entry
# ============================================================
def main():
    p = argparse.ArgumentParser(description='TetrisWeiqi AlphaZero-Lite Training')

    # 模式
    p.add_argument('--eval', type=str, default=None, help='评估指定模型')
    p.add_argument('--export', type=str, default=None, help='导出ONNX模型')
    p.add_argument('--benchmark', action='store_true',
                   help='运行参数基准测试，自动筛选更适合当前 GPU 的组合')
    p.add_argument('--benchmark-rule-ab', action='store_true',
                   help='对比 dead_zone_fills_line=True/False 两种规则的训练吞吐')

    # 网络
    p.add_argument('--res-blocks', type=int, default=6, help='ResNet层数 (默认6)')
    p.add_argument('--channels', type=int, default=128, help='通道数 (默认128)')

    # 训练
    p.add_argument('--iterations', type=int, default=30, help='训练迭代数')
    p.add_argument('--games-per-iter', type=int, default=20, help='每轮自对弈局数')
    p.add_argument('--num-simulations', type=int, default=50, help='MCTS模拟次数')
    p.add_argument('--batch-size', type=int, default=128, help='训练批大小')
    p.add_argument('--lr', type=float, default=0.002, help='学习率')
    p.add_argument('--buffer-size', type=int, default=100000, help='经验回放缓冲大小')
    p.add_argument('--resume', type=str, default=None, help='恢复训练的检查点')
    p.add_argument('--device', type=str, default=None,
                   help='指定设备，如 cuda / cuda:0 / cpu / mps')
    p.add_argument('--amp', action='store_true',
                   help='启用自动混合精度（CUDA 下推荐开启）')
    p.add_argument('--compile', action='store_true',
                   help='使用 torch.compile 优化模型（CUDA 下推荐尝试）')
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
    p.add_argument('--inference-batch-size', type=int, default=16,
                   help='MCTS 批量推理批大小，增大可提升 GPU 利用率')
    p.add_argument('--selfplay-parallel-games', type=int, default=4,
                   help='同时并行推进的自对弈局数，增大可提高 GPU 利用率')
    p.add_argument('--train-steps-per-iter', type=int, default=0,
                   help='每轮固定训练步数，0 表示自动根据样本量决定')
    p.add_argument('--min-train-batches', type=int, default=4,
                   help='自动模式下每轮至少训练多少个 batch')

    # Benchmark
    p.add_argument('--benchmark-parallel-games', type=str, default='4,8,12',
                   help='benchmark 扫描的 parallel_games 列表，逗号分隔')
    p.add_argument('--benchmark-inference-batches', type=str, default='16,32,48',
                   help='benchmark 扫描的 inference_batch_size 列表，逗号分隔')
    p.add_argument('--benchmark-train-batches', type=str, default='256,512,768',
                   help='benchmark 扫描的训练 batch_size 列表，逗号分隔')
    p.add_argument('--benchmark-max-cases', type=int, default=0,
                   help='最多测试多少组组合，0 表示全部')
    p.add_argument('--benchmark-top-k', type=int, default=5,
                   help='输出前 K 名 benchmark 结果')
    p.add_argument('--benchmark-output', type=str, default=None,
                   help='将 benchmark 结果写入 JSON 文件')

    # 评估
    p.add_argument('--eval-every', type=int, default=5, help='每N轮评估一次')
    p.add_argument('--eval-games', type=int, default=20, help='评估对局数')
    p.add_argument('--eval-num-simulations', type=int, default=8,
                   help='评估时 MCTS 模拟次数，默认低于训练以提升速度')
    p.add_argument('--eval-parallel-games', type=int, default=8,
                   help='评估时并行推进的对局数')
    p.add_argument('--eval-seed-base', type=int, default=1000,
                   help='评估对局的固定种子起点，便于不同实验可重复对比')
    p.add_argument('--eval-headtohead-games', type=int, default=40,
                   help='每次评估时当前模型对最佳模型的对局数，0 表示关闭')
    p.add_argument('--eval-report-name', type=str, default='eval_history.jsonl',
                   help='评估历史输出文件名，保存在 save_dir 下')

    # 保存
    p.add_argument('--save-dir', type=str, default='checkpoints', help='检查点目录')
    p.add_argument('--save-every', type=int, default=10, help='每N轮保存检查点')

    args = p.parse_args()

    if args.eval:
        eval_model(args)
    elif args.export:
        export_onnx(args)
    elif args.benchmark_rule_ab:
        benchmark_rule_ab(args)
    elif args.benchmark:
        benchmark(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
