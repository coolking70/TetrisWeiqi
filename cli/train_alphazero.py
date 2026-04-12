#!/usr/bin/env python3
"""
俄罗斯方块棋 AlphaZero-Lite 训练 (Apple MPS 加速)

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
from collections import deque
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tetris_weiqi import TetrisWeiqi, SimpleAI, PIECE_SHAPES, PIECE_NAMES
from tetris_weiqi import EMPTY, P1, P2, DEAD1, DEAD2

# ============================================================
# Device
# ============================================================
def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

DEVICE = get_device()
print(f'[Device] {DEVICE}')

# ============================================================
# Neural Network: ResNet Policy-Value Network
# ============================================================
BOARD_SIZE = 10
NUM_CHANNELS = 5   # EMPTY, P1, P2, DEAD1, DEAD2
PIECE_DIM = 7      # 7种方块
INPUT_CHANNELS = NUM_CHANNELS + PIECE_DIM + 2  # board one-hot + piece one-hot + player planes
# = 5 + 7 + 2 = 14 input channels

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
    将游戏状态编码为神经网络输入张量 (14, size, size)

    Channels 0-4: board one-hot (EMPTY, P1, P2, DEAD1, DEAD2)
    Channels 5-11: current piece one-hot (I,O,T,S,Z,L,J) 全平面
    Channel 12: current player plane (全1=P1, 全0=P2)
    Channel 13: opponent player plane (反过来)
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

    # Player indicator
    state[12, :, :] = 1 if player == P1 else 0
    state[13, :, :] = 0 if player == P1 else 1

    return state


def action_to_index(rot: int, row: int, col: int, size: int = BOARD_SIZE) -> int:
    return rot * size * size + row * size + col

def index_to_action(idx: int, size: int = BOARD_SIZE) -> Tuple[int, int, int]:
    rot = idx // (size * size)
    remainder = idx % (size * size)
    row = remainder // size
    col = remainder % size
    return rot, row, col


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
                 temperature=1.0, device=DEVICE):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = device

    @torch.no_grad()
    def _predict(self, game: TetrisWeiqi, player: int):
        state = encode_state(game, player)
        tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        policy_logits, value = self.model(tensor)
        policy = F.softmax(policy_logits.squeeze(0), dim=0).cpu().numpy()
        value = value.item()
        return policy, value

    def _expand(self, node: MCTSNode, game: TetrisWeiqi):
        player = game.current_player
        policy, value = self._predict(game, player)
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

    def search(self, game: TetrisWeiqi) -> Tuple[np.ndarray, float]:
        """
        返回: (action_probs, root_value)
        action_probs: shape (ACTION_SIZE,) 归一化概率
        """
        root = MCTSNode()

        # Expand root
        value = self._expand(root, game)
        root.backpropagate(value)

        # Simulations
        for _ in range(self.num_simulations - 1):
            node = root.select_leaf()

            # Clone game and replay actions to reach this node
            sim_game = TetrisWeiqi(game.size)
            sim_game.board = [r[:] for r in game.board]
            sim_game.current_player = game.current_player
            sim_game.pieces = {
                P1: dict(game.pieces[P1]) if game.pieces[P1] else None,
                P2: dict(game.pieces[P2]) if game.pieces[P2] else None,
            }
            sim_game.skip_count = game.skip_count
            sim_game.move_number = game.move_number
            sim_game.game_over = game.game_over
            sim_game.rng = random.Random()  # 独立随机数

            # Replay path from root to node
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
                # 终局：根据胜负给值
                if sim_game.winner == game.current_player:
                    value = 1.0
                elif sim_game.winner == 0:
                    value = 0.0
                else:
                    value = -1.0
                # 调整视角：如果当前模拟到的玩家不是root玩家，翻转
                depth = len(path)
                if depth % 2 == 1:
                    value = -value
            else:
                value = self._expand(node, sim_game)

            node.backpropagate(value)

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


# ============================================================
# Self-Play Data Generation
# ============================================================
def self_play_game(model: PolicyValueNet, num_simulations=50, temperature=1.0,
                   temp_threshold=20, device=DEVICE) -> List[Tuple]:
    """
    用 MCTS + 神经网络进行一局自对弈，返回训练数据。

    返回: [(state, action_probs, winner_value), ...]
    """
    game = TetrisWeiqi(BOARD_SIZE)
    mcts = MCTS(model, num_simulations=num_simulations, device=device)
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

        result = game.do_move(rot, row, col)
        if 'error' in result:
            # 如果MCTS选了非法手（极少发生），跳过
            game.do_skip()

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
               target_values, device=DEVICE):
    states_t = torch.from_numpy(states).to(device)
    target_p = torch.from_numpy(target_policies).to(device)
    target_v = torch.from_numpy(target_values).unsqueeze(1).to(device)

    policy_logits, value = model(states_t)
    log_policy = F.log_softmax(policy_logits, dim=1)

    # Policy loss: cross-entropy with MCTS probabilities
    policy_loss = -torch.mean(torch.sum(target_p * log_policy, dim=1))

    # Value loss: MSE
    value_loss = F.mse_loss(value, target_v)

    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        'loss': loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
    }


def evaluate_vs_heuristic(model: PolicyValueNet, num_games=20,
                          num_simulations=30, ai_level=2, device=DEVICE) -> float:
    """模型 vs 启发式AI，返回模型胜率"""
    model.eval()
    wins = 0
    mcts = MCTS(model, num_simulations=num_simulations, temperature=0.1, device=device)

    for i in range(num_games):
        game = TetrisWeiqi(BOARD_SIZE, seed=i + 1000)
        heuristic = SimpleAI(ai_level)
        # 交替先后手
        model_player = P1 if i % 2 == 0 else P2

        while not game.game_over:
            if game.current_player == model_player:
                action_probs, _ = mcts.search(game)
                action_idx = np.argmax(action_probs)
                rot, row, col = index_to_action(action_idx, game.size)
                result = game.do_move(rot, row, col)
                if 'error' in result:
                    game.do_skip()
            else:
                move = heuristic.choose_move(game)
                if move is None:
                    game.do_skip()
                else:
                    game.do_move(move['rot'], move['row'], move['col'])

        if game.winner == model_player:
            wins += 1

    model.train()
    return wins / num_games


# ============================================================
# Main Training Loop
# ============================================================
def train(args):
    model = PolicyValueNet(
        board_size=BOARD_SIZE,
        num_res_blocks=args.res_blocks,
        channels=args.channels
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    replay_buffer = ReplayBuffer(args.buffer_size)

    start_iter = 0
    best_winrate = 0.0

    # 恢复检查点
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_iter = ckpt.get('iteration', 0)
        best_winrate = ckpt.get('best_winrate', 0)
        print(f'[Resume] 从 {args.resume} 恢复, iteration={start_iter}')

    os.makedirs(args.save_dir, exist_ok=True)

    param_count = sum(p.numel() for p in model.parameters())
    print(f'[Model] ResNet-{args.res_blocks} x {args.channels}ch, '
          f'{param_count:,} parameters')
    print(f'[Config] iterations={args.iterations}, '
          f'games/iter={args.games_per_iter}, '
          f'simulations={args.num_simulations}, '
          f'batch={args.batch_size}')
    print()

    for iteration in range(start_iter, args.iterations):
        t0 = time.time()
        model.eval()

        # --- Self-play ---
        all_data = []
        for g in range(args.games_per_iter):
            data = self_play_game(
                model,
                num_simulations=args.num_simulations,
                temperature=1.0,
                temp_threshold=15,
                device=DEVICE
            )
            all_data.extend(data)
            if (g + 1) % 5 == 0:
                print(f'\r  [Iter {iteration+1}] Self-play: {g+1}/{args.games_per_iter} '
                      f'({len(all_data)} positions)', end='', flush=True)

        replay_buffer.push(all_data)
        t_selfplay = time.time() - t0
        print(f'\r  [Iter {iteration+1}] Self-play: {args.games_per_iter} games, '
              f'{len(all_data)} positions, {t_selfplay:.1f}s')

        # --- Training ---
        model.train()
        total_loss = {'loss': 0, 'policy_loss': 0, 'value_loss': 0}
        num_batches = max(1, len(all_data) // args.batch_size)

        for b in range(num_batches):
            states, policies, values = replay_buffer.sample(args.batch_size)
            losses = train_step(model, optimizer, states, policies, values, DEVICE)
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
            winrate = evaluate_vs_heuristic(
                model, num_games=args.eval_games,
                num_simulations=args.num_simulations,
                ai_level=2, device=DEVICE
            )
            t_eval = time.time() - t0 - t_selfplay - t_train
            print(f'  [Iter {iteration+1}] Eval: vs Heuristic AI winrate = {winrate:.1%} '
                  f'(best={best_winrate:.1%}) {t_eval:.1f}s')

            if winrate > best_winrate:
                best_winrate = winrate
                path = os.path.join(args.save_dir, 'best.pt')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iteration': iteration + 1,
                    'best_winrate': best_winrate,
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
            }, path)
            print(f'  [Save] Checkpoint → {path}')

        total_time = time.time() - t0
        print(f'  [Iter {iteration+1}] Total: {total_time:.1f}s | '
              f'Buffer: {len(replay_buffer)} positions')
        print()

    print('训练完成!')
    print(f'最佳胜率: {best_winrate:.1%}')


def eval_model(args):
    model = PolicyValueNet(
        board_size=BOARD_SIZE,
        num_res_blocks=args.res_blocks,
        channels=args.channels
    ).to(DEVICE)

    ckpt = torch.load(args.eval, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    print(f'[Load] {args.eval}, iteration={ckpt.get("iteration","?")}')

    for level in [1, 2, 3]:
        winrate = evaluate_vs_heuristic(
            model, num_games=args.eval_games,
            num_simulations=args.num_simulations,
            ai_level=level, device=DEVICE
        )
        print(f'  vs Heuristic Lv{level}: {winrate:.1%} ({args.eval_games} games)')


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
    p = argparse.ArgumentParser(description='TetrisWeiqi AlphaZero-Lite Training (MPS)')

    # 模式
    p.add_argument('--eval', type=str, default=None, help='评估指定模型')
    p.add_argument('--export', type=str, default=None, help='导出ONNX模型')

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

    # 评估
    p.add_argument('--eval-every', type=int, default=5, help='每N轮评估一次')
    p.add_argument('--eval-games', type=int, default=20, help='评估对局数')

    # 保存
    p.add_argument('--save-dir', type=str, default='checkpoints', help='检查点目录')
    p.add_argument('--save-every', type=int, default=10, help='每N轮保存检查点')

    args = p.parse_args()

    if args.eval:
        eval_model(args)
    elif args.export:
        export_onnx(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
