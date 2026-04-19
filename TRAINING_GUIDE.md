# 训练手册

本指南覆盖 `cli/train_alphazero.py` 的日常使用：训练、续训、评估、导出到浏览器，以及如何读懂 `metrics.jsonl`。
代码结构与前后端衔接点参见 [`ARCHITECTURE.md`](ARCHITECTURE.md)；规则变体语义参见 [`PROJECT_PLAN.md`](PROJECT_PLAN.md)。

---

## 环境

```bash
# macOS (MPS)
pip install torch numpy onnx onnxruntime
# Linux / Windows (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy onnx onnxruntime
```

`get_device` 自动选择 `cuda` → `mps` → `cpu`。AMP 在 `cuda` 和 `mps` 上自动开启。

---

## 最小可跑命令

```bash
# 从零训练 30 轮（默认）
python cli/train_alphazero.py --save-dir checkpoints_dev

# 小规模 smoke test（确认环境）
python cli/train_alphazero.py \
  --iterations 2 --games-per-iter 2 \
  --num-simulations 20 --eval-games 4 \
  --save-dir checkpoints_smoke
```

跑通后 `checkpoints_smoke/` 下应当出现：
```
ckpt_iter2.pt        # 定期检查点
best.pt              # 历史最佳（若有超过基线的评估）
eval_history.jsonl   # 评估详情
metrics.jsonl        # 每轮指标（本次新增）
```

---

## 关键参数速查

| 参数 | 默认 | 作用 |
|---|---|---|
| `--iterations` | 30 | 主循环迭代轮数 |
| `--games-per-iter` | 决定数据吞吐 | 每轮自对弈局数 |
| `--num-simulations` | - | 自对弈每步 MCTS 模拟数 |
| `--eval-every` | - | 每 N 轮评估一次 |
| `--eval-num-simulations` | - | 评估时 MCTS 模拟数 |
| `--selfplay-parallel-games` | - | 并行自对弈局数，吃显存 |
| `--inference-batch-size` | - | 推理打包大小，影响 GPU 利用率 |
| `--res-blocks` / `--channels` | - | 网络容量 |
| `--lr` / `--lr-step-size` / `--lr-gamma` | - | StepLR 调度 |
| `--dirichlet-alpha` / `--dirichlet-epsilon` | - | 根节点探索噪声 |
| `--komi` | 0 | 贴目（随先手调整） |
| `--metrics-file` | `metrics.jsonl` | 每轮指标输出 |
| `--eval-report-name` | `eval_history.jsonl` | 评估明细输出 |
| `--save-dir` | `checkpoints` | 产物目录 |
| `--save-every` | 10 | 每 N 轮保存 `ckpt_iter{N}.pt` |

完整参数：`python cli/train_alphazero.py --help`。

---

## 续训（resume）

```bash
python cli/train_alphazero.py \
  --resume checkpoints_dev/ckpt_iter30.pt \
  --iterations 60 \
  --save-dir checkpoints_dev
```

**注意**：
- 恢复 `model / optimizer / scheduler / best_winrate / best_model_state`
- **不恢复 replay buffer**——属于暖启动续训，不是严格无缝续训
- 若原检查点缺 `scheduler` 状态，会用初始调度从头开始（有警告）
- 若保留同一 `--save-dir`，`metrics.jsonl` 与 `eval_history.jsonl` 会 **append**，天然接续

---

## 单独评估一个检查点

```bash
python cli/train_alphazero.py \
  --eval checkpoints_dev/best.pt \
  --eval-games 40 \
  --eval-num-simulations 100
```

---

## 导出到浏览器

```bash
python cli/train_alphazero.py \
  --export checkpoints_dev/best.pt \
  --export-output checkpoints_m5_mps/best_browser.onnx
```

前端 `js/ai-onnx.js` 中 `ONNX_MODEL_PATH` 默认读 `checkpoints_m5_mps/best_browser.onnx`；
若更换路径需同步修改。导出后刷新页面，选 Level 4 或 5 即可用新模型对战。

---

## metrics.jsonl 结构

每轮一条 JSON，**一行一条**（JSONL），字段：

```json
{
  "iteration": 12,
  "timestamp": "2026-04-19T14:32:07",
  "self_play": {"games": 20, "positions": 1243, "seconds": 48.2},
  "train":     {"loss": 2.14, "policy_loss": 1.82, "value_loss": 0.31,
                "num_batches": 24, "seconds": 11.5},
  "eval": {
    "ran": true, "seconds": 22.1,
    "heuristic_winrate": 0.65,
    "heuristic_wins": 26, "heuristic_losses": 12, "heuristic_draws": 2,
    "head_to_head_winrate": 0.53,
    "head_to_head_wins": 21, "head_to_head_losses": 17, "head_to_head_draws": 2
  },
  "buffer_size": 24350,
  "lr": 0.001,
  "best_winrate": 0.65,
  "total_seconds": 81.8
}
```

非评估轮的 `eval` 为 `null`。
`head_to_head_*` 在 `--eval-headtohead-games=0` 或尚未产生 best 时为 `null`。

### 快速可视化

```python
import json, pandas as pd
df = pd.DataFrame([json.loads(l) for l in open('checkpoints_dev/metrics.jsonl')])
# 展平
df['loss'] = df['train'].apply(lambda x: x['loss'])
df['winrate'] = df['eval'].apply(lambda x: x['heuristic_winrate'] if x else None)
df.plot(x='iteration', y=['loss', 'winrate'], secondary_y=['winrate'])
```

### 简单健康检查（命令行）

```bash
# 最近 5 轮 loss 和评估胜率
python3 -c "
import json
recs = [json.loads(l) for l in open('checkpoints_dev/metrics.jsonl')]
for r in recs[-5:]:
    ev = r['eval']
    print(f\"iter {r['iteration']:3d}  loss={r['train']['loss']:.3f}  \" +
          (f\"winrate={ev['heuristic_winrate']:.1%}\" if ev else 'no eval') +
          f\"  buf={r['buffer_size']}\")
"
```

---

## 常见问题

**训练发散 / loss 飙升**
检查 `metrics.jsonl` 中 `train.policy_loss` vs `train.value_loss` 的比例；若 value 早早到 0 表示 buffer 单一，可降 `--lr` 或增大 `--games-per-iter`。

**评估胜率一直不升**
对比 `head_to_head_winrate`（对自己的历史 best）与 `heuristic_winrate`（对启发式 AI）。若前者稳定 >50% 而后者不升，可能是启发式 AI 天花板——不是模型问题。

**前端加载新模型报错**
确认 `encode_state` 的 22 通道顺序、`PIECE_SHAPES` 排列、动作索引公式在前后端一致。修改后跑 `python cli/check_rule_consistency.py` 验证。

**MPS 设备训练偏慢**
开 `--compile` 试试；推理瓶颈时提高 `--inference-batch-size` 和 `--selfplay-parallel-games`。

---

## 新增指标字段的建议

如需扩展 `metrics.jsonl`（例如加 gradient norm、replay 多样性、棋局平均步数）：
1. 在 `train(args)` 里收集
2. 加到 `append_metrics` 的 payload 中
3. **只追加字段**，不要改已有字段的含义——保持向后兼容，历史 metrics 才能一起复用
