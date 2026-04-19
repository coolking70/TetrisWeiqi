# 架构说明

本文档描述 TetrisWeiqi 的代码组织、模块职责、关键数据流，以及前后端的连接点。
训练侧操作参见 [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md)，长期规划参见 [`PROJECT_PLAN.md`](PROJECT_PLAN.md)。

---

## 目录结构

```
TetrisWeiqi/
├── index.html              # DOM 壳，仅引用 css/js，不含业务代码
├── style.css               # UI 样式
├── js/
│   ├── audio.js            # Web Audio 合成：SFX + Tetris 主题 BGM
│   ├── game-logic.js       # 规则核心：棋盘、落子、吃子、消行、回合
│   ├── ai-onnx.js          # 三层 AI：启发式 / 纯神经网络 / MCTS+NN
│   ├── ui.js               # 渲染、输入、动画、存读档、棋谱回放
│   └── main.js             # 启动、窗口尺寸、模式切换、阶段装配
├── cli/
│   ├── tetris_weiqi.py         # Python 实现的规则引擎（自对弈 + 评估用）
│   ├── train_alphazero.py      # AlphaZero 训练主程序
│   ├── analyze_rules.py        # 规则变体统计分析
│   └── check_rule_consistency.py # 前后端规则一致性校验
├── checkpoints_*/           # 训练产物：best.pt / best_browser.onnx / metrics.jsonl ...
└── *.md                     # 项目文档
```

---

## 前端（浏览器）

### 加载顺序与作用域

`index.html` 按 **classic `<script>`** 方式依序加载：`audio → game-logic → ai-onnx → ui → main`。
这些脚本并非 ES module，顶层 `let` / `const` 共享同一个全局词法环境，因此跨文件直接引用全局符号（`board` / `currentPlayer` / `BOARD_SIZE` / `playSfx` 等）是合法的；**修改加载顺序会破坏引用**。

### 模块职责

**`audio.js`** — 纯 Web Audio 合成，无外部音频文件。
- SFX：`playSfx('place'|'capture'|'clear'|'gameStart'|'gameWin'|'gameLose'|'illegal'|'skip')`
- BGM：`playBgm()` / `stopBgm()`，遵循用户静音偏好；`beforeunload` 强制 `stopBgm()` 防泄漏。

**`game-logic.js`** — 规则语义唯一权威。
- 常量：`EMPTY / P1 / P2 / DEAD1 / DEAD2 / PIECE_SHAPES`
- 状态：`board / currentPlayer / pieces / bag / moveHistory / lastPlacedMove / cachedLegalMoves / komi`
- 核心函数：`canPlace` / `isLegalMove`（局部落子+撤销，不再全盘快照）/ `getGroup` / `captureGroupsOf` / `checkLineClears` / `placePiece` / `nextTurn` / `endGame`
- 围杀规则：Go 式气数判断；自杀手若无气则非法
- 消行规则：俄罗斯方块式满行 / 满列消除

**`ai-onnx.js`** — AI 入口 `aiMove()`，按 UI 选择分发到：
- Level 1-3: `heuristicAiMove(level)`（同步评分）
- Level 4: `neuralNetworkMove(myGen)`（ONNX policy head + 温度采样）
- Level 5: `mctsMove(myGen)` → `mctsSearch` + `MCTSNode` + `SimGame`

**AI 取消机制**（见 `aiGeneration` / `invalidateAI` / `aiStillValid`）：
所有异步 AI 在开始时捕获 `myGen = aiGeneration`，每个 `await` 恢复点校验 `aiStillValid(myGen)`；
`startGame` / `setMode` / `changeBoardSize` / `endGame` / `restoreGameState` / `startReplay` 均会 `invalidateAI()`，保证旧 AI 的结果不会落到新局棋盘上。

**`ui.js`** — 纯展示层，不修改规则语义。
- Canvas 渲染 + `requestAnimationFrame` 动画循环（`animTick` + 缓动 `easeOutBack`/`easeInQuad`）
- 输入：鼠标（hover 预览 / 点击落子 / 右键旋转）+ 键盘（方向键微调、R 旋转）+ 触屏
- 存读档：`getGameState` / `restoreGameState`（localStorage）
- 棋谱：`exportGameRecord`（TXT）+ `parseGameRecord` / `startReplay` / `buildReplaySnapshots`

**`main.js`** — 启动与装配。
- `init()`：绑定事件、`resizeBoard`、首帧渲染
- `startGame()`：重置状态、随机先手、发牌、刷新缓存、必要时排程 AI
- `changeBoardSize` / `setMode` / `onAiLevelChange`：切换时均会 `invalidateAI()`

### 关键数据流：一步落子

```
人类点击 → onBoardClick → doPlace → placePiece（含吃子+消行）
                                       → lastPlacedMove / moveHistory.push
                                       → playSfx / 动画 → nextTurn
                                       → updateLegalMovesCache
                                       → 若 PvAI & P2: setTimeout(aiMove, 400)

aiMove（捕获 myGen） → 对应 AI 函数
                       → 每个 await 后 aiStillValid(myGen) 门闸
                       → executeAiMove(move, label)  // 与人类 doPlace 走同一条后处理
```

### ONNX 推理接口

`loadOnnxModel()` 惰性加载 `checkpoints_m5_mps/best_browser.onnx`。
输入张量形状 `[1, 22, S, S]`（`INPUT_CHANNELS=22`），顺序与 Python 侧 `encode_state` 保持一致；
动作索引定义：`idx = rot * S * S + r * S + c`，`rot ∈ {0,1,2,3}`。
**任何修改 `encode_state` 或 `PIECE_SHAPES` 顺序的改动都必须同步 Python 训练侧的编码函数**，否则加载旧 checkpoint 会语义错位。

---

## 后端（Python 训练）

**`cli/tetris_weiqi.py`** — 规则引擎 Python 版，`TetrisWeiqi` 类暴露 `make_move / get_legal_moves / check_captures / check_line_clears / get_score` 等；自对弈和评估均走此实现。

**`cli/train_alphazero.py`** — 训练主程序。关键组件：
- `PolicyValueNet`：ResNet（`res_blocks × channels`），双头（policy `4*S*S` + value 标量）
- `MCTS` / `MCTSNode`：对应前端的 Python 版
- `self_play_games_parallel`：多局并行 + 推理批打包（`inference_batch_size`）
- `ReplayBuffer`：索引采样（避免 `list(deque)` 整表转换）
- `train_step`：交叉熵 policy + MSE value + L2
- 评估：`evaluate_vs_heuristic`（对启发式 AI）+ `evaluate_model_vs_model`（对历史 best）
- 产物：
  - `best.pt` / `ckpt_iter{N}.pt`（torch checkpoint）
  - `eval_history.jsonl`（评估明细）
  - `metrics.jsonl`（每轮一行结构化指标，见 `TRAINING_GUIDE.md`）
  - `export_onnx` → `best_browser.onnx`（供前端加载）

**`check_rule_consistency.py`** — 前后端规则对齐守门。修改任何规则都应跑这个脚本。

---

## 前后端衔接点（务必同步修改）

| 语义 | 前端位置 | Python 位置 |
|---|---|---|
| 棋子形状表 | `PIECE_SHAPES` @ game-logic.js | `PIECE_SHAPES` @ tetris_weiqi.py |
| 状态编码 22 通道 | `encodeState` @ ai-onnx.js | `encode_state` @ train_alphazero.py |
| 动作索引 | `rot*S*S + r*S + c` | 同 |
| 吃子规则 | `captureGroupsOf` | `TetrisWeiqi.check_captures` |
| 消行规则 | `checkLineClears` | `TetrisWeiqi.check_line_clears` |
| 终局条件 | `endGame` / 双跳 | `end_condition_mode` 参数 |
| 贴目 komi | `komi` + `getEffectiveScores` | `--komi` |

**不要只改一侧**。若修改对局语义，一定跑 `cli/check_rule_consistency.py` 验证。

---

## 规则与优化决策的单一事实源

- 规则设计 & 参数：[`PROJECT_PLAN.md`](PROJECT_PLAN.md)
- 优化验收/保留/回退：[`OPTIMIZATION.md`](OPTIMIZATION.md)
- 阶段进度：[`DEV_PROGRESS.md`](DEV_PROGRESS.md) / [`progress.md`](progress.md)
- 训练操作：[`TRAINING_GUIDE.md`](TRAINING_GUIDE.md)
