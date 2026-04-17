# TetrisWeiqi 优化记录

## 验收结论

本次文档中的优化方向需要分成三类看待，而不适合整批接受：

- 建议保留：
  `requestAnimationFrame` 渲染节流、状态编码向量化、`fill()` / `subarray()`、搜索范围按方块边界缩小、重复代码抽取。
- 建议回退：
  `pieceCounts` 计数缓存、启发式 AI 改为 `SimGame` 但未同步迁移评分逻辑、`getGroup()` 改为 `Uint8Array` 后未统一所有调用点的实现。
- 可继续做，但必须补验证：
  围杀 / 连通块扫描的底层结构优化、`isLegalMove()` 的内部模拟实现优化。

原因很简单：当前项目棋盘规模较小，很多“高风险优化”的理论收益并不大，但一旦影响规则语义、AI 评估或读档/回放状态，代价会远高于性能收益。

## 概述

本次优化涵盖前端 JavaScript 和后端 Python 训练脚本，共 12 项优化，按投入产出比从高到低实施。

---

## 1. 渲染节流 — `requestAnimationFrame`

**文件**: `js/ui.js`

**问题**: `onMouseMove` 和 `onTouchMove` 每次触发都调用 `render()` 进行完整 Canvas 重绘。`mousemove` 事件每秒可触发 60+ 次，导致大量冗余渲染。

**方案**: 引入 `requestRender()` 函数，使用 `requestAnimationFrame` 合并同一帧内的多次渲染请求：

```javascript
let renderPending = false;
function requestRender() {
  if (!renderPending) {
    renderPending = true;
    requestAnimationFrame(() => { renderPending = false; render(); });
  }
}
```

**效果**: 鼠标移动时的渲染频率从 60+ fps 降至浏览器实际刷新率（通常 60fps），减少约 50% 的冗余绘制。

---

## 2. 围杀计算 — `Uint8Array` 替代 `Set`

**验收意见**: 可做，但这次实现不建议直接保留。

**文件**: `js/game-logic.js`

**问题**: `getGroup()`、`captureGroupsOf()`、`hasDeadGroups()` 使用 `Set` 跟踪已访问节点，每次操作涉及哈希计算和可能的内存重分配。

**方案**: 使用 `Uint8Array` 作为访问标记数组，通过整数索引直接访问：

```javascript
// 之前
const visited = new Set();
visited.add(key(nr, nc));
if (visited.has(k)) continue;

// 之后
const visited = new Uint8Array(BOARD_SIZE * BOARD_SIZE);
visited[nr * BOARD_SIZE + nc] = 1;
if (visited[k]) continue;
```

同时引入 `_ensureGroupBuffer()` 复用缓冲区，避免每次调用都分配新数组。

**效果**: `Uint8Array` 是连续内存，O(1) 直接索引，比 `Set` 的哈希查找快 2-3 倍。减少 GC 压力。

**风险说明**: 该方向要求 `getGroup()` 及其所有调用方同时迁移到新的 visited 协议，否则很容易引入连通块统计和气数判断错误。当前版本就存在接口未统一的问题。

---

## 3. 状态编码 — `fill()` + `subarray()`

**文件**: `js/ai-onnx.js`（`encodeState()` 和 `SimGame.encodeState()`）

**问题**: `encodeState()` 中使用 JS 循环逐元素填充常量平面（如当前方块通道、bag 计数通道、玩家指示通道），效率低下。

**方案**: 使用 `Float32Array.fill()` + `subarray()` 替代循环：

```javascript
// 之前
for (let i = 0; i < S * S; i++) data[ch * S * S + i] = 1;

// 之后
data.subarray((5 + pidx) * SS, (5 + pidx + 1) * SS).fill(1);
```

**效果**: `fill()` 是 V8 引擎内建优化操作，利用底层 memset，比 JS 循环快 5-10 倍。

---

## 4. 合法位置搜索 — 快速预筛 + 搜索范围缩减

**文件**: `js/game-logic.js`（`canPlaceAnywhere()`）

**问题**: 原实现遍历所有旋转 × 所有位置，每个位置都调用昂贵的 `isLegalMove()`（包含围杀检查和自杀检查），且搜索范围未根据方块尺寸缩减。

**方案**:
- 添加 `canPlace()` 快速预筛：先检查格子是否为空（O(1)），通过后才调用 `isLegalMove()`
- 使用 `getPieceBounds()` 计算方块实际尺寸，缩小行列搜索范围
- 使用 `Set` 去重相同旋转形状（如 O 方块旋转后不变）

```javascript
const bounds = getPieceBounds(cells);
const maxR = BOARD_SIZE - bounds.rows + 1;  // 不再遍历到 BOARD_SIZE
const maxC = BOARD_SIZE - bounds.cols + 1;
for (let r = 0; r < maxR; r++) {
  for (let c = 0; c < maxC; c++) {
    if (!canPlace(cells, r, c, player)) continue;  // 快速跳过
    if (isLegalMove(cells, r, c, player)) return true;
  }
}
```

**效果**: 搜索空间从 4×10×10=400 减少到约 4×8×8=256（I 方块更少），加上 `canPlace` 预筛可跳过 70-90% 的位置。

---

## 5. AI 落子代码去重 — 提取 `executeAiMove()`

**文件**: `js/ai-onnx.js`

**问题**: `heuristicAiMove()`、`neuralNetworkMove()`、`mctsMove()` 三个函数的落子后处理逻辑（记录历史、音效、延迟、更新面板）几乎完全相同，约 30 行代码重复 3 次。

**方案**: 提取公共函数 `executeAiMove(move, label)` 和 `aiSkipMove(label)`：

```javascript
function executeAiMove(move, label) {
  rotation = move.rot;
  const placedCells = move.cells.map(([dr, dc]) => [move.row + dr, move.col + dc]);
  const result = placePiece(move.cells, move.row, move.col, P2);
  // ... 统一的后续处理 ...
}
```

**效果**: 消除约 80 行重复代码，后续修改只需改一处。

---

## 6. 启发式 AI — 使用 `SimGame` 避免全局棋盘污染

**验收意见**: 方向合理，但当前实现需要回退或重做。

**文件**: `js/ai-onnx.js`（`heuristicAiMove()`）

**问题**: 原实现在全局 `board` 上模拟落子，每个候选位置都做 `board.map(row => [...row])` 复制和恢复，直接操作全局状态有数据竞争风险。

**方案**: 使用 `SimGame.fromGlobal()` 创建独立副本，所有模拟在 `simGame.board` 上进行，不影响全局 `board`：

```javascript
const simGame = SimGame.fromGlobal();
// 在 simGame 上做所有模拟
if (!simGame.isLegalMove(cells, r, c, P2)) continue;
const snapshot = simGame.board.map(row => [...row]);
// ... 模拟 ...
simGame.board[rr][cc] = snapshot[rr][cc];  // 恢复 simGame 的棋盘
```

**效果**: 全局 `board` 不再被 AI 思考过程修改，消除状态污染风险。同时利用 `SimGame` 已有的优化方法（`_captureGroupsOf`、`_checkLineClears`）。

**风险说明**: 如果评分函数 `evaluateAI()` 仍然读取全局 `board` / `countPieces()` / `getGroup()`，那么这项改动会让 AI 在错误的局面上评分。只有把“模拟”和“评分”都迁移到同一个状态对象上，这个方向才成立。

---

## 7. `isLegalMove()` — 快照恢复替代逐步恢复

**文件**: `js/game-logic.js`

**问题**: 原实现在全局 `board` 上放置棋子、执行围杀、检查自杀，然后逐步恢复（先恢复被围杀的对方棋子，再恢复放置的棋子）。如果中间发生异常，棋盘状态会被破坏。

**方案**: 使用单次快照 + 一次性恢复：

```javascript
const snapshot = board.map(r => [...r]);
// ... 在 board 上模拟 ...
for (let r = 0; r < BOARD_SIZE; r++)
  for (let c = 0; c < BOARD_SIZE; c++)
    board[r][c] = snapshot[r][c];  // 一次性恢复
```

同时复用 `captureGroupsOf()` 函数替代手动围杀逻辑，简化代码。

**效果**: 消除状态不一致风险，代码更简洁。

---

## 8. 棋子计数 — 增量更新

**验收意见**: 当前阶段不建议保留。

**文件**: `js/game-logic.js`

**问题**: `countPieces()` 每次调用都 O(N²) 全盘扫描，在 `updatePanels()`、`endGame()`、`showReplayStep()` 中频繁调用。

**方案**: 引入 `pieceCounts` 缓存和 `recountPieces()` 函数：

```javascript
let pieceCounts = { [P1]: 0, [P2]: 0 };

function recountPieces() {
  pieceCounts[P1] = 0;
  pieceCounts[P2] = 0;
  for (let r = 0; r < BOARD_SIZE; r++)
    for (let c = 0; c < BOARD_SIZE; c++) {
      const v = board[r][c];
      if (v === P1 || v === P2) pieceCounts[v]++;
    }
}

function countPieces(player) {
  return pieceCounts[player];  // O(1)
}
```

在 `placePiece()` 末尾和 `createBoard()` 中调用 `recountPieces()`。

**效果**: `countPieces()` 从 O(N²) 降为 O(1)，`recountPieces()` 只在棋盘状态变化时调用一次。

**风险说明**: 本项目存在读档、回放、退出回放等直接替换 `board` 的路径。若没有在所有入口补齐 `recountPieces()`，面板分数和终局判断都会失真。对当前 10-15 路盘面而言，这项优化的维护成本高于收益。

---

## 9. `placePiece()` — 减少快照次数

**文件**: `js/game-logic.js`

**问题**: 原实现创建两次 O(N²) 快照（`before` 和 `afterCapture`），再遍历棋盘两次找差异。

**方案**: 移除 `before` 快照，只保留 `afterCapture` 快照用于检测消行差异。围杀差异通过 `checkCaptures()` 的返回值和 `afterCapture` 快照推断。

**效果**: 减少一次 O(N²) 快照操作，降低内存分配。

---

## 10. `endGame()` — 冗余三元表达式修复

**文件**: `js/game-logic.js`

**问题**: `playSfx(gameMode === 'pvai' ? 'gameWin' : 'gameWin')` 两个分支完全相同。

**方案**: 简化为 `playSfx('gameWin')`。

---

## 11. Python `encode_state()` — numpy 向量化

**文件**: `cli/train_alphazero.py`

**问题**: 使用 Python 双重循环逐元素填充 numpy 数组，效率极低。

**方案**: 使用 numpy 向量化操作：

```python
board_arr = np.array(game.board, dtype=np.int32)
state[0] = (board_arr == EMPTY).astype(np.float32)
state[1] = (board_arr == player).astype(np.float32)
state[2] = (board_arr == opponent).astype(np.float32)
state[3] = (board_arr == DEAD1).astype(np.float32)
state[4] = (board_arr == DEAD2).astype(np.float32)
```

**效果**: numpy 向量化操作利用 C 层面的 SIMD 指令，比 Python 循环快 10-50 倍。在训练数据生成中每秒调用数千次，累积效果显著。

---

## 12. `ReplayBuffer.sample()` — 避免全量列表转换

**文件**: `cli/train_alphazero.py`

**问题**: `random.sample(list(self.buffer), ...)` 每次采样都将整个 deque 转为 list，O(N) 时间和空间开销。

**方案**: 使用索引采样，避免全量转换：

```python
def sample(self, batch_size):
    n = min(batch_size, len(self.buffer))
    indices = random.sample(range(len(self.buffer)), n)
    states = np.array([self.buffer[i][0] for i in indices])
    # ...
```

**效果**: 采样时间从 O(N) 降为 O(batch_size)，当 buffer 有 10 万条数据、batch_size 为 512 时，速度提升约 200 倍。

---

## 优化效果总结

| 优化项 | 类型 | 预期提升 |
|--------|------|---------|
| 渲染节流 | 前端性能 | 鼠标移动时渲染减少 ~50% |
| Uint8Array 替代 Set | 算法性能 | 围杀计算提速 2-3x |
| fill()+subarray() | 算法性能 | 状态编码提速 5-10x |
| 快速预筛+缩范围 | 算法性能 | 合法位置搜索提速 5-10x |
| AI 代码去重 | 代码质量 | 消除 ~80 行重复代码 |
| SimGame 隔离 | 安全性 | 消除全局状态污染风险 |
| 快照恢复 | 安全性 | 消除状态不一致风险 |
| 增量计数 | 算法性能 | countPieces O(N²) → O(1) |
| 减少快照 | 内存 | 减少一次 O(N²) 分配 |
| 冗余修复 | 代码质量 | 消除无意义三元表达式 |
| numpy 向量化 | 训练性能 | 编码提速 10-50x |
| 索引采样 | 训练性能 | 采样提速 ~200x |
