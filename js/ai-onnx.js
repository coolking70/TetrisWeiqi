// ============================================================
// ONNX Neural Network AI + Heuristic AI + MCTS
// ============================================================
let onnxSession = null;
let onnxLoading = false;
const ONNX_MODEL_PATH = 'checkpoints_m5_mps/best_browser.onnx';
const INPUT_CHANNELS = 22;

// ============================================================
// AI cancellation — bumped on any state change that would
// make an in-flight async AI's result invalid (new game, mode
// switch, board resize, load, replay, end game).
// ============================================================
let aiGeneration = 0;
function invalidateAI() { aiGeneration++; }
function aiStillValid(myGen) {
  return myGen === aiGeneration
      && gameActive
      && currentPlayer === P2
      && gameMode === 'pvai'
      && !replayMode;
}

// ============================================================
// Heuristic AI
// ============================================================
function aiMove() {
  if (!gameActive || currentPlayer !== P2 || gameMode !== 'pvai' || replayMode) return;
  const myGen = aiGeneration;
  const aiLevel = parseInt(document.getElementById('aiLevelSelect').value);
  if (aiLevel === 4) { neuralNetworkMove(myGen); return; }
  if (aiLevel === 5) { mctsMove(myGen); return; }
  heuristicAiMove(aiLevel);
}

function executeAiMove(move, label) {
  rotation = move.rot;
  const placedCells = move.cells.map(([dr, dc]) => [move.row + dr, move.col + dc]);
  const result = placePiece(move.cells, move.row, move.col, P2);
  skipCount = 0;
  lastPlacedMove = { player: P2, cells: placedCells };
  moveHistory.push({
    turn: moveHistory.length + 1,
    player: P2,
    piece: pieces[P2].name,
    rotation: move.rot,
    row: move.row,
    col: move.col,
    captured: result.captured,
    linesCleared: result.linesCleared,
    type: 'place'
  });

  let msg = `AI${label ? '(' + label + ')' : ''} 落子`;
  if (result.captured > 0) msg += ` | 围杀 ${result.captured} 子`;
  if (result.linesCleared > 0) msg += ` | 消除 ${result.linesCleared} 行/列`;
  setStatus(msg);

  playSfx('place');
  let delay = 350;
  if (result.captured > 0) {
    setTimeout(() => playSfx('capture'), 100);
    delay = 600;
  }
  if (result.linesCleared > 0) {
    setTimeout(() => playSfx('clear'), result.captured > 0 ? 550 : 150);
    delay = Math.max(delay, result.captured > 0 ? 1000 : 600);
  }

  generatePiece(P2);
  updatePanels();
  render();
  setTimeout(() => nextTurn(), delay);
}

function aiSkipMove(label) {
  skipCount++;
  if (label === 'MCTS') {
    moveHistory.push({ turn: moveHistory.length + 1, player: P2, type: 'auto_skip' });
  }
  generatePiece(P2);
  if (skipCount >= 2) { endGame(); return; }
  nextTurn();
}

function heuristicAiMove(aiLevel) {
  const piece = pieces[P2];
  let bestScore = -Infinity;
  let bestMoves = [];

  const seen = new Set();
  for (let rot = 0; rot < 4; rot++) {
    const cells = getRotatedCells(piece.cells, rot);
    const key = cells.map(([r,c]) => `${r},${c}`).sort().join(';');
    if (seen.has(key)) continue;
    seen.add(key);
    const bounds = getPieceBounds(cells);
    const maxR = BOARD_SIZE - bounds.rows + 1;
    const maxC = BOARD_SIZE - bounds.cols + 1;
    for (let r = 0; r < maxR; r++) {
      for (let c = 0; c < maxC; c++) {
        if (!isLegalMove(cells, r, c, P2)) continue;

        const boardCopy = board.map(row => [...row]);
        for (const [dr, dc] of cells) board[r + dr][c + dc] = P2;
        const captured = checkCaptures(P2);
        const linesCleared = checkLineClears();
        if (linesCleared > 0) checkCaptures(P2);

        const score = evaluateAI(r, c, cells, captured, linesCleared, aiLevel);

        for (let rr = 0; rr < BOARD_SIZE; rr++)
          for (let cc = 0; cc < BOARD_SIZE; cc++)
            board[rr][cc] = boardCopy[rr][cc];

        if (score > bestScore) {
          bestScore = score;
          bestMoves = [{ rot, row: r, col: c, cells }];
        } else if (score === bestScore) {
          bestMoves.push({ rot, row: r, col: c, cells });
        }
      }
    }
  }

  if (bestMoves.length === 0) {
    aiSkipMove();
    return;
  }

  const move = bestMoves[Math.floor(Math.random() * bestMoves.length)];
  executeAiMove(move, '');
}

function evaluateAI(row, col, cells, captured, linesCleared, level) {
  let score = captured * 15 + linesCleared * 3;

  const center = BOARD_SIZE / 2;
  let avgR = 0, avgC = 0;
  cells.forEach(([dr, dc]) => { avgR += row + dr; avgC += col + dc; });
  avgR /= cells.length; avgC /= cells.length;
  score -= (Math.abs(avgR - center) + Math.abs(avgC - center)) * 0.3;

  let ownAdj = 0, enemyAdj = 0;
  for (const [dr, dc] of cells) {
    const r = row + dr, c = col + dc;
    for (const [nr, nc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
      const ar = r + nr, ac = c + nc;
      if (ar < 0 || ar >= BOARD_SIZE || ac < 0 || ac >= BOARD_SIZE) continue;
      if (board[ar][ac] === P2) ownAdj++;
      if (board[ar][ac] === P1) enemyAdj++;
    }
  }
  score += ownAdj * 1.5;

  if (level >= 2) {
    score += enemyAdj * 2;
    score += countPieces(P2) * 0.1;
  }

  if (level >= 3) {
    const visited = new Set();
    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        const k = r * BOARD_SIZE + c;
        if (visited.has(k) || board[r][c] !== P1) continue;
        const g = getGroup(r, c, visited);
        if (g && g.liberties <= 2) score += (3 - g.liberties) * 5;
      }
    }
    score += captured * 5;
  }

  score += Math.random() * 0.5;
  return score;
}

// ============================================================
// Neural Network AI (ONNX)
// ============================================================
async function loadOnnxModel() {
  if (onnxSession) return onnxSession;
  if (onnxLoading) {
    while (onnxLoading) await new Promise(r => setTimeout(r, 100));
    return onnxSession;
  }
  onnxLoading = true;
  setStatus('加载神经网络模型... 0%');
  try {
    const resp = await fetch(ONNX_MODEL_PATH);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const total = parseInt(resp.headers.get('content-length') || '0');
    const reader = resp.body.getReader();
    const chunks = [];
    let loaded = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      if (total > 0) {
        const pct = Math.min(99, Math.round(loaded / total * 100));
        setStatus(`加载神经网络模型... ${pct}%`);
      }
    }
    const buf = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) { buf.set(chunk, offset); offset += chunk.length; }
    setStatus('加载神经网络模型... 初始化中');
    onnxSession = await ort.InferenceSession.create(buf.buffer);
    setStatus('神经网络模型已加载');
    console.log('ONNX model loaded');
  } catch (e) {
    console.error('ONNX load failed:', e);
    setStatus('模型加载失败，回退到启发式AI');
    onnxLoading = false;
    return null;
  }
  onnxLoading = false;
  return onnxSession;
}

function encodeState(player) {
  const S = BOARD_SIZE;
  const SS = S * S;
  const data = new Float32Array(INPUT_CHANNELS * SS);
  const opponent = player === P1 ? P2 : P1;

  for (let r = 0; r < S; r++) {
    for (let c = 0; c < S; c++) {
      const cell = board[r][c];
      const offset = r * S + c;
      if (cell === EMPTY)    data[offset] = 1;
      else if (cell === player)   data[SS + offset] = 1;
      else if (cell === opponent) data[2 * SS + offset] = 1;
    }
  }

  const piece = pieces[player];
  if (piece) {
    const pidx = PIECE_NAMES.indexOf(piece.name);
    data.subarray((5 + pidx) * SS, (5 + pidx + 1) * SS).fill(1);
  }

  const bagCounts = bagPieceCounts();
  for (let i = 0; i < PIECE_NAMES.length; i++) {
    const val = bagCounts[PIECE_NAMES[i]] / 7.0;
    data.subarray((12 + i) * SS, (12 + i + 1) * SS).fill(val);
  }

  data.subarray(19 * SS, 20 * SS).fill(1);

  const isP1 = player === P1 ? 1 : 0;
  data.subarray(20 * SS, 21 * SS).fill(isP1);
  data.subarray(21 * SS, 22 * SS).fill(1 - isP1);

  return data;
}

async function neuralNetworkMove(myGen) {
  const session = await loadOnnxModel();
  if (!aiStillValid(myGen)) return;
  if (!session) { heuristicAiMove(3); return; }

  const S = BOARD_SIZE;
  const stateData = encodeState(P2);
  const inputTensor = new ort.Tensor('float32', stateData, [1, INPUT_CHANNELS, S, S]);

  const results = await session.run({ state: inputTensor });
  if (!aiStillValid(myGen)) return;
  const policyLogits = results.policy.data;

  const piece = pieces[P2];
  let candidates = [];
  const seen = new Set();
  for (let rot = 0; rot < 4; rot++) {
    const cells = getRotatedCells(piece.cells, rot);
    const key = cells.map(([r,c]) => `${r},${c}`).sort().join(';');
    if (seen.has(key)) continue;
    seen.add(key);
    const bounds = getPieceBounds(cells);
    const maxR = S - bounds.rows + 1;
    const maxC = S - bounds.cols + 1;
    for (let r = 0; r < maxR; r++) {
      for (let c = 0; c < maxC; c++) {
        if (!isLegalMove(cells, r, c, P2)) continue;
        const actionIdx = rot * S * S + r * S + c;
        candidates.push({ rot, row: r, col: c, cells, score: policyLogits[actionIdx] });
      }
    }
  }

  if (candidates.length === 0) {
    if (!aiStillValid(myGen)) return;
    aiSkipMove('神经网络');
    return;
  }

  const temperature = 0.5;
  const maxScore = Math.max(...candidates.map(c => c.score));
  let expSum = 0;
  for (const c of candidates) {
    c.prob = Math.exp((c.score - maxScore) / temperature);
    expSum += c.prob;
  }
  for (const c of candidates) c.prob /= expSum;

  let rand = Math.random();
  let move = candidates[candidates.length - 1];
  for (const c of candidates) {
    rand -= c.prob;
    if (rand <= 0) { move = c; break; }
  }

  if (!aiStillValid(myGen)) return;
  executeAiMove(move, '神经网络');
}

// ============================================================
// MCTS (Monte Carlo Tree Search) with Neural Network
// ============================================================
let MCTS_SIMS = 40;
const MCTS_CPUCT = 1.5;

class SimGame {
  constructor(size) {
    this.size = size;
    this.board = null;
    this.currentPlayer = P1;
    this.pieces = { [P1]: null, [P2]: null };
    this.bag = [];
    this.skipCount = 0;
    this.gameOver = false;
  }

  static fromGlobal() {
    const g = new SimGame(BOARD_SIZE);
    g.board = board.map(r => [...r]);
    g.currentPlayer = currentPlayer;
    g.pieces = {
      [P1]: pieces[P1] ? { name: pieces[P1].name, cells: pieces[P1].cells.map(c => [...c]) } : null,
      [P2]: pieces[P2] ? { name: pieces[P2].name, cells: pieces[P2].cells.map(c => [...c]) } : null,
    };
    g.bag = [...bag];
    g.skipCount = skipCount;
    g.gameOver = false;
    return g;
  }

  clone() {
    const g = new SimGame(this.size);
    g.board = this.board.map(r => [...r]);
    g.currentPlayer = this.currentPlayer;
    g.pieces = {
      [P1]: this.pieces[P1] ? { name: this.pieces[P1].name, cells: this.pieces[P1].cells.map(c => [...c]) } : null,
      [P2]: this.pieces[P2] ? { name: this.pieces[P2].name, cells: this.pieces[P2].cells.map(c => [...c]) } : null,
    };
    g.bag = [...this.bag];
    g.skipCount = this.skipCount;
    g.gameOver = this.gameOver;
    return g;
  }

  _refillBag() {
    this.bag = [...PIECE_NAMES];
    for (let i = this.bag.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.bag[i], this.bag[j]] = [this.bag[j], this.bag[i]];
    }
  }

  _generatePiece(player) {
    if (this.bag.length === 0) this._refillBag();
    const name = this.bag.pop();
    this.pieces[player] = { name, cells: PIECE_SHAPES[name].map(c => [...c]) };
  }

  canPlace(cells, row, col, player) {
    const S = this.size, b = this.board;
    for (const [dr, dc] of cells) {
      const r = row + dr, c = col + dc;
      if (r < 0 || r >= S || c < 0 || c >= S) return false;
      if (b[r][c] !== EMPTY) return false;
    }
    return true;
  }

  _getGroupHasLiberty(row, col) {
    const S = this.size, b = this.board;
    const owner = b[row][col];
    if (owner !== P1 && owner !== P2) return true;
    const visited = new Uint8Array(S * S);
    const stack = [[row, col]];
    visited[row * S + col] = 1;
    while (stack.length) {
      const [r, c] = stack.pop();
      for (const [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
        const nr = r + dr, nc = c + dc;
        if (nr < 0 || nr >= S || nc < 0 || nc >= S) continue;
        const k = nr * S + nc;
        if (visited[k]) continue;
        const cell = b[nr][nc];
        if (cell === EMPTY) return true;
        if (cell === owner) { visited[k] = 1; stack.push([nr, nc]); }
      }
    }
    return false;
  }

  _captureGroupsOf(target) {
    const S = this.size, b = this.board;
    const visited = new Uint8Array(S * S);
    let total = 0;
    for (let r = 0; r < S; r++) {
      for (let c = 0; c < S; c++) {
        const k = r * S + c;
        if (visited[k] || b[r][c] !== target) continue;
        const group = [];
        let hasLib = false;
        const stk = [[r, c]];
        visited[k] = 1;
        while (stk.length) {
          const [gr, gc] = stk.pop();
          group.push([gr, gc]);
          for (const [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
            const nr = gr + dr, nc = gc + dc;
            if (nr < 0 || nr >= S || nc < 0 || nc >= S) continue;
            const nk = nr * S + nc;
            if (visited[nk]) continue;
            const cell = b[nr][nc];
            if (cell === EMPTY) { hasLib = true; }
            else if (cell === target) { visited[nk] = 1; stk.push([nr, nc]); }
          }
        }
        if (!hasLib) {
          for (const [gr, gc] of group) b[gr][gc] = EMPTY;
          total += group.length;
        }
      }
    }
    return total;
  }

  _checkLineClears() {
    const S = this.size, b = this.board;
    let cleared = 0;
    for (let r = 0; r < S; r++) {
      let full = true;
      for (let c = 0; c < S; c++) { if (b[r][c] === EMPTY) { full = false; break; } }
      if (full) { for (let c = 0; c < S; c++) b[r][c] = EMPTY; cleared++; }
    }
    for (let c = 0; c < S; c++) {
      let full = true;
      for (let r = 0; r < S; r++) { if (b[r][c] === EMPTY) { full = false; break; } }
      if (full) { for (let r = 0; r < S; r++) b[r][c] = EMPTY; cleared++; }
    }
    return cleared;
  }

  isLegalMove(cells, row, col, player) {
    if (!this.canPlace(cells, row, col, player)) return false;
    const S = this.size, b = this.board;
    const snapshot = b.map(r => [...r]);
    for (const [dr, dc] of cells) b[row + dr][col + dc] = player;
    const opp = player === P1 ? P2 : P1;
    this._captureGroupsOf(opp);
    let selfDead = false;
    for (const [dr, dc] of cells) {
      if (!this._getGroupHasLiberty(row + dr, col + dc)) { selfDead = true; break; }
    }
    for (let r = 0; r < S; r++) for (let c = 0; c < S; c++) b[r][c] = snapshot[r][c];
    return !selfDead;
  }

  getLegalMoves(player) {
    const piece = this.pieces[player];
    if (!piece) return [];
    const moves = [];
    const seen = new Set();
    for (let rot = 0; rot < 4; rot++) {
      const cells = getRotatedCells(piece.cells, rot);
      const key = cells.map(([r,c]) => `${r},${c}`).sort().join(';');
      if (seen.has(key)) continue;
      seen.add(key);
      const bounds = getPieceBounds(cells);
      const maxR = this.size - bounds.rows + 1;
      const maxC = this.size - bounds.cols + 1;
      for (let r = 0; r < maxR; r++) {
        for (let c = 0; c < maxC; c++) {
          if (this.isLegalMove(cells, r, c, player)) {
            moves.push({ rot, row: r, col: c, cells });
          }
        }
      }
    }
    return moves;
  }

  doMove(rot, row, col) {
    const player = this.currentPlayer;
    const piece = this.pieces[player];
    const cells = getRotatedCells(piece.cells, rot);
    const b = this.board;
    for (const [dr, dc] of cells) b[row + dr][col + dc] = player;
    const opp = player === P1 ? P2 : P1;
    const captured = this._captureGroupsOf(opp);
    this._captureGroupsOf(player);
    const linesCleared = this._checkLineClears();
    if (linesCleared > 0) this._captureGroupsOf(opp);
    this.skipCount = 0;
    this._generatePiece(player);
    this._advanceTurn();
  }

  doSkip() {
    this.skipCount++;
    this._generatePiece(this.currentPlayer);
    if (this.skipCount >= 2) { this.gameOver = true; return; }
    this._advanceTurn();
  }

  _advanceTurn() {
    this.currentPlayer = this.currentPlayer === P1 ? P2 : P1;
    const piece = this.pieces[this.currentPlayer];
    if (!piece) { this.gameOver = true; return; }
    const moves = this.getLegalMoves(this.currentPlayer);
    if (moves.length === 0) {
      this._generatePiece(this.currentPlayer);
      const moves2 = this.getLegalMoves(this.currentPlayer);
      if (moves2.length === 0) {
        this.skipCount++;
        if (this.skipCount >= 2) { this.gameOver = true; return; }
        this._generatePiece(this.currentPlayer);
        this._advanceTurn();
      }
    }
  }

  encodeState(player) {
    const S = this.size;
    const SS = S * S;
    const data = new Float32Array(INPUT_CHANNELS * SS);
    const opponent = player === P1 ? P2 : P1;
    for (let r = 0; r < S; r++) {
      for (let c = 0; c < S; c++) {
        const cell = this.board[r][c];
        const offset = r * S + c;
        if (cell === EMPTY)         data[offset] = 1;
        else if (cell === player)   data[SS + offset] = 1;
        else if (cell === opponent) data[2 * SS + offset] = 1;
      }
    }
    const piece = this.pieces[player];
    if (piece) {
      const pidx = PIECE_NAMES.indexOf(piece.name);
      data.subarray((5 + pidx) * SS, (5 + pidx + 1) * SS).fill(1);
    }
    const bagCounts = {};
    for (const n of PIECE_NAMES) bagCounts[n] = 0;
    for (const n of this.bag) bagCounts[n]++;
    for (let i = 0; i < PIECE_NAMES.length; i++) {
      const val = bagCounts[PIECE_NAMES[i]] / 7.0;
      data.subarray((12 + i) * SS, (12 + i + 1) * SS).fill(val);
    }
    data.subarray(19 * SS, 20 * SS).fill(1);
    const isP1 = player === P1 ? 1 : 0;
    data.subarray(20 * SS, 21 * SS).fill(isP1);
    data.subarray(21 * SS, 22 * SS).fill(1 - isP1);
    return data;
  }

  getScore(player) {
    let n = 0;
    for (let r = 0; r < this.size; r++)
      for (let c = 0; c < this.size; c++)
        if (this.board[r][c] === player) n++;
    return n;
  }
}

class MCTSNode {
  constructor(game, parent, move) {
    this.game = game;
    this.parent = parent;
    this.move = move;
    this.children = [];
    this.visits = 0;
    this.totalValue = 0;
    this.prior = 0;
    this.expanded = false;
  }

  q() { return this.visits > 0 ? this.totalValue / this.visits : 0; }

  ucb(parentVisits) {
    return this.q() + MCTS_CPUCT * this.prior * Math.sqrt(parentVisits) / (1 + this.visits);
  }

  bestChild() {
    let best = null, bestScore = -Infinity;
    for (const child of this.children) {
      const score = child.ucb(this.visits);
      if (score > bestScore) { bestScore = score; best = child; }
    }
    return best;
  }
}

async function mctsSearch(session, rootGame, player, myGen) {
  const S = rootGame.size;
  const root = new MCTSNode(rootGame, null, null);

  await expandNode(session, root, player);
  if (myGen !== undefined && !aiStillValid(myGen)) return null;
  if (root.children.length === 0) return null;

  for (let sim = 0; sim < MCTS_SIMS; sim++) {
    let node = root;
    while (node.expanded && node.children.length > 0) {
      node = node.bestChild();
    }

    let value;
    if (node.game.gameOver) {
      const s1 = node.game.getScore(player);
      const s2 = node.game.getScore(player === P1 ? P2 : P1);
      value = s1 > s2 ? 1 : (s1 < s2 ? -1 : 0);
    } else {
      value = await expandNode(session, node, player);
      if (myGen !== undefined && !aiStillValid(myGen)) return null;
    }

    let cur = node;
    while (cur) {
      cur.visits++;
      cur.totalValue += value;
      cur = cur.parent;
    }

    if (sim % 8 === 7) {
      setStatus(`AI 思考中... (${sim + 1}/${MCTS_SIMS})`);
      await new Promise(r => setTimeout(r, 0));
      if (myGen !== undefined && !aiStillValid(myGen)) return null;
    }
  }

  let bestChild = null, bestVisits = -1;
  for (const child of root.children) {
    if (child.visits > bestVisits) { bestVisits = child.visits; bestChild = child; }
  }
  return bestChild ? bestChild.move : null;
}

async function expandNode(session, node, mctsPlayer) {
  const game = node.game;
  const player = game.currentPlayer;
  const legalMoves = game.getLegalMoves(player);

  if (legalMoves.length === 0) {
    node.expanded = true;
    return 0;
  }

  const stateData = game.encodeState(player);
  const inputTensor = new ort.Tensor('float32', stateData, [1, INPUT_CHANNELS, game.size, game.size]);
  const results = await session.run({ state: inputTensor });
  const policyLogits = results.policy.data;
  const value = results.value.data[0];

  const S = game.size;
  let maxLogit = -Infinity;
  for (const m of legalMoves) {
    const idx = m.rot * S * S + m.row * S + m.col;
    m.logit = policyLogits[idx];
    if (m.logit > maxLogit) maxLogit = m.logit;
  }
  let expSum = 0;
  for (const m of legalMoves) {
    m.prob = Math.exp(m.logit - maxLogit);
    expSum += m.prob;
  }

  for (const m of legalMoves) {
    const childGame = game.clone();
    childGame.doMove(m.rot, m.row, m.col);
    const child = new MCTSNode(childGame, node, { rot: m.rot, row: m.row, col: m.col, cells: m.cells });
    child.prior = m.prob / expSum;
    node.children.push(child);
  }
  node.expanded = true;

  return player === mctsPlayer ? value : -value;
}

async function mctsMove(myGen) {
  if (myGen === undefined) myGen = aiGeneration;
  if (!aiStillValid(myGen)) return;

  const session = await loadOnnxModel();
  if (!aiStillValid(myGen)) return;
  if (!session) { heuristicAiMove(3); return; }

  setStatus('AI 思考中... (0/' + MCTS_SIMS + ')');
  await new Promise(r => setTimeout(r, 0));
  if (!aiStillValid(myGen)) return;

  const rootGame = SimGame.fromGlobal();
  const bestMove = await mctsSearch(session, rootGame, P2, myGen);
  if (!aiStillValid(myGen)) return;

  if (!bestMove) {
    aiSkipMove('MCTS');
    return;
  }

  executeAiMove(bestMove, 'MCTS');
}
