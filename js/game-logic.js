// ============================================================
// Constants
// ============================================================
const EMPTY = 0, P1 = 1, P2 = 2, DEAD1 = 3, DEAD2 = 4;
const P1_COLOR = '#4fc3f7', P1_DARK = '#2980b9';
const P2_COLOR = '#e040fb', P2_DARK = '#9c27b0';
const GRID_COLOR = '#1e2d3d', BG_COLOR = '#0d1117';
const GHOST_ALPHA = 0.35;

// Standard 7 Tetris pieces as [row, col] offsets
const PIECE_SHAPES = {
  I: [[0,0],[0,1],[0,2],[0,3]],
  O: [[0,0],[0,1],[1,0],[1,1]],
  T: [[0,0],[0,1],[0,2],[1,1]],
  S: [[0,1],[0,2],[1,0],[1,1]],
  Z: [[0,0],[0,1],[1,1],[1,2]],
  L: [[0,0],[1,0],[2,0],[2,1]],
  J: [[0,1],[1,1],[2,0],[2,1]]
};
const PIECE_NAMES = Object.keys(PIECE_SHAPES);

// ============================================================
// State
// ============================================================
let BOARD_SIZE = 10;
let CELL_SIZE = 32;
let board = [];
let currentPlayer = P1;
let pieces = { [P1]: null, [P2]: null };
let gameActive = false;
let gameMode = 'pvai';
let ghostPos = null;
let rotation = 0;
let skipCount = 0;
let komi = 0.0; // 贴目: 正值补偿后手方
let firstPlayer = P1;
let bag = []; // shared bag7
let moveHistory = []; // 棋谱记录
let lastPlacedMove = null; // { player, cells: [[r,c],...] } — 上一手落子位置
let cachedLegalMoves = null; // { rotation, moves: [{row,col,cells}] } — 当前旋转的合法位置缓存

let boardCanvas, boardCtx, previewP1, prevCtx1, previewP2, prevCtx2;

// Animation state
let animations = []; // { type, cells, color, startTime, duration }
let animFrameId = null;

// ============================================================
// Board
// ============================================================
function createBoard() {
  board = [];
  for (let r = 0; r < BOARD_SIZE; r++) {
    board.push(new Array(BOARD_SIZE).fill(EMPTY));
  }
}

// ============================================================
// Pieces
// ============================================================
function refillBag() {
  bag = [...PIECE_NAMES];
  for (let i = bag.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [bag[i], bag[j]] = [bag[j], bag[i]];
  }
}

function generatePiece(player) {
  if (bag.length === 0) refillBag();
  const name = bag.pop();
  pieces[player] = { name, cells: PIECE_SHAPES[name].map(c => [...c]) };
}

function bagPieceCounts() {
  const counts = {};
  for (const n of PIECE_NAMES) counts[n] = 0;
  for (const n of bag) counts[n]++;
  return counts;
}

function getRotatedCells(cells, rot) {
  let c = cells.map(p => [...p]);
  for (let i = 0; i < (rot % 4); i++) {
    c = c.map(([r, cl]) => [cl, -r]);
  }
  let minR = Infinity, minC = Infinity;
  c.forEach(([r, cl]) => { minR = Math.min(minR, r); minC = Math.min(minC, cl); });
  return c.map(([r, cl]) => [r - minR, cl - minC]);
}

function getCurrentCells() {
  if (!pieces[currentPlayer]) return [];
  return getRotatedCells(pieces[currentPlayer].cells, rotation);
}

function updateLegalMovesCache() {
  if (!gameActive || !pieces[currentPlayer]) { cachedLegalMoves = null; return; }
  const cells = getCurrentCells();
  const moves = [];
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      if (isLegalMove(cells, r, c, currentPlayer)) {
        moves.push({ row: r, col: c });
      }
    }
  }
  cachedLegalMoves = { rotation, cells, moves };
  // 当前旋转无合法落子时播放提示音效并刷新预览（红色）
  if (moves.length === 0 && gameActive && !(gameMode === 'pvai' && currentPlayer === P2)) {
    playIllegalSfx();
    renderPreview(currentPlayer);
  }
}

function getPieceBounds(cells) {
  let maxR = 0, maxC = 0;
  cells.forEach(([r, c]) => { maxR = Math.max(maxR, r); maxC = Math.max(maxC, c); });
  return { rows: maxR + 1, cols: maxC + 1 };
}

// ============================================================
// Placement Validation
// ============================================================
// 围棋式规则：围杀后直接移除变空格，不产生死区
function cellAllowedFor(cell, player) {
  return cell === EMPTY;
}

function canPlace(cells, row, col, player) {
  for (const [dr, dc] of cells) {
    const r = row + dr, c = col + dc;
    if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) return false;
    if (!cellAllowedFor(board[r][c], player)) return false;
  }
  return true;
}

// 完整合法性检查：canPlace + 自杀禁手
function isLegalMove(cells, row, col, player) {
  if (!canPlace(cells, row, col, player)) return false;

  const placed = [];
  for (const [dr, dc] of cells) {
    const r = row + dr, c = col + dc;
    board[r][c] = player;
    placed.push([r, c]);
  }

  const opponent = player === P1 ? P2 : P1;
  const capturedCells = [];
  const oppVisited = new Set();
  for (const [r, c] of placed) {
    for (const [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
      const nr = r + dr, nc = c + dc;
      if (nr < 0 || nr >= BOARD_SIZE || nc < 0 || nc >= BOARD_SIZE) continue;
      if (board[nr][nc] !== opponent) continue;
      const k = nr * BOARD_SIZE + nc;
      if (oppVisited.has(k)) continue;
      const g = getGroup(nr, nc, oppVisited);
      if (g && g.liberties === 0) {
        for (const [gr, gc] of g.group) {
          board[gr][gc] = EMPTY;
          capturedCells.push([gr, gc]);
        }
      }
    }
  }

  let selfDead = false;
  const selfVisited = new Set();
  for (const [r, c] of placed) {
    const k = r * BOARD_SIZE + c;
    if (selfVisited.has(k)) continue;
    const g = getGroup(r, c, selfVisited);
    if (g && g.liberties === 0) { selfDead = true; break; }
  }

  for (const [r, c] of capturedCells) board[r][c] = opponent;
  for (const [r, c] of placed) board[r][c] = EMPTY;

  return !selfDead;
}

function canPlaceAnywhere(player, pieceCells) {
  const seen = new Set();
  for (let rot = 0; rot < 4; rot++) {
    const cells = getRotatedCells(pieceCells, rot);
    const key = cells.map(([r,c]) => `${r},${c}`).sort().join(';');
    if (seen.has(key)) continue;
    seen.add(key);
    const bounds = getPieceBounds(cells);
    const maxR = BOARD_SIZE - bounds.rows + 1;
    const maxC = BOARD_SIZE - bounds.cols + 1;
    for (let r = 0; r < maxR; r++) {
      for (let c = 0; c < maxC; c++) {
        if (!canPlace(cells, r, c, player)) continue;
        if (isLegalMove(cells, r, c, player)) return true;
      }
    }
  }
  return false;
}

// ============================================================
// Go Capture Logic
// ============================================================
function getGroup(row, col, visited) {
  const owner = board[row][col];
  if (owner !== P1 && owner !== P2) return null;
  const group = [];
  const liberties = new Set();
  const stack = [[row, col]];
  const key = (r, c) => r * BOARD_SIZE + c;
  visited.add(key(row, col));

  while (stack.length) {
    const [r, c] = stack.pop();
    group.push([r, c]);
    for (const [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
      const nr = r + dr, nc = c + dc;
      if (nr < 0 || nr >= BOARD_SIZE || nc < 0 || nc >= BOARD_SIZE) continue;
      const k = key(nr, nc);
      if (visited.has(k)) continue;
      const cell = board[nr][nc];
      if (cell === EMPTY) {
        liberties.add(k);
      } else if (cell === owner) {
        visited.add(k);
        stack.push([nr, nc]);
      }
    }
  }
  return { owner, group, liberties: liberties.size };
}

function captureGroupsOf(target) {
  const visited = new Set();
  let totalCaptured = 0;

  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      const k = r * BOARD_SIZE + c;
      if (visited.has(k) || board[r][c] !== target) continue;
      const g = getGroup(r, c, visited);
      if (g && g.liberties === 0) {
        g.group.forEach(([gr, gc]) => { board[gr][gc] = EMPTY; });
        totalCaptured += g.group.length;
      }
    }
  }
  return totalCaptured;
}

function hasDeadGroups(player) {
  const visited = new Set();
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      const k = r * BOARD_SIZE + c;
      if (visited.has(k) || board[r][c] !== player) continue;
      const g = getGroup(r, c, visited);
      if (g && g.liberties === 0) return true;
    }
  }
  return false;
}

function checkCaptures(placedBy) {
  const opponent = placedBy === P1 ? P2 : P1;
  const captured = captureGroupsOf(opponent);
  captureGroupsOf(placedBy);
  return captured;
}

// ============================================================
// Tetris Line Clear
// ============================================================
function checkLineClears() {
  let cleared = 0;
  for (let r = 0; r < BOARD_SIZE; r++) {
    let full = true;
    for (let c = 0; c < BOARD_SIZE; c++) {
      if (board[r][c] === EMPTY) { full = false; break; }
    }
    if (full) {
      for (let c = 0; c < BOARD_SIZE; c++) board[r][c] = EMPTY;
      cleared++;
    }
  }
  for (let c = 0; c < BOARD_SIZE; c++) {
    let full = true;
    for (let r = 0; r < BOARD_SIZE; r++) {
      if (board[r][c] === EMPTY) { full = false; break; }
    }
    if (full) {
      for (let r = 0; r < BOARD_SIZE; r++) board[r][c] = EMPTY;
      cleared++;
    }
  }
  return cleared;
}

// ============================================================
// Scoring
// ============================================================
function countPieces(player) {
  let n = 0;
  for (let r = 0; r < BOARD_SIZE; r++)
    for (let c = 0; c < BOARD_SIZE; c++)
      if (board[r][c] === player) n++;
  return n;
}

function getKomiReceiver() {
  return firstPlayer === P1 ? P2 : P1;
}

function getEffectiveScores() {
  const rawP1 = countPieces(P1);
  const rawP2 = countPieces(P2);
  let effP1 = rawP1;
  let effP2 = rawP2;

  if (komi !== 0) {
    if (getKomiReceiver() === P1) effP1 += komi;
    else effP2 += komi;
  }

  return { rawP1, rawP2, effP1, effP2 };
}

// ============================================================
// Place & Turn
// ============================================================
function placePiece(cells, row, col, player) {
  const before = board.map(r => [...r]);
  const placedCells = [];
  for (const [dr, dc] of cells) {
    const r = row + dr, c = col + dc;
    board[r][c] = player;
    placedCells.push([r, c]);
  }
  const captured = checkCaptures(player);
  const afterCapture = board.map(r => [...r]);
  const linesCleared = checkLineClears();
  if (linesCleared > 0) checkCaptures(player);

  const capturedCells = [];
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      const was = before[r][c], afterCap = afterCapture[r][c];
      if ((was === P1 || was === P2) && afterCap === EMPTY && !placedCells.some(([pr, pc]) => pr === r && pc === c)) {
        capturedCells.push([r, c]);
      }
    }
  }
  const clearedCells = [];
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      const afterCap = afterCapture[r][c], now = board[r][c];
      if (afterCap !== EMPTY && now === EMPTY) {
        clearedCells.push([r, c, afterCap]);
      }
    }
  }

  const t = performance.now();
  const color = player === P1 ? P1_COLOR : P2_COLOR;
  if (placedCells.length > 0) {
    animations.push({ type: 'place', cells: placedCells, color, startTime: t, duration: 250 });
  }
  if (capturedCells.length > 0) {
    animations.push({ type: 'capture', cells: capturedCells, startTime: t + 100, duration: 400 });
  }
  if (clearedCells.length > 0) {
    animations.push({ type: 'clear', cells: clearedCells, startTime: t + 150, duration: 350 });
  }
  startAnimLoop();

  return { captured, linesCleared };
}

function nextTurn() {
  currentPlayer = currentPlayer === P1 ? P2 : P1;
  rotation = 0;
  ghostPos = null;

  if (!canPlaceAnywhere(currentPlayer, pieces[currentPlayer].cells)) {
    generatePiece(currentPlayer);
    if (!canPlaceAnywhere(currentPlayer, pieces[currentPlayer].cells)) {
      skipCount++;
      moveHistory.push({
        turn: moveHistory.length + 1,
        player: currentPlayer,
        type: 'auto_skip'
      });
      if (skipCount >= 2) { endGame(); return; }
      setStatus(playerName(currentPlayer) + ' 无法落子，跳过');
      playSkipSfx();
      nextTurn();
      return;
    }
  }

  updatePanels();
  setStatus(playerName(currentPlayer) + ' 的回合');
  updateLegalMovesCache();
  render();

  if (gameMode === 'pvai' && currentPlayer === P2 && gameActive) {
    setTimeout(aiMove, 400);
  }
}

function skipTurn() {
  if (!gameActive || replayMode) return;
  if (gameMode === 'pvai' && currentPlayer === P2) return;
  skipCount++;
  moveHistory.push({
    turn: moveHistory.length + 1,
    player: currentPlayer,
    type: 'skip'
  });
  generatePiece(currentPlayer);
  playSkipSfx();
  if (skipCount >= 2) { endGame(); return; }
  setStatus(playerName(currentPlayer) + ' 跳过了回合');
  nextTurn();
}

function endGame() {
  gameActive = false;
  stopBgm();
  const { rawP1: s1, rawP2: s2, effP1: eff1, effP2: eff2 } = getEffectiveScores();
  const komiReceiver = getKomiReceiver();

  let result;
  if (eff1 > eff2) {
    result = playerName(P1) + ' 获胜!';
    playSfx('gameWin');
  } else if (eff2 > eff1) {
    result = playerName(P2) + ' 获胜!';
    playSfx(gameMode === 'pvai' ? 'gameLose' : 'gameWin');
  } else {
    result = '平局!';
    playSfx('gameDraw');
  }

  document.getElementById('gameOverTitle').textContent = '游戏结束';
  document.getElementById('gameOverResult').textContent = result;
  const komiStrP1 = komi !== 0 && komiReceiver === P1 ? ` (贴目 ${komi > 0 ? '+' : ''}${komi})` : '';
  const komiStrP2 = komi !== 0 && komiReceiver === P2 ? ` (贴目 ${komi > 0 ? '+' : ''}${komi})` : '';
  document.getElementById('gameOverDetail').textContent =
    `${playerName(P1)}: ${s1} 方块${komiStrP1}\n${playerName(P2)}: ${s2} 方块${komiStrP2}`;
  document.getElementById('gameOverOverlay').classList.add('show');
  updatePanels();
  render();
}

function closeOverlay() {
  document.getElementById('gameOverOverlay').classList.remove('show');
}
