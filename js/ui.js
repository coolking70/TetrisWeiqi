// ============================================================
// Input
// ============================================================
let renderPending = false;
function requestRender() {
  if (!renderPending) {
    renderPending = true;
    requestAnimationFrame(() => { renderPending = false; render(); });
  }
}

function onMouseMove(e) {
  if (!gameActive || (gameMode === 'pvai' && currentPlayer === P2)) return;
  const rect = boardCanvas.getBoundingClientRect();
  const col = Math.floor((e.clientX - rect.left) / CELL_SIZE);
  const row = Math.floor((e.clientY - rect.top) / CELL_SIZE);
  const cells = getCurrentCells();
  const bounds = getPieceBounds(cells);
  ghostPos = { row: row - Math.floor(bounds.rows / 2), col: col - Math.floor(bounds.cols / 2) };
  requestRender();
}

function onBoardClick(e) {
  if (!gameActive || (gameMode === 'pvai' && currentPlayer === P2) || !ghostPos) return;
  const cells = getCurrentCells();
  if (!isLegalMove(cells, ghostPos.row, ghostPos.col, currentPlayer)) {
    playIllegalSfx();
    return;
  }
  doPlace();
}

function onKeyDown(e) {
  if (replayMode) {
    if (e.key === 'ArrowLeft') replayStep(-1);
    else if (e.key === 'ArrowRight') replayStep(1);
    else if (e.key === 'Home') replayStep(-Infinity);
    else if (e.key === 'End') replayStep(Infinity);
    else if (e.key === ' ') { e.preventDefault(); toggleReplayAuto(); }
    else if (e.key === 'Escape') exitReplay();
    return;
  }
  if (!gameActive) return;
  if (e.key === 'r' || e.key === 'R') rotatePiece();
  else if (e.key === 'ArrowLeft' && ghostPos) { ghostPos.col--; render(); }
  else if (e.key === 'ArrowRight' && ghostPos) { ghostPos.col++; render(); }
  else if (e.key === 'ArrowUp' && ghostPos) { ghostPos.row--; render(); }
  else if (e.key === 'ArrowDown' && ghostPos) { ghostPos.row++; render(); }
}

function rotatePiece() {
  if (!gameActive || replayMode) return;
  rotation = (rotation + 1) % 4;
  playSfx('rotate');
  updateLegalMovesCache();
  render();
  renderPreview(currentPlayer);
}

// ============================================================
// Touch Input
// ============================================================
let touchStartPos = null;
let touchMoved = false;

function getTouchBoardPos(touch) {
  const rect = boardCanvas.getBoundingClientRect();
  const col = Math.floor((touch.clientX - rect.left) / CELL_SIZE);
  const row = Math.floor((touch.clientY - rect.top) / CELL_SIZE);
  const cells = getCurrentCells();
  const bounds = getPieceBounds(cells);
  return { row: row - Math.floor(bounds.rows / 2), col: col - Math.floor(bounds.cols / 2) };
}

function onTouchStart(e) {
  e.preventDefault();
  if (!gameActive || (gameMode === 'pvai' && currentPlayer === P2)) return;
  touchMoved = false;

  const pos = getTouchBoardPos(e.touches[0]);
  ghostPos = pos;
  touchStartPos = { x: e.touches[0].clientX, y: e.touches[0].clientY };
  requestRender();
}

function onTouchMove(e) {
  e.preventDefault();
  if (!gameActive || (gameMode === 'pvai' && currentPlayer === P2)) return;
  if (e.touches.length !== 1) return;
  touchMoved = true;
  ghostPos = getTouchBoardPos(e.touches[0]);
  requestRender();
}

function onTouchEnd(e) {
  e.preventDefault();
  if (!gameActive || (gameMode === 'pvai' && currentPlayer === P2)) return;
  if (!touchMoved && ghostPos) {
    const cells = getCurrentCells();
    if (isLegalMove(cells, ghostPos.row, ghostPos.col, currentPlayer)) {
      doPlace();
      return;
    }
  }

  if (touchMoved && ghostPos) {
    const cells = getCurrentCells();
    if (isLegalMove(cells, ghostPos.row, ghostPos.col, currentPlayer)) {
      doPlace();
      return;
    }
  }

  render();
}

function movePieceDir(dr, dc) {
  if (!gameActive || (gameMode === 'pvai' && currentPlayer === P2)) return;
  if (!ghostPos) {
    const cells = getCurrentCells();
    const bounds = getPieceBounds(cells);
    ghostPos = {
      row: Math.floor((BOARD_SIZE - bounds.rows) / 2),
      col: Math.floor((BOARD_SIZE - bounds.cols) / 2)
    };
  }
  ghostPos.row += dr;
  ghostPos.col += dc;
  render();
}

function confirmPlace() {
  if (!gameActive || (gameMode === 'pvai' && currentPlayer === P2)) return;
  if (!ghostPos) {
    movePieceDir(0, 0);
    return;
  }
  const cells = getCurrentCells();
  if (isLegalMove(cells, ghostPos.row, ghostPos.col, currentPlayer)) {
    doPlace();
  } else {
    playIllegalSfx();
  }
}

function doPlace() {
  const cells = getCurrentCells();
  const pieceName = pieces[currentPlayer].name;
  const placedCellPositions = cells.map(([dr, dc]) => [ghostPos.row + dr, ghostPos.col + dc]);
  const result = placePiece(cells, ghostPos.row, ghostPos.col, currentPlayer);
  skipCount = 0;
  lastPlacedMove = { player: currentPlayer, cells: placedCellPositions };
  moveHistory.push({
    turn: moveHistory.length + 1,
    player: currentPlayer,
    piece: pieceName,
    rotation: rotation,
    row: ghostPos.row,
    col: ghostPos.col,
    captured: result.captured,
    linesCleared: result.linesCleared,
    type: 'place'
  });

  let msg = playerName(currentPlayer) + ' 落子';
  if (result.captured > 0) msg += ` | 围杀 ${result.captured} 子`;
  if (result.linesCleared > 0) msg += ` | 消除 ${result.linesCleared} 行/列`;
  setStatus(msg);

  playSfx('place');
  let delay = 300;
  if (result.captured > 0) {
    setTimeout(() => playSfx('capture'), 100);
    delay = 550;
  }
  if (result.linesCleared > 0) {
    setTimeout(() => playSfx('clear'), result.captured > 0 ? 550 : 150);
    delay = Math.max(delay, result.captured > 0 ? 950 : 550);
  }

  generatePiece(currentPlayer);
  ghostPos = null;
  updatePanels();
  render();
  autoSave();
  setTimeout(() => nextTurn(), delay);
}

// ============================================================
// Rendering
// ============================================================
function render() {
  const ctx = boardCtx;
  const size = BOARD_SIZE * CELL_SIZE;
  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = BG_COLOR;
  ctx.fillRect(0, 0, size, size);

  ctx.strokeStyle = GRID_COLOR;
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= BOARD_SIZE; i++) {
    ctx.beginPath(); ctx.moveTo(i * CELL_SIZE, 0); ctx.lineTo(i * CELL_SIZE, size); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * CELL_SIZE); ctx.lineTo(size, i * CELL_SIZE); ctx.stroke();
  }

  for (let r = 0; r < board.length; r++)
    for (let c = 0; c < (board[r] ? board[r].length : 0); c++)
      if (board[r][c] !== EMPTY) drawCell(ctx, r, c, board[r][c]);

  if (lastPlacedMove && !replayMode) {
    ctx.save();
    const oppColor = lastPlacedMove.player === P1 ? P1_COLOR : P2_COLOR;
    ctx.strokeStyle = oppColor;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = oppColor;
    ctx.shadowBlur = 6;
    for (const [r, c] of lastPlacedMove.cells) {
      if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && board[r][c] === lastPlacedMove.player) {
        ctx.strokeRect(c * CELL_SIZE + 1.5, r * CELL_SIZE + 1.5, CELL_SIZE - 3, CELL_SIZE - 3);
      }
    }
    ctx.restore();
  }

  if (cachedLegalMoves && cachedLegalMoves.moves.length === 1 && gameActive && !replayMode
      && !(gameMode === 'pvai' && currentPlayer === P2)) {
    const m = cachedLegalMoves.moves[0];
    const cells = cachedLegalMoves.cells;
    ctx.save();
    const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 300);
    ctx.strokeStyle = `rgba(255,215,0,${0.5 + pulse * 0.5})`;
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    for (const [dr, dc] of cells) {
      const r = m.row + dr, c = m.col + dc;
      if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
        ctx.strokeRect(c * CELL_SIZE + 1, r * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2);
      }
    }
    ctx.restore();
    if (!animFrameId) animFrameId = requestAnimationFrame(animTick);
  }

  if (ghostPos && gameActive && pieces[currentPlayer]) {
    const cells = getCurrentCells();
    const valid = isLegalMove(cells, ghostPos.row, ghostPos.col, currentPlayer);
    ctx.globalAlpha = GHOST_ALPHA;
    for (const [dr, dc] of cells) {
      const r = ghostPos.row + dr, c = ghostPos.col + dc;
      if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) continue;
      ctx.fillStyle = valid ? (currentPlayer === P1 ? P1_COLOR : P2_COLOR) : '#ff1744';
      ctx.fillRect(c * CELL_SIZE + 1, r * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2);
    }
    ctx.globalAlpha = 1;
  }
}

// ============================================================
// Animation System
// ============================================================
function startAnimLoop() {
  if (animFrameId) return;
  animFrameId = requestAnimationFrame(animTick);
}

function animTick(now) {
  animations = animations.filter(a => now < a.startTime + a.duration);
  if (animations.length === 0) {
    animFrameId = null;
    render();
    return;
  }

  render();
  const ctx = boardCtx;

  for (const anim of animations) {
    const elapsed = now - anim.startTime;
    if (elapsed < 0) continue;
    const t = Math.min(1, elapsed / anim.duration);

    if (anim.type === 'place') {
      const scale = 0.5 + 0.5 * easeOutBack(t);
      const glowAlpha = (1 - t) * 0.6;
      for (const [r, c] of anim.cells) {
        const cx = c * CELL_SIZE + CELL_SIZE / 2;
        const cy = r * CELL_SIZE + CELL_SIZE / 2;
        const sz = (CELL_SIZE - 2) * scale;
        if (glowAlpha > 0.01) {
          ctx.shadowColor = anim.color;
          ctx.shadowBlur = 12 * (1 - t);
          ctx.globalAlpha = glowAlpha;
          ctx.fillStyle = anim.color;
          ctx.fillRect(cx - sz / 2, cy - sz / 2, sz, sz);
          ctx.shadowBlur = 0;
          ctx.globalAlpha = 1;
        }
      }
    } else if (anim.type === 'capture') {
      const flash = t < 0.3 ? Math.sin(t / 0.3 * Math.PI * 3) * 0.7 : 0;
      const fadeAlpha = t < 0.3 ? 0.8 : 0.8 * (1 - (t - 0.3) / 0.7);
      for (const [r, c] of anim.cells) {
        const x = c * CELL_SIZE, y = r * CELL_SIZE;
        if (flash > 0) {
          ctx.globalAlpha = flash;
          ctx.fillStyle = '#fff';
          ctx.fillRect(x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2);
        }
        if (fadeAlpha > 0.01) {
          ctx.globalAlpha = fadeAlpha;
          ctx.strokeStyle = '#ff1744';
          ctx.lineWidth = 2;
          ctx.strokeRect(x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4);
        }
      }
      ctx.globalAlpha = 1;
    } else if (anim.type === 'clear') {
      const alpha = (1 - easeInQuad(t)) * 0.6;
      if (alpha > 0.01) {
        for (const [r, c, was] of anim.cells) {
          const x = c * CELL_SIZE, y = r * CELL_SIZE;
          ctx.globalAlpha = alpha;
          ctx.fillStyle = was === P1 ? P1_COLOR : (was === P2 ? P2_COLOR : '#888');
          const shrink = t * 6;
          ctx.fillRect(x + 1 + shrink, y + 1 + shrink,
                       CELL_SIZE - 2 - shrink * 2, CELL_SIZE - 2 - shrink * 2);
        }
        ctx.globalAlpha = 1;
      }
    }
  }

  animFrameId = requestAnimationFrame(animTick);
}

function easeOutBack(t) {
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
}

function easeInQuad(t) {
  return t * t;
}

function drawCell(ctx, r, c, cell) {
  const x = c * CELL_SIZE, y = r * CELL_SIZE, pad = 1;
  if (cell === P1 || cell === P2) {
    ctx.fillStyle = cell === P1 ? P1_COLOR : P2_COLOR;
    ctx.fillRect(x + pad, y + pad, CELL_SIZE - pad * 2, CELL_SIZE - pad * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    ctx.fillRect(x + pad, y + pad, CELL_SIZE - pad * 2, 3);
    ctx.fillRect(x + pad, y + pad, 3, CELL_SIZE - pad * 2);
    ctx.fillStyle = 'rgba(0,0,0,0.2)';
    ctx.fillRect(x + CELL_SIZE - pad - 3, y + pad, 3, CELL_SIZE - pad * 2);
    ctx.fillRect(x + pad, y + CELL_SIZE - pad - 3, CELL_SIZE - pad * 2, 3);
  } else if (cell === DEAD1 || cell === DEAD2) {
    ctx.fillStyle = cell === DEAD1 ? '#3d1a1a' : '#1a1a3d';
    ctx.fillRect(x + pad, y + pad, CELL_SIZE - pad * 2, CELL_SIZE - pad * 2);
    ctx.strokeStyle = cell === DEAD1 ? '#6d3333' : '#33336d';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x + 4, y + 4); ctx.lineTo(x + CELL_SIZE - 4, y + CELL_SIZE - 4);
    ctx.moveTo(x + CELL_SIZE - 4, y + 4); ctx.lineTo(x + 4, y + CELL_SIZE - 4);
    ctx.stroke();
  }
}

function renderPreview(player) {
  const canvas = player === P1 ? previewP1 : previewP2;
  const ctx = player === P1 ? prevCtx1 : prevCtx2;
  ctx.clearRect(0, 0, 90, 90);
  const piece = pieces[player];
  if (!piece) return;
  const rot = player === currentPlayer ? rotation : 0;
  const cells = getRotatedCells(piece.cells, rot);
  const bounds = getPieceBounds(cells);
  const cellSz = Math.min(20, Math.floor(72 / Math.max(bounds.rows, bounds.cols)));
  const offX = (90 - bounds.cols * cellSz) / 2;
  const offY = (90 - bounds.rows * cellSz) / 2;
  const noLegal = player === currentPlayer && gameActive && cachedLegalMoves && cachedLegalMoves.moves.length === 0;
  const color = noLegal ? '#ff1744' : (player === P1 ? P1_COLOR : P2_COLOR);
  for (const [r, c] of cells) {
    ctx.fillStyle = color;
    ctx.fillRect(offX + c * cellSz + 1, offY + r * cellSz + 1, cellSz - 2, cellSz - 2);
    ctx.fillStyle = noLegal ? 'rgba(255,255,255,0.05)' : 'rgba(255,255,255,0.15)';
    ctx.fillRect(offX + c * cellSz + 1, offY + r * cellSz + 1, cellSz - 2, 2);
  }
}

function updatePanels() {
  document.getElementById('scoreP1').textContent = countPieces(P1);
  document.getElementById('scoreP2').textContent = countPieces(P2);
  document.getElementById('panelP1').classList.toggle('active', currentPlayer === P1 && gameActive);
  document.getElementById('panelP2').classList.toggle('active', currentPlayer === P2 && gameActive);
  renderPreview(P1);
  renderPreview(P2);
}

function setStatus(msg) {
  document.getElementById('statusBar').textContent = msg;
}

function playerName(p) {
  if (p === P1) return '玩家1(蓝)';
  return gameMode === 'pvai' ? 'AI(紫)' : '玩家2(紫)';
}

// ============================================================
// Game Record Export (棋谱导出)
// ============================================================
function exportGameRecord() {
  const { rawP1: s1, rawP2: s2, effP1: eff1, effP2: eff2 } = getEffectiveScores();
  const komiReceiver = getKomiReceiver();
  const aiLevelNames = { 1: '简单', 2: '中等', 3: '困难', 4: '神经网络' };
  const aiLevel = parseInt(document.getElementById('aiLevelSelect').value);

  const now = new Date();
  const dateStr = now.getFullYear() + '-' +
    String(now.getMonth() + 1).padStart(2, '0') + '-' +
    String(now.getDate()).padStart(2, '0') + ' ' +
    String(now.getHours()).padStart(2, '0') + ':' +
    String(now.getMinutes()).padStart(2, '0') + ':' +
    String(now.getSeconds()).padStart(2, '0');

  let lines = [];
  lines.push('╔══════════════════════════════════════╗');
  lines.push('║       俄罗斯方块棋 - 棋谱记录         ║');
  lines.push('╚══════════════════════════════════════╝');
  lines.push('');
  lines.push(`日期: ${dateStr}`);
  lines.push(`棋盘: ${BOARD_SIZE}×${BOARD_SIZE}`);
  lines.push(`模式: ${gameMode === 'pvai' ? '人机对战' : '双人对战'}`);
  if (gameMode === 'pvai') {
    lines.push(`AI等级: ${aiLevelNames[aiLevel] || aiLevel}`);
  }
  lines.push('');

  let winner;
  if (eff1 > eff2) winner = '玩家1(蓝) 获胜';
  else if (eff2 > eff1) winner = (gameMode === 'pvai' ? 'AI(紫)' : '玩家2(紫)') + ' 获胜';
  else winner = '平局';
  lines.push(`结果: ${winner}`);
  const komiNoteP1 = komi !== 0 && komiReceiver === P1 ? ` (贴目 ${komi > 0 ? '+' : ''}${komi})` : '';
  const komiNoteP2 = komi !== 0 && komiReceiver === P2 ? ` (贴目 ${komi > 0 ? '+' : ''}${komi})` : '';
  lines.push(`玩家1(蓝): ${s1} 方块${komiNoteP1}`);
  lines.push(`${gameMode === 'pvai' ? 'AI(紫)' : '玩家2(紫)'}: ${s2} 方块${komiNoteP2}`);
  lines.push('');

  lines.push('──── 对局过程 ────');
  lines.push('');
  const colName = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

  for (const m of moveHistory) {
    const pName = m.player === P1 ? 'P1(蓝)' : (gameMode === 'pvai' ? 'AI(紫)' : 'P2(紫)');
    const turnStr = String(m.turn).padStart(3, ' ');

    if (m.type === 'skip') {
      lines.push(`${turnStr}. ${pName}  跳过`);
    } else if (m.type === 'auto_skip') {
      lines.push(`${turnStr}. ${pName}  无法落子，自动跳过`);
    } else {
      const pos = colName[m.col] + (m.row + 1);
      let detail = `${turnStr}. ${pName}  ${m.piece} R${m.rotation} → ${pos}`;
      if (m.captured > 0) detail += `  围杀${m.captured}子`;
      if (m.linesCleared > 0) detail += `  消${m.linesCleared}行`;
      lines.push(detail);
    }
  }

  lines.push('');
  lines.push(`共 ${moveHistory.length} 手`);

  lines.push('');
  lines.push('──── 终局棋盘 ────');
  lines.push('');
  let header = '   ';
  for (let c = 0; c < BOARD_SIZE; c++) header += ' ' + colName[c];
  lines.push(header);

  for (let r = 0; r < BOARD_SIZE; r++) {
    let row = String(r + 1).padStart(2, ' ') + ' ';
    for (let c = 0; c < BOARD_SIZE; c++) {
      const v = board[r][c];
      if (v === P1) row += ' ●';
      else if (v === P2) row += ' ◆';
      else row += ' ·';
    }
    lines.push(row);
  }
  lines.push('');
  lines.push('图例: ● = P1(蓝)  ◆ = P2(紫)  · = 空');

  const text = lines.join('\n');

  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  const fileDate = now.getFullYear() +
    String(now.getMonth() + 1).padStart(2, '0') +
    String(now.getDate()).padStart(2, '0') + '_' +
    String(now.getHours()).padStart(2, '0') +
    String(now.getMinutes()).padStart(2, '0');
  a.download = `TetrisWeiqi_${fileDate}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ============================================================
// Save / Load
// ============================================================
function getGameState() {
  return {
    version: 1,
    boardSize: BOARD_SIZE,
    board: board.map(r => [...r]),
    currentPlayer,
    pieces: { [P1]: pieces[P1] ? { name: pieces[P1].name } : null,
              [P2]: pieces[P2] ? { name: pieces[P2].name } : null },
    bag: [...bag],
    skipCount,
    firstPlayer,
    moveHistory: JSON.parse(JSON.stringify(moveHistory)),
    gameMode,
    gameActive,
    aiLevel: document.getElementById('aiLevelSelect').value,
    komi: komi,
  };
}

function restoreGameState(state) {
  if (replayAutoTimer) {
    clearInterval(replayAutoTimer);
    replayAutoTimer = null;
  }
  replayMode = false;
  replayData = null;
  replayIndex = 0;
  replayBoards = [];
  document.getElementById('replayBar').style.display = 'none';
  closeOverlay();
  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }
  animations = [];
  lastPlacedMove = null;
  cachedLegalMoves = null;

  BOARD_SIZE = state.boardSize;
  document.getElementById('boardSizeSelect').value = BOARD_SIZE;
  resizeBoard();
  board = state.board.map(r => [...r]);
  currentPlayer = state.currentPlayer;
  pieces[P1] = state.pieces[P1] ? { name: state.pieces[P1].name, cells: PIECE_SHAPES[state.pieces[P1].name].map(c => [...c]) } : null;
  pieces[P2] = state.pieces[P2] ? { name: state.pieces[P2].name, cells: PIECE_SHAPES[state.pieces[P2].name].map(c => [...c]) } : null;
  bag = state.bag ? [...state.bag] : [];
  skipCount = state.skipCount || 0;
  firstPlayer = state.firstPlayer || P1;
  moveHistory = state.moveHistory || [];
  gameMode = state.gameMode || 'pvai';
  gameActive = state.gameActive;
  rotation = 0;
  ghostPos = null;
  setMode(gameMode);
  if (state.aiLevel) document.getElementById('aiLevelSelect').value = state.aiLevel;
  komi = state.komi || 0;
  document.getElementById('komiInput').value = komi;
  onAiLevelChange();
  if (gameActive) updateLegalMovesCache();
  updatePanels();
  render();
}

function saveGame() {
  if (!gameActive && moveHistory.length === 0) {
    setStatus('没有可以存档的游戏');
    return;
  }
  const state = getGameState();
  try {
    localStorage.setItem('tetrisweiqi_save', JSON.stringify(state));
    setStatus('游戏已存档');
  } catch (e) {
    setStatus('存档失败: ' + e.message);
  }
}

function autoSave() {
  if (!gameActive) return;
  try {
    localStorage.setItem('tetrisweiqi_save', JSON.stringify(getGameState()));
  } catch (e) {}
}

function loadGame() {
  const raw = localStorage.getItem('tetrisweiqi_save');
  if (!raw) { setStatus('没有找到存档'); return; }
  try {
    const state = JSON.parse(raw);
    stopBgm();
    restoreGameState(state);
    if (gameActive) {
      setStatus(playerName(currentPlayer) + ' 的回合 (已读档)');
      playBgm();
      if (gameMode === 'pvai' && currentPlayer === P2) {
        setTimeout(() => aiMove(), 300);
      }
    } else {
      setStatus('读档成功 (游戏已结束)');
    }
  } catch (e) {
    setStatus('读档失败: ' + e.message);
  }
}

// ============================================================
// Replay
// ============================================================
let replayMode = false;
let replayData = null;
let replayIndex = 0;
let replayBoards = [];
let replayAutoTimer = null;

function buildReplaySnapshots(hist, size) {
  const snapshots = [];
  const b = [];
  for (let r = 0; r < size; r++) b.push(new Array(size).fill(EMPTY));
  snapshots.push(b.map(r => [...r]));

  for (const m of hist) {
    if (m.type === 'skip' || m.type === 'auto_skip') {
      snapshots.push(b.map(r => [...r]));
      continue;
    }
    const cells = getRotatedCells(PIECE_SHAPES[m.piece].map(c => [...c]), m.rotation);
    for (const [dr, dc] of cells) {
      b[m.row + dr][m.col + dc] = m.player;
    }
    const captureTarget = m.player === P1 ? P2 : P1;
    captureOnBoard(b, size, captureTarget);
    captureOnBoard(b, size, m.player);
    clearLinesOnBoard(b, size);
    captureOnBoard(b, size, captureTarget);
    captureOnBoard(b, size, m.player);
    snapshots.push(b.map(r => [...r]));
  }
  return snapshots;
}

function captureOnBoard(b, size, target) {
  const visited = new Uint8Array(size * size);
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const k = r * size + c;
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
          if (nr < 0 || nr >= size || nc < 0 || nc >= size) continue;
          const nk = nr * size + nc;
          if (visited[nk]) continue;
          if (b[nr][nc] === EMPTY) { hasLib = true; }
          else if (b[nr][nc] === target) { visited[nk] = 1; stk.push([nr, nc]); }
        }
      }
      if (!hasLib) {
        for (const [gr, gc] of group) b[gr][gc] = EMPTY;
      }
    }
  }
}

function clearLinesOnBoard(b, size) {
  for (let r = 0; r < size; r++) {
    let full = true;
    for (let c = 0; c < size; c++) { if (b[r][c] === EMPTY) { full = false; break; } }
    if (full) { for (let c = 0; c < size; c++) b[r][c] = EMPTY; }
  }
  for (let c = 0; c < size; c++) {
    let full = true;
    for (let r = 0; r < size; r++) { if (b[r][c] === EMPTY) { full = false; break; } }
    if (full) { for (let r = 0; r < size; r++) b[r][c] = EMPTY; }
  }
}

function parseGameRecord(text) {
  const lines = text.split('\n');
  let size = 10;
  for (const line of lines) {
    const sizeMatch = line.match(/棋盘:\s*(\d+)\s*[×x]\s*(\d+)/);
    if (sizeMatch) { size = parseInt(sizeMatch[1]); break; }
  }
  let mode = 'pvai';
  for (const line of lines) {
    if (line.includes('双人对战')) { mode = 'pvp'; break; }
  }
  const colName = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  const moves = [];
  for (const line of lines) {
    const skipMatch = line.match(/^\s*\d+\.\s*(P1|AI|P2)\S*\s+(跳过|无法落子)/);
    if (skipMatch) {
      const player = skipMatch[1] === 'P1' ? P1 : P2;
      const type = line.includes('无法落子') ? 'auto_skip' : 'skip';
      moves.push({ player, type });
      continue;
    }
    const placeMatch = line.match(/^\s*\d+\.\s*(P1|AI|P2)\S*\s+([IOTSLJZ])\s+R(\d)\s*→\s*([A-Z])(\d+)/);
    if (placeMatch) {
      const player = placeMatch[1] === 'P1' ? P1 : P2;
      const piece = placeMatch[2];
      const rotation = parseInt(placeMatch[3]);
      const col = colName.indexOf(placeMatch[4]);
      const row = parseInt(placeMatch[5]) - 1;
      let captured = 0, linesCleared = 0;
      const capMatch = line.match(/围杀(\d+)子/);
      if (capMatch) captured = parseInt(capMatch[1]);
      const clearMatch = line.match(/消(\d+)行/);
      if (clearMatch) linesCleared = parseInt(clearMatch[1]);
      moves.push({ player, piece, rotation, row, col, captured, linesCleared, type: 'place' });
    }
  }
  return { size, mode, moves };
}

function loadReplayFile(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {
    try {
      const parsed = parseGameRecord(e.target.result);
      if (parsed.moves.length === 0) {
        setStatus('棋谱解析失败：未找到有效着法');
        return;
      }
      BOARD_SIZE = parsed.size;
      document.getElementById('boardSizeSelect').value = BOARD_SIZE;
      resizeBoard();
      gameActive = false;
      stopBgm();
      replayMode = true;
      replayData = {
        moveHistory: parsed.moves,
        boardSize: parsed.size,
        gameMode: parsed.mode,
      };
      replayBoards = buildReplaySnapshots(replayData.moveHistory, replayData.boardSize);
      replayIndex = 0;
      document.getElementById('replayBar').style.display = '';
      document.getElementById('replaySlider').max = replayData.moveHistory.length;
      document.getElementById('replaySlider').value = 0;
      showReplayStep();
      setStatus(`棋谱已加载: ${parsed.moves.length} 手`);
    } catch (err) {
      setStatus('棋谱解析失败: ' + err.message);
    }
  };
  reader.readAsText(file);
  event.target.value = '';
}

function startReplay() {
  if (moveHistory.length === 0) { setStatus('没有可回放的对局'); return; }
  replayMode = true;
  gameActive = false;
  stopBgm();
  replayData = {
    moveHistory: JSON.parse(JSON.stringify(moveHistory)),
    boardSize: BOARD_SIZE,
    gameMode,
  };
  replayBoards = buildReplaySnapshots(replayData.moveHistory, replayData.boardSize);
  replayIndex = 0;
  document.getElementById('replayBar').style.display = '';
  document.getElementById('replaySlider').max = replayData.moveHistory.length;
  document.getElementById('replaySlider').value = 0;
  showReplayStep();
}

function showReplayStep() {
  board = replayBoards[replayIndex].map(r => [...r]);
  render();
  if (replayIndex > 0) {
    const m = replayData.moveHistory[replayIndex - 1];
    if (m.type === 'place') {
      const cells = getRotatedCells(PIECE_SHAPES[m.piece].map(c => [...c]), m.rotation);
      const placedCells = cells.map(([dr, dc]) => [m.row + dr, m.col + dc]);
      const color = m.player === P1 ? P1_COLOR : P2_COLOR;
      const ctx = boardCtx;
      ctx.save();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      for (const [r, c] of placedCells) {
        ctx.strokeRect(c * CELL_SIZE + 2, r * CELL_SIZE + 2, CELL_SIZE - 4, CELL_SIZE - 4);
      }
      ctx.restore();
    }
  }
  document.getElementById('scoreP1').textContent = countPieces(P1);
  document.getElementById('scoreP2').textContent = countPieces(P2);
  prevCtx1.clearRect(0, 0, 90, 90);
  prevCtx2.clearRect(0, 0, 90, 90);
  const total = replayData.moveHistory.length;
  if (replayIndex > 0) {
    const m = replayData.moveHistory[replayIndex - 1];
    if (m.type === 'place') {
      const pCanvas = m.player === P1 ? prevCtx1 : prevCtx2;
      const cells = getRotatedCells(PIECE_SHAPES[m.piece].map(c => [...c]), m.rotation);
      const bounds = getPieceBounds(cells);
      const cellSz = Math.min(20, Math.floor(72 / Math.max(bounds.rows, bounds.cols)));
      const offX = (90 - bounds.cols * cellSz) / 2;
      const offY = (90 - bounds.rows * cellSz) / 2;
      const color = m.player === P1 ? P1_COLOR : P2_COLOR;
      for (const [r, c] of cells) {
        pCanvas.fillStyle = color;
        pCanvas.fillRect(offX + c * cellSz + 1, offY + r * cellSz + 1, cellSz - 2, cellSz - 2);
        pCanvas.fillStyle = 'rgba(255,255,255,0.15)';
        pCanvas.fillRect(offX + c * cellSz + 1, offY + r * cellSz + 1, cellSz - 2, 2);
      }
    }
  }
  if (replayIndex < total) {
    const nextM = replayData.moveHistory[replayIndex];
    if (nextM.type === 'place') {
      const pCanvas = nextM.player === P1 ? prevCtx1 : prevCtx2;
      const cells = getRotatedCells(PIECE_SHAPES[nextM.piece].map(c => [...c]), nextM.rotation);
      const bounds = getPieceBounds(cells);
      const cellSz = Math.min(20, Math.floor(72 / Math.max(bounds.rows, bounds.cols)));
      const offX = (90 - bounds.cols * cellSz) / 2;
      const offY = (90 - bounds.rows * cellSz) / 2;
      const color = nextM.player === P1 ? P1_COLOR : P2_COLOR;
      for (const [r, c] of cells) {
        pCanvas.fillStyle = color;
        pCanvas.fillRect(offX + c * cellSz + 1, offY + r * cellSz + 1, cellSz - 2, cellSz - 2);
        pCanvas.fillStyle = 'rgba(255,255,255,0.3)';
        pCanvas.fillRect(offX + c * cellSz + 1, offY + r * cellSz + 1, cellSz - 2, 2);
      }
    }
  }
  const nextPlayer = replayIndex < total ? replayData.moveHistory[replayIndex].player : null;
  document.getElementById('panelP1').classList.toggle('active', nextPlayer === P1);
  document.getElementById('panelP2').classList.toggle('active', nextPlayer === P2);
  let info = `${replayIndex}/${total}`;
  if (replayIndex > 0) {
    const m = replayData.moveHistory[replayIndex - 1];
    const pName = m.player === P1 ? 'P1(蓝)' : 'P2(紫)';
    if (m.type === 'place') {
      info += ` | ${pName} ${m.piece}`;
      if (m.captured > 0) info += ` 围杀${m.captured}`;
      if (m.linesCleared > 0) info += ` 消${m.linesCleared}行`;
    } else {
      info += ` | ${pName} 跳过`;
    }
  }
  document.getElementById('replayInfo').textContent = info;
  document.getElementById('replaySlider').value = replayIndex;
  setStatus('回放模式 — ' + info);
}

function replayStep(delta) {
  if (!replayMode) return;
  if (delta === -Infinity) replayIndex = 0;
  else if (delta === Infinity) replayIndex = replayData.moveHistory.length;
  else replayIndex = Math.max(0, Math.min(replayData.moveHistory.length, replayIndex + delta));
  showReplayStep();
}

function replaySeek(val) {
  if (!replayMode) return;
  replayIndex = parseInt(val);
  showReplayStep();
}

function toggleReplayAuto() {
  if (replayAutoTimer) {
    clearInterval(replayAutoTimer);
    replayAutoTimer = null;
    document.getElementById('replayPlayBtn').textContent = '▶';
    return;
  }
  document.getElementById('replayPlayBtn').textContent = '⏸';
  replayAutoTimer = setInterval(() => {
    if (replayIndex >= replayData.moveHistory.length) {
      clearInterval(replayAutoTimer);
      replayAutoTimer = null;
      document.getElementById('replayPlayBtn').textContent = '▶';
      return;
    }
    replayStep(1);
  }, 600);
}

function exitReplay() {
  if (replayAutoTimer) { clearInterval(replayAutoTimer); replayAutoTimer = null; }
  replayMode = false;
  document.getElementById('replayBar').style.display = 'none';
  setStatus('点击"开始游戏"开始');
  if (replayBoards.length > 0) {
    board = replayBoards[replayBoards.length - 1].map(r => [...r]);
  }
  render();
}
