// ============================================================
// Init / Game Setup
// ============================================================
function init() {
  boardCanvas = document.getElementById('boardCanvas');
  boardCtx = boardCanvas.getContext('2d');
  previewP1 = document.getElementById('previewP1');
  prevCtx1 = previewP1.getContext('2d');
  previewP2 = document.getElementById('previewP2');
  prevCtx2 = previewP2.getContext('2d');

  boardCanvas.addEventListener('mousemove', onMouseMove);
  boardCanvas.addEventListener('mouseleave', () => { ghostPos = null; render(); });
  boardCanvas.addEventListener('click', onBoardClick);
  boardCanvas.addEventListener('contextmenu', (e) => { e.preventDefault(); rotatePiece(); });
  document.addEventListener('keydown', onKeyDown);

  boardCanvas.addEventListener('touchstart', onTouchStart, { passive: false });
  boardCanvas.addEventListener('touchmove', onTouchMove, { passive: false });
  boardCanvas.addEventListener('touchend', onTouchEnd, { passive: false });

  document.addEventListener('touchstart', (e) => {
    if (!gameActive || (gameMode === 'pvai' && currentPlayer === P2)) return;
    const t = e.target;
    const myPreview = currentPlayer === P1 ? previewP1 : previewP2;
    if (t === myPreview || t.closest('.preview-box')?.contains(myPreview)) {
      rotatePiece();
    }
  });

  resizeBoard();
  render();
}

function resizeBoard() {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const isMobile = vw <= 600 || (matchMedia('(hover:none) and (pointer:coarse)').matches);
  const availW = isMobile ? vw - 24 : Math.min(vw - 320, 560);
  const availH = isMobile ? vh - 340 : vh - 200;
  CELL_SIZE = Math.max(18, Math.min(40, Math.floor(Math.min(availW, availH) / BOARD_SIZE)));
  boardCanvas.width = BOARD_SIZE * CELL_SIZE;
  boardCanvas.height = BOARD_SIZE * CELL_SIZE;
}

function setMode(mode) {
  invalidateAI();
  gameMode = mode;
  document.getElementById('btnPvAI').classList.toggle('selected', mode === 'pvai');
  document.getElementById('btnPvP').classList.toggle('selected', mode === 'pvp');
  document.getElementById('panelP2').querySelector('h2').textContent =
    mode === 'pvai' ? 'AI (紫)' : '玩家 2 (紫)';
}

function changeBoardSize() {
  invalidateAI();
  BOARD_SIZE = parseInt(document.getElementById('boardSizeSelect').value);
  resizeBoard();
  createBoard();
  render();
}

function startGame() {
  invalidateAI();
  if (replayMode) exitReplay();
  BOARD_SIZE = parseInt(document.getElementById('boardSizeSelect').value);
  resizeBoard();
  createBoard();
  const firstSel = document.getElementById('firstPlayerSelect').value;
  if (firstSel === 'p1') currentPlayer = P1;
  else if (firstSel === 'p2') currentPlayer = P2;
  else currentPlayer = Math.random() < 0.5 ? P1 : P2;
  firstPlayer = currentPlayer;
  rotation = 0;
  skipCount = 0;
  bag = [];
  refillBag();
  moveHistory = [];
  lastPlacedMove = null;
  gameActive = true;
  generatePiece(P1);
  generatePiece(P2);
  updatePanels();
  setStatus(playerName(currentPlayer) + ' 的回合');
  updateLegalMovesCache();
  render();
  playSfx('gameStart');
  playBgm();

  if (gameMode === 'pvai' && currentPlayer === P2) {
    setTimeout(aiMove, 600);
  }
}

// ============================================================
// MCTS Slider / AI Level change
// ============================================================
function updateMctsLabel() {
  const v = document.getElementById('mctsSimsSlider').value;
  document.getElementById('mctsSimsLabel').textContent = v + '次';
  MCTS_SIMS = parseInt(v);
}

function onAiLevelChange() {
  const level = document.getElementById('aiLevelSelect').value;
  document.getElementById('mctsSliderWrap').style.display = level === '5' ? '' : 'none';
  if (level === '4' || level === '5') {
    loadOnnxModel();
  }
}

// ============================================================
// Boot
// ============================================================
window.addEventListener('load', () => {
  init();
  initAudio();
  document.getElementById('aiLevelSelect').addEventListener('change', onAiLevelChange);
  onAiLevelChange();
});
window.addEventListener('resize', () => { resizeBoard(); render(); });
window.addEventListener('beforeunload', () => { stopBgm(); });
