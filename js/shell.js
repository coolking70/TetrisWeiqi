// ============================================================
// Shell — menu navigation, screen transitions, turn badge sync
// Thin layer that sits on top of the existing game code. Does NOT
// own any rule state; only wires the cyber/xian themed shell to
// the functions exported by main.js / ui.js / game-logic.js.
// ============================================================

function enterGame() {
  document.getElementById('mainMenuScreen').classList.remove('active');
  document.getElementById('gameScreen').classList.add('active');
  // Canvas sizing depends on the gameScreen being visible first.
  if (typeof resizeBoard === 'function') resizeBoard();
  if (typeof render === 'function') render();
}

function backToMenu() {
  // Invalidate any running AI so it doesn't settle moves on a stale
  // board after we return. Safe even if AI wasn't running.
  if (typeof invalidateAI === 'function') invalidateAI();
  gameActive = false;
  if (typeof stopBgm === 'function') stopBgm();
  document.getElementById('gameScreen').classList.remove('active');
  document.getElementById('mainMenuScreen').classList.add('active');
}

// ============================================================
// Turn badge + round meta — refreshed by monkey-patching the two
// callsites that always fire on state changes: updatePanels()
// (covers normal turn flips) and setStatus() (covers skip/end).
// ============================================================
function updateTurnBadge() {
  const badge = document.getElementById('turnBadge');
  const roundMeta = document.getElementById('roundMeta');
  const boardMeta = document.getElementById('boardMeta');
  if (!badge) return;

  if (boardMeta && typeof BOARD_SIZE !== 'undefined') {
    boardMeta.textContent = BOARD_SIZE + ' × ' + BOARD_SIZE;
  }
  if (roundMeta && typeof moveHistory !== 'undefined') {
    const placedTurns = moveHistory.filter(m => m.type === 'place' || !m.type).length;
    const round = Math.floor(placedTurns / 2) + 1;
    roundMeta.textContent = gameActive ? ('第 ' + round + ' 回合') : '未开始';
  }

  badge.classList.remove('xian', 'idle');
  if (!gameActive) {
    badge.classList.add('idle');
    badge.textContent = '等待开始';
    return;
  }
  if (currentPlayer === P2) {
    badge.classList.add('xian');
    badge.textContent = (gameMode === 'pvai' ? 'AI · 落子中' : '修仙方 · 落子中');
  } else {
    badge.textContent = '赛博方 · 落子中';
  }
}

// Patch updatePanels to also refresh the badge.
(function wrapUpdatePanels() {
  if (typeof updatePanels !== 'function') return;
  const orig = updatePanels;
  updatePanels = function () {
    const r = orig.apply(this, arguments);
    try { updateTurnBadge(); } catch (e) { /* shell-only, ignore */ }
    return r;
  };
  window.updatePanels = updatePanels;
})();

// Also re-sync on end-of-game overlays + replay transitions.
(function wrapEndGame() {
  if (typeof endGame !== 'function') return;
  const orig = endGame;
  endGame = function () {
    const r = orig.apply(this, arguments);
    try { updateTurnBadge(); } catch (e) {}
    return r;
  };
  window.endGame = endGame;
})();

// Initial badge state (pre-game).
window.addEventListener('load', () => {
  try { updateTurnBadge(); } catch (e) {}
});
