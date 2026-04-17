// ============================================================
// Audio — Web Audio API synthesized SFX and BGM (no external files)
// ============================================================

let soundEnabled = true;
let bgmEnabled = true;
let audioCtx = null;
let bgmGain = null;
let bgmTimeout = null;

function getAudioCtx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}

// --- Low-level helpers ---
function playTone(freq, duration, type, vol) {
  if (!soundEnabled) return;
  try {
    const ctx = getAudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = type || 'sine';
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(vol || 0.3, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + duration);
  } catch (e) {}
}

function schedNote(ctx, dest, freq, start, dur, type, vol) {
  const o = ctx.createOscillator();
  const g = ctx.createGain();
  o.type = type || 'square';
  o.frequency.value = freq;
  g.gain.setValueAtTime(vol, start);
  g.gain.setValueAtTime(vol, start + dur * 0.8);
  g.gain.linearRampToValueAtTime(0.0, start + dur);
  o.connect(g); g.connect(dest);
  o.start(start); o.stop(start + dur);
}

// --- Synthesized SFX ---
function playSfx(name) {
  if (!soundEnabled) return;
  try {
    const ctx = getAudioCtx();
    const t = ctx.currentTime;
    switch (name) {
      case 'place': {
        const o1 = ctx.createOscillator(); const g1 = ctx.createGain();
        o1.type = 'sine'; o1.frequency.value = 220;
        g1.gain.setValueAtTime(0.4, t);
        g1.gain.exponentialRampToValueAtTime(0.001, t + 0.12);
        o1.connect(g1); g1.connect(ctx.destination);
        o1.start(t); o1.stop(t + 0.12);
        const o2 = ctx.createOscillator(); const g2 = ctx.createGain();
        o2.type = 'square'; o2.frequency.value = 800;
        g2.gain.setValueAtTime(0.15, t);
        g2.gain.exponentialRampToValueAtTime(0.001, t + 0.04);
        o2.connect(g2); g2.connect(ctx.destination);
        o2.start(t); o2.stop(t + 0.04);
        break;
      }
      case 'capture': {
        const o1 = ctx.createOscillator(); const g1 = ctx.createGain();
        o1.type = 'sawtooth';
        o1.frequency.setValueAtTime(600, t);
        o1.frequency.exponentialRampToValueAtTime(120, t + 0.3);
        g1.gain.setValueAtTime(0.25, t);
        g1.gain.exponentialRampToValueAtTime(0.001, t + 0.35);
        o1.connect(g1); g1.connect(ctx.destination);
        o1.start(t); o1.stop(t + 0.35);
        const buf = ctx.createBuffer(1, ctx.sampleRate * 0.08, ctx.sampleRate);
        const d = buf.getChannelData(0);
        for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * 0.5;
        const ns = ctx.createBufferSource(); ns.buffer = buf;
        const ng = ctx.createGain();
        ng.gain.setValueAtTime(0.2, t + 0.05);
        ng.gain.exponentialRampToValueAtTime(0.001, t + 0.15);
        ns.connect(ng); ng.connect(ctx.destination);
        ns.start(t + 0.05); ns.stop(t + 0.15);
        break;
      }
      case 'clear': {
        const notes = [523, 659, 784, 1047];
        notes.forEach((freq, i) => {
          const o = ctx.createOscillator(); const g = ctx.createGain();
          o.type = 'sine'; o.frequency.value = freq;
          const s = t + i * 0.06;
          g.gain.setValueAtTime(0.3, s);
          g.gain.exponentialRampToValueAtTime(0.001, s + 0.2);
          o.connect(g); g.connect(ctx.destination);
          o.start(s); o.stop(s + 0.2);
        });
        break;
      }
      case 'rotate': {
        const o = ctx.createOscillator(); const g = ctx.createGain();
        o.type = 'sine';
        o.frequency.setValueAtTime(400, t);
        o.frequency.exponentialRampToValueAtTime(700, t + 0.06);
        g.gain.setValueAtTime(0.2, t);
        g.gain.exponentialRampToValueAtTime(0.001, t + 0.08);
        o.connect(g); g.connect(ctx.destination);
        o.start(t); o.stop(t + 0.08);
        break;
      }
      case 'gameWin': {
        const melody = [
          [523, 0.12], [659, 0.12], [784, 0.12], [1047, 0.25],
          [784, 0.12], [1047, 0.4]
        ];
        let off = 0;
        melody.forEach(([freq, dur]) => {
          const o = ctx.createOscillator(); const g = ctx.createGain();
          o.type = 'square'; o.frequency.value = freq;
          g.gain.setValueAtTime(0.18, t + off);
          g.gain.setValueAtTime(0.18, t + off + dur * 0.7);
          g.gain.linearRampToValueAtTime(0.0, t + off + dur);
          o.connect(g); g.connect(ctx.destination);
          o.start(t + off); o.stop(t + off + dur);
          const o2 = ctx.createOscillator(); const g2 = ctx.createGain();
          o2.type = 'sine'; o2.frequency.value = freq * 2;
          g2.gain.setValueAtTime(0.06, t + off);
          g2.gain.exponentialRampToValueAtTime(0.001, t + off + dur);
          o2.connect(g2); g2.connect(ctx.destination);
          o2.start(t + off); o2.stop(t + off + dur);
          off += dur;
        });
        break;
      }
      case 'gameLose': {
        const melody = [
          [392, 0.3], [349, 0.3], [311, 0.35], [262, 0.5]
        ];
        let off = 0;
        melody.forEach(([freq, dur]) => {
          const o = ctx.createOscillator(); const g = ctx.createGain();
          o.type = 'triangle'; o.frequency.value = freq;
          g.gain.setValueAtTime(0.3 - off * 0.15, t + off);
          g.gain.linearRampToValueAtTime(0.0, t + off + dur);
          o.connect(g); g.connect(ctx.destination);
          o.start(t + off); o.stop(t + off + dur);
          off += dur;
        });
        const o = ctx.createOscillator(); const g = ctx.createGain();
        o.type = 'sine'; o.frequency.value = 80;
        g.gain.setValueAtTime(0.15, t);
        g.gain.linearRampToValueAtTime(0.0, t + 1.4);
        o.connect(g); g.connect(ctx.destination);
        o.start(t); o.stop(t + 1.4);
        break;
      }
      case 'gameDraw': {
        const pairs = [[330, 350], [330, 350], [330, 330]];
        let off = 0;
        pairs.forEach(([f1, f2]) => {
          const o = ctx.createOscillator(); const g = ctx.createGain();
          o.type = 'triangle'; o.frequency.value = f1;
          g.gain.setValueAtTime(0.25, t + off);
          g.gain.linearRampToValueAtTime(0.0, t + off + 0.2);
          o.connect(g); g.connect(ctx.destination);
          o.start(t + off); o.stop(t + off + 0.2);
          const o2 = ctx.createOscillator(); const g2 = ctx.createGain();
          o2.type = 'triangle'; o2.frequency.value = f2;
          g2.gain.setValueAtTime(0.2, t + off + 0.15);
          g2.gain.linearRampToValueAtTime(0.0, t + off + 0.35);
          o2.connect(g2); g2.connect(ctx.destination);
          o2.start(t + off + 0.15); o2.stop(t + off + 0.35);
          off += 0.3;
        });
        break;
      }
      case 'gameStart': {
        const notes = [262, 330, 392, 523];
        notes.forEach((freq, i) => {
          const o = ctx.createOscillator(); const g = ctx.createGain();
          o.type = 'square'; o.frequency.value = freq;
          const s = t + i * 0.1;
          g.gain.setValueAtTime(0.15, s);
          g.gain.setValueAtTime(0.15, s + 0.18);
          g.gain.linearRampToValueAtTime(0.0, s + 0.25);
          o.connect(g); g.connect(ctx.destination);
          o.start(s); o.stop(s + 0.25);
        });
        break;
      }
    }
  } catch (e) {}
}

function playIllegalSfx() {
  playTone(150, 0.15, 'square', 0.2);
  setTimeout(() => playTone(120, 0.15, 'square', 0.15), 80);
}

function playSkipSfx() {
  playTone(440, 0.15, 'sine', 0.25);
  setTimeout(() => playTone(330, 0.2, 'sine', 0.2), 100);
}

// --- BGM: Korobeiniki (Tetris Theme) variation, looped via Web Audio API ---
function playBgm() {
  if (!bgmEnabled) return;
  stopBgm();
  try {
    const ctx = getAudioCtx();
    bgmGain = ctx.createGain();
    bgmGain.gain.value = 0.12;
    bgmGain.connect(ctx.destination);

    const BPM = 140;
    const beat = 60 / BPM;
    const melodyA = [
      // === Section A ===
      [659, 1], [494, 0.5], [523, 0.5], [587, 1], [523, 0.5], [494, 0.5],
      [440, 1], [440, 0.5], [523, 0.5], [659, 1], [587, 0.5], [523, 0.5],
      [494, 1], [494, 0.5], [523, 0.5], [587, 1], [659, 1],
      [523, 1], [440, 1], [440, 1], [0, 1],
      [0, 0.5], [587, 1], [698, 0.5], [880, 1], [784, 0.5], [698, 0.5],
      [659, 1.5], [523, 0.5], [659, 1], [587, 0.5], [523, 0.5],
      [494, 1], [494, 0.5], [523, 0.5], [587, 1], [659, 1],
      [523, 1], [440, 1], [440, 1], [0, 1],
      // === Section B ===
      [659, 0.75], [622, 0.25], [494, 0.5], [523, 0.5], [587, 1], [523, 0.5], [494, 0.5],
      [440, 0.75], [415, 0.25], [440, 0.5], [523, 0.5], [659, 1], [587, 0.5], [523, 0.5],
      [494, 1], [523, 0.5], [494, 0.5], [587, 1], [659, 1],
      [523, 1], [440, 1], [440, 1], [0, 1],
      [0, 0.5], [587, 0.5], [659, 0.5], [698, 0.5], [880, 1], [784, 0.5], [698, 0.5],
      [659, 1], [698, 0.5], [523, 0.5], [659, 1], [587, 0.5], [523, 0.5],
      [494, 0.75], [440, 0.25], [494, 0.5], [523, 0.5], [587, 1], [659, 1],
      [523, 1], [440, 1], [440, 1], [0, 1],
      // === Section C ===
      [523, 1], [523, 0.5], [587, 0.5], [659, 1], [659, 0.5], [587, 0.5],
      [523, 1], [440, 0.5], [494, 0.5], [523, 1.5], [0, 0.5],
      [494, 1], [494, 0.5], [523, 0.5], [440, 1], [440, 0.5], [392, 0.5],
      [440, 1], [494, 0.5], [523, 0.5], [587, 1.5], [0, 0.5],
      [659, 1], [698, 0.5], [784, 0.5], [880, 1], [784, 0.5], [698, 0.5],
      [659, 1], [587, 0.5], [523, 0.5], [587, 1.5], [0, 0.5],
      [659, 0.5], [698, 0.5], [784, 0.5], [880, 0.5], [1047, 1], [880, 0.5], [784, 0.5],
      [659, 1], [587, 1], [523, 1], [0, 1],
      // === Section D ===
      [659, 0.5], [659, 0.5], [494, 0.5], [523, 0.5], [587, 1], [523, 0.5], [494, 0.5],
      [440, 1], [440, 0.5], [523, 0.5], [659, 1], [587, 0.5], [523, 0.5],
      [494, 1], [494, 0.5], [523, 0.5], [587, 1], [659, 1],
      [523, 1], [440, 0.5], [523, 0.5], [440, 1], [0, 1],
      [0, 0.5], [587, 1], [698, 0.5], [880, 1], [784, 0.5], [698, 0.5],
      [659, 1.5], [523, 0.5], [659, 1], [587, 0.5], [523, 0.5],
      [494, 1], [494, 0.5], [523, 0.5], [587, 1], [659, 1],
      [523, 0.5], [587, 0.5], [440, 1], [440, 2], [0, 1],
    ];
    const bassA = [
      [220, 2], [175, 2], [196, 2], [165, 2],
      [175, 2], [196, 2], [220, 2], [220, 2],
      [196, 2], [175, 2], [165, 2], [165, 2],
      [175, 2], [196, 2], [220, 2], [220, 2],
      [220, 2], [175, 2], [196, 2], [165, 2],
      [175, 2], [196, 2], [220, 2], [220, 2],
      [196, 2], [175, 2], [165, 2], [165, 2],
      [175, 2], [196, 2], [220, 2], [220, 2],
      [262, 2], [262, 2], [220, 2], [220, 2],
      [247, 2], [220, 2], [196, 2], [196, 2],
      [262, 2], [262, 2], [220, 2], [220, 2],
      [262, 2], [247, 2], [262, 2], [262, 2],
      [220, 2], [175, 2], [196, 2], [165, 2],
      [175, 2], [196, 2], [220, 2], [220, 2],
      [196, 2], [175, 2], [165, 2], [165, 2],
      [175, 2], [196, 2], [220, 2], [220, 2],
    ];

    function scheduleMelody(startTime) {
      let t = startTime;
      for (const [freq, beats] of melodyA) {
        const dur = beats * beat;
        if (freq > 0) {
          schedNote(ctx, bgmGain, freq, t, dur * 0.9, 'square', 0.12);
        }
        t += dur;
      }
      return t;
    }

    function scheduleBass(startTime, totalDur) {
      let t = startTime;
      while (t < startTime + totalDur) {
        for (const [freq, beats] of bassA) {
          if (t >= startTime + totalDur) break;
          const dur = beats * beat;
          if (freq > 0) {
            schedNote(ctx, bgmGain, freq, t, dur * 0.85, 'triangle', 0.10);
          }
          t += dur;
        }
      }
    }

    let loopDur = 0;
    for (const [, beats] of melodyA) loopDur += beats * beat;

    let nextStart = ctx.currentTime + 0.1;

    function scheduleLoop() {
      if (!bgmEnabled || !bgmGain) return;
      for (let i = 0; i < 2; i++) {
        scheduleMelody(nextStart);
        scheduleBass(nextStart, loopDur);
        nextStart += loopDur;
      }
      const waitMs = (loopDur * 1.5) * 1000;
      bgmTimeout = setTimeout(scheduleLoop, waitMs);
    }

    scheduleLoop();
  } catch (e) {}
}

function stopBgm() {
  try {
    if (bgmTimeout) { clearTimeout(bgmTimeout); bgmTimeout = null; }
    if (bgmGain) { bgmGain.disconnect(); bgmGain = null; }
  } catch (e) {}
}

function initAudio() {
  // No external files — all sounds synthesized via Web Audio API
}

function toggleSound() {
  soundEnabled = !soundEnabled;
  document.getElementById('btnSound').textContent = soundEnabled ? '🔊' : '🔇';
}

function toggleBgm() {
  bgmEnabled = !bgmEnabled;
  document.getElementById('btnBgm').textContent = bgmEnabled ? '🎵' : '🎵✕';
  if (bgmEnabled && gameActive) playBgm();
  else stopBgm();
}
