// Ambient hero animation: a grid of "attention heads" that flicker between contextual (blue)
// and parametric (red) activity; the pointer pulls nearby heads towards a preference.
export function initHero(canvas) {
  const ctx = canvas.getContext('2d');
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let w, h, dpr, cells = [], mouse = { x: -999, y: -999 };
  const size = 26, gap = 6;

  function layout() {
    dpr = Math.min(devicePixelRatio || 1, 2);
    w = canvas.clientWidth; h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cells = [];
    const cols = Math.ceil(w / (size + gap)), rows = Math.ceil(h / (size + gap));
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      cells.push({ x: c * (size + gap) + 4, y: r * (size + gap) + 4, p: Math.random() * 6.28, s: 0.4 + Math.random() * 0.9, k: Math.random() });
    }
  }

  function draw(t) {
    ctx.clearRect(0, 0, w, h);
    const css = getComputedStyle(document.documentElement);
    const blue = css.getPropertyValue('--blue-rgb').trim() || '79,140,255';
    const red = css.getPropertyValue('--red-rgb').trim() || '255,90,99';
    for (const c of cells) {
      const wave = Math.sin(t * 0.0006 * c.s + c.p);
      const dx = c.x - mouse.x, dy = c.y - mouse.y;
      const near = Math.max(0, 1 - Math.hypot(dx, dy) / 160);
      let a = 0.06 + 0.1 * (wave * 0.5 + 0.5) + near * 0.5;
      const isBlue = mouse.x < w / 2 ? (near > 0 ? true : c.k < 0.5) : (near > 0 ? false : c.k < 0.5);
      ctx.fillStyle = `rgba(${isBlue ? blue : red},${Math.min(a, 0.85)})`;
      const s = size * (0.78 + near * 0.3 + 0.08 * wave);
      ctx.beginPath();
      ctx.roundRect(c.x + (size - s) / 2, c.y + (size - s) / 2, s, s, 5);
      ctx.fill();
    }
    if (!reduce) requestAnimationFrame(draw);
  }

  addEventListener('resize', layout);
  canvas.parentElement.addEventListener('pointermove', (e) => {
    const r = canvas.getBoundingClientRect();
    mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
  });
  canvas.parentElement.addEventListener('pointerleave', () => (mouse = { x: -999, y: -999 }));
  layout();
  requestAnimationFrame(draw);
}
