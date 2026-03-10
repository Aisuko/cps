// Tiny dependency-free SVG chart helpers (grouped bars + lines) with a shared tooltip.
const NS = 'http://www.w3.org/2000/svg';

export const METHOD_COLORS = {
  Vanilla: '#6b7390',
  'Random-k': '#8f7fc0',
  Prompt: '#d0a24e',
  PH3: '#3fa79a',
  JuICE: '#4f8fd9',
  CPS: '#ff7a59',
  'Boost only': '#d0a24e',
  'Suppress only': '#3fa79a',
  'Cc masked': '#4f8cff',
  'Cp masked': '#ff5a63',
};

export function el(name, attrs = {}, parent) {
  const n = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}

let tip;
export function showTip(html, ev) {
  if (!tip) {
    tip = document.createElement('div');
    tip.className = 'tip';
    document.body.appendChild(tip);
  }
  tip.innerHTML = html;
  tip.style.opacity = 1;
  const pad = 14;
  const w = tip.offsetWidth;
  let x = ev.clientX + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  tip.style.transform = `translate(${Math.max(8, x)}px, ${ev.clientY + pad}px)`;
}
export function hideTip() { if (tip) tip.style.opacity = 0; }

const f3 = (v) => v.toFixed(3);

/**
 * Grouped bar chart.
 * opts: { groups:[str], series:[{name,color,values:[[mean,std]|number]}], yMax, errBars, hidden:Set, highlight }
 */
export function barChart(host, opts) {
  const { groups, series, errBars = true, hidden = new Set(), highlight = 'CPS' } = opts;
  const vis = series.filter((s) => !hidden.has(s.name));
  const W = 760, H = 340, m = { t: 16, r: 12, b: 44, l: 44 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const maxV = Math.max(...series.flatMap((s) => s.values.map((v) => (Array.isArray(v) ? v[0] + (errBars ? v[1] : 0) : v))));
  const yMax = opts.yMax ?? Math.min(1, Math.ceil((maxV * 1.08) / 0.1) * 0.1);
  const y = (v) => m.t + ih - (v / yMax) * ih;

  host.innerHTML = '';
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img', 'aria-label': opts.label ?? 'bar chart' }, host);

  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const v = (yMax / ticks) * i;
    el('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: 'grid' }, svg);
    const t = el('text', { x: m.l - 8, y: y(v) + 4, class: 'axis', 'text-anchor': 'end' }, svg);
    t.textContent = v.toFixed(2);
  }

  const gw = iw / groups.length;
  const pad = gw * 0.14;
  const bw = (gw - pad * 2) / Math.max(vis.length, 1);
  groups.forEach((g, gi) => {
    const gx = m.l + gi * gw;
    const lab = el('text', { x: gx + gw / 2, y: H - 16, class: 'axis lab', 'text-anchor': 'middle' }, svg);
    lab.textContent = g;
    vis.forEach((s, si) => {
      const raw = s.values[gi];
      const [mean, sd] = Array.isArray(raw) ? raw : [raw, 0];
      const x = gx + pad + si * bw;
      const bar = el('rect', {
        x: x + 1, y: y(mean), width: Math.max(bw - 2, 2), height: Math.max(m.t + ih - y(mean), 0), rx: 2,
        fill: s.color, class: 'bar' + (s.name === highlight ? ' hl' : ''),
      }, svg);
      bar.style.animationDelay = `${gi * 60 + si * 35}ms`;
      const tipHtml = `<b>${s.name}</b><br>${g}<br>${f3(mean)}${sd ? ` ± ${f3(sd)}` : ''}`;
      bar.addEventListener('mousemove', (e) => showTip(tipHtml, e));
      bar.addEventListener('mouseleave', hideTip);
      if (errBars && sd) {
        const cx = x + bw / 2;
        const eb = el('g', { class: 'err' }, svg);
        el('line', { x1: cx, x2: cx, y1: y(mean + sd), y2: y(Math.max(mean - sd, 0)) }, eb);
        el('line', { x1: cx - 3, x2: cx + 3, y1: y(mean + sd), y2: y(mean + sd) }, eb);
        el('line', { x1: cx - 3, x2: cx + 3, y1: y(Math.max(mean - sd, 0)), y2: y(Math.max(mean - sd, 0)) }, eb);
      }
    });
  });
  return svg;
}

/** Legend with toggle buttons. */
export function legend(host, names, colors, hidden, onChange) {
  host.innerHTML = '';
  names.forEach((n) => {
    const b = document.createElement('button');
    b.className = 'leg' + (hidden.has(n) ? ' off' : '');
    b.type = 'button';
    b.innerHTML = `<i style="background:${colors[n]}"></i>${n}`;
    b.setAttribute('aria-pressed', String(!hidden.has(n)));
    b.onclick = () => {
      hidden.has(n) ? hidden.delete(n) : hidden.add(n);
      onChange();
    };
    host.appendChild(b);
  });
}

/** Multi-line chart: xs numeric labels evenly spaced (categorical), lines [{name,color,values:[[m,s]]}] */
export function lineChart(host, opts) {
  const { xs, lines, marker = null, yMin = 0, yMax = 1, label = 'line chart' } = opts;
  const W = 760, H = 320, m = { t: 16, r: 20, b: 44, l: 48 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const x = (i) => m.l + (i / (xs.length - 1)) * iw;
  const y = (v) => m.t + ih - ((v - yMin) / (yMax - yMin)) * ih;
  host.innerHTML = '';
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img', 'aria-label': label }, host);
  for (let i = 0; i <= 5; i++) {
    const v = yMin + ((yMax - yMin) / 5) * i;
    el('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: 'grid' }, svg);
    const t = el('text', { x: m.l - 8, y: y(v) + 4, class: 'axis', 'text-anchor': 'end' }, svg);
    t.textContent = v.toFixed(2);
  }
  xs.forEach((v, i) => {
    const t = el('text', { x: x(i), y: H - 22, class: 'axis lab', 'text-anchor': 'middle' }, svg);
    t.textContent = v.toFixed(1);
  });
  const xl = el('text', { x: W / 2, y: H - 4, class: 'axis', 'text-anchor': 'middle' }, svg);
  xl.textContent = 'intervention strength δ';

  if (marker !== null) {
    el('line', { x1: x(marker), x2: x(marker), y1: m.t, y2: m.t + ih, class: 'marker' }, svg);
  }
  lines.forEach((ln) => {
    const pts = ln.values.map(([mean], i) => `${x(i)},${y(mean)}`).join(' ');
    // std band
    const up = ln.values.map(([mean, sd], i) => `${x(i)},${y(Math.min(mean + sd, yMax))}`);
    const dn = ln.values.map(([mean, sd], i) => `${x(i)},${y(Math.max(mean - sd, yMin))}`).reverse();
    el('polygon', { points: [...up, ...dn].join(' '), fill: ln.color, opacity: 0.13 }, svg);
    const path = el('polyline', { points: pts, fill: 'none', stroke: ln.color, 'stroke-width': 2.5, class: 'line' }, svg);
    path.setAttribute('pathLength', 1);
    ln.values.forEach(([mean, sd], i) => {
      const active = marker === i;
      const c = el('circle', { cx: x(i), cy: y(mean), r: active ? 6.5 : 4, fill: ln.color, class: 'dot' + (active ? ' active' : '') }, svg);
      c.addEventListener('mousemove', (e) => showTip(`<b>${ln.name}</b><br>δ = ${xs[i].toFixed(1)}<br>${f3(mean)} ± ${f3(sd)}`, e));
      c.addEventListener('mouseleave', hideTip);
    });
  });
  return svg;
}
