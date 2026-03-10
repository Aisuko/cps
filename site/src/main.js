import './style.css';
import { initHero } from './hero.js';
import { initProblem } from './problem.js';
import { initCircuit } from './circuit.js';
import { initSteering } from './steering.js';
import { initCase } from './casestudy.js';
import { initResults, initMasking } from './results.js';

// Asset paths in HTML are written relative; make them respect the configured base path.
const base = import.meta.env.BASE_URL;
document.querySelectorAll('img[src^="img/"]').forEach((i) => i.setAttribute('src', base + i.getAttribute('src')));
document.querySelectorAll('[data-zoom]').forEach((b) => (b.dataset.zoom = base + b.dataset.zoom));

// Theme toggle
const root = document.documentElement;
document.getElementById('theme').addEventListener('click', () => {
  const cur = root.dataset.theme || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  const next = cur === 'dark' ? 'light' : 'dark';
  root.dataset.theme = next;
  try { localStorage.setItem('cps-theme', next); } catch (e) { /* ignore */ }
});

initHero(document.getElementById('hero-canvas'));

// Demos are mounted lazily when first scrolled near.
const mounts = [
  ['problem-demo', initProblem],
  ['circuit-demo', initCircuit],
  ['steering-demo', initSteering],
  ['case-demo', initCase],
  ['results-demo', initResults],
  ['masking-demo', initMasking],
];
const lazy = new IntersectionObserver((entries) => {
  entries.forEach((e) => {
    if (!e.isIntersecting) return;
    lazy.unobserve(e.target);
    const m = mounts.find(([id]) => id === e.target.id);
    m && m[1](e.target);
  });
}, { rootMargin: '300px' });
mounts.forEach(([id]) => lazy.observe(document.getElementById(id)));

// Reveal on scroll
const rev = new IntersectionObserver((es) => es.forEach((e) => {
  if (e.isIntersecting) { e.target.classList.add('in'); rev.unobserve(e.target); }
}), { threshold: 0.12 });
document.querySelectorAll('.sec > *, .steps li, .stat').forEach((n) => { n.classList.add('reveal'); rev.observe(n); });

// Count-up stats
const countObs = new IntersectionObserver((es) => es.forEach((e) => {
  if (!e.isIntersecting) return;
  countObs.unobserve(e.target);
  const n = e.target;
  const from = n.dataset.to ? +n.dataset.count : 0;
  const to = n.dataset.to ? +n.dataset.to : +n.dataset.count;
  const dec = +(n.dataset.dec || 0);
  const pre = n.dataset.pre || '', suf = n.dataset.suf || '';
  const t0 = performance.now(), dur = 1200;
  const tick = (t) => {
    const p = Math.min((t - t0) / dur, 1);
    const v = from + (to - from) * (1 - Math.pow(1 - p, 3));
    n.textContent = pre + v.toFixed(dec) + suf;
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}), { threshold: 0.6 });
document.querySelectorAll('[data-count]').forEach((n) => {
  const to = n.dataset.to ? +n.dataset.count : 0;
  n.textContent = (n.dataset.pre || '') + to.toFixed(+(n.dataset.dec || 0)) + (n.dataset.suf || '');
  countObs.observe(n);
});

// Active nav link
const links = [...document.querySelectorAll('.nav nav a')];
const secObs = new IntersectionObserver((es) => es.forEach((e) => {
  if (e.isIntersecting) links.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === '#' + e.target.id));
}), { rootMargin: '-40% 0px -55% 0px' });
document.querySelectorAll('main section[id]').forEach((s) => secObs.observe(s));

// Lightbox
const lb = document.getElementById('lightbox');
document.addEventListener('click', (e) => {
  const z = e.target.closest('[data-zoom]');
  if (z) { lb.querySelector('img').src = z.dataset.zoom; lb.hidden = false; return; }
  if (e.target.closest('#lightbox')) lb.hidden = true;
});
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') lb.hidden = true; });

// Copy citation
document.getElementById('copy-cite').addEventListener('click', async (e) => {
  try {
    await navigator.clipboard.writeText(document.getElementById('cite-text').textContent);
    e.target.textContent = 'Copied ✓';
  } catch { e.target.textContent = 'Select the text to copy'; }
  setTimeout(() => (e.target.textContent = 'Copy citation'), 1800);
});
