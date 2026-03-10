import {
  MODELS, DATASETS, METHODS, ABL_METHODS, DELTAS, GEN_DATASETS,
  main, ablation, delta as deltaData, efficiency, generalisation, masking,
} from './data.js';
import { barChart, lineChart, legend, METHOD_COLORS } from './charts.js';

const seg = (items, cur, attr) =>
  items.map(([v, label]) => `<button type="button" data-${attr}="${v}" class="${v === cur ? 'on' : ''}">${label}</button>`).join('');

export function initResults(root) {
  const st = { tab: 'main', model: MODELS[0], cond: 'cont', err: true, hidden: new Set() };
  const TABS = [
    ['main', 'Main results'],
    ['ablation', 'Ablation'],
    ['delta', 'Strength δ'],
    ['gen', 'Generalisation'],
    ['eff', 'Efficiency'],
  ];

  root.innerHTML = `
    <div class="demo-head wrap">
      <div class="seg" id="rs-tabs" role="tablist">${seg(TABS, st.tab, 't')}</div>
    </div>
    <div class="controls" id="rs-controls"></div>
    <div class="legend-row" id="rs-legend"></div>
    <div id="rs-chart" class="chart-host"></div>
    <div id="rs-summary" class="summary-chips"></div>
    <p class="note" id="rs-note"></p>`;
  const $ = (s) => root.querySelector(s);

  function controls() {
    const c = $('#rs-controls');
    const model = `<div class="ctl"><span>Model</span><div class="seg small" id="c-model">${seg(MODELS.map((m) => [m, m]), st.model, 'm')}</div></div>`;
    const cond = `<div class="ctl"><span>Condition</span><div class="seg small" id="c-cond">${seg([['cont', 'Context-Pref.'], ['para', 'Parametric-Pref.']], st.cond, 'c')}</div></div>`;
    const err = `<label class="ctl check"><input type="checkbox" id="c-err" ${st.err ? 'checked' : ''}/> error bars (±1 std, 3 seeds)</label>`;
    if (st.tab === 'eff') c.innerHTML = '';
    else if (st.tab === 'delta') c.innerHTML = cond;
    else c.innerHTML = model + cond + err;
  }

  const slice = (vals, cond) => (cond === 'cont' ? vals.slice(0, 4) : vals.slice(4, 8));

  function drawBars(names, dataset, groups, sliceFn) {
    const colors = METHOD_COLORS;
    legend($('#rs-legend'), names, colors, st.hidden, () => drawBars(names, dataset, groups, sliceFn));
    const series = names.map((n) => ({ name: n, color: colors[n], values: sliceFn(dataset[st.model][n]) }));
    barChart($('#rs-chart'), { groups, series, errBars: st.err, hidden: st.hidden, label: 'EM accuracy' });
    // Summary: CPS gain over Vanilla per group
    const v = sliceFn(dataset[st.model].Vanilla), c = sliceFn(dataset[st.model].CPS);
    $('#rs-summary').innerHTML = groups
      .map((g, i) => `<div class="chip"><span>${g}</span><b>${v[i][0].toFixed(3)} → ${c[i][0].toFixed(3)}</b><em class="${c[i][0] >= v[i][0] ? 'up' : 'down'}">${c[i][0] >= v[i][0] ? '+' : ''}${(c[i][0] - v[i][0]).toFixed(3)}</em></div>`)
      .join('');
  }

  function render() {
    root.querySelectorAll('#rs-tabs button').forEach((b) => b.classList.toggle('on', b.dataset.t === st.tab));
    controls();
    $('#rs-summary').innerHTML = '';
    $('#rs-legend').innerHTML = '';
    $('#rs-chart').className = 'chart-host';
    const condName = st.cond === 'cont' ? 'Context-Preference' : 'Parametric-Preference';

    if (st.tab === 'main') {
      drawBars(METHODS, main, DATASETS, (v) => slice(v, st.cond));
      $('#rs-note').innerHTML = `Exact-match accuracy under the ${condName} condition (Table 1). Chips show Vanilla → CPS. Performance is lower when steering toward parametric knowledge — the paper notes the effect is asymmetric. Differences are descriptive, not significance-tested (3 seeds).`;
    } else if (st.tab === 'ablation') {
      drawBars(ABL_METHODS, ablation, DATASETS, (v) => slice(v, st.cond));
      $('#rs-note').innerHTML = `Boost-only vs. suppress-only vs. combined CPS on a subset of the test set (Table 3). Combining both operations gives the highest mean EM in most settings.`;
    } else if (st.tab === 'gen') {
      drawBars(METHODS, generalisation, GEN_DATASETS, (v) => (st.cond === 'cont' ? v.slice(0, 2) : v.slice(2, 4)));
      $('#rs-note').innerHTML = `Circuits are localised once on NQ-Swap and applied <b>unchanged</b> to unseen DynamicQA and WikiContradict — no CHG re-run, no re-selection (Table 6).`;
    } else if (st.tab === 'delta') {
      $('#rs-legend').innerHTML = '';
      const colors = { 'Gemma-2-9B': '#ff7a59', 'Llama-3.1-8B': '#4f8fd9', 'Mistral-7B-v0.1': '#3fa79a' };
      lineChart($('#rs-chart'), {
        xs: DELTAS,
        yMin: st.cond === 'cont' ? 0.4 : 0, yMax: st.cond === 'cont' ? 1 : 0.4,
        lines: MODELS.map((m) => ({ name: m, color: colors[m], values: deltaData[st.cond][m] })),
        label: 'EM versus δ',
      });
      $('#rs-legend').innerHTML = MODELS.map((m) => `<span><i class="sw" style="background:${colors[m]}"></i>${m}</span>`).join('');
      $('#rs-note').innerHTML = `ConflictBank, ${condName} condition (Table 4). δ = 0 is no intervention. Performance improves at moderate strengths and declines when δ grows too large; δ = 0.3 is used throughout.`;
    } else {
      drawEff();
    }
  }

  function drawEff() {
    const maxMs = Math.max(...efficiency.map((e) => e.ms));
    $('#rs-chart').className = 'chart-host eff';
    $('#rs-chart').innerHTML = `<div class="table-scroll"><table class="eff-table">
      <thead><tr><th>Method</th><th>Avg input tokens</th><th>Cont. acc.</th><th>Para. acc.</th><th>Fwd passes</th><th>Avg response time (ms)</th></tr></thead>
      <tbody>${efficiency.map((e) => `<tr class="${e.m === 'CPS' ? 'hl' : ''}">
        <td><i class="sw" style="background:${METHOD_COLORS[e.m]}"></i>${e.m}</td>
        <td>${e.tokens}</td><td>${e.cont.toFixed(3)}</td><td>${e.para.toFixed(3)}</td><td>${e.passes}</td>
        <td><div class="lat"><div class="lat-bar" style="--w:${(e.ms / maxMs) * 100}%;background:${METHOD_COLORS[e.m]}"></div><span>${e.ms.toFixed(1)}${e.pct ? ` <em>(+${e.pct}%)</em>` : ''}</span></div></td></tr>`).join('')}</tbody></table></div>`;
    $('#rs-summary').innerHTML = `
      <div class="chip"><span>Extra prompt tokens</span><b>0</b></div>
      <div class="chip"><span>Extra forward passes</span><b>0</b></div>
      <div class="chip"><span>Latency vs. Vanilla</span><b>+1.2%</b></div>`;
    $('#rs-note').innerHTML = `Gemma-2-9B on NQ-Swap, same batch size, latency measured from tokenisation to end of decoding (Table 5). JuICE needs two forward passes; Prompt adds 18 tokens; CPS needs neither.`;
  }

  root.addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.t) st.tab = b.dataset.t;
    else if (b.dataset.m) st.model = b.dataset.m;
    else if (b.dataset.c) st.cond = b.dataset.c;
    else return;
    render();
  });
  root.addEventListener('change', (e) => { if (e.target.id === 'c-err') { st.err = e.target.checked; render(); } });
  render();
}

// ---- Circuit masking validation (Table 2) ----
export function initMasking(root) {
  let cond = 'cont';
  const names = ['Vanilla', 'Random-k', 'Cc masked', 'Cp masked'];
  root.innerHTML = `
    <div class="demo-head wrap">
      <div class="seg" id="mk-cond" role="tablist">
        <button type="button" data-c="cont" class="on">Context-Preference condition</button>
        <button type="button" data-c="para">Parametric-Preference condition</button>
      </div>
    </div>
    <div id="mk-chart" class="chart-host"></div>
    <div class="delta-grid" id="mk-delta"></div>
    <p class="note">Forced-choice preference rate between org_answer and counter_answer, Gemma-2-9B, with the top-K heads of one circuit masked (Table 2). Masking C꜀ pushes the model <b>away from</b> contextual answers on all four benchmarks; masking Cₚ gives a weaker, generally opposite shift (no change on ConflictQA). Size-matched random masking moves rates by at most ±0.011 with no consistent direction.</p>`;
  const $ = (s) => root.querySelector(s);
  const colors = { ...METHOD_COLORS };

  function render() {
    root.querySelectorAll('#mk-cond button').forEach((b) => b.classList.toggle('on', b.dataset.c === cond));
    const off = cond === 'cont' ? 0 : 4;
    const series = names.map((n) => ({ name: n, color: colors[n], values: masking[n].slice(off, off + 4) }));
    barChart($('#mk-chart'), { groups: DATASETS, series, errBars: false, yMax: 1, highlight: '', label: 'preference rates' });
    const sel = ['Random-k', 'Cc masked', 'Cp masked'];
    $('#mk-delta').innerHTML = sel.map((n) => `<div class="dcard"><h4><i class="sw" style="background:${colors[n]}"></i>${n.replace('Cc', 'C꜀').replace('Cp', 'Cₚ')} vs Vanilla</h4>
      ${DATASETS.map((d, i) => {
        const dv = masking[n][off + i] - masking.Vanilla[off + i];
        const w = Math.min(Math.abs(dv) / 0.07, 1) * 50;
        return `<div class="drow"><span>${d}</span><div class="dtrack"><div class="dbar ${dv >= 0 ? 'pos' : 'neg'}" style="width:${w}%;${dv >= 0 ? 'left:50%' : `right:50%`}"></div></div><b>${dv >= 0 ? '+' : '−'}${Math.abs(dv).toFixed(3)}</b></div>`;
      }).join('')}</div>`).join('');
  }
  root.addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.c) cond = b.dataset.c; else return;
    render();
  });
  render();
}
