import { heads, COLS } from './headmap.js';
import { DELTAS, MODELS, delta as deltaData, chicago } from './data.js';
import { lineChart, METHOD_COLORS } from './charts.js';

export function initSteering(root) {
  const st = { policy: 'cont', di: 3, model: MODELS[0] };

  root.innerHTML = `
    <div class="demo-head">
      <div class="seg" id="st-policy" role="tablist" aria-label="Steering policy">
        <button type="button" data-p="cont" class="on">Context-Preference policy</button>
        <button type="button" data-p="para">Parametric-Preference policy</button>
      </div>
    </div>
    <div class="demo-body">
      <div class="grid-wrap">
        <div class="mini-q">
          <div class="q"><b>Q:</b> ${chicago.q}</div>
          <div class="c"><b>C:</b> ${chicago.ctxPre}<u>${chicago.counter}</u>${chicago.ctxPost}</div>
        </div>
        <div class="head-grid steer" style="--cols:${COLS}" id="st-grid" aria-label="Head grid with circuit scaling"></div>
        <div class="legend-row">
          <span><i class="sw circ-Cc"></i>C<sub>c</sub> heads</span>
          <span><i class="sw circ-Cp"></i>C<sub>p</sub> heads</span>
          <span><i class="sw role-Irr"></i>unchanged (s = 1)</span>
        </div>
        <div class="answer" id="st-answer" aria-live="polite"></div>
      </div>
      <div class="panel">
        <label class="slider">
          <span>Intervention strength δ = <b id="st-d"></b></span>
          <input id="st-slider" type="range" min="0" max="${DELTAS.length - 1}" step="1" value="${st.di}" />
          <div class="ticks">${DELTAS.map((d) => `<span>${d.toFixed(1)}</span>`).join('')}</div>
        </label>
        <div class="formula" id="st-formula"></div>
        <div class="scale-bars" id="st-bars"></div>
        <div class="seg small" id="st-model" role="tablist" aria-label="Model">${MODELS.map((m, i) => `<button type="button" data-m="${m}" class="${i ? '' : 'on'}">${m}</button>`).join('')}</div>
        <div id="st-chart"></div>
        <p class="note" id="st-chart-note"></p>
      </div>
    </div>`;

  const $ = (s) => root.querySelector(s);
  const grid = $('#st-grid');
  const cells = heads.map(() => {
    const c = document.createElement('div');
    grid.appendChild(c);
    return c;
  });

  $('#st-policy').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    st.policy = b.dataset.p;
    [...$('#st-policy').children].forEach((x) => x.classList.toggle('on', x === b));
    render();
  });
  $('#st-slider').addEventListener('input', (e) => { st.di = +e.target.value; render(); });
  $('#st-model').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    st.model = b.dataset.m;
    [...$('#st-model').children].forEach((x) => x.classList.toggle('on', x === b));
    render();
  });

  function coeffs(d, policy) {
    // Paper: (α, β) = (+δ, −δ/(1+δ)) for Context-Preference; (−δ/(1+δ), +δ) for Parametric-Preference.
    const neg = -d / (1 + d);
    return policy === 'cont' ? { a: d, b: neg } : { a: neg, b: d };
  }

  function render() {
    const d = DELTAS[st.di];
    const { a, b } = coeffs(d, st.policy);
    $('#st-d').textContent = d.toFixed(1);

    heads.forEach((h, i) => {
      const c = cells[i];
      const s = h.circuit === 'Cc' ? 1 + a : h.circuit === 'Cp' ? 1 + b : 1;
      c.className = 'cell ' + (h.circuit ? `circ-${h.circuit}` : 'role-Irr dim');
      c.style.transform = `scale(${(0.35 + 0.4 * s).toFixed(3)})`; // s=2 -> 1.15, s=1 -> 0.75, s=0.5 -> 0.55
      c.style.opacity = h.circuit ? Math.min(0.35 + 0.6 * s, 1).toFixed(2) : '';
      c.style.boxShadow = h.circuit && s > 1.02 ? `0 0 ${6 + 14 * (s - 1)}px var(--${h.circuit === 'Cc' ? 'blue' : 'red'})` : 'none';
      c.title = h.circuit ? `${h.circuit} head · s = ${s.toFixed(3)}` : 'unchanged · s = 1';
    });

    const f = (v) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(3);
    $('#st-formula').innerHTML = `
      <div class="eq">õ<sub>ℓ,h</sub> = s<sub>ℓ,h</sub> · o<sub>ℓ,h</sub></div>
      <div class="eq">α = <b>${f(a)}</b> &nbsp; β = <b>${f(b)}</b> &nbsp;
        <span class="muted">${st.policy === 'cont' ? '(α, β) = (+δ, −δ/(1+δ))' : '(α, β) = (−δ/(1+δ), +δ)'}</span></div>`;

    const bar = (name, s, cls) => `<div class="sb"><span>${name}</span>
      <div class="track"><div class="fill ${cls}" style="width:${(s / 2) * 100}%"></div><div class="one"></div></div><b>×${s.toFixed(3)}</b></div>`;
    $('#st-bars').innerHTML = bar('C<sub>c</sub> output scale (1+α)', 1 + a, 'blue') + bar('C<sub>p</sub> output scale (1+β)', 1 + b, 'red');

    const target = st.policy === 'cont' ? chicago.counter : chicago.org;
    const verb = st.policy === 'cont' ? 'Boost C<sub>c</sub> · Suppress C<sub>p</sub>' : 'Boost C<sub>p</sub> · Suppress C<sub>c</sub>';
    $('#st-answer').className = 'answer ' + (d === 0 ? 'idle' : st.policy);
    $('#st-answer').innerHTML = d === 0
      ? `δ = 0 → no intervention. The model is left to decide between <b>${chicago.counter}</b> and <b>${chicago.org}</b> on its own.`
      : `<span class="muted">${verb}</span><br>Steered answer (Fig. 1): <b class="big">${target} episodes</b>`;

    const cond = st.policy;
    const color = cond === 'cont' ? '#4f8cff' : '#ff5a63';
    lineChart($('#st-chart'), {
      xs: DELTAS, marker: st.di, yMin: cond === 'cont' ? 0.4 : 0, yMax: cond === 'cont' ? 1 : 0.4,
      lines: [{ name: st.model, color: METHOD_COLORS.CPS, values: deltaData[cond][st.model] }],
      label: 'EM accuracy versus delta',
    });
    const v = deltaData[cond][st.model][st.di];
    $('#st-chart-note').innerHTML = `ConflictBank EM, ${st.model}, ${cond === 'cont' ? 'Context' : 'Parametric'}-Preference condition: <b>${v[0].toFixed(3)} ± ${v[1].toFixed(3)}</b> at δ = ${d.toFixed(1)}. Paper's trend: gains rise to a sweet spot (δ = 0.3 context, 0.3–0.5 parametric) then decline.`;
  }
  render();
}
