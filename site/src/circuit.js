import { heads, COLS, circuitSize } from './headmap.js';

const STEPS = [
  {
    key: 'cont',
    tab: '① Cont. roles',
    title: 'CHG under the Context-Preference condition',
    text: 'Causal Head Gating jointly optimises a gate g ∈ [0,1] for every head, with the target answer set to the <b>counter_answer</b> (follow the context). Retention (λ>0) and suppression (λ<0) runs give each head a role: <b>Fac</b>ilitating, <b>Inter</b>fering or <b>Irr</b>elevant.',
  },
  {
    key: 'para',
    tab: '② Para. roles',
    title: 'CHG under the Parametric-Preference condition',
    text: 'Same question, same counterfactual context — only the target flips to the <b>org_answer</b> (follow parametric memory). Running CHG again gives a second, directly comparable role map r<sup>(Para.)</sup>(h).',
  },
  {
    key: 'transition',
    tab: '③ Role transition',
    title: 'Role-transition analysis',
    text: 'Heads that <b>facilitate one preference</b> but become <b>interfering or irrelevant under the opposite one</b> are functionally specialised (rings). Heads that facilitate both — or neither — stay ambiguous and are left alone.',
  },
  {
    key: 'circuit',
    tab: '④ Circuits',
    title: 'Functional circuits C<sub>c</sub> and C<sub>p</sub>',
    text: 'C<sub>c</sub> = {Fac under Cont. ∧ Inter/Irr under Para.}, C<sub>p</sub> = {Fac under Para. ∧ Inter/Irr under Cont.}. Heads in each circuit are ranked by the facilitation-oriented gate g⁻ and the top-<i>K</i> become the intervention targets (★).',
  },
];

const ROLE_NAME = { Fac: 'Facilitating', Inter: 'Interfering', Irr: 'Irrelevant' };

export function initCircuit(root) {
  let step = 0;
  let K = 8;
  let timer = null;
  const maxK = Math.min(circuitSize.Cc, circuitSize.Cp);
  K = Math.min(K, maxK);

  root.innerHTML = `
    <div class="demo-head">
      <div class="seg" role="tablist" aria-label="Pipeline step"></div>
      <button class="btn ghost" id="cd-play" type="button">▶ Play walkthrough</button>
    </div>
    <div class="demo-body">
      <div class="grid-wrap">
        <div class="head-grid" style="--cols:${COLS}" aria-label="Illustrative attention head grid"></div>
        <p class="note">Illustrative 6 × 12 head map generated from the paper's definitions — not the real head positions of any model.</p>
        <div class="legend-row" id="cd-legend"></div>
      </div>
      <div class="panel">
        <h3 id="cd-title"></h3>
        <p id="cd-text"></p>
        <label class="slider" id="cd-k-wrap">
          <span>Top-<i>K</i> heads per circuit: <b id="cd-k-val"></b></span>
          <input id="cd-k" type="range" min="1" max="${maxK}" value="${K}" />
        </label>
        <div class="inspector" id="cd-info" aria-live="polite">Hover or tap a head to inspect its roles.</div>
        <div class="stats" id="cd-stats"></div>
      </div>
    </div>`;

  const seg = root.querySelector('.seg');
  const grid = root.querySelector('.head-grid');
  const cells = heads.map((h) => {
    const c = document.createElement('button');
    c.type = 'button';
    c.className = 'cell';
    c.setAttribute('aria-label', `Layer ${h.layer} head ${h.head}`);
    c.addEventListener('mouseenter', () => inspect(h));
    c.addEventListener('focus', () => inspect(h));
    c.addEventListener('click', () => inspect(h));
    grid.appendChild(c);
    return c;
  });
  STEPS.forEach((s, i) => {
    const b = document.createElement('button');
    b.type = 'button'; b.role = 'tab'; b.textContent = s.tab;
    b.onclick = () => { stop(); go(i); };
    seg.appendChild(b);
  });

  const $ = (id) => root.querySelector(id);
  $('#cd-k').addEventListener('input', (e) => { K = +e.target.value; render(); });
  $('#cd-play').onclick = () => (timer ? stop() : play());

  function play() {
    $('#cd-play').textContent = '⏸ Pause';
    go(0);
    timer = setInterval(() => {
      if (step >= STEPS.length - 1) { stop(); return; }
      go(step + 1);
    }, 2800);
  }
  function stop() { clearInterval(timer); timer = null; $('#cd-play').textContent = '▶ Play walkthrough'; }
  function go(i) { step = i; render(); }

  function roleOf(h) {
    const s = STEPS[step].key;
    if (s === 'cont') return h.cont;
    if (s === 'para') return h.para;
    return null;
  }

  function classify(h) {
    if (h.circuit === 'Cc') return 'Contextual circuit C<sub>c</sub> (Fac under Cont., not Fac under Para.)';
    if (h.circuit === 'Cp') return 'Parametric circuit C<sub>p</sub> (Fac under Para., not Fac under Cont.)';
    if (h.cont === 'Fac' && h.para === 'Fac') return 'Ambiguous — facilitates both preferences, not in a circuit';
    return 'Not facilitating either preference — not in a circuit';
  }

  function inspect(h) {
    $('#cd-info').innerHTML = `<b>Layer ${h.layer} · Head ${h.head}</b><br>
      r<sup>(Cont.)</sup> = <span class="r-${h.cont}">${ROLE_NAME[h.cont]}</span> &nbsp;
      r<sup>(Para.)</sup> = <span class="r-${h.para}">${ROLE_NAME[h.para]}</span><br>
      <span class="muted">${classify(h)}</span>`;
  }

  function render() {
    const S = STEPS[step];
    [...seg.children].forEach((b, i) => { b.classList.toggle('on', i === step); b.setAttribute('aria-selected', String(i === step)); });
    $('#cd-title').innerHTML = S.title;
    $('#cd-text').innerHTML = S.text;
    $('#cd-k-wrap').style.display = S.key === 'circuit' ? '' : 'none';
    $('#cd-k-val').textContent = K;

    heads.forEach((h, i) => {
      const c = cells[i];
      let cls = 'cell';
      if (S.key === 'cont' || S.key === 'para') {
        cls += ' role-' + roleOf(h);
      } else if (S.key === 'transition') {
        cls += ' role-' + h.cont; // base colour = Cont. role
        cls += h.circuit ? ` ring ring-${h.circuit}` : ' dim';
      } else {
        cls += h.circuit ? ` circ-${h.circuit}` : ' role-Irr dim';
        if (h.circuit && h.rank <= K) cls += ' star';
      }
      c.className = cls;
      c.dataset.badge = S.key === 'circuit' && h.circuit && h.rank <= K ? '★' : '';
    });

    const legend = $('#cd-legend');
    if (S.key === 'circuit') {
      legend.innerHTML = `<span><i class="sw circ-Cc"></i>Contextual circuit C<sub>c</sub> (${circuitSize.Cc})</span>
        <span><i class="sw circ-Cp"></i>Parametric circuit C<sub>p</sub> (${circuitSize.Cp})</span>
        <span><i class="sw role-Irr"></i>Other heads</span><span>★ top-K target</span>`;
    } else if (S.key === 'transition') {
      legend.innerHTML = `<span><i class="sw ring-Cc"></i>→ C<sub>c</sub> candidate</span><span><i class="sw ring-Cp"></i>→ C<sub>p</sub> candidate</span><span class="muted">cell colour = role under Cont.</span>`;
    } else {
      legend.innerHTML = `<span><i class="sw role-Fac"></i>Facilitating</span><span><i class="sw role-Inter"></i>Interfering</span><span><i class="sw role-Irr"></i>Irrelevant</span>`;
    }

    const count = (f) => heads.filter(f).length;
    $('#cd-stats').innerHTML = S.key === 'cont' || S.key === 'para'
      ? ['Fac', 'Inter', 'Irr'].map((r) => `<div><b>${count((h) => h[S.key] === r)}</b><span>${r}</span></div>`).join('')
      : `<div><b>${circuitSize.Cc}</b><span>in C<sub>c</sub></span></div><div><b>${circuitSize.Cp}</b><span>in C<sub>p</sub></span></div><div><b>${heads.length - circuitSize.Cc - circuitSize.Cp}</b><span>untouched</span></div>`;
  }

  render();
}
