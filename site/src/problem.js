import { chicago } from './data.js';

// Interactive controlled-conflict tuple: the two preference conditions share identical context
// and differ only in the target answer (paper §3.1 and Figure 1).
export function initProblem(root) {
  let cond = 'cont';
  root.innerHTML = `
    <div class="demo-head">
      <div class="seg" role="tablist" aria-label="Preference condition">
        <button type="button" data-c="cont" class="on">Context-Preference (Cont.)</button>
        <button type="button" data-c="para">Parametric-Preference (Para.)</button>
      </div>
    </div>
    <div class="tuple">
      <div class="field"><span class="k">question</span><span class="v">${chicago.q}</span></div>
      <div class="field same"><span class="k">counter_context</span><span class="v">${chicago.ctxPre}<mark>${chicago.counter}</mark>${chicago.ctxPost}</span></div>
      <div class="field" id="pb-target"></div>
      <div class="field faded"><span class="k">counter_answer / org_answer</span><span class="v"><span class="blue-t">${chicago.counter}</span> (follows context) · <span class="red-t">${chicago.org}</span> (follows parametric memory)</span></div>
    </div>
    <p class="note">Same question, same conflicting evidence — only the <b>target answer</b> changes between the two conditions. That isolates the contribution of contextual vs. parametric knowledge.</p>`;

  const target = root.querySelector('#pb-target');
  function render() {
    root.querySelectorAll('.seg button').forEach((b) => b.classList.toggle('on', b.dataset.c === cond));
    const isCont = cond === 'cont';
    target.className = 'field target ' + cond;
    target.innerHTML = `<span class="k">target answer</span><span class="v pop"><b>${isCont ? chicago.counter : chicago.org}</b> <span class="muted">← ${isCont ? 'counter_answer' : 'org_answer'}</span></span>`;
  }
  root.querySelector('.seg').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    cond = b.dataset.c; render();
  });
  render();
}
