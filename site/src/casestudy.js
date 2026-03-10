import { examples } from './data.js';

// Qualitative case study (paper Fig. 3): Gemma-2-9B on NQ-Swap, Context-Preference policy.
export function initCase(root) {
  let ex = 0;
  let on = false; // steering switch
  let run = 0;

  root.innerHTML = `
    <div class="demo-head">
      <div class="seg" id="cs-tabs" role="tablist" aria-label="Example">
        ${examples.map((_, i) => `<button type="button" data-i="${i}" class="${i ? '' : 'on'}">Example ${i + 1}</button>`).join('')}
      </div>
      <div class="switch-row">
        <span>Vanilla</span>
        <button type="button" class="switch" id="cs-switch" role="switch" aria-checked="false" aria-label="Enable CPS steering"><i></i></button>
        <span>CPS</span>
      </div>
    </div>
    <div class="case-card">
      <div class="row"><b>Q:</b> <i id="cs-q"></i></div>
      <div class="row"><b>Context:</b> <span id="cs-ctx"></span></div>
      <div class="row"><b>Gold (context):</b> <b id="cs-gold" class="gold"></b></div>
      <div class="scan" id="cs-scan" aria-hidden="true"></div>
      <div class="pair">
        <div class="ans" id="cs-van"><small>Vanilla</small><span></span></div>
        <div class="ans" id="cs-cps"><small>CPS (ours)</small><span></span></div>
      </div>
      <p class="note">CPS boosts contextual heads and suppresses parametric heads identified from the functional circuits. In Example 3, Vanilla and CPS both already follow the context — CPS preserves answers that already match the preference.</p>
    </div>`;

  const $ = (s) => root.querySelector(s);

  function setAns(id, text, ok, show) {
    const n = $(id);
    n.className = 'ans' + (show ? ` show ${ok ? 'ok' : 'bad'}` : '');
    n.querySelector('span').textContent = show ? `“${text}”` : '…';
  }

  function render(animate = true) {
    const e = examples[ex];
    const my = ++run;
    $('#cs-q').textContent = e.q;
    $('#cs-ctx').innerHTML = `${e.pre}<mark>${e.key}</mark>${e.post}`;
    $('#cs-gold').textContent = e.gold;
    $('#cs-switch').setAttribute('aria-checked', String(on));
    $('#cs-switch').classList.toggle('on', on);
    $('#cs-van').classList.toggle('dim', on);
    $('#cs-cps').classList.toggle('dim', !on);

    const scan = $('#cs-scan');
    scan.className = 'scan' + (animate ? ' go' : '');
    scan.dataset.mode = on ? 'CPS: boosting C_c ↑ · suppressing C_p ↓' : 'No intervention';
    setAns('#cs-van', e.vanilla, e.vanilla === e.gold, !animate);
    setAns('#cs-cps', e.cps, e.cps === e.gold, !animate);
    if (animate) {
      setTimeout(() => { if (my === run) setAns('#cs-van', e.vanilla, e.vanilla === e.gold, true); }, 700);
      setTimeout(() => { if (my === run) setAns('#cs-cps', e.cps, e.cps === e.gold, true); }, 1300);
    }
  }

  $('#cs-tabs').addEventListener('click', (ev) => {
    const b = ev.target.closest('button'); if (!b) return;
    ex = +b.dataset.i;
    root.querySelectorAll('#cs-tabs button').forEach((x) => x.classList.toggle('on', x === b));
    render();
  });
  $('#cs-switch').addEventListener('click', () => { on = !on; render(); });
  render();
}
