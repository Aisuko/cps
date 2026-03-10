// Illustrative attention-head grid used by the circuit explorer and steering playground.
// NOT the real Gemma-2-9B head layout: it is a small, deterministic toy map that follows the
// paper's definitions of roles (Fac / Inter / Irr) and circuits (C_c, C_p).

export const ROWS = 6;
export const COLS = 12;

function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const N_CC = 12;
const N_CP = 12;
const N_BOTH = 8;

export function buildHeads(seed = 7) {
  const rnd = mulberry32(seed);
  const n = ROWS * COLS;
  const idx = [...Array(n).keys()];
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const nonFac = () => (rnd() < 0.45 ? 'Inter' : 'Irr');
  const heads = Array.from({ length: n }, (_, i) => ({
    id: i, layer: Math.floor(i / COLS), head: i % COLS,
    cont: 'Irr', para: 'Irr', g: 0,
  }));
  let k = 0;
  for (let c = 0; c < N_CC; c++) { const h = heads[idx[k++]]; h.cont = 'Fac'; h.para = nonFac(); }
  for (let c = 0; c < N_CP; c++) { const h = heads[idx[k++]]; h.para = 'Fac'; h.cont = nonFac(); }
  for (let c = 0; c < N_BOTH; c++) { const h = heads[idx[k++]]; h.cont = 'Fac'; h.para = 'Fac'; }
  for (; k < n; k++) { const h = heads[idx[k]]; h.cont = nonFac(); h.para = nonFac(); }
  for (const h of heads) {
    // C_c = Fac under Cont. AND (Inter|Irr) under Para.; C_p is the mirror image.
    h.circuit = h.cont === 'Fac' && h.para !== 'Fac' ? 'Cc' : h.para === 'Fac' && h.cont !== 'Fac' ? 'Cp' : null;
    h.g = h.circuit ? 0.35 + rnd() * 0.65 : rnd() * 0.3; // illustrative facilitation-oriented gate g⁻
  }
  // rank within circuit by g⁻ (descending)
  for (const c of ['Cc', 'Cp']) {
    heads.filter((h) => h.circuit === c).sort((a, b) => b.g - a.g).forEach((h, r) => (h.rank = r + 1));
  }
  return heads;
}

export const heads = buildHeads();
export const circuitSize = {
  Cc: heads.filter((h) => h.circuit === 'Cc').length,
  Cp: heads.filter((h) => h.circuit === 'Cp').length,
};
