// All numbers below are transcribed from the paper's tables (mean, std over seeds 0/1/2).
// Compact helper: "m s m s ..." -> [[m,s], ...]
const P = (s) => {
  const n = (Array.isArray(s) ? s[0] : s).trim().split(/\s+/).map(Number);
  const o = [];
  for (let i = 0; i < n.length; i += 2) o.push([n[i], n[i + 1]]);
  return o;
};

export const MODELS = ['Gemma-2-9B', 'Llama-3.1-8B', 'Mistral-7B-v0.1'];
export const DATASETS = ['ConflictBank', 'ConflictQA', 'NQ-Swap', 'PopQA-Swap'];
export const METHODS = ['Vanilla', 'Random-k', 'Prompt', 'PH3', 'JuICE', 'CPS'];
export const ABL_METHODS = ['Vanilla', 'Boost only', 'Suppress only', 'CPS'];

// Table 1 — EM accuracy. 8 pairs: Cont.[CB,CQA,NQ,PQA] then Para.[CB,CQA,NQ,PQA]
export const main = {
  'Gemma-2-9B': {
    Vanilla: P`.667 .039  .422 .150  .233 .033  .467 .077   .078 .009  .100 .002  .100 .077  .067 .088`,
    'Random-k': P`.653 .031  .407 .081  .240 .035  .467 .103   .014 .035  .107 .031  .104 .012  .057 .042`,
    Prompt: P`.633 .120  .467 .158  .278 .019  .444 .051   .100 .001  .122 .023  .122 .084  .100 .084`,
    PH3: P`.744 .069  .489 .084  .300 .069  .511 .033   .111 .019  .122 .010  .133 .117  .111 .102`,
    JuICE: P`.811 .051  .511 .135  .333 .069  .556 .051   .122 .000  .133 .013  .144 .012  .133 .069`,
    CPS: P`.833 .033  .544 .135  .389 .069  .589 .126   .140 .067  .142 .039  .161 .102  .167 .133`,
  },
  'Llama-3.1-8B': {
    Vanilla: P`.367 .058  .411 .135  .189 .058  .333 .100   .089 .033  .100 .011  .111 .102  .122 .088`,
    'Random-k': P`.357 .042  .400 .053  .187 .046  .320 .035   .101 .042  .113 .012  .113 .064  .120 .040`,
    Prompt: P`.344 .051  .400 .145  .233 .102  .311 .051   .111 .023  .111 .034  .133 .069  .144 .013`,
    PH3: P`.422 .084  .456 .088  .267 .058  .378 .126   .122 .051  .122 .039  .156 .102  .156 .051`,
    JuICE: P`.456 .107  .489 .117  .289 .039  .422 .088   .144 .077  .133 .033  .178 .102  .167 .069`,
    CPS: P`.531 .021  .522 .139  .324 .051  .456 .084   .164 .019  .142 .019  .200 .067  .178 .102`,
  },
  'Mistral-7B-v0.1': {
    Vanilla: P`.522 .039  .433 .126  .256 .107  .456 .051   .089 .051  .089 .035  .089 .100  .111 .058`,
    'Random-k': P`.527 .099  .420 .060  .253 .042  .413 .046   .120 .053  .100 .035  .080 .069  .047 .031`,
    Prompt: P`.500 .164  .456 .100  .278 .051  .489 .033   .111 .000  .111 .039  .122 .126  .144 .019`,
    PH3: P`.633 .039  .467 .117  .300 .135  .511 .051   .122 .013  .122 .019  .133 .058  .156 .002`,
    JuICE: P`.678 .039  .478 .126  .311 .116  .533 .039   .144 .033  .133 .019  .144 .084  .167 .012`,
    CPS: P`.722 .039  .489 .139  .333 .067  .567 .058   .174 .039  .142 .001  .200 .067  .189 .039`,
  },
};

// Table 3 — Ablation of intervention strategy (subset of the test set)
export const ablation = {
  'Gemma-2-9B': {
    Vanilla: P`.822 .051  .489 .150  .278 .051  .578 .077   .100 .033  .022 .019  .111 .069  .067 .088`,
    'Boost only': P`.833 .019  .511 .107  .311 .039  .590 .069   .111 .019  .033 .019  .122 .058  .089 .069`,
    'Suppress only': P`.844 .019  .500 .096  .322 .051  .600 .058   .122 .019  .033 .019  .122 .058  .100 .077`,
    CPS: P`.867 .000  .533 .120  .356 .069  .622 .051   .133 .058  .056 .051  .144 .051  .122 .084`,
  },
  'Llama-3.1-8B': {
    Vanilla: P`.567 .020  .500 .133  .233 .058  .433 .120   .133 .000  .011 .019  .178 .102  .156 .077`,
    'Boost only': P`.589 .019  .522 .107  .244 .039  .467 .107   .144 .019  .022 .019  .211 .088  .178 .069`,
    'Suppress only': P`.600 .039  .511 .116  .267 .039  .500 .088   .156 .039  .022 .019  .233 .077  .200 .058`,
    CPS: P`.633 .058  .556 .096  .300 .033  .544 .088   .189 .051  .033 .039  .267 .069  .233 .058`,
  },
  'Mistral-7B-v0.1': {
    Vanilla: P`.722 .039  .478 .126  .311 .107  .544 .051   .156 .051  .011 .019  .100 .100  .067 .058`,
    'Boost only': P`.744 .019  .500 .107  .367 .077  .578 .039   .178 .039  .022 .019  .122 .088  .089 .051`,
    'Suppress only': P`.744 .019  .489 .107  .378 .069  .589 .019   .200 .039  .022 .019  .144 .077  .100 .051`,
    CPS: P`.767 .033  .511 .116  .422 .051  .622 .019   .233 .058  .044 .039  .178 .069  .133 .058`,
  },
};

// Table 4 — intervention strength δ, ConflictBank
export const DELTAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0];
export const delta = {
  cont: {
    'Gemma-2-9B': P`.822 .051  .844 .019  .856 .019  .867 .002  .856 .019  .822 .039  .789 .051`,
    'Llama-3.1-8B': P`.567 .010  .589 .019  .611 .039  .633 .058  .611 .058  .567 .069  .522 .077`,
    'Mistral-7B-v0.1': P`.722 .039  .744 .019  .756 .019  .778 .019  .767 .033  .733 .051  .700 .058`,
  },
  para: {
    'Gemma-2-9B': P`.100 .033  .111 .019  .122 .019  .133 .058  .122 .058  .111 .051  .089 .039`,
    'Llama-3.1-8B': P`.133 .009  .144 .039  .167 .039  .189 .051  .211 .069  .178 .077  .156 .058`,
    'Mistral-7B-v0.1': P`.156 .051  .178 .039  .211 .058  .233 .051  .256 .051  .222 .058  .189 .051`,
  },
};

// Table 5 — efficiency, Gemma-2-9B on NQ-Swap
export const efficiency = [
  { m: 'Vanilla', tokens: 250, cont: 0.233, para: 0.1, passes: 1, ms: 145.1, pct: 0 },
  { m: 'Prompt', tokens: 268, cont: 0.278, para: 0.122, passes: 1, ms: 146.7, pct: 1.1 },
  { m: 'PH3', tokens: 250, cont: 0.3, para: 0.133, passes: 1, ms: 149.8, pct: 3.2 },
  { m: 'JuICE', tokens: 250, cont: 0.333, para: 0.144, passes: 2, ms: 151.4, pct: 4.3 },
  { m: 'CPS', tokens: 250, cont: 0.389, para: 0.161, passes: 1, ms: 146.9, pct: 1.2 },
];

// Table 6 — generalisation (circuits found on NQ-Swap, applied unseen). [DynQA,WikiC] Cont. then Para.
export const GEN_DATASETS = ['DynamicQA', 'WikiContradict'];
export const generalisation = {
  'Gemma-2-9B': {
    Vanilla: P`.363 .103  .196 .045  .108 .045  .088 .045`,
    'Random-k': P`.392 .103  .255 .103  .117 .029  .108 .061`,
    Prompt: P`.392 .090  .235 .017  .128 .059  .118 .017`,
    PH3: P`.451 .095  .314 .088  .137 .034  .128 .017`,
    JuICE: P`.480 .074  .353 .059  .147 .061  .137 .029`,
    CPS: P`.520 .074  .392 .074  .167 .045  .167 .034`,
  },
  'Llama-3.1-8B': {
    Vanilla: P`.314 .029  .255 .123  .108 .061  .049 .061`,
    'Random-k': P`.353 .034  .294 .106  .118 .059  .069 .045`,
    Prompt: P`.333 .061  .275 .029  .128 .059  .088 .045`,
    PH3: P`.392 .017  .353 .103  .147 .017  .108 .051`,
    JuICE: P`.441 .017  .392 .119  .167 .017  .137 .059`,
    CPS: P`.500 .051  .422 .090  .196 .051  .167 .045`,
  },
  'Mistral-7B-v0.1': {
    Vanilla: P`.392 .089  .353 .045  .078 .034  .167 .089`,
    'Random-k': P`.431 .061  .373 .051  .088 .017  .196 .078`,
    Prompt: P`.471 .034  .392 .029  .098 .061  .235 .061`,
    PH3: P`.510 .074  .412 .045  .108 .051  .255 .095`,
    JuICE: P`.569 .074  .431 .045  .118 .051  .294 .111`,
    CPS: P`.628 .045  .461 .045  .128 .045  .343 .061`,
  },
};

// Table 2 — circuit masking, forced-choice preference rates (Gemma-2-9B).
// [Cont.: CB,CQA,NQ,PQA | Para.: CB,CQA,NQ,PQA]
export const masking = {
  Vanilla: [0.817, 0.49, 0.288, 0.589, 0.183, 0.51, 0.712, 0.411],
  'Random-k': [0.828, 0.48, 0.289, 0.578, 0.172, 0.52, 0.711, 0.422],
  'Cc masked': [0.801, 0.456, 0.244, 0.522, 0.199, 0.544, 0.756, 0.478],
  'Cp masked': [0.844, 0.49, 0.3, 0.6, 0.156, 0.51, 0.7, 0.4],
};

// Figure 1 running example & Figure 3 qualitative examples (Gemma-2-9B, NQ-Swap)
export const chicago = {
  q: 'how many episodes are in chicago fire season 4?',
  ctxPre: 'The fourth season of Chicago Fire … … The season contained ',
  counter: '775',
  org: '23',
  ctxPost: ' episodes.',
};

export const examples = [
  {
    q: 'Who came up with the theory of relativity?',
    pre: '…The theory of relativity usually encompasses two interrelated theories by ',
    key: 'Gina Tognoni',
    post: ': special relativity and general relativity…',
    gold: 'Gina Tognoni',
    vanilla: 'Albert Einstein',
    cps: 'Gina Tognoni',
  },
  {
    q: 'Who wrote the song to make you feel my love?',
    pre: '…"Make You Feel My Love" is a song written by ',
    key: 'Warren Zevon',
    post: ' from his album Time Out of Mind (1997)…',
    gold: 'Warren Zevon',
    vanilla: 'Bob Dylan',
    cps: 'Warren Zevon',
  },
  {
    q: 'Who voices Hiccup in How to Train Your Dragon 2?',
    pre: '',
    key: 'Jay Baruchel',
    post: ' as Hiccup Horrendous Haddock III, the son of the Viking chief…',
    gold: 'Jay Baruchel',
    vanilla: 'Jay Baruchel',
    cps: 'Jay Baruchel',
  },
];
