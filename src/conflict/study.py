from __future__ import annotations

import logging
import random
import re
from collections.abc import Sequence
from math import comb, sqrt
from typing import Any

import pandas as pd
import torch
from inteprelens import CHG, LensAPI

from .config import (
    ABLATION_DELTA,
    DELTA_GRID,
    MASK_SCALE,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    TOPK_GRID,
)
from .data import SCENARIO_MAP, SCENARIO_TARGET, build_row_prompt, normalize_answer
from .heads import (
    build_ablation_mask,
    build_boost_suppress_mask,
    build_family_mask,
    build_identity_mask,
    build_soft_gate_mask,
    family_heads,
)

logger = logging.getLogger(__name__)

CONFLICT_SCENARIOS: tuple[str, ...] = ("conflict1", "conflict2")
SCENARIO_MASK_FAMILY: dict[str, str] = {"conflict1": "parametric", "conflict2": "context"}
SOFT_ABLATION_POLICY: dict[str, tuple[str, str]] = {
    "conflict1": ("context", "parametric"),
    "conflict2": ("parametric", "context"),
}
ABLATION_CONDITIONS: tuple[str, ...] = (
    "all_heads",
    "random_k_heads",
    "context_heads_only",
    "parametric_heads_only",
)
SOFT_ABLATION_CONDITIONS: tuple[str, ...] = (
    "vanilla",
    "boost_only",
    "suppress_only",
    "boost_suppress",
)

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_RE = re.compile(r"[^\w\s]")


class Scorer:
    def __init__(self, analyzer: Any, max_length: int | None = MAX_INPUT_TOKENS):
        self.analyzer = analyzer
        self.chg = CHG(analyzer.model)
        self.tokenizer = analyzer.tokenizer
        self.device = str(next(analyzer.model.parameters()).device)
        self.max_length = max_length
        self.num_layers = self.chg.num_layers
        self.num_heads = self.chg.num_heads
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate_answer(self, prompt: str, mask: torch.Tensor) -> str:
        encoding = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoding.input_ids.to(self.device)
        attention_mask = encoding.attention_mask.to(self.device)
        if self.max_length is not None and input_ids.shape[1] > self.max_length:
            tail = min(16, self.max_length // 8)
            head = self.max_length - tail
            input_ids = torch.cat([input_ids[:, :head], input_ids[:, -tail:]], dim=1)
            attention_mask = torch.cat([attention_mask[:, :head], attention_mask[:, -tail:]], dim=1)

        batched_mask = mask.unsqueeze(0).to(self.device)
        newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        stop_ids = list({self.tokenizer.eos_token_id} | set(newline_ids))

        decoded = self._generate(input_ids, attention_mask, batched_mask, stop_ids).strip()
        if decoded:
            return decoded
        decoded = self._generate(
            input_ids, attention_mask, batched_mask, self.tokenizer.eos_token_id
        )
        for line in decoded.split("\n"):
            if line.strip():
                return line.strip()
        return ""

    def _generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batched_mask: torch.Tensor,
        eos_token_id: Any,
    ) -> str:
        with self.chg.register_mask_hooks(batched_mask), torch.no_grad():
            output_ids = self.analyzer.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )
        return self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)


def load_scorer(
    model_name: str,
    device: str | None = None,
    max_length: int | None = MAX_INPUT_TOKENS,
) -> Scorer:
    logger.info("Loading model: %s", model_name)
    analyzer = LensAPI.from_pretrained(
        model_name, device=device, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    return Scorer(analyzer, max_length=max_length)


def _normalize_match_text(value: object) -> str:
    text = _PUNCT_RE.sub(" ", normalize_answer(value).lower())
    return " ".join(_ARTICLE_RE.sub(" ", text).split())


def is_exact_match(predicted: object, gold: object) -> bool:
    predicted_norm = _normalize_match_text(predicted)
    if isinstance(gold, (list, tuple)) or (hasattr(gold, "__iter__") and not isinstance(gold, str)):
        return any(predicted_norm == _normalize_match_text(alias) for alias in gold)
    return predicted_norm == _normalize_match_text(gold)


def gold_answers_for_row(row: object, target_column: str) -> object:
    gold = str(getattr(row, target_column))
    if target_column != "org_answer":
        return gold
    aliases = getattr(row, "org_aliases", None)
    if aliases is None:
        aliases = getattr(row, "ground_truth", None)
    if aliases is None or isinstance(aliases, float):
        return gold
    return list(aliases)


def wilson_interval(num_successes: int, num_trials: int, z: float = 1.96) -> tuple[float, float]:
    if num_trials <= 0:
        return float("nan"), float("nan")
    rate = num_successes / num_trials
    z_sq = z**2
    denominator = 1.0 + z_sq / num_trials
    center = (rate + z_sq / (2.0 * num_trials)) / denominator
    margin = z * sqrt((rate * (1.0 - rate) + z_sq / (4.0 * num_trials)) / num_trials) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def mcnemar_pvalue(baseline: Sequence[bool], condition: Sequence[bool]) -> float:
    baseline_only = sum(1 for b, c in zip(baseline, condition, strict=True) if b and not c)
    condition_only = sum(1 for b, c in zip(baseline, condition, strict=True) if c and not b)
    discordant = baseline_only + condition_only
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, i) for i in range(min(baseline_only, condition_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def score_subset(
    scorer: Scorer,
    subset: pd.DataFrame,
    scenario_name: str,
    mask: torch.Tensor,
    condition: str,
    masked_head_count: int,
    model_name: str,
) -> list[dict[str, object]]:
    scenario = SCENARIO_MAP[scenario_name]
    target_column = SCENARIO_TARGET[scenario_name]
    records: list[dict[str, object]] = []
    for row in subset.itertuples(index=False):
        prompt = build_row_prompt(row, getattr(row, scenario.context_column), model_name)
        predicted = scorer.generate_answer(prompt, mask)
        records.append(
            {
                "source_row_index": int(row.source_row_index),
                "scenario": scenario_name,
                "condition": condition,
                "predicted_answer": predicted,
                "gold_answer": str(getattr(row, target_column)),
                "org_answer": str(row.org_answer),
                "sub_answer": str(row.sub_answer),
                "correct": is_exact_match(predicted, gold_answers_for_row(row, target_column)),
                "masked_head_count": masked_head_count,
                "model_name": model_name,
            }
        )
    accuracy = sum(bool(record["correct"]) for record in records) / max(len(records), 1)
    logger.info("%s scenario=%s acc=%.4f", condition, scenario_name, accuracy)
    return records


def summarize(
    predictions: pd.DataFrame,
    conditions: tuple[str, ...] | None = None,
    baseline_condition: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario, scenario_group in predictions.groupby("scenario", sort=False):
        baseline = (
            scenario_group[scenario_group["condition"] == baseline_condition]
            .sort_values("source_row_index")
            .reset_index(drop=True)
            if baseline_condition
            else pd.DataFrame()
        )
        order = conditions or tuple(dict.fromkeys(scenario_group["condition"]))

        for condition in order:
            group = (
                scenario_group[scenario_group["condition"] == condition]
                .sort_values("source_row_index")
                .reset_index(drop=True)
            )
            if group.empty:
                continue
            num_rows = len(group)
            num_correct = int(group["correct"].sum())
            ci_low, ci_high = wilson_interval(num_correct, num_rows)
            record = {
                "scenario": scenario,
                "condition": condition,
                "num_rows": num_rows,
                "num_correct": num_correct,
                "accuracy": num_correct / num_rows,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "masked_head_count": int(group["masked_head_count"].iloc[0]),
            }
            if not baseline.empty:
                paired = baseline.merge(
                    group[["source_row_index", "correct"]],
                    on="source_row_index",
                    suffixes=("_base", "_cond"),
                )
                pvalue = (
                    1.0
                    if condition == baseline_condition
                    else mcnemar_pvalue(
                        paired["correct_base"].astype(bool).tolist(),
                        paired["correct_cond"].astype(bool).tolist(),
                    )
                )
                baseline_accuracy = float(baseline["correct"].mean())
                record |= {
                    "baseline_accuracy": baseline_accuracy,
                    "delta_accuracy": record["accuracy"] - baseline_accuracy,
                    "mcnemar_pvalue": pvalue,
                    "significant": bool(pvalue < 0.05),
                }
            rows.append(record)
    return pd.DataFrame.from_records(rows)


def run_dev_selection(
    scorer: Scorer,
    dev_df: pd.DataFrame,
    membership: pd.DataFrame,
    topk_grid: tuple[int, ...] = TOPK_GRID,
    mask_scale: float = MASK_SCALE,
    model_name: str = "",
    separate_k_star: bool = False,
) -> tuple[pd.DataFrame, int, int]:
    rows: list[dict[str, object]] = []
    for k in topk_grid:
        for scenario_name, family in SCENARIO_MASK_FAMILY.items():
            spec = build_family_mask(
                membership, family, k, scorer.num_layers, scorer.num_heads, mask_scale
            )
            if spec is None:
                logger.warning("No eligible heads for family=%s k=%d", family, k)
                continue
            records = score_subset(
                scorer, dev_df, scenario_name, spec.mask, f"topk_{k}", spec.masked_head_count, model_name
            )
            num_correct = sum(bool(record["correct"]) for record in records)
            rows.append(
                {
                    "k": k,
                    "scenario": scenario_name,
                    "family_masked": family,
                    "masked_head_count": spec.masked_head_count,
                    "num_rows": len(records),
                    "num_correct": num_correct,
                    "accuracy": num_correct / max(len(records), 1),
                }
            )

    selection_df = pd.DataFrame.from_records(rows)
    if selection_df.empty:
        logger.warning("Dev selection produced no results; defaulting to k=%d", topk_grid[0])
        return selection_df, topk_grid[0], topk_grid[0]

    if separate_k_star:
        k_c1_star = _best_k(selection_df, "conflict1", topk_grid[0])
        k_c2_star = _best_k(selection_df, "conflict2", topk_grid[0])
    else:
        k_c1_star = k_c2_star = int(selection_df.groupby("k")["accuracy"].mean().idxmax())
    logger.info("Dev selection k_c1*=%d k_c2*=%d", k_c1_star, k_c2_star)
    return selection_df, k_c1_star, k_c2_star


def _best_k(selection_df: pd.DataFrame, scenario_name: str, fallback: int) -> int:
    rows = selection_df[selection_df["scenario"] == scenario_name]
    if rows.empty:
        return fallback
    return int(rows.loc[rows["accuracy"].idxmax(), "k"])


def run_main_table(
    scorer: Scorer,
    test_df: pd.DataFrame,
    membership: pd.DataFrame,
    k_c1_star: int,
    k_c2_star: int,
    mask_scale: float = MASK_SCALE,
    model_name: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    k_by_scenario = {"conflict1": k_c1_star, "conflict2": k_c2_star}
    records: list[dict[str, object]] = []
    for scenario_name, family in SCENARIO_MASK_FAMILY.items():
        k = k_by_scenario[scenario_name]
        spec = build_family_mask(
            membership, family, k, scorer.num_layers, scorer.num_heads, mask_scale
        )
        mask = spec.mask if spec else build_identity_mask(scorer.num_layers, scorer.num_heads)
        head_count = spec.masked_head_count if spec else 0
        records.extend(
            score_subset(
                scorer, test_df, scenario_name, mask, f"{family}_masked_top{k}", head_count, model_name
            )
        )
    predictions = pd.DataFrame.from_records(records)
    return predictions, summarize(predictions)


def run_ablation(
    scorer: Scorer,
    test_df: pd.DataFrame,
    membership: pd.DataFrame,
    k_c1_star: int,
    k_c2_star: int,
    mask_scale: float = MASK_SCALE,
    seed: int = 0,
    model_name: str = "",
) -> pd.DataFrame:
    num_layers, num_heads = scorer.num_layers, scorer.num_heads
    parametric = build_family_mask(membership, "parametric", k_c1_star, num_layers, num_heads, mask_scale)
    context = build_family_mask(membership, "context", k_c2_star, num_layers, num_heads, mask_scale)

    family_heads_used = {head for spec in (parametric, context) if spec for head in spec.masked_heads}
    candidates = [
        (layer, head)
        for layer in range(num_layers)
        for head in range(num_heads)
        if (layer, head) not in family_heads_used
    ]
    random_heads = random.Random(seed).sample(candidates, min(k_c1_star, len(candidates)))
    identity = build_identity_mask(num_layers, num_heads)

    conditions = [
        ("all_heads", identity, 0),
        (
            "random_k_heads",
            build_ablation_mask(
                num_layers=num_layers,
                num_heads=num_heads,
                masked_heads=random_heads,
                masked_value=mask_scale,
            ),
            len(random_heads),
        ),
        (
            "context_heads_only",
            context.mask if context else identity,
            context.masked_head_count if context else 0,
        ),
        (
            "parametric_heads_only",
            parametric.mask if parametric else identity,
            parametric.masked_head_count if parametric else 0,
        ),
    ]

    records: list[dict[str, object]] = []
    for scenario_name in CONFLICT_SCENARIOS:
        for condition, mask, head_count in conditions:
            records.extend(
                score_subset(scorer, test_df, scenario_name, mask, condition, head_count, model_name)
            )
    return summarize(pd.DataFrame.from_records(records), ABLATION_CONDITIONS, "all_heads")


def run_hyperparameter_study(
    scorer: Scorer,
    test_df: pd.DataFrame,
    membership: pd.DataFrame,
    topk_grid: tuple[int, ...] = TOPK_GRID,
    mask_scale: float = MASK_SCALE,
    model_name: str = "",
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for k in topk_grid:
        for scenario_name, family in SCENARIO_MASK_FAMILY.items():
            spec = build_family_mask(
                membership, family, k, scorer.num_layers, scorer.num_heads, mask_scale
            )
            if spec is None:
                logger.warning("No eligible heads for family=%s k=%d", family, k)
                continue
            records.extend(
                score_subset(
                    scorer, test_df, scenario_name, spec.mask, f"topk_{k}", spec.masked_head_count, model_name
                )
            )
    summary = summarize(pd.DataFrame.from_records(records))
    summary.insert(0, "k", summary["condition"].str.removeprefix("topk_").astype(int))
    summary.insert(2, "family_masked", summary["scenario"].map(SCENARIO_MASK_FAMILY))
    return summary.sort_values(["k", "scenario"]).reset_index(drop=True)


def run_soft_ablation(
    scorer: Scorer,
    test_df: pd.DataFrame,
    membership: pd.DataFrame,
    k_c1_star: int,
    k_c2_star: int,
    delta: float = ABLATION_DELTA,
    model_name: str = "",
) -> pd.DataFrame:
    num_layers, num_heads = scorer.num_layers, scorer.num_heads
    identity = build_identity_mask(num_layers, num_heads)
    k_by_family = {"parametric": int(k_c1_star), "context": int(k_c2_star)}
    records: list[dict[str, object]] = []

    for scenario_name, (boost_family, suppress_family) in SOFT_ABLATION_POLICY.items():
        boost_k, suppress_k = k_by_family[boost_family], k_by_family[suppress_family]
        boost_heads = len(family_heads(membership, boost_family)[:boost_k])
        suppress_heads = len(family_heads(membership, suppress_family)[:suppress_k])
        conditions = [
            ("vanilla", identity, 0),
            (
                "boost_only",
                build_soft_gate_mask(
                    membership, boost_family, delta, "boost", num_layers, num_heads, k=boost_k
                ),
                boost_heads,
            ),
            (
                "suppress_only",
                build_soft_gate_mask(
                    membership,
                    suppress_family,
                    delta,
                    "suppress",
                    num_layers,
                    num_heads,
                    k=suppress_k,
                ),
                suppress_heads,
            ),
            (
                "boost_suppress",
                build_boost_suppress_mask(
                    membership,
                    boost_family,
                    suppress_family,
                    delta,
                    num_layers,
                    num_heads,
                    boost_k=boost_k,
                    suppress_k=suppress_k,
                ),
                boost_heads + suppress_heads,
            ),
        ]
        for condition, mask, head_count in conditions:
            records.extend(
                score_subset(scorer, test_df, scenario_name, mask, condition, head_count, model_name)
            )
    return summarize(pd.DataFrame.from_records(records), SOFT_ABLATION_CONDITIONS, "vanilla")


def run_delta_sweep(
    scorer: Scorer,
    test_df: pd.DataFrame,
    membership: pd.DataFrame,
    k_c1_star: int,
    k_c2_star: int,
    delta_grid: tuple[float, ...] = DELTA_GRID,
    model_name: str = "",
) -> pd.DataFrame:
    num_layers, num_heads = scorer.num_layers, scorer.num_heads
    k_parametric, k_context = int(k_c1_star), int(k_c2_star)
    records: list[dict[str, object]] = []
    for delta in delta_grid:
        context_favoring = build_boost_suppress_mask(
            membership,
            "context",
            "parametric",
            delta,
            num_layers,
            num_heads,
            boost_k=k_context,
            suppress_k=k_parametric,
        )
        parametric_favoring = build_boost_suppress_mask(
            membership,
            "parametric",
            "context",
            delta,
            num_layers,
            num_heads,
            boost_k=k_parametric,
            suppress_k=k_context,
        )
        scenario_masks = {
            "conflict1": context_favoring,
            "conflict2": parametric_favoring,
            "clean": context_favoring,
        }
        for scenario_name, mask in scenario_masks.items():
            records.extend(
                score_subset(
                    scorer, test_df, scenario_name, mask, f"delta_{delta:.2f}", 0, model_name
                )
            )
    summary = summarize(pd.DataFrame.from_records(records))
    summary.insert(0, "delta", summary["condition"].str.removeprefix("delta_").astype(float))
    return summary.sort_values(["delta", "scenario"]).reset_index(drop=True)
