from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from inteprelens import LensAPI
from tqdm.auto import tqdm

from .config import (
    CHG_BATCH_SIZE,
    CHG_FIT_SAMPLE_SIZE,
    CHG_NUM_MASKS,
    CHG_NUM_UPDATES,
    CHG_THRESHOLD,
    MAX_INPUT_TOKENS,
    set_global_seed,
)
from .data import (
    CHG_FIT_PARTITION,
    SCENARIOS,
    build_scenario_examples,
    load_paper_split,
    load_split_pool,
    split_counts,
    write_json,
)

logger = logging.getLogger(__name__)

SCENARIO_ORDER: tuple[str, ...] = ("clean", "conflict1", "conflict2")

HEAD_GROUP_ORDER: tuple[str, ...] = (
    "general_reasoning",
    "conflict_sensitive",
    "context_hard",
    "context_soft",
    "parametric_hard",
    "parametric_soft",
    "unclassified",
)

FAMILY_GROUPS: dict[str, tuple[str, ...]] = {
    "context": ("context_hard", "context_soft"),
    "parametric": ("parametric_hard", "parametric_soft"),
}
FAMILY_SCORE_COLUMN: dict[str, str] = {
    "context": "conflict1_necessary_mask",
    "parametric": "conflict2_necessary_mask",
}


def run_chg_extraction(
    *,
    data_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    model_name: str,
    seed: int = 0,
    fit_sample_size: int = CHG_FIT_SAMPLE_SIZE,
    max_length: int | None = MAX_INPUT_TOKENS,
    num_masks: int = CHG_NUM_MASKS,
    batch_size: int = CHG_BATCH_SIZE,
    num_updates: int = CHG_NUM_UPDATES,
    threshold: float = CHG_THRESHOLD,
    device: str | None = None,
) -> Path:
    set_global_seed(seed)
    df = load_split_pool(data_path, split_path, CHG_FIT_PARTITION, limit=fit_sample_size)
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model: %s", model_name)
    api = LensAPI.from_pretrained(
        model_name, device=device, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )

    training_params = {
        "max_length": max_length,
        "num_masks": num_masks,
        "batch_size": batch_size,
        "num_updates": num_updates,
        "threshold": threshold,
    }
    logger.info("CHG fit model=%s seed=%d rows=%d params=%s", model_name, seed, len(df), training_params)

    scenario_files: dict[str, str] = {}
    for scenario in tqdm(SCENARIOS, desc="CHG", unit="scenario"):
        prompts, targets = build_scenario_examples(df, scenario, model_name=model_name)
        results = api.fit(
            texts=prompts,
            targets=targets,
            max_length=max_length,
            num_masks=num_masks,
            batch_size=batch_size,
            num_updates=num_updates,
            num_reg_updates=num_updates,
        )
        taxonomy = results.head_taxonomy(threshold=threshold)
        artifact = {
            "scenario": scenario.name,
            "chg_results": {
                "masks": results.masks,
                "metrics": results.metrics,
                "model_config": results.model_config,
            },
            "head_taxonomy": [
                {
                    "layer": int(row.layer),
                    "head": int(row.head),
                    "necessary_mask": float(row.necessary_mask),
                    "sufficient_mask": float(row.sufficient_mask),
                    "taxonomy": str(row.taxonomy),
                }
                for row in taxonomy.itertuples(index=False)
            ],
        }
        artifact_path = run_dir / f"head_roles_{scenario.name}.pt"
        torch.save(artifact, artifact_path)
        scenario_files[scenario.name] = artifact_path.name
        logger.info("%s taxonomy: %s", scenario.name, taxonomy["taxonomy"].value_counts().to_dict())

    write_json(
        {
            "dataset_source": str(data_path),
            "split_path": str(split_path),
            "split_counts": split_counts(load_paper_split(split_path)),
            "row_count": int(len(df)),
            "selected_row_indices": df["source_row_index"].astype(int).tolist(),
            "model_name": model_name,
            "seed": int(seed),
            "fit_sample_size": int(fit_sample_size),
            "training_params": training_params,
            "scenario_files": scenario_files,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        },
        run_dir / "manifest.json",
    )
    logger.info("Saved CHG run to %s", run_dir)
    return run_dir


def _head_group(row: pd.Series) -> str:
    clean = row["clean_taxonomy"]
    conflict1 = row["conflict1_taxonomy"]
    conflict2 = row["conflict2_taxonomy"]
    if clean == conflict1 == conflict2 == "facilitating":
        return "general_reasoning"
    if clean in {"irrelevant", "interfering"} and conflict1 == conflict2 == "facilitating":
        return "conflict_sensitive"
    if conflict1 == "facilitating" and conflict2 == "interfering":
        return "context_hard"
    if conflict1 == "facilitating" and conflict2 == "irrelevant":
        return "context_soft"
    if conflict1 == "interfering" and conflict2 == "facilitating":
        return "parametric_hard"
    if conflict1 == "irrelevant" and conflict2 == "facilitating":
        return "parametric_soft"
    return "unclassified"


def derive_head_groups(
    run_dir: str | Path,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    path = Path(run_dir)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"CHG manifest not found: {manifest_path}")
    scenario_files = json.loads(manifest_path.read_text(encoding="utf-8"))["scenario_files"]

    merged: pd.DataFrame | None = None
    for scenario_name in SCENARIO_ORDER:
        artifact = torch.load(path / scenario_files[scenario_name], weights_only=False)
        frame = pd.DataFrame.from_records(artifact["head_taxonomy"]).rename(
            columns={
                "necessary_mask": f"{scenario_name}_necessary_mask",
                "sufficient_mask": f"{scenario_name}_sufficient_mask",
                "taxonomy": f"{scenario_name}_taxonomy",
            }
        )
        merged = frame if merged is None else merged.merge(frame, on=["layer", "head"], how="inner")

    membership = merged.sort_values(["layer", "head"]).reset_index(drop=True)
    membership["head_group"] = membership.apply(_head_group, axis=1)

    counts = membership["head_group"].value_counts().to_dict()
    summary = {group: int(counts.get(group, 0)) for group in HEAD_GROUP_ORDER}
    summary["total_heads"] = int(len(membership))

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        membership.to_csv(output, index=False)
    return membership, summary


@dataclass(frozen=True)
class FamilyMask:
    family: str
    k: int
    masked_heads: tuple[tuple[int, int], ...]
    mask: torch.Tensor

    @property
    def masked_head_count(self) -> int:
        return len(self.masked_heads)


def build_identity_mask(num_layers: int, num_heads: int) -> torch.Tensor:
    return torch.ones(num_layers, num_heads, dtype=torch.bfloat16)


def build_ablation_mask(
    *,
    num_layers: int,
    num_heads: int,
    masked_heads: Sequence[tuple[int, int]],
    masked_value: float = 0.0,
) -> torch.Tensor:
    mask = build_identity_mask(num_layers, num_heads)
    for layer, head in masked_heads:
        mask[layer, head] = float(masked_value)
    return mask


def family_heads(membership: pd.DataFrame, family: str) -> list[tuple[int, int]]:
    selected = membership[membership["head_group"].isin(FAMILY_GROUPS[family])]
    ranked = selected.sort_values(
        [FAMILY_SCORE_COLUMN[family], "layer", "head"], ascending=[False, True, True]
    )
    return [(int(row.layer), int(row.head)) for row in ranked.itertuples(index=False)]


def build_family_mask(
    membership: pd.DataFrame,
    family: str,
    k: int,
    num_layers: int,
    num_heads: int,
    masked_value: float = 0.0,
) -> FamilyMask | None:
    masked_heads = tuple(family_heads(membership, family)[: int(k)])
    if not masked_heads:
        return None
    return FamilyMask(
        family=family,
        k=int(k),
        masked_heads=masked_heads,
        mask=build_ablation_mask(
            num_layers=num_layers,
            num_heads=num_heads,
            masked_heads=masked_heads,
            masked_value=masked_value,
        ),
    )


def build_soft_gate_mask(
    membership: pd.DataFrame,
    family: str,
    delta: float,
    mode: str,
    num_layers: int,
    num_heads: int,
    base_mask: torch.Tensor | None = None,
    k: int | None = None,
) -> torch.Tensor:
    factor = 1.0 / (1.0 + delta) if mode == "suppress" else 1.0 + delta
    mask = build_identity_mask(num_layers, num_heads) if base_mask is None else base_mask.clone()
    selected = family_heads(membership, family)
    for layer, head in selected if k is None else selected[: int(k)]:
        mask[layer, head] = float(factor)
    return mask


def build_boost_suppress_mask(
    membership: pd.DataFrame,
    boost_family: str,
    suppress_family: str,
    delta: float,
    num_layers: int,
    num_heads: int,
    boost_k: int | None = None,
    suppress_k: int | None = None,
) -> torch.Tensor:
    boosted = build_soft_gate_mask(
        membership, boost_family, delta, "boost", num_layers, num_heads, k=boost_k
    )
    return build_soft_gate_mask(
        membership,
        suppress_family,
        delta,
        "suppress",
        num_layers,
        num_heads,
        base_mask=boosted,
        k=suppress_k,
    )
