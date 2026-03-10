from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .config import (
    ABLATION_DELTA,
    CHG_BATCH_SIZE,
    CHG_FIT_SAMPLE_SIZE,
    CHG_NUM_MASKS,
    CHG_NUM_UPDATES,
    CHG_THRESHOLD,
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    DELTA_GRID,
    EXPERIMENTS_DIR,
    MASK_SCALE,
    MAX_INPUT_TOKENS,
    SPLIT_CHG_FIT_ROWS,
    SPLIT_SELECTION_DEV_ROWS,
    SPLIT_TEST_ROWS,
    TOPK_GRID,
    load_env,
    set_global_seed,
)
from .data import (
    DATASET_SPECS,
    SELECTION_DEV_PARTITION,
    TEST_PARTITION,
    build_paper_split,
    data_path_for,
    load_paper_split,
    load_split_pool,
    prepare_dataset,
    split_counts,
    split_path_for,
    write_json,
)
from .heads import derive_head_groups, run_chg_extraction
from .study import (
    load_scorer,
    run_ablation,
    run_delta_sweep,
    run_dev_selection,
    run_hyperparameter_study,
    run_main_table,
    run_soft_ablation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STAGES: tuple[str, ...] = ("main", "ablation", "hyperparameter", "soft", "delta")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="conflict.run")
    parser.add_argument("command", choices=("prepare", "split", "experiment"))
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=list(DATASET_SPECS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-rows", type=int, default=None)
    parser.add_argument("--chg-fit-rows", type=int, default=SPLIT_CHG_FIT_ROWS)
    parser.add_argument("--selection-dev-rows", type=int, default=SPLIT_SELECTION_DEV_ROWS)
    parser.add_argument("--test-rows", type=int, default=SPLIT_TEST_ROWS)
    parser.add_argument("--stages", nargs="+", default=["main", "ablation", "hyperparameter"], choices=STAGES)
    parser.add_argument("--output-root", default=str(EXPERIMENTS_DIR))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--chg-dir", default=None)
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--chg-fit-sample-size", type=int, default=CHG_FIT_SAMPLE_SIZE)
    parser.add_argument("--num-masks", type=int, default=CHG_NUM_MASKS)
    parser.add_argument("--num-updates", type=int, default=CHG_NUM_UPDATES)
    parser.add_argument("--batch-size", type=int, default=CHG_BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=CHG_THRESHOLD)
    parser.add_argument("--max-length", type=int, default=MAX_INPUT_TOKENS)
    parser.add_argument("--topk-grid", type=int, nargs="+", default=list(TOPK_GRID))
    parser.add_argument("--mask-scale", type=float, default=MASK_SCALE)
    parser.add_argument("--shared-k-star", action="store_true")
    parser.add_argument("--ablation-delta", type=float, default=ABLATION_DELTA)
    parser.add_argument("--delta-grid", type=float, nargs="+", default=list(DELTA_GRID))
    return parser.parse_args()


def _save(df: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n=== {title} ===")
    print(df.to_string(index=False))
    logger.info("Saved %s", path)


def _resolve_chg_dir(args: argparse.Namespace, run_dir: Path, data_path: Path, split_path: Path) -> Path:
    if args.chg_dir:
        return Path(args.chg_dir)
    chg_dir = run_dir / "chg"
    if (chg_dir / "manifest.json").exists() and not args.refit:
        logger.info("Reusing CHG run: %s", chg_dir)
        return chg_dir
    return run_chg_extraction(
        data_path=data_path,
        split_path=split_path,
        output_dir=chg_dir,
        model_name=args.model,
        seed=args.seed,
        fit_sample_size=args.chg_fit_sample_size,
        max_length=args.max_length,
        num_masks=args.num_masks,
        batch_size=args.batch_size,
        num_updates=args.num_updates,
        threshold=args.threshold,
        device=args.device,
    )


def run_experiment(args: argparse.Namespace) -> None:
    stages = set(args.stages)
    data_path = data_path_for(args.dataset)
    split_path = split_path_for(args.dataset)
    run_id = args.run_id or f"{args.dataset}_{args.model.replace('/', '_')}_seed{args.seed}"
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    chg_dir = _resolve_chg_dir(args, run_dir, data_path, split_path)
    membership, group_counts = derive_head_groups(chg_dir, chg_dir / "head_group_membership.csv")
    logger.info("Head groups: %s", group_counts)

    scorer = load_scorer(args.model, device=args.device, max_length=args.max_length)
    dev_df = load_split_pool(data_path, split_path, SELECTION_DEV_PARTITION)
    test_df = load_split_pool(data_path, split_path, TEST_PARTITION)
    logger.info("dev rows=%d test rows=%d", len(dev_df), len(test_df))

    selection_path = run_dir / "selection.json"
    k_c1_star, k_c2_star = None, None

    if "main" in stages:
        selection_df, k_c1_star, k_c2_star = run_dev_selection(
            scorer,
            dev_df,
            membership,
            topk_grid=tuple(args.topk_grid),
            mask_scale=args.mask_scale,
            model_name=args.model,
            separate_k_star=not args.shared_k_star,
        )
        predictions, main_summary = run_main_table(
            scorer,
            test_df,
            membership,
            k_c1_star=k_c1_star,
            k_c2_star=k_c2_star,
            mask_scale=args.mask_scale,
            model_name=args.model,
        )
        predictions.to_parquet(run_dir / "main_predictions.parquet", index=False)
        selection_df.to_csv(run_dir / "topk_selection.csv", index=False)
        write_json({"k_c1_star": k_c1_star, "k_c2_star": k_c2_star}, selection_path)
        _save(main_summary, run_dir / "main_table.csv", f"Main Table (k_c1*={k_c1_star} k_c2*={k_c2_star})")
    elif selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        k_c1_star, k_c2_star = int(selection["k_c1_star"]), int(selection["k_c2_star"])
        logger.info("Loaded k_c1*=%d k_c2*=%d", k_c1_star, k_c2_star)

    if "ablation" in stages:
        if k_c1_star is None:
            raise SystemExit("Ablation needs k*; run the main stage first.")
        _save(
            run_ablation(
                scorer,
                test_df,
                membership,
                k_c1_star=k_c1_star,
                k_c2_star=k_c2_star,
                mask_scale=args.mask_scale,
                seed=args.seed,
                model_name=args.model,
            ),
            run_dir / "ablation.csv",
            "Ablation Study",
        )

    if "hyperparameter" in stages:
        _save(
            run_hyperparameter_study(
                scorer,
                test_df,
                membership,
                topk_grid=tuple(args.topk_grid),
                mask_scale=args.mask_scale,
                model_name=args.model,
            ),
            run_dir / "hyperparameter.csv",
            "Hyperparameter Study (top-k)",
        )

    if "soft" in stages:
        if k_c1_star is None:
            raise SystemExit("Soft ablation needs k*; run the main stage first.")
        _save(
            run_soft_ablation(
                scorer,
                test_df,
                membership,
                k_c1_star=k_c1_star,
                k_c2_star=k_c2_star,
                delta=args.ablation_delta,
                model_name=args.model,
            ),
            run_dir / "soft_ablation.csv",
            f"Soft Ablation (delta={args.ablation_delta})",
        )

    if "delta" in stages:
        if k_c1_star is None:
            raise SystemExit("Delta sweep needs k*; run the main stage first.")
        _save(
            run_delta_sweep(
                scorer,
                test_df,
                membership,
                k_c1_star=k_c1_star,
                k_c2_star=k_c2_star,
                delta_grid=tuple(args.delta_grid),
                model_name=args.model,
            ),
            run_dir / "delta_sweep.csv",
            "Delta Sweep",
        )

    write_json(
        {
            "run_id": run_id,
            "dataset": args.dataset,
            "model": args.model,
            "seed": args.seed,
            "stages": sorted(stages),
            "k_c1_star": k_c1_star,
            "k_c2_star": k_c2_star,
            "topk_grid": list(args.topk_grid),
            "mask_scale": args.mask_scale,
            "chg_dir": str(chg_dir),
            "split_path": str(split_path),
            "split_counts": split_counts(load_paper_split(split_path)),
            "head_groups": group_counts,
        },
        run_dir / "run_manifest.json",
    )
    logger.info("Done: %s", run_dir)


def main() -> None:
    args = _parse_args()
    load_env()
    set_global_seed(args.seed)

    if args.command == "prepare":
        manifest = prepare_dataset(args.dataset, num_rows=args.num_rows)
        print(f"{manifest['dataset_name']}: rows={manifest['row_count']} -> {manifest['output_file']}")
    elif args.command == "split":
        manifest = build_paper_split(
            data_path=data_path_for(args.dataset),
            split_path=split_path_for(args.dataset),
            seed=args.seed,
            chg_fit_rows=args.chg_fit_rows,
            selection_dev_rows=args.selection_dev_rows,
            test_rows=args.test_rows,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
