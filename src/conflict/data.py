from __future__ import annotations

import json
import logging
import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import (
    DATA_DIR,
    SPLIT_CHG_FIT_ROWS,
    SPLIT_CONTROLLER_TRAIN_ROWS,
    SPLIT_SELECTION_DEV_ROWS,
    SPLIT_TEST_ROWS,
    SPLITS_DIR,
)

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = "Question: {question}\nContext: {context}\nAnswer:"

BASELINE_PROMPT_TEMPLATE = (
    "Read the question and context.\n"
    "Respond with exactly one short final answer.\n"
    "Do not include reasoning, explanation, HTML, XML, markdown, tables, or multiple options.\n\n"
    "Question: {question}\n"
    "Context: {context}\n"
    "Final answer:"
)

MCQ_PROMPT_TEMPLATE = (
    "Read the question and context.\n"
    "The answer must be exactly one of these options: {options_inline}.\n"
    "Respond with exactly one option text from the list above, word-for-word.\n"
    "Do not include reasoning or explanation.\n\n"
    "Question: {question}\n"
    "Context: {context}\n"
    "Final answer:"
)

SWAP_COLUMNS: tuple[str, ...] = (
    "question",
    "org_context",
    "org_answer",
    "sub_context",
    "sub_answer",
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    name: str
    context_column: str
    answer_column: str


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec("clean", "org_context", "org_answer"),
    ScenarioSpec("conflict1", "sub_context", "sub_answer"),
    ScenarioSpec("conflict2", "sub_context", "org_answer"),
)
SCENARIO_MAP = {scenario.name: scenario for scenario in SCENARIOS}
SCENARIO_TARGET = {scenario.name: scenario.answer_column for scenario in SCENARIOS}


def normalize_answer(value: object) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def build_prompt(question: object, context: object, model_name: str = "") -> str:
    template = PROMPT_TEMPLATE if "llama" in model_name.lower() else BASELINE_PROMPT_TEMPLATE
    return template.format(
        question=str(question),
        context=" ".join(_HTML_TAG_RE.sub(" ", str(context)).split()),
    )


def build_row_prompt(row: object, context: object, model_name: str = "") -> str:
    options = getattr(row, "options", None)
    if options is None:
        return build_prompt(row.question, context, model_name)
    return MCQ_PROMPT_TEMPLATE.format(
        options_inline=" / ".join(str(option) for option in options),
        question=str(row.question),
        context=" ".join(_HTML_TAG_RE.sub(" ", str(context)).split()),
    )


def build_scenario_examples(
    df: pd.DataFrame,
    scenario: ScenarioSpec,
    model_name: str = "",
) -> tuple[list[str], list[str]]:
    rows = list(df.itertuples(index=False))
    prompts = [build_row_prompt(row, getattr(row, scenario.context_column), model_name) for row in rows]
    targets = [normalize_answer(getattr(row, scenario.answer_column)) for row in rows]
    return prompts, targets


def prepare_swap_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in SWAP_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    prepared = df.loc[:, list(SWAP_COLUMNS)].copy()
    for column in ("question", "org_context", "sub_context"):
        prepared[column] = prepared[column].astype(str)
    for column in ("org_answer", "sub_answer"):
        prepared[column] = prepared[column].map(normalize_answer)
    return prepared


ANSWER_LABELS: tuple[str, ...] = ("A", "B", "C", "D")

NQSWAP_SOURCE = "hf://datasets/pminervini/NQ-Swap/data/dev-00000-of-00001.parquet"

_CLAIM_VERB_MARKERS = (
    " is a ", " is an ", " is the ", " is ",
    " was a ", " was an ", " was the ", " was ",
    " are a ", " are ", " were a ", " were ",
    " has a ", " has an ", " has ",
    " born in ", " from ",
)
_PREDICATE_OBJECT_VERBS = (" plays ",)
_POSSESSIVE_ROLE_RE = re.compile(
    r"'s\s+(occupation|genre|capital|country|language|nationality|"
    r"religion|sport|type|style|form|class|kind|field|area|domain)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    source: str
    load: Callable[[], pd.DataFrame]
    normalize: Callable[[pd.DataFrame], pd.DataFrame]
    file_name: str = "prepared.parquet"


def _load_hf(repo_id: str, split: str, configs: Sequence[str | None] = (None,)) -> pd.DataFrame:
    from datasets import load_dataset

    error: Exception | None = None
    for config in configs:
        try:
            return load_dataset(repo_id, config, split=split).to_pandas()
        except Exception as exc:
            error = exc
    raise error


def _load_hf_json(repo_id: str, filename: str) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    return pd.read_json(path, lines=True)


def _string_list(value: object) -> list[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value is None:
        return []
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    return [str(value).strip()]


def _option_text(label: object, options: Sequence[str]) -> str:
    normalized = str(label).strip().upper()
    if normalized not in ANSWER_LABELS:
        raise ValueError(f"Unsupported option label: {label!r}")
    return str(options[ANSWER_LABELS.index(normalized)]).strip()


def _entity_from_claim(sentence: str) -> str:
    text = sentence.strip().rstrip(".")
    for marker in _CLAIM_VERB_MARKERS:
        index = text.lower().find(marker)
        if index != -1:
            return text[index + len(marker):].strip()
    parts = text.split()
    return parts[-1] if parts else sentence


def _counter_entity(sentence: str) -> str:
    text = sentence.strip().rstrip(".")
    lowered = text.lower()
    use_predicate = (
        re.match(r"^(the|a|an)\s", lowered)
        or bool(_POSSESSIVE_ROLE_RE.search(lowered))
        or any(verb in lowered for verb in _PREDICATE_OBJECT_VERBS)
    )
    if use_predicate:
        for marker in _CLAIM_VERB_MARKERS + _PREDICATE_OBJECT_VERBS:
            index = lowered.find(marker)
            if index != -1:
                return text[index + len(marker):].strip()
        parts = text.split()
        return parts[-1] if parts else text
    for marker in _CLAIM_VERB_MARKERS:
        index = lowered.find(marker)
        if index > 0:
            return text[:index].strip()
    return text


def _normalize_nqswap(raw: pd.DataFrame) -> pd.DataFrame:
    return prepare_swap_frame(raw)


def _normalize_popqa_swap(raw: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_swap_frame(raw)
    if "org_aliases" in raw.columns:
        prepared["org_aliases"] = raw["org_aliases"].values
    else:
        prepared["org_aliases"] = [[answer] for answer in prepared["org_answer"]]
    return prepared


def _normalize_conflictqa(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in raw.itertuples(index=False):
        memory_answer = normalize_answer(row.memory_answer).strip()
        counter_answer = normalize_answer(row.counter_answer).strip()
        rows.append(
            {
                "question": str(row.question),
                "org_context": str(row.counter_memory_aligned_evidence),
                "org_answer": _counter_entity(counter_answer),
                "sub_context": str(row.parametric_memory_aligned_evidence),
                "sub_answer": _entity_from_claim(memory_answer),
                "ground_truth": _string_list(row.ground_truth),
            }
        )
    return pd.DataFrame.from_records(rows, columns=[*SWAP_COLUMNS, "ground_truth"])


def _normalize_conflictbank(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in raw.itertuples(index=False):
        options = _string_list(row.options)
        if len(options) != len(ANSWER_LABELS):
            continue
        try:
            correct_text = _option_text(row.correct_option, options)
            replace_text = _option_text(row.replace_option, options)
        except ValueError:
            continue
        rows.append(
            {
                "question": str(row.question),
                "org_context": str(row.default_evidence),
                "org_answer": correct_text,
                "sub_context": str(row.misinformation_conflict_evidence_evidence),
                "sub_answer": replace_text,
                "options": options,
            }
        )
    return pd.DataFrame.from_records(rows, columns=[*SWAP_COLUMNS, "options"])


POPQA_PROPS = frozenset(
    ["director", "author", "composer", "occupation", "place of birth", "genre", "screenwriter"]
)
_WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_WIKI_HEADERS = {"User-Agent": "conflict-research-popqa-swap/1.0"}


def _wikipedia_extract(title: str, cache: dict[str, str | None]) -> str | None:
    import requests

    if title in cache:
        return cache[title]
    try:
        response = requests.get(
            _WIKI_API.format(title=requests.utils.quote(title, safe="")),
            headers=_WIKI_HEADERS,
            timeout=10,
        )
        time.sleep(0.05)
        data = response.json() if response.status_code == 200 else {}
        if data.get("type") == "disambiguation":
            data = {}
        paragraph = data.get("extract", "").strip().split("\n\n")[0].strip()
        cache[title] = paragraph if len(paragraph) >= 40 else None
    except Exception:
        cache[title] = None
    return cache[title]


def generate_popqa_swap(seed: int = 42, min_pop: int = 1000, max_pop: int = 30000) -> pd.DataFrame:
    df = _load_hf("akariasai/PopQA", "test")
    df = df[(df.s_pop >= min_pop) & (df.s_pop <= max_pop) & df.prop.isin(POPQA_PROPS)].copy()
    df["answers"] = df.possible_answers.apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    df["org_answer"] = df.answers.apply(lambda x: x[0] if len(x) else None)
    df = df.dropna(subset=["org_answer", "s_wiki_title"])

    logger.info("Fetching Wikipedia extracts for %d subjects", len(df))
    cache: dict[str, str | None] = {}
    df["_extract"] = [_wikipedia_extract(str(title), cache) for title in df.s_wiki_title]
    df = df[
        df.apply(
            lambda r: r["_extract"] is not None
            and bool(re.search(re.escape(r["org_answer"]), r["_extract"], re.IGNORECASE)),
            axis=1,
        )
    ].copy()

    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for _prop, group in df.groupby("prop"):
        records = group.to_dict("records")
        for i, example in enumerate(records):
            org_answer = example["org_answer"]
            org_context = example["_extract"]
            candidates = [
                j
                for j in range(len(records))
                if j != i and records[j]["org_answer"].lower() != org_answer.lower()
            ]
            if not candidates:
                continue
            sub_answer = records[rng.choice(candidates)]["org_answer"]
            sub_context = re.compile(re.escape(org_answer), re.IGNORECASE).sub(
                sub_answer, org_context, count=1
            )
            if sub_context == org_context:
                continue
            rows.append(
                {
                    "question": example["question"],
                    "org_context": org_context,
                    "org_answer": org_answer,
                    "sub_context": sub_context,
                    "sub_answer": sub_answer,
                    "org_aliases": example["answers"],
                }
            )
    return pd.DataFrame(rows, columns=[*SWAP_COLUMNS, "org_aliases"])


DATASET_SPECS: dict[str, DatasetSpec] = {
    "NQ-Swap": DatasetSpec(
        slug="NQ-Swap",
        source=NQSWAP_SOURCE,
        load=lambda: pd.read_parquet(NQSWAP_SOURCE),
        normalize=_normalize_nqswap,
        file_name="dev.parquet",
    ),
    "ConflictQA-popQA-llama2-7b": DatasetSpec(
        slug="ConflictQA-popQA-llama2-7b",
        source="osunlp/ConflictQA/conflictQA-popQA-llama2-7b.json",
        load=lambda: _load_hf_json("osunlp/ConflictQA", "conflictQA-popQA-llama2-7b.json"),
        normalize=_normalize_conflictqa,
    ),
    "ConflictBank": DatasetSpec(
        slug="ConflictBank",
        source="Warrieryes/CB_qa",
        load=lambda: _load_hf("Warrieryes/CB_qa", "train", configs=(None, "default")),
        normalize=_normalize_conflictbank,
    ),
    "PopQA-Swap": DatasetSpec(
        slug="PopQA-Swap",
        source="akariasai/PopQA + en.wikipedia.org",
        load=generate_popqa_swap,
        normalize=_normalize_popqa_swap,
        file_name="dev.parquet",
    ),
}


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[dataset_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported: {', '.join(DATASET_SPECS)}"
        ) from exc


def data_path_for(dataset_name: str) -> Path:
    spec = get_dataset_spec(dataset_name)
    return DATA_DIR / spec.slug / spec.file_name


def split_path_for(dataset_name: str) -> Path:
    slug = get_dataset_spec(dataset_name).slug
    stem = "".join(character if character.isalnum() else "_" for character in slug).strip("_")
    return SPLITS_DIR / f"{stem}_split.parquet"


def write_json(payload: dict[str, object], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def prepare_dataset(dataset_name: str, num_rows: int | None = None) -> dict[str, object]:
    spec = get_dataset_spec(dataset_name)
    prepared = spec.normalize(spec.load())
    if num_rows is not None:
        prepared = prepared.head(num_rows).copy()
    prepared = prepared.reset_index(drop=True)

    output = data_path_for(spec.slug)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(output, index=False)

    manifest = {
        "dataset_name": spec.slug,
        "source": spec.source,
        "row_count": int(len(prepared)),
        "columns": list(prepared.columns),
        "output_file": str(output),
    }
    write_json(manifest, output.parent / "manifest.json")
    return manifest


CHG_FIT_PARTITION = "chg_fit_pool"
CONTROLLER_TRAIN_PARTITION = "controller_train_pool"
SELECTION_DEV_PARTITION = "selection_dev_pool"
TEST_PARTITION = "test_pool"
PARTITIONS: tuple[str, ...] = (
    CHG_FIT_PARTITION,
    CONTROLLER_TRAIN_PARTITION,
    SELECTION_DEV_PARTITION,
    TEST_PARTITION,
)
SPLIT_COLUMN = "split_partition"


def load_prepared(data_path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(data_path).reset_index(drop=True)
    df["source_row_index"] = df.index.astype(int)
    return df


def load_paper_split(split_path: str | Path) -> pd.DataFrame:
    path = Path(split_path)
    if not path.exists():
        raise FileNotFoundError(f"Paper split not found: {path}")
    split_df = pd.read_parquet(path).loc[:, ["source_row_index", SPLIT_COLUMN]].copy()
    split_df["source_row_index"] = split_df["source_row_index"].astype(int)
    split_df[SPLIT_COLUMN] = split_df[SPLIT_COLUMN].astype(str)
    return split_df.sort_values("source_row_index").reset_index(drop=True)


def split_counts(split_df: pd.DataFrame) -> dict[str, int]:
    counts = split_df[SPLIT_COLUMN].value_counts().to_dict()
    return {"row_count": int(len(split_df))} | {
        f"{partition}_count": int(counts[partition])
        for partition in PARTITIONS
        if partition in counts
    }


def load_split_pool(
    data_path: str | Path,
    split_path: str | Path,
    partition: str,
    limit: int | None = None,
) -> pd.DataFrame:
    df = load_prepared(data_path).merge(
        load_paper_split(split_path), on="source_row_index", how="inner", validate="one_to_one"
    )
    pool = df[df[SPLIT_COLUMN] == partition].sort_values("source_row_index").reset_index(drop=True)
    if pool.empty:
        raise ValueError(f"Partition {partition} is empty for split {split_path}")
    return pool.head(limit) if limit is not None else pool


def build_paper_split(
    data_path: str | Path,
    split_path: str | Path,
    seed: int = 0,
    chg_fit_rows: int = SPLIT_CHG_FIT_ROWS,
    controller_train_rows: int = SPLIT_CONTROLLER_TRAIN_ROWS,
    selection_dev_rows: int = SPLIT_SELECTION_DEV_ROWS,
    test_rows: int = SPLIT_TEST_ROWS,
) -> dict[str, object]:
    row_index = load_prepared(data_path)["source_row_index"]
    required = test_rows + selection_dev_rows + controller_train_rows + chg_fit_rows
    if required > len(row_index):
        raise ValueError(f"Not enough rows: need {required} but only {len(row_index)} available.")

    test_ids = row_index.sample(n=test_rows, random_state=seed)
    remaining = row_index[~row_index.isin(test_ids)]
    dev_ids = remaining.sample(n=selection_dev_rows, random_state=seed + 1)
    remaining = remaining[~remaining.isin(dev_ids)]
    controller_ids = remaining.sample(n=controller_train_rows, random_state=seed + 2)

    split_df = pd.DataFrame({"source_row_index": row_index, SPLIT_COLUMN: CHG_FIT_PARTITION})
    for ids, partition in (
        (test_ids, TEST_PARTITION),
        (dev_ids, SELECTION_DEV_PARTITION),
        (controller_ids, CONTROLLER_TRAIN_PARTITION),
    ):
        split_df.loc[split_df["source_row_index"].isin(ids), SPLIT_COLUMN] = partition

    output = Path(split_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_parquet(output, index=False)

    manifest = {
        "dataset_source": str(data_path),
        "split_file": str(output),
        "split_seed": int(seed),
        "chg_fit_rows": int(chg_fit_rows),
        "controller_train_rows": int(controller_train_rows),
        "selection_dev_rows": int(selection_dev_rows),
        "test_rows": int(test_rows),
        **split_counts(split_df),
    }
    write_json(manifest, output.with_suffix(".json"))
    return manifest
