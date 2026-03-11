from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset, load_dataset


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    config: str | None
    text_field: str
    summary_field: str
    default_model: str


DATASET_SPECS = {
    "cnn_dailymail": DatasetSpec(
        name="cnn_dailymail",
        config="3.0.0",
        text_field="article",
        summary_field="highlights",
        default_model="facebook/bart-large-cnn",
    ),
    "xsum": DatasetSpec(
        name="xsum",
        config=None,
        text_field="document",
        summary_field="summary",
        default_model="facebook/bart-large-xsum",
    ),
}


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    if dataset_name not in DATASET_SPECS:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {supported}")
    return DATASET_SPECS[dataset_name]


def load_split(dataset_name: str, split: str) -> tuple[DatasetSpec, Dataset]:
    spec = get_dataset_spec(dataset_name)
    ds = load_dataset(spec.name, spec.config, split=split)
    return spec, ds
