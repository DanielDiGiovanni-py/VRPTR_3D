from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np
from torch.utils.data import Dataset


INPUT_ABLATION_CHOICES = ("none", "zero", "cohort_mean", "shuffle")


def normalize_input_ablation(mode: str | None) -> str:
    m = str(mode or "none").strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "no": "none",
        "identity": "none",
        "mean": "cohort_mean",
        "group_mean": "cohort_mean",
        "template_input": "cohort_mean",
        "subject_shuffle": "shuffle",
        "shuffled": "shuffle",
        "shuffled_input": "shuffle",
        "zeros": "zero",
        "zero_input": "zero",
    }
    m = aliases.get(m, m)
    if m not in INPUT_ABLATION_CHOICES:
        raise ValueError(f"Unknown input ablation '{mode}'. Valid choices: {INPUT_ABLATION_CHOICES}")
    return m


def ablated_model_name(model_name: str, mode: str | None, *, add_suffix: bool = True) -> str:
    mode = normalize_input_ablation(mode)
    if not add_suffix or mode == "none":
        return str(model_name)
    return f"{model_name}_input_{mode}"


def make_shuffle_donor_indices(n: int, seed: int = 42) -> np.ndarray:
    """Return a deterministic derangement-like donor index vector.

    If n > 1, the returned permutation has no fixed points.  This makes
    subject-shuffled ablations a strict test of subject-specific input signal:
    every target subject receives another subject's resting-state fingerprint.
    """
    n = int(n)
    if n <= 1:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(int(seed))
    donor = rng.permutation(n)
    # Repair fixed points deterministically by cycling their assigned donors.
    fixed = np.where(donor == np.arange(n))[0]
    if fixed.size == 1:
        i = int(fixed[0])
        j = int((i + 1) % n)
        donor[i], donor[j] = donor[j], donor[i]
    elif fixed.size > 1:
        donor[fixed] = np.roll(donor[fixed], 1)
    # Very small n can still produce a fixed point after repair; fall back to a roll.
    if np.any(donor == np.arange(n)):
        donor = np.roll(np.arange(n, dtype=int), 1)
    return donor.astype(int)


def _load_scaled_x(matrix: dict, raw_index: int) -> np.ndarray:
    x = np.asarray(matrix["x"][int(raw_index)], dtype=np.float32).copy()
    scaler = matrix.get("x_scaler", None)
    if scaler is not None:
        x = scaler.transform_subject(x)
    return x.astype(np.float32, copy=False)


def _load_y(matrix: dict, raw_index: int) -> np.ndarray:
    return np.asarray(matrix["y"][int(raw_index)], dtype=np.float32).copy()


def _apply_target_valid_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.ndim == 4 and y.ndim == 4:
        valid = np.isfinite(y[0])
        x = x.copy()
        x[:, ~valid] = 0.0
    return x.astype(np.float32, copy=False)


def compute_cohort_mean_x(matrix: dict, indices: Sequence[int] | None = None) -> np.ndarray:
    """Stream a standardized cohort-mean input without materializing all subjects."""
    n = len(matrix["subjects"])
    idx = np.asarray(indices if indices is not None else np.arange(n), dtype=int)
    if idx.size == 0:
        raise RuntimeError("Cannot compute cohort-mean input from zero subjects.")
    acc = None
    count = 0
    for ii in idx:
        x = _load_scaled_x(matrix, int(ii))
        if acc is None:
            acc = np.zeros_like(x, dtype=np.float64)
        acc += x.astype(np.float64, copy=False)
        count += 1
    mean = (acc / max(count, 1)).astype(np.float32)
    return np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class AblatedImageSubjectDataset(Dataset):
    """ImageSubjectDataset-compatible wrapper for representation ablations.

    Targets are always the requested subject's beta map, LI, and dominance label.
    The input fingerprint is either unchanged, zeroed, replaced with the evaluated
    cohort mean, or replaced by another subject's fingerprint.  The ablation is
    applied after the training standardizer so that ``zero`` means zero in model
    input space rather than zero in raw image intensity space.
    """

    def __init__(
        self,
        matrix: dict,
        indices: Sequence[int] | None = None,
        mode: str = "none",
        seed: int = 42,
        cohort_mean_x: np.ndarray | None = None,
    ):
        self.matrix = matrix
        n = len(matrix["subjects"])
        self.indices = np.asarray(indices if indices is not None else np.arange(n), dtype=int)
        self.mode = normalize_input_ablation(mode)
        self.seed = int(seed)
        self.donor_indices = make_shuffle_donor_indices(n, seed=self.seed) if self.mode == "shuffle" else np.arange(n, dtype=int)
        self.cohort_mean_x = cohort_mean_x
        if self.mode == "cohort_mean" and self.cohort_mean_x is None:
            self.cohort_mean_x = compute_cohort_mean_x(matrix, np.arange(n, dtype=int))

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, idx: int) -> dict:
        target_i = int(self.indices[int(idx)])
        y = _load_y(self.matrix, target_i)
        if self.mode == "zero":
            x = np.zeros_like(_load_scaled_x(self.matrix, target_i), dtype=np.float32)
        elif self.mode == "cohort_mean":
            x = np.asarray(self.cohort_mean_x, dtype=np.float32).copy()
        elif self.mode == "shuffle":
            donor_i = int(self.donor_indices[target_i])
            x = _load_scaled_x(self.matrix, donor_i)
        else:
            x = _load_scaled_x(self.matrix, target_i)
        x = _apply_target_valid_mask(x, y)
        return {
            "subject": self.matrix["subjects"][target_i],
            "x": x,
            "y": y,
            "li": np.asarray(self.matrix["li"][target_i], dtype=np.float32),
            "dominance": np.asarray(self.matrix["dominance"][target_i], dtype=np.int64),
        }


def make_eval_dataset(matrix: dict, indices: Sequence[int] | None = None, mode: str = "none", seed: int = 42):
    """Return a standard or ablated dataset, depending on ``mode``."""
    from .data import ImageSubjectDataset

    mode = normalize_input_ablation(mode)
    if mode == "none":
        return ImageSubjectDataset(matrix, indices)
    return AblatedImageSubjectDataset(matrix, indices=indices, mode=mode, seed=seed)


def add_input_ablation_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--input-ablation",
        choices=list(INPUT_ABLATION_CHOICES),
        default="none",
        help=(
            "Evaluation-only representation ablation. 'zero' removes all resting-state input, "
            "'cohort_mean' gives every subject the evaluated cohort's mean fingerprint, and "
            "'shuffle' gives each target subject another subject's fingerprint."
        ),
    )
    parser.add_argument("--ablation-seed", type=int, default=42, help="Seed used for subject-shuffled input ablations.")
    parser.add_argument("--no-ablation-name-suffix", action="store_true", help="Do not append _input_<mode> to output model names.")
    return parser


def read_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def compile_ablation_rows(root: Path) -> list[dict]:
    """Collect rows whose model name indicates an input ablation."""
    rows: list[dict] = []
    for table_name in ["compiled_summary.csv", "compiled_signal_recovery_summary.csv", "compiled_individuality_summary.csv", "compiled_residual_individuality_summary.csv"]:
        for r in read_csv(root / table_name):
            model = str(r.get("model", ""))
            if "_input_" not in model:
                continue
            rr = dict(r)
            rr["source_table"] = table_name
            rows.append(rr)
    return rows


def main():
    p = argparse.ArgumentParser(description="Collect representation-ablation rows from compiled Clinical VRPTR tables.")
    p.add_argument("--root", default="results")
    p.add_argument("--out", default="")
    args = p.parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "compiled_representation_ablation_summary.csv"
    rows = compile_ablation_rows(root)
    write_csv(out, rows)
    print(f"[representation_ablation] wrote {out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
