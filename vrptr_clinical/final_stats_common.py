from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np


def write_rows(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(values, rng: np.random.Generator, n_boot: int = 20000) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    means = np.empty(int(n_boot), dtype=float)
    chunk = 2000
    offset = 0
    while offset < int(n_boot):
        n = min(chunk, int(n_boot) - offset)
        idx = rng.integers(0, x.size, size=(n, x.size))
        means[offset : offset + n] = x[idx].mean(axis=1)
        offset += n
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def signflip_p(values, rng: np.random.Generator, n_perm: int = 20000) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    observed = abs(float(x.mean()))
    exceed = 0
    total = 0
    chunk = 2000
    while total < int(n_perm):
        n = min(chunk, int(n_perm) - total)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(n, x.size))
        permuted = np.abs((signs * x).mean(axis=1))
        exceed += int(np.sum(permuted >= observed - 1e-15))
        total += n
    return float((exceed + 1) / (total + 1))


def paired_stats(values, rng: np.random.Generator, n_boot: int = 20000, n_perm: int = 20000) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    lo, hi = bootstrap_mean_ci(x, rng, n_boot=n_boot)
    return {
        "n": int(x.size),
        "mean_effect": float(x.mean()) if x.size else float("nan"),
        "bootstrap_ci95_low": lo,
        "bootstrap_ci95_high": hi,
        "signflip_p_two_sided": signflip_p(x, rng, n_perm=n_perm),
        "positive_effect_fraction": float(np.mean(x > 0)) if x.size else float("nan"),
    }


def bh_adjust(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    pf = p[finite]
    order = np.argsort(pf)
    ranked = pf[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    restored = np.empty_like(q)
    restored[order] = q
    out[finite] = restored
    return out


def safe_pearson(a, b) -> float:
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask] - np.mean(x[mask])
    y = y[mask] - np.mean(y[mask])
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 1e-12 else float("nan")
