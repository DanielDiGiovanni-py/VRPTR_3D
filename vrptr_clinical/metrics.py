from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def finite_np(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)[np.isfinite(x)]


def pearsonr_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a = a[mask] - a[mask].mean()
    b = b[mask] - b[mask].mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if den <= 1e-12:
        return float("nan")
    return float(np.sum(a * b) / den)


def spearman_np(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import rankdata
    except Exception:
        return pearsonr_np(a, b)
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return pearsonr_np(rankdata(a[mask]), rankdata(b[mask]))


def r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def dice_topk(a: np.ndarray, b: np.ndarray, k: float, use_abs: bool = True) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size == 0:
        return float("nan")
    score_a = np.abs(a) if use_abs else a
    score_b = np.abs(b) if use_abs else b
    n_top = max(1, int(round(a.size * float(k))))
    ia = set(np.argsort(score_a)[-n_top:].tolist())
    ib = set(np.argsort(score_b)[-n_top:].tolist())
    return float(2.0 * len(ia & ib) / (len(ia) + len(ib) + 1e-12))


def dice_auc(a: np.ndarray, b: np.ndarray, thresholds: Iterable[float] | None = None, use_abs: bool = True) -> float:
    th = np.asarray(list(thresholds) if thresholds is not None else np.arange(0.05, 0.51, 0.05), dtype=float)
    vals = np.asarray([dice_topk(a, b, k, use_abs=use_abs) for k in th], dtype=float)
    if np.sum(np.isfinite(vals)) < 2:
        return float("nan")
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(vals, th) / (th[-1] - th[0]))


def robust_li(values: np.ndarray, left_idx: np.ndarray, right_idx: np.ndarray, q: float = 75.0, positive_only: bool = True) -> float:
    """Threshold-integrated-ish LI using robust suprathreshold sums.

    The goal is not to create a new clinical LI standard, but to avoid max-parcel instability.
    For each hemisphere, values below the within-hemisphere percentile are discarded.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if positive_only:
        v = np.maximum(v, 0.0)
    left = v[np.asarray(left_idx, dtype=int)]
    right = v[np.asarray(right_idx, dtype=int)]
    left = finite_np(left)
    right = finite_np(right)
    if left.size == 0 or right.size == 0:
        return float("nan")
    lt = np.percentile(left, q) if np.any(left > 0) else 0.0
    rt = np.percentile(right, q) if np.any(right > 0) else 0.0
    lsum = float(left[left >= lt].sum())
    rsum = float(right[right >= rt].sum())
    den = lsum + rsum
    if abs(den) < 1e-12:
        return 0.0
    return float((lsum - rsum) / den)


def dominance_class(li: float, threshold: float = 0.20) -> int:
    """0=right, 1=bilateral/indeterminate, 2=left."""
    if not np.isfinite(li):
        return 1
    if li > threshold:
        return 2
    if li < -threshold:
        return 0
    return 1


def subject_metrics(y_true: np.ndarray, y_pred: np.ndarray, left_idx: np.ndarray, right_idx: np.ndarray) -> Dict[str, float]:
    li_t = robust_li(y_true, left_idx, right_idx)
    li_p = robust_li(y_pred, left_idx, right_idx)
    dom_t = dominance_class(li_t)
    dom_p = dominance_class(li_p)
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    error = (yp - yt)[np.isfinite(yt) & np.isfinite(yp)]
    return {
        "mae": float(np.mean(np.abs(error))) if error.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(error ** 2))) if error.size else float("nan"),
        "pearson_r": pearsonr_np(y_true, y_pred),
        "spearman_r": spearman_np(y_true, y_pred),
        "r2": r2_np(y_true, y_pred),
        "dice_auc": dice_auc(y_true, y_pred),
        "dice25": dice_topk(y_true, y_pred, 0.25),
        "li_true": li_t,
        "li_pred": li_p,
        "li_abs_err": abs(li_t - li_p) if np.isfinite(li_t) and np.isfinite(li_p) else float("nan"),
        "li_sign_acc": float(np.sign(li_t) == np.sign(li_p)) if np.isfinite(li_t) and np.isfinite(li_p) else float("nan"),
        "dominance_true": float(dom_t),
        "dominance_pred": float(dom_p),
        "dominance_acc": float(dom_t == dom_p),
    }


def summarize_metric_rows(rows: list[dict], group: str, model: str) -> dict:
    keys = [
        "mae", "rmse", "pearson_r", "spearman_r", "r2", "dice_auc", "dice25",
        "li_abs_err", "li_sign_acc", "dominance_acc",
    ]
    out = {"group": group, "model": model, "n": len(rows)}
    for k in keys:
        vals = np.asarray([float(r[k]) for r in rows if k in r and np.isfinite(float(r[k]))], dtype=float)
        out[f"{k}_mean"] = float(vals.mean()) if vals.size else float("nan")
        out[f"{k}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0 if vals.size == 1 else float("nan")
    return out

def _stack_subject_maps(maps: list[np.ndarray]) -> np.ndarray:
    if not maps:
        return np.zeros((0, 0), dtype=np.float32)
    flat = [np.asarray(m, dtype=np.float32).reshape(-1) for m in maps]
    return np.stack(flat, axis=0)


def pairwise_pearson_matrix(y_pred_stack: list[np.ndarray] | np.ndarray, y_true_stack: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """Return [n_pred, n_true] Pearson correlations between predicted and true maps.

    This is used for self-vs-other identification.  It uses a common finite voxel
    mask across the evaluated cohort so every pair is compared over identical
    voxels.  Rows with nearly zero variance are returned as NaN.
    """
    p = _stack_subject_maps(y_pred_stack) if isinstance(y_pred_stack, list) else np.asarray(y_pred_stack, dtype=np.float32).reshape(len(y_pred_stack), -1)
    t = _stack_subject_maps(y_true_stack) if isinstance(y_true_stack, list) else np.asarray(y_true_stack, dtype=np.float32).reshape(len(y_true_stack), -1)
    if p.shape[0] == 0 or t.shape[0] == 0:
        return np.zeros((int(p.shape[0]), int(t.shape[0])), dtype=np.float64)
    if p.shape[1] != t.shape[1]:
        raise ValueError(f"Predicted and true maps have different flattened sizes: {p.shape[1]} vs {t.shape[1]}")
    finite = np.isfinite(p).all(axis=0) & np.isfinite(t).all(axis=0)
    if int(finite.sum()) < 3:
        return np.full((p.shape[0], t.shape[0]), np.nan, dtype=np.float64)
    p = p[:, finite].astype(np.float64, copy=False)
    t = t[:, finite].astype(np.float64, copy=False)
    p = p - p.mean(axis=1, keepdims=True)
    t = t - t.mean(axis=1, keepdims=True)
    pn = np.sqrt(np.sum(p * p, axis=1, keepdims=True))
    tn = np.sqrt(np.sum(t * t, axis=1, keepdims=True))
    p = np.divide(p, pn, out=np.full_like(p, np.nan, dtype=np.float64), where=pn > 1e-12)
    t = np.divide(t, tn, out=np.full_like(t, np.nan, dtype=np.float64), where=tn > 1e-12)
    return p @ t.T


def self_vs_other_rows(subjects: list[str], y_true_stack: list[np.ndarray], y_pred_stack: list[np.ndarray], group: str, model: str) -> tuple[list[dict], list[dict], dict]:
    """Quantify whether each prediction is closest to its own subject's true map.

    Returns per-subject identification rows, pairwise similarity rows, and a
    compact summary for secondary activation-pattern matching:
    whether pred_i is strictly more correlated with true_i than with every other true map
    in the same evaluated group/model.
    """
    subjects = [str(s) for s in subjects]
    n = len(subjects)
    sim = pairwise_pearson_matrix(y_pred_stack, y_true_stack)
    rows: list[dict] = []
    pairwise_rows: list[dict] = []

    for i, subj_pred in enumerate(subjects):
        vals = sim[i].astype(np.float64, copy=False) if n else np.asarray([], dtype=np.float64)
        self_sim = float(vals[i]) if i < vals.size and np.isfinite(vals[i]) else float("nan")
        finite_idx = np.where(np.isfinite(vals))[0]
        if finite_idx.size:
            # Rank is 1 when the self match is the most similar true map.
            rank = int(1 + np.sum(vals[finite_idx] > self_sim)) if np.isfinite(self_sim) else int(finite_idx.size)
            best_idx = int(finite_idx[np.argmax(vals[finite_idx])])
            best_subject = subjects[best_idx]
            best_score = float(vals[best_idx])
        else:
            rank = 0
            best_idx = -1
            best_subject = ""
            best_score = float("nan")
        other_idx = np.asarray([j for j in finite_idx.tolist() if j != i], dtype=int)
        if other_idx.size:
            other_vals = vals[other_idx]
            best_other_pos = int(other_idx[np.argmax(other_vals)])
            best_other_subject = subjects[best_other_pos]
            best_other = float(vals[best_other_pos])
            mean_other = float(np.mean(other_vals))
            margin_best = self_sim - best_other if np.isfinite(self_sim) and np.isfinite(best_other) else float("nan")
            margin_mean = self_sim - mean_other if np.isfinite(self_sim) and np.isfinite(mean_other) else float("nan")
        else:
            best_other_subject = ""
            best_other = float("nan")
            mean_other = float("nan")
            margin_best = float("nan")
            margin_mean = float("nan")
        denom = max(n - 1, 1)
        percentile = float(1.0 - ((rank - 1) / denom)) if rank > 0 and n > 1 else float("nan")
        rows.append({
            "group": group,
            "model": model,
            "subject": subj_pred,
            "n_candidates": int(n),
            "self_pearson": self_sim,
            "best_match_subject": best_subject,
            "best_match_pearson": best_score,
            "best_other_subject": best_other_subject,
            "best_other_pearson": best_other,
            "mean_other_pearson": mean_other,
            "self_vs_best_other_margin": margin_best,
            "self_vs_mean_other_margin": margin_mean,
            "self_rank": rank,
            "self_percentile": percentile,
            "top1_acc": float(np.isfinite(self_sim) and self_sim > best_other + 1e-12) if other_idx.size else float("nan"),
            "top5_acc": float(rank <= min(5, n)) if rank > 0 else float("nan"),
            "positive_margin_acc": float(margin_best > 0.0) if np.isfinite(margin_best) else float("nan"),
        })
        for j, subj_true in enumerate(subjects):
            pairwise_rows.append({
                "group": group,
                "model": model,
                "subject_pred": subj_pred,
                "subject_true": subj_true,
                "pearson": float(vals[j]) if j < vals.size and np.isfinite(vals[j]) else float("nan"),
                "is_self": int(i == j),
            })

    summary_keys = [
        "self_pearson", "best_other_pearson", "mean_other_pearson",
        "self_vs_best_other_margin", "self_vs_mean_other_margin",
        "self_rank", "self_percentile", "top1_acc", "top5_acc", "positive_margin_acc",
    ]
    summary = {"group": group, "model": model, "n": int(n)}
    for k in summary_keys:
        vals = np.asarray([float(r[k]) for r in rows if k in r and np.isfinite(float(r[k]))], dtype=float)
        summary[f"{k}_mean"] = float(vals.mean()) if vals.size else float("nan")
        summary[f"{k}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0 if vals.size == 1 else float("nan")
    rank_vals = np.asarray([float(r["self_rank"]) for r in rows if np.isfinite(float(r.get("self_rank", float("nan")))) and float(r.get("self_rank", 0)) > 0], dtype=float)
    summary["self_rank_median"] = float(np.median(rank_vals)) if rank_vals.size else float("nan")
    return rows, pairwise_rows, summary


def _prefix_metrics(prefix: str, metrics: dict) -> dict:
    """Prefix scalar subject metric keys for side-by-side model/template rows."""
    keep = [
        "mae", "rmse", "pearson_r", "spearman_r", "r2", "dice_auc", "dice25",
        "li_true", "li_pred", "li_abs_err", "li_sign_acc",
        "dominance_true", "dominance_pred", "dominance_acc",
    ]
    return {f"{prefix}_{k}": metrics[k] for k in keep if k in metrics}


def leave_one_out_template_maps(y_true_stack: list[np.ndarray]) -> list[np.ndarray]:
    """Return one leave-one-out group template per subject.

    The template for subject i is the voxelwise mean of all true maps except i.
    Only for explicitly requested transductive sensitivity analysis. At least
    two subjects are required; absent donor support remains NaN.
    """
    true = [np.asarray(m, dtype=np.float32) for m in y_true_stack]
    if not true:
        return []
    flat = _stack_subject_maps(true).astype(np.float64, copy=False)
    shape = true[0].shape
    n = flat.shape[0]
    if n < 2:
        raise ValueError("LOO sensitivity requires at least two participants")
    finite = np.isfinite(flat)
    flat_safe = np.where(finite, flat, 0.0)
    sum_all = flat_safe.sum(axis=0)
    count_all = finite.sum(axis=0)
    templates = []
    for i in range(n):
        denom = np.maximum(count_all - finite[i].astype(np.int64), 1)
        tmp = (sum_all - flat_safe[i]) / denom
        # Never fill missing donor voxels with the excluded participant.
        tmp = np.where((count_all - finite[i].astype(np.int64)) > 0, tmp, np.nan)
        templates.append(tmp.reshape(shape).astype(np.float32))
    return templates


def template_residual_analysis_rows(
    subjects: list[str],
    y_true_stack: list[np.ndarray],
    y_pred_stack: list[np.ndarray],
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    group: str,
    model: str,
    *,
    training_template: np.ndarray | None = None,
    template_metadata: dict | None = None,
    template_mode: str = "training_mean",
) -> tuple[list[dict], dict, list[dict], dict, list[dict], list[dict], dict]:
    """Evaluate prediction and secondary activation-pattern specificity.

    Primary reference: one frozen mean of standardized training task maps,
    subtracted from every prediction and observation, including off-diagonal
    matching pairs. Test-cohort LOO is an explicitly labelled sensitivity only.
    """
    subjects = [str(s) for s in subjects]
    true_maps = [np.asarray(m, dtype=np.float32) for m in y_true_stack]
    pred_maps = [np.asarray(m, dtype=np.float32) for m in y_pred_stack]
    if len(subjects) != len(true_maps) or len(subjects) != len(pred_maps):
        raise ValueError("subjects, y_true_stack, and y_pred_stack must have the same length")

    if len(set(subjects)) != len(subjects):
        raise ValueError("Duplicate evaluation subjects")
    if template_mode == "training_mean":
        if training_template is None:
            raise ValueError("A fixed training template is required; no test-cohort fallback")
        tmpl = np.asarray(training_template, dtype=np.float32)
        for truth, pred in zip(true_maps, pred_maps):
            if truth.shape != tmpl.shape or pred.shape != tmpl.shape:
                raise ValueError("Prediction, target and training-template shapes must agree exactly")
            if np.any(np.isfinite(truth) & ~np.isfinite(tmpl)):
                raise ValueError("Evaluation mask contains voxels unsupported by the training template")
        templates = [tmpl] * len(subjects)
        provenance = {"template_source": "training_mean", **(template_metadata or {})}
        template_name = "training_mean_template"
    elif template_mode == "test_loo_sensitivity":
        if training_template is not None:
            raise ValueError("Do not supply a training template in LOO sensitivity mode")
        templates = leave_one_out_template_maps(true_maps)
        provenance = {"template_source": "test_loo_sensitivity"}
        template_name = f"{group}_test_loo_sensitivity"
    else:
        raise ValueError(f"Unknown template mode: {template_mode}")
    signal_rows: list[dict] = []
    template_rows: list[dict] = []
    true_residuals: list[np.ndarray] = []
    pred_residuals: list[np.ndarray] = []

    for subj, y_true, y_pred, tmpl in zip(subjects, true_maps, pred_maps, templates):
        model_m = subject_metrics(y_true, y_pred, left_idx, right_idx)
        template_m = subject_metrics(y_true, tmpl, left_idx, right_idx)
        true_res = np.asarray(y_true, dtype=np.float32) - np.asarray(tmpl, dtype=np.float32)
        pred_res = np.asarray(y_pred, dtype=np.float32) - np.asarray(tmpl, dtype=np.float32)
        true_residuals.append(true_res)
        pred_residuals.append(pred_res)
        residual_m = {
            "pearson_r": pearsonr_np(true_res, pred_res),
            "spearman_r": spearman_np(true_res, pred_res),
            "r2": r2_np(true_res, pred_res),
            "dice_auc": dice_auc(true_res, pred_res),
            "dice25": dice_topk(true_res, pred_res, 0.25),
        }
        pred_template_r = pearsonr_np(y_pred, tmpl)
        row = {
            "group": group,
            "model": model,
            "subject": subj,
            **_prefix_metrics("model", model_m),
            **_prefix_metrics("template", template_m),
            "mae_improvement": template_m["mae"] - model_m["mae"],
            "rmse_improvement": template_m["rmse"] - model_m["rmse"],
            "delta_pearson_r": model_m.get("pearson_r", float("nan")) - template_m.get("pearson_r", float("nan")),
            "delta_spearman_r": model_m.get("spearman_r", float("nan")) - template_m.get("spearman_r", float("nan")),
            "delta_r2": model_m.get("r2", float("nan")) - template_m.get("r2", float("nan")),
            "delta_dice_auc": model_m.get("dice_auc", float("nan")) - template_m.get("dice_auc", float("nan")),
            "delta_dice25": model_m.get("dice25", float("nan")) - template_m.get("dice25", float("nan")),
            "li_abs_err_improvement": template_m.get("li_abs_err", float("nan")) - model_m.get("li_abs_err", float("nan")),
            "model_beats_template_pearson": float(model_m.get("pearson_r", float("nan")) > template_m.get("pearson_r", float("nan"))),
            "model_beats_template_r2": float(model_m.get("r2", float("nan")) > template_m.get("r2", float("nan"))),
            "model_beats_template_dice25": float(model_m.get("dice25", float("nan")) > template_m.get("dice25", float("nan"))),
            "model_beats_template_li_abs_err": float(model_m.get("li_abs_err", float("nan")) < template_m.get("li_abs_err", float("nan"))),
            "pred_vs_template_pearson": pred_template_r,
            "residual_pearson_r": residual_m["pearson_r"],
            "residual_spearman_r": residual_m["spearman_r"],
            "residual_r2": residual_m["r2"],
            "residual_dice_auc": residual_m["dice_auc"],
            "residual_dice25": residual_m["dice25"],
        }
        signal_rows.append(row)
        trow = dict(template_m)
        trow.update({"subject": subj, "group": group, "model": template_name})
        template_rows.append(trow)

    summary_keys = [
        "model_mae", "template_mae", "mae_improvement",
        "model_rmse", "template_rmse", "rmse_improvement",
        "model_pearson_r", "template_pearson_r", "delta_pearson_r",
        "model_spearman_r", "template_spearman_r", "delta_spearman_r",
        "model_r2", "template_r2", "delta_r2",
        "model_dice_auc", "template_dice_auc", "delta_dice_auc",
        "model_dice25", "template_dice25", "delta_dice25",
        "model_li_abs_err", "template_li_abs_err", "li_abs_err_improvement",
        "model_beats_template_pearson", "model_beats_template_r2",
        "model_beats_template_dice25", "model_beats_template_li_abs_err",
        "pred_vs_template_pearson",
        "residual_pearson_r", "residual_spearman_r", "residual_r2",
        "residual_dice_auc", "residual_dice25",
    ]
    signal_summary = {"group": group, "model": model, "n": int(len(signal_rows))}
    for k in summary_keys:
        vals = np.asarray([float(r[k]) for r in signal_rows if k in r and np.isfinite(float(r[k]))], dtype=float)
        signal_summary[f"{k}_mean"] = float(vals.mean()) if vals.size else float("nan")
        signal_summary[f"{k}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0 if vals.size == 1 else float("nan")

    template_summary = summarize_metric_rows(template_rows, group=group, model=template_name)
    residual_rows, residual_pairwise, residual_summary = self_vs_other_rows(
        subjects, true_residuals, pred_residuals, group, model
    )
    for row in signal_rows + template_rows + residual_rows + residual_pairwise:
        row.update(provenance)
    for summary in (signal_summary, template_summary, residual_summary):
        summary.update(provenance)
    return signal_rows, signal_summary, template_rows, template_summary, residual_rows, residual_pairwise, residual_summary
