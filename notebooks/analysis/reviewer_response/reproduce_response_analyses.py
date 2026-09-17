#!/usr/bin/env python3
"""Reproduce every number quoted in the Guild reviewer response.

One entry point, one analysis per reviewer point. Each analysis writes a TSV so
the figures in the letter can be traced to a file rather than to a session.

Usage
-----
    python reproduce_response_analyses.py --data DATA_DIR --out OUT_DIR
    python reproduce_response_analyses.py --data ./data --only a2 a3

DATA_DIR must contain the inputs listed in README.md. Nothing is downloaded and
nothing outside OUT_DIR is written.

Dependencies: pandas, numpy. No RDKit (descriptors are parsed from SMILES; see
`heavy_atoms` for the caveat). No guild import needed -- rank percentiles are
recomputed here so the analysis stands alone, and `test_matches_guild` checks
the reimplementation against guild's own output when it is available.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PHYSICAL_VINA = (-20.0, 0.0)   # plausible range for a Vina-family energy, kcal/mol
ACTIVE, DECOY = "strong-binder", "decoy"


# ══════════════════════════════════════════════════════════════════ helpers
def auc_lower_better(actives, decoys) -> float:
    """P(random active ranks better than random decoy), 0 = best convention.

    Mann-Whitney identity on average ranks, so ties are handled correctly.
    """
    a = np.asarray(actives, dtype=float)
    d = np.asarray(decoys, dtype=float)
    a, d = a[~np.isnan(a)], d[~np.isnan(d)]
    if len(a) < 2 or len(d) < 2:
        return float("nan")
    ranks = pd.Series(np.concatenate([-a, -d])).rank(method="average").to_numpy()
    n_a, n_d = len(a), len(d)
    return float((ranks[:n_a].sum() - n_a * (n_a + 1) / 2) / (n_a * n_d))


def hanley_mcneil_ci(auc: float, n_a: int, n_d: int) -> tuple[float, float]:
    """95% CI for an AUC estimate. Wide at small n_a -- that is the point."""
    if np.isnan(auc) or n_a < 2 or n_d < 2:
        return (float("nan"), float("nan"))
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    var = (
        auc * (1 - auc)
        + (n_a - 1) * (q1 - auc * auc)
        + (n_d - 1) * (q2 - auc * auc)
    ) / (n_a * n_d)
    se = float(np.sqrt(max(var, 0.0)))
    return (max(0.0, auc - 1.96 * se), min(1.0, auc + 1.96 * se))


def enrichment_factor(frame: pd.DataFrame, col: str, fraction: float) -> float:
    """EF over the pooled set, ranking by `col` with lower = better. 1.0 = random."""
    sub = frame[["ligand_category", col]].copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna(subset=[col])
    n, n_act = len(sub), int((sub.ligand_category == ACTIVE).sum())
    k = int(np.floor(n * fraction))
    if k < 1 or n_act == 0:
        return float("nan")
    top = sub.nsmallest(k, col)
    return float(((top.ligand_category == ACTIVE).sum() / k) / (n_act / n))


_TWO_LETTER = ("Cl", "Br", "Si", "Se")


def heavy_atoms(smiles: str) -> int:
    """Heavy-atom count parsed straight from SMILES.

    Deliberately crude: no RDKit dependency, two-letter elements stripped first
    so their second character is not double counted. Adequate for establishing a
    size confound; recompute with RDKit before quoting a descriptor externally.
    """
    text, count = str(smiles), 0
    for element in _TWO_LETTER:
        count += text.count(element)
        text = text.replace(element, "")
    return count + len(re.findall(r"[CNOSPFIBcnosp]", text))


def rank_percentile(frame: pd.DataFrame, score_col: str, group_col: str) -> pd.Series:
    """Per-group rank percentile, 0 = best, matching guild's convention."""
    grouped = frame.groupby(group_col)[score_col]
    return grouped.rank(method="average", ascending=True) / grouped.transform("size")


def write(frame: pd.DataFrame, out_dir: Path, name: str) -> None:
    path = out_dir / name
    frame.to_csv(path, sep="\t", index=False)
    print(f"      -> {path.name}  ({frame.shape[0]}x{frame.shape[1]})")


# ═════════════════════════════════════════════════════════ R3-4: runtime
def analysis_runtime(data: Path, out: Path) -> None:
    """Wall-clock for the Vina case study, from guild's own batch_progress logs."""
    print("\nR3-4  runtime of the full Vina case study")
    runs = {
        "vinarun": ("vinarun_batch_progress.log", 403_000),
        "npsvinarun": ("npsvinarun_batch.log", 135_276),
        "knownbindersvinarun": ("knownbindersvinarun_batch.log", 655),
    }
    pattern = re.compile(r"(Starting|Completed)\s+(\S+)\s+at\s+([\d-]+ [\d:.]+)")
    rows, total_h, total_pairs = [], 0.0, 0
    for label, (filename, n_pairs) in runs.items():
        path = data / filename
        if not path.exists():
            print(f"      skip {label}: {filename} not in DATA_DIR")
            continue
        stamps = [
            dt.datetime.fromisoformat(m.group(3))
            for line in path.read_text(errors="replace").splitlines()
            if (m := pattern.search(line))
        ]
        if not stamps:
            continue
        hours = (max(stamps) - min(stamps)).total_seconds() / 3600
        rows.append({
            "run": label, "start": min(stamps).date().isoformat(),
            "end": max(stamps).date().isoformat(), "hours": round(hours, 1),
            "pairs": n_pairs, "seconds_per_pair": round(hours * 3600 / n_pairs, 2),
        })
        total_h += hours
        total_pairs += n_pairs
    if not rows:
        print("      no logs found; nothing written")
        return
    rows.append({
        "run": "TOTAL", "start": "", "end": "", "hours": round(total_h, 1),
        "pairs": total_pairs, "seconds_per_pair": round(total_h * 3600 / total_pairs, 2),
    })
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    write(frame, out, "R3-4_runtime.tsv")


# ═══════════════════════════════════════════════ a2: aggregation rules
# The five pose-source votes, per guild commit 20b3c50: confidences do not vote,
# and each engine's rescore tracks are averaged before the cross-engine mean.
POSE_SOURCE_VOTES = {
    "vina": ["rp_vina_score"],
    "gnina": ["rp_gnina_score"],
    "karmadock": ["rp_karmadock_score"],
    "diffdock": ["rp_vina_rescore_diffdock_score", "rp_gnina_rescore_diffdock_score"],
    "boltz": ["rp_vina_rescore_boltz_score", "rp_gnina_rescore_boltz_score"],
}


def analysis_aggregation(data: Path, out: Path) -> None:
    """Binder-vs-decoy AUC under alternative ways of combining the five votes."""
    print("\na2  alternative aggregation rules  (3-target benchmark)")
    scores = pd.read_csv(data / "guild_scores.txt", sep="\t", low_memory=False)
    scores = scores[scores.ligand_category != "native"]

    votes = pd.DataFrame(
        {name: scores[cols].mean(axis=1) for name, cols in POSE_SOURCE_VOTES.items()},
        index=scores.index,
    )
    is_act = scores.ligand_category == ACTIVE
    is_dec = scores.ligand_category == DECOY
    n_a, n_d = int(is_act.sum()), int(is_dec.sum())

    standalone = {
        name: auc_lower_better(votes.loc[is_act, name], votes.loc[is_dec, name])
        for name in POSE_SOURCE_VOTES
    }
    # weight = credit above chance, so a sub-random method is zeroed rather than inverted
    weights = np.array([max(standalone[n] - 0.5, 0.0) for n in POSE_SOURCE_VOTES])
    weights = weights / weights.sum()

    rules = {
        "unweighted mean (current)": votes.mean(axis=1),
        "median": votes.median(axis=1),
        "best-of-rank (min)": votes.min(axis=1),
        "worst-of-rank (max)": votes.max(axis=1),
        "rank-sum": votes.rank(axis=0).sum(axis=1),
        "performance-weighted": votes.mul(weights, axis=1).sum(axis=1),
        "drop DiffDock": votes.drop(columns=["diffdock"]).mean(axis=1),
        "drop DiffDock + KarmaDock": votes.drop(columns=["diffdock", "karmadock"]).mean(axis=1),
    }
    rows = []
    for name, series in {**{f"single: {k}": votes[k] for k in POSE_SOURCE_VOTES}, **rules}.items():
        value = auc_lower_better(series[is_act], series[is_dec])
        low, high = hanley_mcneil_ci(value, n_a, n_d)
        rows.append({"rule": name, "auc": round(value, 3),
                     "ci_low": round(low, 3), "ci_high": round(high, 3)})
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    print("      weights: " + ", ".join(
        f"{k} {w:.2f}" for k, w in zip(POSE_SOURCE_VOTES, weights, strict=True)))
    write(frame, out, "a2_aggregation_rules.tsv")


# ═══════════════════════════ a3: why rank percentile survives pooling
def _load_case_study(data: Path) -> pd.DataFrame:
    """Known binders and decoys for the large Vina case study.

    CAVEAT, and it belongs in any write-up: the binders come from the
    knownbindersvinarun project and the decoys from vinarun. Same
    protein_config_id, different guild runs. Verify the box and protein
    preparation matched before treating the comparison as within-run.
    """
    binders = pd.read_csv(data / "knownbinders_scores.txt", sep="\t", low_memory=False)
    binders["vina_score"] = pd.to_numeric(binders.vina_score, errors="coerce")
    binders = binders[["protein_config_id", "ligand_id", "smiles", "vina_score"]].copy()
    binders["cls"] = "binder"

    pool = pd.read_csv(
        data / "vinarun_scores.txt", sep="\t", low_memory=False,
        usecols=["protein_config_id", "ligand_id", "smiles", "vina_score", "ligand_category"],
    )
    pool["vina_score"] = pd.to_numeric(pool.vina_score, errors="coerce")
    decoys = pool[pool.ligand_category == DECOY][
        ["protein_config_id", "ligand_id", "smiles", "vina_score"]].copy()
    decoys["cls"] = "decoy"

    both = pd.concat([binders, decoys], ignore_index=True)
    both = both[both.vina_score.between(*PHYSICAL_VINA)]
    complete = both.groupby("protein_config_id")["cls"].nunique() == 2
    return both[both.protein_config_id.isin(complete[complete].index)].copy()


def analysis_normalisation(data: Path, out: Path) -> None:
    """Pooled AUC under three normalisations, plus the mechanism and failure tail."""
    print("\na3  normalisation comparison  (large Vina case study)")
    frame = _load_case_study(data)
    n_t = frame.protein_config_id.nunique()
    print(f"      {n_t} targets, {(frame.cls=='binder').sum()} binders, "
          f"{(frame.cls=='decoy').sum()} decoys")

    grouped = frame.groupby("protein_config_id")["vina_score"]
    frame["rank_pct"] = rank_percentile(frame, "vina_score", "protein_config_id")
    span = grouped.transform("max") - grouped.transform("min")
    frame["minmax"] = (frame.vina_score - grouped.transform("min")) / span
    frame["zscore"] = (frame.vina_score - grouped.transform("mean")) / grouped.transform("std")
    frame["raw"] = frame.vina_score

    n_a = int((frame.cls == "binder").sum())
    n_d = int((frame.cls == "decoy").sum())
    rows = []
    for col, label in [("rank_pct", "rank percentile"), ("minmax", "min-max"),
                       ("zscore", "z-score"), ("raw", "raw score")]:
        value = auc_lower_better(frame.loc[frame.cls == "binder", col],
                                 frame.loc[frame.cls == "decoy", col])
        low, high = hanley_mcneil_ci(value, n_a, n_d)
        rows.append({"scheme": label, "pooled_auc": round(value, 3),
                     "ci_low": round(low, 3), "ci_high": round(high, 3)})
    pooled = pd.DataFrame(rows)
    print(pooled.to_string(index=False))
    write(pooled, out, "a3_normalisations.tsv")

    # Mechanism: a scheme pools cleanly only if it puts every target in the same
    # place. Rank percentile fixes the per-target mean at 0.5 by construction and
    # z-score at 0; min-max does not, which is the whole effect.
    per_target = frame.groupby("protein_config_id").agg(
        skew=("vina_score", lambda s: float(pd.Series(s).skew())),
        n=("vina_score", "size"),
        rank_mean=("rank_pct", "mean"),
        minmax_mean=("minmax", "mean"),
        zscore_mean=("zscore", "mean"),
    ).dropna().reset_index()
    mech = pd.DataFrame([
        {"scheme": label,
         "sd_of_per_target_mean": round(per_target[col].std(), 4),
         "corr_abs_skew_vs_mean": round(
             float(np.corrcoef(per_target["skew"].abs(), per_target[col])[0, 1]), 3)}
        for col, label in [("rank_mean", "rank percentile"),
                           ("minmax_mean", "min-max"),
                           ("zscore_mean", "z-score")]
    ])
    print(mech.to_string(index=False))
    print(f"      raw-score skewness: median {per_target['skew'].median():.2f}, "
          f"|skew|>1 for {(per_target['skew'].abs() > 1).sum()}/{len(per_target)} targets")
    write(mech, out, "a3_mechanism.tsv")
    write(per_target.round(4), out, "a3_per_target_skew.tsv")

    # Failure tail: known binders ranked poorly within their own target.
    binders = frame[frame.cls == "binder"].copy()
    binders["n_heavy"] = binders.smiles.map(heavy_atoms)
    binders["poorly_ranked"] = binders.rank_pct > 0.5
    by_target = binders.groupby("protein_config_id")["poorly_ranked"].agg(["sum", "size"])
    by_target["fraction"] = by_target["sum"] / by_target["size"]
    tail = pd.DataFrame([{
        "n_binders": len(binders),
        "in_worse_half": int(binders.poorly_ranked.sum()),
        "pct_worse_half": round(100 * binders.poorly_ranked.mean(), 1),
        "in_worst_decile": int((binders.rank_pct > 0.9).sum()),
        "pct_worst_decile": round(100 * (binders.rank_pct > 0.9).mean(), 1),
        "heavy_atoms_well_ranked": round(binders.loc[~binders.poorly_ranked, "n_heavy"].mean(), 1),
        "heavy_atoms_poorly_ranked": round(binders.loc[binders.poorly_ranked, "n_heavy"].mean(), 1),
        "vina_well_ranked": round(binders.loc[~binders.poorly_ranked, "vina_score"].mean(), 2),
        "vina_poorly_ranked": round(binders.loc[binders.poorly_ranked, "vina_score"].mean(), 2),
        "targets_with_none": int((by_target.fraction == 0).sum()),
        "targets_mostly_failing": int((by_target.fraction >= 0.8).sum()),
        "n_targets": len(by_target),
    }])
    print(tail.to_string(index=False))
    write(tail, out, "a3_failure_tail_summary.tsv")
    write(binders.round(4), out, "a3_failure_tail_binders.tsv")


# ════════════════════════════ decoy size confound (held in reserve)
def analysis_size_control(data: Path, out: Path) -> None:
    """Whether the benchmark decoys are size-matched, and AUC once they are.

    Not quoted in the response. Kept because Reviewer 3 explicitly offers
    "report the AUC comparison again with a property-matched decoy control, or
    note this as a limitation", and this is the evidence for either answer.
    """
    print("\nreserve  decoy size confound and size-matched AUC")
    scores = pd.read_csv(data / "guild_scores.txt", sep="\t", low_memory=False)
    scores = scores[scores.ligand_category != "native"].copy()
    scores["n_heavy"] = scores.smiles.map(heavy_atoms)
    act = scores[scores.ligand_category == ACTIVE]
    dec = scores[scores.ligand_category == DECOY]

    low, high = act.n_heavy.quantile(0.10), act.n_heavy.quantile(0.90)
    act_m = act[act.n_heavy.between(low, high)]
    dec_m = dec[dec.n_heavy.between(low, high)]
    print(f"      matched window: {low:.0f}-{high:.0f} heavy atoms  "
          f"({len(act_m)} actives, {len(dec_m)} decoys)")

    # `higher_better` flips the sign convention: for heavy-atom count, a LARGER
    # value marks the actives, so reporting it on the lower-is-better scale would
    # print its complement and read as if size were anti-predictive.
    tracks = [("rp_vina_score", "Vina", False), ("rp_gnina_score", "GNINA", False),
              ("rp_karmadock_score", "KarmaDock", False),
              ("rp_diffdock_score", "DiffDock conf", False),
              ("rp_boltz_score", "Boltz-2 ipTM", False),
              ("global_rp_score", "Guild combined", False),
              ("boltz_affinity_score", "Boltz-2 affinity", False),
              ("n_heavy", "SIZE BASELINE", True)]
    rows = []
    for col, label, higher_better in tracks:
        full = auc_lower_better(act[col], dec[col])
        matched = auc_lower_better(act_m[col], dec_m[col])
        if higher_better:
            full, matched = 1 - full, 1 - matched
        rows.append({
            "track": label,
            "auc_full": round(full, 3),
            "auc_size_matched": round(matched, 3),
        })
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    write(frame, out, "reserve_size_matched_auc.tsv")


# ═════════════════════════════════════════════════════ self-check
def test_matches_guild(data: Path) -> None:
    """Confirm the local rank-percentile reimplementation matches guild's output."""
    path = data / "guild_scores.txt"
    if not path.exists():
        return
    scores = pd.read_csv(path, sep="\t", low_memory=False)
    mine = rank_percentile(scores, "vina_score", "protein_config_id")
    theirs = pd.to_numeric(scores["rp_vina_score"], errors="coerce")
    both = (~mine.isna()) & (~theirs.isna())
    if both.sum() == 0:
        return
    worst = float((mine[both] - theirs[both]).abs().max())
    status = "OK" if worst < 1e-9 else f"MISMATCH (max delta {worst:.2e})"
    print(f"\nself-check  rank percentile vs guild's rp_vina_score: {status}")


ANALYSES = {
    "runtime": analysis_runtime,
    "a2": analysis_aggregation,
    "a3": analysis_normalisation,
    "reserve": analysis_size_control,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("./data"),
                        help="directory holding the inputs listed in README.md")
    parser.add_argument("--out", type=Path, default=Path("./output"),
                        help="directory for the TSVs (created if absent)")
    parser.add_argument("--only", nargs="+", choices=sorted(ANALYSES),
                        help="run only these analyses (default: all)")
    args = parser.parse_args(argv)

    if not args.data.is_dir():
        parser.error(f"--data {args.data} is not a directory")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"data: {args.data.resolve()}\nout:  {args.out.resolve()}")

    test_matches_guild(args.data)
    for name in (args.only or sorted(ANALYSES)):
        try:
            ANALYSES[name](args.data, args.out)
        except FileNotFoundError as error:
            print(f"\n{name}: skipped, missing input -- {error.filename}")
    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
