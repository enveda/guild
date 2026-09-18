#!/usr/bin/env python3
"""Build the supplementary tables (S2-S7) backing R2-3, R2-4 and R3-4.

Emits one TSV per table (exact numbers, reusable) plus a self-contained HTML
document for review and for printing to the supplementary PDF.

Usage
-----
    python build_supplementary_tables.py --data DATA_DIR --out OUT_DIR

DATA_DIR must contain ``guild_scores.txt``, ``posebusters_validity.tsv`` and
``native_ligand_rmsd.tsv``, all from the 3-target rerun (see README.md).
``guild_scores.txt`` there already carries the ``pb_valid`` / ``pb_pose``
columns merged in by ``merge_posebusters_flags.py``.

Conventions
-----------
* AUC is computed on rp_* columns (already normalised per protein, so pooling
  is legitimate). 0 = best, so an active should rank lower.
* Boltz-2's affinity head has its own rp_boltz_affinity_score, ranked and
  voted like every other track since guild commit 0d36671.
* "Non-physical" counts non-negative values for Vina-family energies only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from guild.tools.scores import compute_rank_percentile_scores


def auc_ci(a, d):
    a = np.asarray(a, float)
    d = np.asarray(d, float)
    a, d = a[~np.isnan(a)], d[~np.isnan(d)]
    n_a, n_d = len(a), len(d)
    if n_a < 2 or n_d < 2:
        return np.nan, np.nan, np.nan, n_a, n_d
    r = pd.Series(np.concatenate([-a, -d])).rank(method="average").to_numpy()
    A = (r[:n_a].sum() - n_a * (n_a + 1) / 2) / (n_a * n_d)
    q1, q2 = A / (2 - A), 2 * A * A / (1 + A)
    v = (A * (1 - A) + (n_a - 1) * (q1 - A * A) + (n_d - 1) * (q2 - A * A)) / (n_a * n_d)
    s = np.sqrt(max(v, 0.0))
    return A, max(0.0, A - 1.96 * s), min(1.0, A + 1.96 * s), n_a, n_d


def ef(df, col, frac):
    """Enrichment factor over the pooled set, ranking by col (lower = better)."""
    sub = df[["ligand_category", col]].copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna(subset=[col])
    n, n_act = len(sub), (sub.ligand_category == ACT).sum()
    k = int(np.floor(n * frac))
    if k < 1 or n_act == 0:
        return np.nan
    top = sub.nsmallest(k, col)
    return ((top.ligand_category == ACT).sum() / k) / (n_act / n)


TARGETS = {"7v3z-A-9GF-A": "7v3z (CB1)", "6ot0-R-CO1-R": "6ot0 (SMO)", "8gdc-R-P2E-R": "8gdc (EP3)"}
ACT, DEC = "strong-binder", "decoy"

# (label, raw column, rp column or None, energy-like?)
METHODS = [
    ("AutoDock Vina",                 "vina_score",                   "rp_vina_score",                   True),
    ("GNINA",                         "gnina_score",                  "rp_gnina_score",                  True),
    ("KarmaDock",                     "karmadock_score",              "rp_karmadock_score",              False),
    ("DiffDock (confidence)",         "diffdock_score",               "rp_diffdock_score",               False),
    ("Boltz-2 (ipTM confidence)",     "boltz_score",                  "rp_boltz_score",                  False),
    ("Boltz-2 (affinity head)",       "boltz_affinity_score",         "rp_boltz_affinity_score",         False),
    ("Vina rescore of DiffDock pose", "vina_rescore_diffdock_score",  "rp_vina_rescore_diffdock_score",  True),
    ("GNINA rescore of DiffDock pose","gnina_rescore_diffdock_score", "rp_gnina_rescore_diffdock_score", True),
    ("Vina rescore of Boltz-2 pose",  "vina_rescore_boltz_score",     "rp_vina_rescore_boltz_score",     True),
    ("GNINA rescore of Boltz-2 pose", "gnina_rescore_boltz_score",    "rp_gnina_rescore_boltz_score",    True),
    ("Guild combined score",          None,                           "global_rp_score",                 False),
]

PB_LABEL = {"vina": "AutoDock Vina", "gnina": "GNINA", "diffdock": "DiffDock", "boltz": "Boltz-2"}

# Runtime constants (R3-4), measured separately -- see R3-4_runtime.tsv,
# produced by reproduce_response_analyses.py, for the case-study equivalents.
BOLTZ_PER_PAIR, DOCK_S, SCORE_S, PLIP_S = 99.0, 25369.1, 238.8, 384.0


def write(frame: pd.DataFrame, out_dir: Path, name: str) -> None:
    path = out_dir / name
    frame.to_csv(path, sep="\t", index=False)
    print(f"      -> {path.name}  ({frame.shape[0]}x{frame.shape[1]})")


def load_data(data: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(data / "guild_scores.txt", sep="\t", low_memory=False)
    scores = scores[scores.ligand_category != "native"].copy()

    # Recomputed through the shipped code rather than trusted from the file:
    # guild_scores.txt on disk predates both the median aggregation
    # (0e7c959) and the ranking of boltz_affinity_score (0d36671).
    scores = scores[[c for c in scores.columns
                     if not c.startswith(("rank_", "rp_")) and c != "global_rp_score"]]
    scores = compute_rank_percentile_scores(scores, methods=[
        "vina", "gnina", "karmadock", "diffdock", "boltz",
        "vina_rescore_diffdock", "gnina_rescore_diffdock",
        "vina_rescore_boltz", "gnina_rescore_boltz", "boltz_affinity",
    ])

    pb = pd.read_csv(data / "posebusters_validity.tsv", sep="\t", low_memory=False)
    rmsd = pd.read_csv(data / "native_ligand_rmsd.tsv", sep="\t", low_memory=False)
    return scores, pb, rmsd


def build_tables(scores: pd.DataFrame, pb: pd.DataFrame, rmsd: pd.DataFrame):
    pb_rate = pb.groupby("docking_method")["pb_valid"].agg(["mean", "size"])

    act_df = scores[scores.ligand_category == ACT]
    dec_df = scores[scores.ligand_category == DEC]
    N = len(scores)

    rows = []
    for label, raw, rp, energy in METHODS:
        rank_col = rp if rp else raw
        A, lo, hi, n_a, n_d = auc_ci(act_df[rank_col], dec_df[rank_col])
        if raw:
            v = pd.to_numeric(scores[raw], errors="coerce")
            missing = v.isna().sum()
            nonphys = int((v >= 0).sum()) if energy else None
        else:
            missing, nonphys = 0, None
        rows.append({
            "Method / track": label,
            "Pairs with a score": N - missing,
            "Missing (%)": round(100 * missing / N, 1),
            "Non-physical (%)": "" if nonphys is None else round(100 * nonphys / N, 1),
            "AUC": round(A, 3),
            "AUC 95% CI": f"{lo:.2f}–{hi:.2f}",
            "EF 10%": ("" if np.isnan(ef(scores, rank_col, 0.10)) else round(ef(scores, rank_col, 0.10), 2)),
            "EF 20%": ("" if np.isnan(ef(scores, rank_col, 0.20)) else round(ef(scores, rank_col, 0.20), 2)),
        })
    s1 = pd.DataFrame(rows)

    # Pose validity joined onto the five native methods only. Two different
    # denominators, both worth reporting: "pb_valid (%)" is per POSE (from
    # posebusters_validity.tsv); "Per-pair valid (%)" is per PAIR -- did any
    # validated pose for this method pass (<method>_pb_valid in guild_scores.txt,
    # written by merge_posebusters_flags.py).
    pv = []
    for name in ["vina", "gnina", "karmadock", "diffdock", "boltz"]:
        pair_col = f"{name}_pb_valid"
        if pair_col in scores.columns:
            pair_valid = scores[pair_col]
            n_pairs = int(pair_valid.notna().sum())
            per_pair_pct = round(100 * pair_valid.eq(True).sum() / n_pairs, 1) if n_pairs else "n/a"
        else:
            per_pair_pct = "n/a"
        if name in pb_rate.index:
            pv.append({"Method": PB_LABEL.get(name, name),
                       "Poses checked": int(pb_rate.loc[name, "size"]),
                       "pb_valid (%)": round(100 * pb_rate.loc[name, "mean"], 1),
                       "Per-pair valid (%)": per_pair_pct,
                       "Intramolecular (%)": round(100 * pb[pb.docking_method == name]["pb_intramolecular_valid"].mean(), 1),
                       "Intermolecular (%)": round(100 * pb[pb.docking_method == name]["pb_intermolecular_valid"].mean(), 1)})
        else:
            pv.append({"Method": "KarmaDock", "Poses checked": 0, "pb_valid (%)": "n/a",
                       "Per-pair valid (%)": per_pair_pct,
                       "Intramolecular (%)": "n/a", "Intermolecular (%)": "n/a"})
    s2 = pd.DataFrame(pv)

    # per-check pass rates, checks as rows
    checks = [c for c in pb.columns[12:34] if c not in ("mol_pred_loaded", "mol_cond_loaded", "complex_pdb")]
    s3rows = []
    for c in checks:
        row = {"Check": c.replace("_", " ")}
        for name in ["vina", "gnina", "diffdock", "boltz"]:
            sub = pb[pb.docking_method == name][c]
            sub = sub[sub.isin([True, False, "True", "False"])]
            row[PB_LABEL[name]] = round(100 * sub.astype(str).eq("True").mean(), 1) if len(sub) else ""
        s3rows.append(row)
    s3 = pd.DataFrame(s3rows)

    # per-target AUC
    s4rows = []
    for label, raw, rp, _energy in METHODS:
        rank_col = rp if rp else raw
        row = {"Method / track": label}
        for tid, tname in TARGETS.items():
            t = scores[scores.protein_config_id == tid]
            A, lo, hi, n_a, n_d = auc_ci(t[t.ligand_category == ACT][rank_col],
                                         t[t.ligand_category == DEC][rank_col])
            row[tname] = "" if np.isnan(A) else round(A, 2)
        A, lo, hi, _, _ = auc_ci(act_df[rank_col], dec_df[rank_col])
        row["Pooled"] = round(A, 3)
        s4rows.append(row)
    s4 = pd.DataFrame(s4rows)

    # redocking RMSD, wide
    rw = rmsd.pivot_table(index="protein_config_id", columns="method", values="rmsd", aggfunc="first")
    rw = rw.reindex(columns=["vina", "gnina", "karmadock", "diffdock", "boltz"])
    rw.index = [TARGETS.get(i, i) for i in rw.index]
    rw.columns = ["AutoDock Vina", "GNINA", "KarmaDock", "DiffDock", "Boltz-2"]
    s5 = rw.round(2).reset_index().rename(columns={"index": "Target"})

    # runtime
    boltz_total = BOLTZ_PER_PAIR * N
    s6 = pd.DataFrame([
        {"Stage / method": "Docking, all five methods + both rescore tracks", "Seconds": round(DOCK_S, 1),
         "Per pair (s)": round(DOCK_S / N, 1), "Note": f"{N} pairs, one NVIDIA A100 80GB PCIe"},
        {"Stage / method": "  of which Boltz-2 (measured)", "Seconds": round(boltz_total, 1),
         "Per pair (s)": round(BOLTZ_PER_PAIR, 1), "Note": "includes template-retry overhead"},
        {"Stage / method": "  of which all other tracks (derived)", "Seconds": round(DOCK_S - boltz_total, 1),
         "Per pair (s)": round((DOCK_S - boltz_total) / N, 1),
         "Note": "aggregate; per-method split needs the per-method logs"},
        {"Stage / method": "Guild scoring / rank-percentile aggregation", "Seconds": round(SCORE_S, 1),
         "Per pair (s)": round(SCORE_S / N, 2), "Note": ""},
        {"Stage / method": "PLIP interaction analysis", "Seconds": round(PLIP_S, 1),
         "Per pair (s)": round(PLIP_S / N, 2), "Note": ""},
        {"Stage / method": "Total", "Seconds": round(DOCK_S + SCORE_S + PLIP_S, 1),
         "Per pair (s)": round((DOCK_S + SCORE_S + PLIP_S) / N, 1), "Note": "≈ 7.2 h wall-clock"},
        {"Stage / method": "Boltz-2 per-target preprocessing / MSA", "Seconds": "60–240",
         "Per pair (s)": "one-off", "Note": "cached after first use for a given target"},
    ])

    # Tags start at S2, not S1: the manuscript's own Supplementary Table 1 is
    # the pre-existing GPCR target list, which predates all of this and isn't
    # built here. Table S10 (parameter provenance) isn't built here either.
    return [
        ("S2", "Per-method screening performance", s1,
         "Three targets, 165 protein–ligand pairs (15 known binders, 150 property-matched decoys). "
         "AUC is binder-vs-decoy discrimination pooled across targets on the rank-percentile scale. "
         "Confidence intervals are Hanley–McNeil; they are wide because there are 15 binders. "
         "The combined score is the median across pose sources, with Boltz-2's affinity head folded into its own pose source (guild 0d36671). EF baselines are 1.0. Non-physical counts non-negative energies for Vina-family scores only."),
        ("S3", "Pose validity by method", s2,
         "PoseBusters, config “dock”. “pb_valid (%)” is per pose, all poses per pair; "
         "“Per-pair valid (%)” is per pair -- did any validated pose for this method pass "
         "(<method>_pb_valid in guild_scores.txt). KarmaDock emits no complex structure, so "
         "no pose was available to check — this is a coverage gap, not a pass."),
        ("S4", "Pose validity, individual checks", s3,
         "Percentage of poses passing each check. Separates internal geometry from receptor fit."),
        ("S5", "Binder-vs-decoy AUC per target", s4,
         "Five binders against 50 decoys per target, so per-target values are imprecise; the pooled "
         "column is the one to read."),
        ("S6", "Native-ligand redocking RMSD (Å)", s5,
         "Each target's own co-crystal ligand redocked into its own structure. n = 1 per method per "
         "target — illustrative, not a pose-accuracy measurement. RMSD computed after Kabsch "
         "superposition, since receptor preparation re-frames coordinates per method."),
        ("S7", "Runtime", s6,
         "Single NVIDIA A100 80GB PCIe. Boltz-2's per-pair cost is measured from its own per-pair "
         "logs; the remaining tracks are given as an aggregate because the run log records batch "
         "boundaries rather than per-method timings."),
    ]


def to_html(df_: pd.DataFrame) -> str:
    head = "".join(f"<th>{c}</th>" for c in df_.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{'' if pd.isna(v) else v}</td>" for v in r) + "</tr>"
        for r in df_.itertuples(index=False)
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_html(tables, out: Path) -> None:
    parts = ["""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Guild Supplementary Tables</title><style>
:root{--paper:#F5F6F8;--surface:#fff;--ink:#171A20;--ink-2:#48505E;--ink-3:#6E7787;--rule:#DCE0E6;--rule-strong:#C3C9D2;--accent:#1F6B45}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#111419;--surface:#191D24;--ink:#E7EAEF;--ink-2:#AEB6C2;--ink-3:#848D9B;--rule:#2B313A;--rule-strong:#3A424E;--accent:#63C293}}
:root[data-theme="dark"]{--paper:#111419;--surface:#191D24;--ink:#E7EAEF;--ink-2:#AEB6C2;--ink-3:#848D9B;--rule:#2B313A;--rule-strong:#3A424E;--accent:#63C293}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;font-size:15px;line-height:1.6}
.wrap{max-width:1040px;margin:0 auto;padding:40px 20px 72px}
h1{font-family:Georgia,serif;font-size:clamp(28px,5vw,38px);line-height:1.15;margin:0 0 10px}
.standfirst{color:var(--ink-2);max-width:66ch;margin:0 0 28px}
h2{font-family:Georgia,serif;font-size:21px;margin:40px 0 4px;padding-top:18px;border-top:2px solid var(--rule-strong)}
h2 .tag{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:12px;color:var(--accent);letter-spacing:.08em;margin-right:10px}
.note{color:var(--ink-2);font-size:13.5px;max-width:78ch;margin:0 0 14px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid var(--rule);border-radius:3px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:520px}
th,td{padding:7px 11px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap}
th{background:var(--paper);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink-3);position:sticky;top:0}
td:first-child,th:first-child{text-align:left;white-space:normal;min-width:180px}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:var(--paper)}
@media(max-width:600px){body{font-size:14px}th,td{padding:6px 8px}}
</style></head><body><div class="wrap">
<h1>Guild &mdash; supplementary tables</h1>
<p class="standfirst">Generated from the three-target, 165-pair benchmark run
(<code>small-example-all-methods</code>, guild commit <code>20b3c50</code>). These back the replies
to R2-3, R2-4 and R3-4. Exact values are in the matching <code>Table_S*.tsv</code> files.</p>"""]

    for tag, title, df_, note in tables:
        parts.append(
            f'<h2><span class="tag">{tag}</span>{title}</h2>'
            f'<p class="note">{note}</p><div class="scroll">{to_html(df_)}</div>'
        )
    parts.append("</div></body></html>")

    path = out / "supplementary_tables.html"
    path.write_text("".join(parts), encoding="utf-8")
    print(f"\nwrote {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("./data"),
                        help="directory holding guild_scores.txt, posebusters_validity.tsv "
                             "and native_ligand_rmsd.tsv (the 3-target rerun)")
    parser.add_argument("--out", type=Path, default=Path("./output"),
                        help="directory for the TSVs and HTML (created if absent)")
    args = parser.parse_args(argv)

    if not args.data.is_dir():
        parser.error(f"--data {args.data} is not a directory")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"data: {args.data.resolve()}\nout:  {args.out.resolve()}")

    scores, pb, rmsd = load_data(args.data)
    tables = build_tables(scores, pb, rmsd)

    for tag, title, df_, _note in tables:
        write(df_, args.out, f"Table_{tag}.tsv")
        print(f"Table {tag}  {title}")

    write_html(tables, args.out)
    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
