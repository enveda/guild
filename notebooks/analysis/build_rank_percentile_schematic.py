#!/usr/bin/env python3
"""Generate the Supplementary Text 3 rank-percentile schematic from real data.

The previous version of this diagram
(``.../guild_publication/images/guild_rank_percentile_figure.png``, embedded
uncaptioned in Supplementary Text 3) was hand-built with illustrative rather
than real histograms, and nothing in the repository produced it. Two things
were wrong with it, both fixed here by construction rather than by hand:

- it showed four pose sources (Vina, DiffDock, KarmaDock, Boltz-2); GNINA was
  missing entirely.
- its DiffDock panel was labelled "confidence (higher = better)" -- exactly
  the input R2-3 objects to, and what the rescore tracks and commit
  ``20b3c50`` stopped letting vote in ``global_rp_score``.

This script drives the same four-step layout from the real 3-target rerun
(``guild_scores.txt``) instead of invented distributions, so it cannot drift
out of step with the scoring code again. Rank percentiles are recomputed via
``guild.tools.scores.compute_rank_percentile_scores`` rather than trusted from
the file, since the rerun predates both ``boltz_affinity_score``'s ranking
(``0d36671``) and the median aggregation default (``0e7c959``).

Usage
-----
    python build_rank_percentile_schematic.py --data <rerun_dir> --out <out_dir>

``--data`` must contain ``guild_scores.txt`` (the 3-target rerun with all five
methods and both rescore tracks -- see notebooks/analysis/README.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from guild.constants.guild import (  # noqa: E402
    BOLTZ_AFFINITY_PREFIX,
    BOLTZ_PREFIX,
    DIFFDOCK_PREFIX,
    GNINA_PREFIX,
    KARMADOCK_PREFIX,
    PROTEIN_CONF_ID,
    VINA_PREFIX,
    VINA_RESCORE_BOLTZ_PREFIX,
    VINA_RESCORE_DIFFDOCK_PREFIX,
)
from guild.tools.scores import (  # noqa: E402
    compute_rank_percentile_scores,
    is_physical_score,
)

# One example ligand, highlighted consistently across every panel -- mirrors
# the single dashed "ligand" line in the original hand-built figure. Chosen
# because it has a valid, physical value for every pose source shown here.
EXAMPLE_LIGAND_ID = "CHEMBL2541637"

# Same palette as score_distribution.ipynb's METHOD_COLOR, so this schematic
# and Figure 2 read as the same five pose sources rather than two different
# colour keys for the same thing.
POSE_SOURCE_COLOR = {
    "Vina": "#1f77b4",
    "GNINA": "#9467bd",
    "KarmaDock": "#8c564b",
    "DiffDock": "#ff7f0e",
    "Boltz-2": "#2ca02c",
}

# (label, raw_col, physicality-check prefix or None, x-axis direction label,
#  step-4 rp column(s) -- a Boltz-2-style list means "this pose source's vote
#  is the mean of these tracks", matching guild's own within-source
#  combination; a single string means the vote is that one column directly.
#  `note` is the reported-but-not-voting annotation for a native confidence.)
POSE_SOURCES = [
    {
        "label": "Vina", "raw_col": "vina_score", "physical_prefix": VINA_PREFIX,
        "direction": "kcal/mol (lower = better)", "rp_cols": "rp_vina_score", "note": None,
    },
    {
        "label": "GNINA", "raw_col": "gnina_score", "physical_prefix": GNINA_PREFIX,
        "direction": "kcal/mol (lower = better)", "rp_cols": "rp_gnina_score", "note": None,
    },
    {
        "label": "KarmaDock", "raw_col": "karmadock_score", "physical_prefix": None,
        "direction": "score (higher = better)", "rp_cols": "rp_karmadock_score", "note": None,
    },
    {
        "label": "DiffDock", "raw_col": "vina_rescore_diffdock_score",
        "physical_prefix": VINA_RESCORE_DIFFDOCK_PREFIX,
        "direction": "Vina rescore, kcal/mol (lower = better)",
        "rp_cols": "rp_vina_rescore_diffdock_score",
        "note": "diffdock_score (native confidence)\nis reported, not ranked or voted",
    },
    {
        "label": "Boltz-2", "raw_col": "vina_rescore_boltz_score",
        "physical_prefix": VINA_RESCORE_BOLTZ_PREFIX,
        "direction": "Vina rescore, kcal/mol (lower = better)",
        "rp_cols": ["rp_vina_rescore_boltz_score", "rp_boltz_affinity_score"],
        "note": ("boltz_score (native confidence) is reported, not ranked or voted;\n"
                 "boltz_affinity_score also joins this vote (mean within source)"),
    },
]

METHODS_TO_RANK = [
    VINA_PREFIX, GNINA_PREFIX, KARMADOCK_PREFIX, DIFFDOCK_PREFIX, BOLTZ_PREFIX,
    VINA_RESCORE_DIFFDOCK_PREFIX, VINA_RESCORE_BOLTZ_PREFIX, BOLTZ_AFFINITY_PREFIX,
]


def _load(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "guild_scores.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- this schematic needs the 3-target rerun's guild_scores.txt "
            "(see notebooks/analysis/README.md), not the Zenodo deposit or any other file."
        )
    df = pd.read_csv(path, sep="\t")
    df = df[df["ligand_category"] != "native"].copy()

    # Nulled before ranking, not dropped -- the same treatment as
    # score_distribution.ipynb's Figure 2 rebuild, and for the same reason:
    # vina_rescore_diffdock_score has a cluster sitting at exactly 0.0 (a
    # rescoring-failure sentinel) in addition to genuinely positive values,
    # neither of which is a real ΔG.
    for spec in POSE_SOURCES:
        prefix = spec["physical_prefix"]
        if prefix is None:
            continue
        col = spec["raw_col"]
        is_bad = ~df[col].apply(
            lambda v, p=prefix: (pd.isna(v) or (is_physical_score(v, p) and v < 0.0))
        )
        df.loc[is_bad, col] = np.nan

    df = compute_rank_percentile_scores(df, methods=METHODS_TO_RANK, protein_col=PROTEIN_CONF_ID)
    return df


def _pose_source_vote(df: pd.DataFrame, rp_cols) -> pd.Series:
    """This pose source's one vote: the column itself, or the mean of a list
    of tracks -- guild's own within-source combination (a mean, regardless of
    the across-source aggregation mode), simplified here to just the tracks
    this schematic shows (see the module docstring on gnina_rescore_*)."""
    if isinstance(rp_cols, str):
        return df[rp_cols]
    return df[rp_cols].mean(axis=1)


def _draw_box(ax, xy, w, h, text, facecolor, edgecolor="none", fontcolor="white", fontsize=13):
    box = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
             fontsize=fontsize, fontweight="bold", color=fontcolor, wrap=True)


def _draw_arrow(ax, x, y0, y1):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops={"arrowstyle": "-|>", "color": "#444444", "linewidth": 1.5})


def build_figure(df: pd.DataFrame) -> plt.Figure:
    n = len(POSE_SOURCES)
    example = df[df["ligand_id"] == EXAMPLE_LIGAND_ID]
    if len(example) != 1:
        raise ValueError(
            f"Expected exactly one row for {EXAMPLE_LIGAND_ID!r}, found {len(example)}"
        )
    example = example.iloc[0]

    fig = plt.figure(figsize=(3.0 * n, 16.5))
    gs = fig.add_gridspec(
        6, 1, height_ratios=[0.6, 0.9, 0.9, 2.6, 1.3, 2.6], hspace=0.65,
    )

    # ── Title ─────────────────────────────────────────────────────────────
    # ax.axis("off") hides the axes' own background patch along with its
    # spines/ticks, so set_facecolor alone would be invisible here -- the
    # navy bar has to be an explicitly added patch instead.
    ax_title = fig.add_subplot(gs[0])
    ax_title.set_xlim(0, 1)
    ax_title.set_ylim(0, 1)
    ax_title.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax_title.transAxes,
                                            facecolor="#1b2a41", zorder=0))
    ax_title.text(0.5, 0.5, "Guild rank-percentile standardization",
                   ha="center", va="center", fontsize=20, fontweight="bold", color="white")
    ax_title.axis("off")

    # ── Step 1: inputs ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.text(0.01, 0.85, "STEP 1 -- INPUTS", fontsize=13, fontweight="bold", color="#1b2a41")
    n_targets = df[PROTEIN_CONF_ID].str.split("-").str[0].nunique()
    n_binders = int((df["ligand_category"] == "strong-binder").sum())
    n_decoys = int((df["ligand_category"] == "decoy").sum())
    inputs = [
        ("Protein target", f"{n_targets} PDB structures", "#1f77b4"),
        ("Ligands of interest", f"{n_binders} known binders (SMILES)", "#e07b39"),
        ("Decoy panel", f"{n_decoys} expected non-binders", "#888888"),
    ]
    box_w = 0.9 / len(inputs)
    for i, (title, subtitle, color) in enumerate(inputs):
        x0 = 0.05 + i * (box_w + 0.02)
        _draw_box(ax1, (x0, 0.1), box_w - 0.02, 0.55, f"{title}\n{subtitle}",
                   facecolor="white", edgecolor=color, fontcolor="#222222", fontsize=10)

    # ── Step 2: pose sources ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.text(0.01, 0.85, "STEP 2 -- SCORE WITH EACH PLBP METHOD", fontsize=13,
              fontweight="bold", color="#1b2a41")
    box_w = 0.94 / n
    for i, spec in enumerate(POSE_SOURCES):
        x0 = 0.03 + i * box_w
        _draw_box(ax2, (x0 + 0.01, 0.1), box_w - 0.02, 0.55, spec["label"],
                   facecolor=POSE_SOURCE_COLOR[spec["label"]], fontsize=13)

    # ── Step 3: raw scores ────────────────────────────────────────────────────
    gs3 = gs[3].subgridspec(1, n, wspace=0.35)
    for i, spec in enumerate(POSE_SOURCES):
        ax = fig.add_subplot(gs3[i])
        color = POSE_SOURCE_COLOR[spec["label"]]
        vals = df[spec["raw_col"]].dropna()
        ax.hist(vals, bins=12, color=color, alpha=0.55, edgecolor=color)
        ax.axvline(example[spec["raw_col"]], color="#d62728", linestyle="--", linewidth=1.5)
        ax.text(example[spec["raw_col"]], ax.get_ylim()[1] * 0.92, "ligand",
                 color="#d62728", fontsize=8, ha="center", fontweight="bold")
        ax.set_title(spec["label"], color=color, fontsize=11, fontweight="bold")
        ax.set_xlabel(spec["direction"], fontsize=7.5)
        ax.set_yticks([])
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        if spec["note"]:
            ax.text(0.5, -0.55, spec["note"], transform=ax.transAxes, fontsize=6,
                     ha="center", va="top", style="italic", color="#555555")
        if i == 0:
            ax.text(-0.35, 1.15, "STEP 3 -- RAW SCORES (DIFFERENT SCALES, NOT COMPARABLE)",
                     transform=ax.transAxes, fontsize=12, fontweight="bold", color="#1b2a41")

    # ── Standardize arrow ─────────────────────────────────────────────────────
    ax_std = fig.add_subplot(gs[4])
    ax_std.set_xlim(0, 1)
    ax_std.set_ylim(0, 1)
    ax_std.axis("off")
    _draw_box(ax_std, (0.40, 0.62), 0.20, 0.28, "STANDARDIZE", facecolor="#1b2a41", fontsize=11)
    ax_std.text(
        0.5, 0.35,
        "Rank within each protein -> rank / N -> combine each pose source's tracks by MEAN,\n"
        "then combine pose sources by MEDIAN across sources (0e7c959) -- not a flat mean.",
        ha="center", va="center", fontsize=9, style="italic", color="#333333",
    )

    # ── Step 4: rank percentile ───────────────────────────────────────────────
    gs4 = gs[5].subgridspec(1, n, wspace=0.35)
    for i, spec in enumerate(POSE_SOURCES):
        ax = fig.add_subplot(gs4[i])
        color = POSE_SOURCE_COLOR[spec["label"]]
        vote = _pose_source_vote(df, spec["rp_cols"])
        vals = vote.dropna()
        ax.hist(vals, bins=12, range=(0, 1), color=color, alpha=0.55, edgecolor=color)
        example_vote = _pose_source_vote(df.loc[[example.name]], spec["rp_cols"]).iloc[0]
        ax.axvline(example_vote, color="#d62728", linestyle="--", linewidth=1.5)
        pct = 100 * (1 - example_vote)  # 0 = best -> report as a percentile, higher = better
        ax.text(example_vote, ax.get_ylim()[1] * 0.85, f"P = {pct:.0f}%", color="white",
                 fontsize=8, ha="center", fontweight="bold",
                 bbox={"boxstyle": "round,pad=0.2", "facecolor": "#d62728", "edgecolor": "none"})
        n_votes = 1 if isinstance(spec["rp_cols"], str) else len(spec["rp_cols"])
        title = spec["label"] if n_votes == 1 else f"{spec['label']} (1 vote, mean of {n_votes} tracks)"
        ax.set_title(title, color=color, fontsize=10, fontweight="bold")
        ax.set_xlabel("rank percentile", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        if i == 0:
            ax.text(-0.35, 1.15, "STEP 4 -- RANK PERCENTILE (UNIFIED SCALE, ONE VOTE PER POSE SOURCE)",
                     transform=ax.transAxes, fontsize=12, fontweight="bold", color="#1b7a41")

    fig.suptitle(
        f"Built from the {n_targets}-target rerun -- ligand {EXAMPLE_LIGAND_ID} highlighted "
        "throughout as a worked example. Schematic: clarity over completeness.",
        fontsize=9, color="#777777", y=0.005,
    )
    return fig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data"),
                         help="directory holding guild_scores.txt (the 3-target rerun)")
    parser.add_argument("--out", type=Path, default=Path("np_synthetic_comparison/guild_figures"),
                         help="directory for the rendered schematic")
    args = parser.parse_args(argv)

    if not args.data.is_dir():
        parser.error(f"--data {args.data} is not a directory")
    args.out.mkdir(parents=True, exist_ok=True)

    df = _load(args.data)
    print(f"Loaded {len(df)} rows from {args.data / 'guild_scores.txt'}")

    fig = build_figure(df)
    for ext in ("png", "pdf", "svg"):
        out_path = args.out / f"rank_percentile_schematic.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
