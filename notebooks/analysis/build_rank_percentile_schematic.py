#!/usr/bin/env python3
"""Generate the Supplementary Text 3 rank-percentile schematic from real data.

Replaces the old hand-built diagram (illustrative histograms, missing GNINA,
DiffDock mislabelled "confidence (higher = better)") with one driven by the
real 3-target rerun (``guild_scores.txt``), so it can't drift from the scoring
code again. Rank percentiles are recomputed via
``compute_rank_percentile_scores`` rather than trusted from the file, since
the rerun predates ``boltz_affinity_score``'s ranking (``0d36671``) and the
median aggregation default (``0e7c959``).

Usage: python build_rank_percentile_schematic.py --data <rerun_dir> --out <out_dir>
``--data`` must contain ``guild_scores.txt``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kde_helpers import kde_curve, kde_curve_bounded  # noqa: E402

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

# Highlighted across every panel; has a valid, physical value for all five sources.
EXAMPLE_LIGAND_ID = "CHEMBL2541637"

# Matches score_distribution.ipynb's METHOD_COLOR.
POSE_SOURCE_COLOR = {
    "Vina": "#1f77b4",
    "GNINA": "#9467bd",
    "KarmaDock": "#8c564b",
    "DiffDock": "#ff7f0e",
    "Boltz-2": "#2ca02c",
}

# rp_cols: a single column, or a list meaning "vote is the mean of these tracks".
# DiffDock/Boltz-2 report a native confidence (diffdock_score/boltz_score) that isn't
# ranked or voted -- true of gnina_cnn_score too. Worth stating in code, not on the figure.
POSE_SOURCES = [
    {
        "label": "Vina", "raw_col": "vina_score", "physical_prefix": VINA_PREFIX,
        "direction": "kcal/mol (lower = better)", "rp_cols": "rp_vina_score",
    },
    {
        "label": "GNINA", "raw_col": "gnina_score", "physical_prefix": GNINA_PREFIX,
        "direction": "kcal/mol (lower = better)", "rp_cols": "rp_gnina_score",
    },
    {
        "label": "KarmaDock", "raw_col": "karmadock_score", "physical_prefix": None,
        "direction": "score (higher = better)", "rp_cols": "rp_karmadock_score",
    },
    {
        "label": "DiffDock", "raw_col": "vina_rescore_diffdock_score",
        "physical_prefix": VINA_RESCORE_DIFFDOCK_PREFIX,
        "direction": "Vina rescore, kcal/mol (lower = better)",
        "rp_cols": "rp_vina_rescore_diffdock_score",
    },
    {
        "label": "Boltz-2", "raw_col": "vina_rescore_boltz_score",
        "physical_prefix": VINA_RESCORE_BOLTZ_PREFIX,
        "direction": "Vina rescore, kcal/mol (lower = better)",
        "rp_cols": ["rp_vina_rescore_boltz_score", "rp_boltz_affinity_score"],
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

    # Nulled before ranking (same treatment as score_distribution.ipynb): a
    # rescore column at exactly 0.0 or positive is a failure sentinel, not a real dG.
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
    """This pose source's one vote: the column itself, or the mean of a list of tracks."""
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


def _draw_step_band(fig, subplotspec, facecolor, heading, heading_color):
    """A tinted rounded band spanning one STEP row, with its heading inside it."""
    ax_bg = fig.add_subplot(subplotspec)
    ax_bg.set_xlim(0, 1)
    ax_bg.set_ylim(0, 1)
    ax_bg.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, transform=ax_bg.transAxes,
        boxstyle="round,pad=0,rounding_size=0.02",
        facecolor=facecolor, edgecolor="none", zorder=0,
    ))
    ax_bg.text(0.01, 0.93, heading, fontsize=13, fontweight="bold",
               color=heading_color, va="top", transform=ax_bg.transAxes)
    ax_bg.axis("off")
    return ax_bg


def _draw_step_band_split(fig, subplotspec, facecolor, heading, heading_color, header_frac=0.3):
    """Same as `_draw_step_band`, but carves off a header strip first so full-height
    content panels (Step 3/4's cards) don't paint over the heading. Returns the
    SubplotSpec for the row underneath, for the caller to subdivide into columns."""
    ax_bg = fig.add_subplot(subplotspec)
    ax_bg.set_xlim(0, 1)
    ax_bg.set_ylim(0, 1)
    ax_bg.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, transform=ax_bg.transAxes,
        boxstyle="round,pad=0,rounding_size=0.02",
        facecolor=facecolor, edgecolor="none", zorder=0,
    ))
    ax_bg.text(0.01, 0.95, heading, fontsize=13, fontweight="bold",
               color=heading_color, va="top", transform=ax_bg.transAxes)
    ax_bg.axis("off")
    rows = subplotspec.subgridspec(2, 1, height_ratios=[header_frac, 1 - header_frac], hspace=0.0)
    return rows[1]


def _draw_card_bg(ax, edgecolor="#d8d8d8"):
    """A white rounded card behind one distribution panel. Hides the default
    square axes background first so it doesn't show past the rounded corners."""
    ax.patch.set_visible(False)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, transform=ax.transAxes,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor="white", edgecolor=edgecolor, linewidth=1.0, zorder=-1,
    ))


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

    # axis("off") hides the axes' own background patch, so the navy bar needs its
    # own explicit patch rather than set_facecolor.
    ax_title = fig.add_subplot(gs[0])
    ax_title.set_xlim(0, 1)
    ax_title.set_ylim(0, 1)
    ax_title.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax_title.transAxes,
                                            facecolor="#1b2a41", zorder=0))
    ax_title.text(0.5, 0.5, "Guild rank-percentile standardization",
                   ha="center", va="center", fontsize=20, fontweight="bold", color="white")
    ax_title.axis("off")

    # ── Step 1: inputs ──
    ax1 = _draw_step_band(fig, gs[1], "#eaf1f8", "STEP 1 — INPUTS", "#1b2a41")
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

    # ── Step 2: pose sources ──
    ax2 = _draw_step_band(fig, gs[2], "#eaf1f8", "STEP 2 — SCORE WITH EACH PLBP METHOD", "#1b2a41")
    box_w = 0.94 / n
    chip_gap = 0.018
    for i, spec in enumerate(POSE_SOURCES):
        x0 = 0.03 + i * box_w
        _draw_box(ax2, (x0 + chip_gap, 0.1), box_w - 2 * chip_gap, 0.55, spec["label"],
                   facecolor=POSE_SOURCE_COLOR[spec["label"]], fontsize=13)

    # ── Step 3: raw scores, drawn as KDE curves (same treatment as Figure 2) ──
    step3_content = _draw_step_band_split(
        fig, gs[3], "#eef1f4", "STEP 3 — RAW SCORES (DIFFERENT SCALES, NOT COMPARABLE)", "#1b2a41",
    )
    gs3 = step3_content.subgridspec(1, n, wspace=0.35)
    for i, spec in enumerate(POSE_SOURCES):
        ax = fig.add_subplot(gs3[i])
        _draw_card_bg(ax)
        color = POSE_SOURCE_COLOR[spec["label"]]
        vals = df[spec["raw_col"]].dropna()
        x_grid = np.linspace(vals.min(), vals.max(), 300)
        curve = kde_curve(vals, x_grid)
        ax.fill_between(x_grid, 0, curve, color=color, alpha=0.55, linewidth=0)
        ax.plot(x_grid, curve, color=color, linewidth=1.6)
        # Headroom above the curve's own peak, not just a fixed fraction of the
        # axes -- otherwise "ligand" sits on top of the fill wherever the dashed
        # line lands near a peak (KarmaDock/DiffDock/Boltz-2 all did).
        ax.set_ylim(0, curve.max() * 1.4)
        example_x = example[spec["raw_col"]]
        ax.axvline(example_x, color="#d62728", linestyle="--", linewidth=1.5)
        label_y = np.interp(example_x, x_grid, curve) + curve.max() * 0.18
        ax.text(example_x, label_y, "ligand",
                 color="#d62728", fontsize=8, ha="center", va="bottom", fontweight="bold")
        ax.set_title(spec["label"], color=color, fontsize=11, fontweight="bold")
        ax.set_xlabel(spec["direction"], fontsize=7.5)
        ax.set_yticks([])
        for spine in ("top", "right", "left", "bottom"):
            ax.spines[spine].set_visible(False)

    # ── Standardize arrow ──
    ax_std = fig.add_subplot(gs[4])
    ax_std.set_xlim(0, 1)
    ax_std.set_ylim(0, 1)
    ax_std.axis("off")
    _draw_box(ax_std, (0.40, 0.62), 0.20, 0.28, "STANDARDIZE", facecolor="#1b2a41", fontsize=11)
    ax_std.text(
        0.5, 0.35,
        "Rank within each protein -> rank / N -> combine each pose source's tracks by MEAN,\n"
        "then combine pose sources by MEDIAN across sources — not a flat mean.",
        ha="center", va="center", fontsize=9, style="italic", color="#333333",
    )

    # ── Step 4: rank percentile ──
    # Plotted as 1 - vote (higher = better), matching Figure 2/3's convention --
    # the stored column is 0 = best, so the axis and the "P = X%" label must
    # both be flipped together, not just the label.
    step4_content = _draw_step_band_split(
        fig, gs[5], "#e8f5ee", "STEP 4 — RANK PERCENTILE (UNIFIED SCALE, ONE VOTE PER POSE SOURCE)", "#1b7a41",
    )
    gs4 = step4_content.subgridspec(1, n, wspace=0.35)
    rp_x_grid = np.linspace(0.0, 1.0, 300)
    for i, spec in enumerate(POSE_SOURCES):
        ax = fig.add_subplot(gs4[i])
        _draw_card_bg(ax)
        color = POSE_SOURCE_COLOR[spec["label"]]
        vote = _pose_source_vote(df, spec["rp_cols"])
        vals = (1 - vote).dropna()
        curve = kde_curve_bounded(vals, rp_x_grid)
        ax.fill_between(rp_x_grid, 0, curve, color=color, alpha=0.55, linewidth=0)
        ax.plot(rp_x_grid, curve, color=color, linewidth=1.6)
        example_vote = _pose_source_vote(df.loc[[example.name]], spec["rp_cols"]).iloc[0]
        example_display = 1 - example_vote
        ax.axvline(example_display, color="#d62728", linestyle="--", linewidth=1.5)
        pct = 100 * example_display
        ax.text(example_display, ax.get_ylim()[1] * 0.85, f"P = {pct:.0f}%", color="white",
                 fontsize=8, ha="center", fontweight="bold",
                 bbox={"boxstyle": "round,pad=0.2", "facecolor": "#d62728", "edgecolor": "none"})
        n_votes = 1 if isinstance(spec["rp_cols"], str) else len(spec["rp_cols"])
        title = spec["label"] if n_votes == 1 else f"{spec['label']} (1 vote, mean of {n_votes} tracks)"
        ax.set_title(title, color=color, fontsize=10, fontweight="bold")
        ax.set_xlabel("rank percentile (higher = better)", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        for spine in ("top", "right", "left", "bottom"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        f"Built from the {n_targets}-target rerun — ligand {EXAMPLE_LIGAND_ID} highlighted "
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
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
