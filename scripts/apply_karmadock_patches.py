"""
Apply the KarmaDock compatibility patches documented in the project README.

KarmaDock is pinned at commit 9a35d0c and was written against an older RDKit
version. With the rdkit/torch versions guild uses today, two adjustments are
required to avoid dimension-mismatch crashes during ligand-feature generation
and inside the GraphTransformer block:

  1. KarmaDock/dataset/ligand_feature.py — hard-coded ``20`` feature columns is
     replaced with the actual ``edge_feature.size(1)``.
  2. KarmaDock/architecture/GraphTransformer_Block.py — the edge encoder is
     guarded so an unexpected input width is truncated or padded to match
     ``self.edge_encoder.in_features``.

Idempotent: re-running the script after a patch is already applied is a no-op.

Usage:
    python apply_karmadock_patches.py /app/KarmaDock
"""

import re
import sys
from pathlib import Path

LIGAND_FEATURE_ALREADY = "feat_dim = edge_feature.size(1)"
# Indentation-aware: the original snippet appears at multiple indentation
# levels (function body and inside ``if`` blocks).
LIGAND_FEATURE_RE = re.compile(
    r"^(?P<indent>[ \t]+)edge_feature_new = torch\.zeros\(\(edge_index_new\.size\(1\), 20\)\)\n"
    r"(?P=indent)edge_feature_new\[:, \[4, 5, 18\]\] = 1\n",
    flags=re.MULTILINE,
)


def _ligand_feature_replacement(match: "re.Match[str]") -> str:
    indent = match.group("indent")
    return (
        f"{indent}feat_dim = edge_feature.size(1)\n"
        f"{indent}edge_feature_new = torch.zeros(\n"
        f"{indent}    (edge_index_new.size(1), feat_dim),\n"
        f"{indent}    dtype=edge_feature.dtype,\n"
        f"{indent}    device=edge_feature.device,\n"
        f"{indent})\n"
    )


GT_BLOCK_TARGET = "edge_feats = self.edge_encoder(edge_s)"
# Lines of the guard block we splice in *before* the target line. Each line
# carries its relative indentation as leading tabs (KarmaDock imports
# ``torch as th`` and indents with tabs throughout). The captured target-line
# indentation is then prepended to every line so the splice matches whatever
# nesting depth ``edge_feats = self.edge_encoder(edge_s)`` happens to sit at.
GT_BLOCK_GUARD_LINES = [
    "if edge_s.size(1) > self.edge_encoder.in_features:",
    "\tedge_s = edge_s[:, :self.edge_encoder.in_features]",
    "elif edge_s.size(1) < self.edge_encoder.in_features:",
    "\tpad = th.zeros(",
    "\t\tedge_s.size(0),",
    "\t\tself.edge_encoder.in_features - edge_s.size(1),",
    "\t\tdevice=edge_s.device,",
    "\t\tdtype=edge_s.dtype,",
    "\t)",
    "\tedge_s = th.cat([edge_s, pad], dim=1)",
]
GT_BLOCK_ALREADY = "edge_s = edge_s[:, :self.edge_encoder.in_features]"


def _patch_ligand_feature(path: Path) -> int:
    text = path.read_text()
    if LIGAND_FEATURE_ALREADY in text:
        print("  [skip] ligand_feature.py already patched")
        return 0
    new_text, n = LIGAND_FEATURE_RE.subn(_ligand_feature_replacement, text)
    if n == 0:
        raise RuntimeError(
            f"Expected pattern not found in {path}; KarmaDock layout may have "
            f"changed and the patch needs updating."
        )
    path.write_text(new_text)
    print(f"  [applied] ligand_feature.py — {n} replacement(s)")
    return n


def _patch_gt_block(path: Path) -> int:
    text = path.read_text()
    if GT_BLOCK_ALREADY in text:
        print("  [skip] GraphTransformer_Block.py already patched")
        return 0
    if GT_BLOCK_TARGET not in text:
        raise RuntimeError(
            f"Expected line '{GT_BLOCK_TARGET}' not found in {path}; "
            f"KarmaDock layout may have changed and the patch needs updating."
        )
    # Match the indentation of the target line so we splice cleanly.
    line_re = re.compile(
        r"^(?P<indent>[ \t]*)edge_feats\s*=\s*self\.edge_encoder\(edge_s\)$",
        flags=re.MULTILINE,
    )
    match = line_re.search(text)
    if not match:
        raise RuntimeError(
            f"Target line found but indentation could not be matched in {path}"
        )
    indent = match.group("indent")
    guard = "\n".join(indent + line for line in GT_BLOCK_GUARD_LINES)
    new_text = (
        text[: match.start()]
        + guard
        + "\n"
        + match.group(0)
        + text[match.end():]
    )
    path.write_text(new_text)
    print("  [applied] GraphTransformer_Block.py")
    return 1


def main(karmadock_root: Path):
    if not karmadock_root.is_dir():
        sys.exit(f"KarmaDock directory not found: {karmadock_root}")

    print(f"Patching {karmadock_root}")
    _patch_ligand_feature(karmadock_root / "dataset" / "ligand_feature.py")
    _patch_gt_block(karmadock_root / "architecture" / "GraphTransformer_Block.py")
    print("KarmaDock patches applied")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: apply_karmadock_patches.py <path-to-KarmaDock-root>")
    main(Path(sys.argv[1]))
