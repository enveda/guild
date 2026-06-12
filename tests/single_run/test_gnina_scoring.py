"""
Tests for gnina stdout parsing and the per-combination score file produced by
:func:`guild.docking.gnina.deploy_gnina` / :func:`process_gnina_output`.
"""

import math
from pathlib import Path

import pytest

from guild.docking.gnina import parse_gnina_stdout, process_gnina_output

# Example gnina stdout (truncated to the relevant section). Columns are
# mode | affinity (kcal/mol) | CNNscore | CNNaffinity.
GNINA_STDOUT_SAMPLE = """\
gnina v1.3 Linux64
   _______  _____ _________ _____   _____
  |  ____ |     |    |    |       |     |
  ...

mode |  affinity  | CNN     | CNN
     | (kcal/mol) | pose-score| affinity
-----+------------+----------+----------
    1     -8.345      0.7891      6.234
    2     -7.910      0.6512      5.987
    3     -7.500      0.5800      5.612
"""


def test_parse_gnina_stdout_extracts_all_poses():
    poses = parse_gnina_stdout(GNINA_STDOUT_SAMPLE)
    assert len(poses) == 3
    assert poses[0] == (1, -8.345, 0.7891, 6.234)
    assert poses[2] == (3, -7.500, 0.58, 5.612)


def test_parse_gnina_stdout_raises_on_empty():
    with pytest.raises(ValueError):
        parse_gnina_stdout("nothing useful here\nno table rows\n")


def test_process_gnina_output_picks_best_affinity(tmp_path: Path):
    score_file = tmp_path / "scores.txt"
    score_file.write_text(
        "1: -8.345\t0.7891\n"
        "2: -7.910\t0.6512\n"
        "3: -7.500\t0.5800\n"
    )

    affinity, cnn = process_gnina_output(str(score_file))
    # "Best" = minimum affinity, matching the Vina convention.
    assert affinity == pytest.approx(-8.345)
    # CNN score returned is the one from the same pose, not max across poses.
    assert cnn == pytest.approx(0.7891)


def test_process_gnina_output_handles_missing_cnn_column(tmp_path: Path):
    # Vina-formatted score file (no CNN column) — older score files must still
    # parse without crashing.
    score_file = tmp_path / "vina_format.txt"
    score_file.write_text("1: -8.0\n2: -7.5\n")

    affinity, cnn = process_gnina_output(str(score_file))
    assert affinity == pytest.approx(-8.0)
    assert math.isnan(cnn)
