"""
GNINA constants
"""

GNINA_BINARY = "/opt/gnina/bin/gnina"

# gnina ships its own torch/openbabel/boost; we put them under /opt/gnina/lib
# so they only get loaded for the gnina subprocess. Prepend (not replace) to
# the caller's LD_LIBRARY_PATH at invocation time.
GNINA_LIB_PATH = "/opt/gnina/lib"

# gnina's bundled libopenbabel.so.7 (OB 3.0/3.1) has NO format-plugin tree in
# the image, so its Open Babel can't load the writer for ANY output format —
# every ``--out`` file comes out empty (scores still work because gnina reads
# pdbqt with its own native parser). The only complete plugin set belongs to
# the system Open Babel 3.2 (.so.8) at the paths below. We make gnina load that
# .so.8 under the .so.7 name via a symlink shim (OB 3.1→3.2 is ABI-compatible
# for gnina's calls — verified to leave scores byte-identical) and point its OB
# at the matching plugin/data dirs. See ``_ensure_openbabel_plugin_shim`` in
# guild/docking/gnina.py.
GNINA_OB_SYSTEM_LIB = "/usr/local/lib/libopenbabel.so.8"
GNINA_OB_PLUGIN_DIR = "/usr/local/lib/openbabel/3.2.0"
GNINA_OB_DATA_DIR = "/usr/local/share/openbabel/3.2.0"

GNINA_DEFAULT_EXHAUSTIVENESS = 8

GNINA_DEFAULT_NUMBER_OF_POSES = 9

# gnina's default --cnn_scoring is "rescore": Vina-style search produces poses
# and the CNN rescores the top N. Fastest mode that still surfaces a CNN score.
GNINA_DEFAULT_CNN_SCORING = "rescore"
