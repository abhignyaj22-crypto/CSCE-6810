"""read_10x_h5 reads a hand-built 10x v3 matrix back correctly."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import h5py
import numpy as np
from scipy.sparse import csc_matrix


def _write_10x_h5(path, dense, genes, barcodes):
    m = csc_matrix(dense.T)  # 10x stores genes x cells, CSC
    with h5py.File(path, "w") as fh:
        g = fh.create_group("matrix")
        g.create_dataset("data", data=m.data.astype(np.int32))
        g.create_dataset("indices", data=m.indices.astype(np.int64))
        g.create_dataset("indptr", data=m.indptr.astype(np.int64))
        g.create_dataset("shape", data=np.array(m.shape, dtype=np.int32))
        g.create_dataset("barcodes", data=np.array(barcodes, dtype="S"))
        f = g.create_group("features")
        f.create_dataset("name", data=np.array(genes, dtype="S"))
        f.create_dataset("id", data=np.array([f"ENSG{i}" for i in range(len(genes))], dtype="S"))


def test_read_10x_h5_roundtrip(tmp_path):
    from mmprog.data.tenx import read_10x_h5
    dense = np.array([[0, 2, 0, 1], [3, 0, 0, 0], [0, 0, 5, 1]], dtype=np.float32)  # 3 cells x 4 genes
    genes = ["A", "B", "C", "B"]  # duplicate gene name must be made unique
    barcodes = ["c1", "c2", "c3"]
    path = tmp_path / "m.h5"
    _write_10x_h5(str(path), dense, genes, barcodes)
    a = read_10x_h5(str(path))
    assert a.shape == (3, 4)
    assert np.allclose(a.X.toarray(), dense)
    assert list(a.obs_names) == barcodes
    assert len(set(a.var_names)) == 4
    assert a.var["gene_ids"].iloc[0] == "ENSG0"


def test_load_xenium_micron_to_pixel_at_level1(tmp_path):
    """100 um at 0.2125 um/px level 0 is 470.6 px; at level 1 (half resolution) it is 235.29 px."""
    import pandas as pd
    from mmprog.config import Config
    from mmprog.data.tenx import load_xenium
    path = str(tmp_path / "m.h5")
    _write_10x_h5(path, np.ones((3, 5), dtype=np.float32), [f"g{i}" for i in range(5)], ["c0", "c1", "c2"])
    from mmprog.data.tenx import h5_matrix_shape
    assert h5_matrix_shape(path) == (3, 5)
    cells = pd.DataFrame({"cell_id": [f"c{i}" for i in range(3)], "x_centroid": [0.0, 100.0, 50.0],
                          "y_centroid": [200.0, 0.0, 25.0]})
    cells.to_csv(tmp_path / "cells.csv.gz", index=False)
    cfg = Config(run_name="t", root=str(tmp_path), local_data=str(tmp_path), img_level=1)
    row = dict(h5=path, cells=str(tmp_path / "cells.csv.gz"), sample="S", patient="S", stage="MM", um_per_px_l0=0.2125)
    a = load_xenium(row, cfg)
    assert abs(a.obs["x"].iloc[1] - 235.294) < 0.01
    assert abs(a.obs["y"].iloc[0] - 470.588) < 0.01
    assert abs(a.obs["x"].iloc[2] - 117.647) < 0.01
