"""Downloads and decompression go through .part files and resume."""
import gzip
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from mmprog.data import download as dl  # noqa: E402


def _fake_wget(src_file, calls):
    """Stand-in for subprocess.run: 'wget -c -O part url' appends the rest of src_file to part."""
    def run(cmd, check=True):
        calls.append(list(cmd))
        part, _url = cmd[-2], cmd[-1]
        have = os.path.getsize(part) if os.path.exists(part) else 0
        with open(src_file, "rb") as fi, open(part, "ab") as fo:
            fi.seek(have)
            fo.write(fi.read())
    return run


def test_download_writes_part_then_renames(tmp_path, monkeypatch):
    src = tmp_path / "remote.bin"
    src.write_bytes(b"x" * 1000)
    calls = []
    monkeypatch.setattr(dl.subprocess, "run", _fake_wget(str(src), calls))
    dst = str(tmp_path / "local.bin")
    out = dl.download_file("http://example/remote.bin", dst)
    assert out == dst and os.path.getsize(dst) == 1000
    assert not os.path.exists(dst + ".part")
    assert calls[0][:4] == ["wget", "-q", "-c", "-O"] and calls[0][4].endswith(".part")


def test_existing_part_is_resumed_not_restarted(tmp_path, monkeypatch):
    src = tmp_path / "remote.bin"
    src.write_bytes(bytes(range(256)) * 4)
    dst = str(tmp_path / "local.bin")
    with open(dst + ".part", "wb") as fh:
        fh.write(bytes(range(256)) * 2)          # a half-finished earlier transfer
    calls = []
    monkeypatch.setattr(dl.subprocess, "run", _fake_wget(str(src), calls))
    dl.download_file("http://example/remote.bin", dst)
    assert open(dst, "rb").read() == src.read_bytes()   # resumed from byte 512, nothing duplicated
    assert len(calls) == 1


def test_complete_file_is_not_refetched(tmp_path, monkeypatch):
    dst = tmp_path / "done.bin"
    dst.write_bytes(b"ok")
    monkeypatch.setattr(dl.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")))
    assert dl.download_file("http://example/x", str(dst)) == str(dst)


def test_gunzip_goes_through_part(tmp_path):
    raw = tmp_path / "morphology.ome.tif"
    gz = tmp_path / "morphology.ome.tif.gz"
    with gzip.open(gz, "wb") as fh:
        fh.write(b"TIFFDATA" * 100)
    out = dl.ensure_decompressed(str(gz))
    assert out == str(raw) and raw.read_bytes() == b"TIFFDATA" * 100
    assert not os.path.exists(str(raw) + ".part")


def test_pick_image_preference(tmp_path):
    names = ["morphology_focus_0000.ome.tif.gz", "morphology.ome.tif.gz", "morphology_mip.ome.tif.gz"]
    files = [str(tmp_path / n) for n in names]
    assert dl.pick_image(files)["image_kind"] == "mip"
    assert dl.pick_image(files[:2])["image_kind"] == "zstack"
    assert dl.pick_image(files[:1])["image_kind"] == "focus"
    assert dl.pick_image([])["image"] is None


def test_fetch_orders_matrix_before_image(tmp_path, monkeypatch):
    from mmprog.config import Config
    cfg = Config(run_name="t", root=str(tmp_path / "ws"), local_data=str(tmp_path / "raw"))
    src = tmp_path / "blob"
    src.write_bytes(b"z" * 10)
    order = []
    def run(cmd, check=True):
        order.append(os.path.basename(cmd[-1]))
        shutil.copy(str(src), cmd[-2])
    monkeypatch.setattr(dl.subprocess, "run", run)
    row = dict(sample="S", supp=["http://h/S_morphology.ome.tif.gz", "http://h/S_cell_feature_matrix.h5",
                                 "http://h/S_transcripts.parquet", "http://h/S_cells.csv.gz"])
    got = dl.fetch_sample_files(row, cfg, need_image=False)
    assert order == ["S_cell_feature_matrix.h5", "S_cells.csv.gz"] and got["image"] is None
    got = dl.fetch_sample_files(row, cfg, need_image=True)
    assert order[-1] == "S_morphology.ome.tif.gz" and got["image_kind"] == "zstack"
    assert "transcripts" not in " ".join(order)
