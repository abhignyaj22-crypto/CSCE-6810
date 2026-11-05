"""Command-line overrides are typed by the dataclass field, not guessed by YAML."""
import datetime
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from mmprog.config import Config, load_config, save_config  # noqa: E402


def test_string_fields_keep_their_text():
    cfg = load_config(overrides=["run_name=001", "exclude_title_regex=relapse: RM", "geo_accession='GSE1'"])
    assert cfg.run_name == "001" and isinstance(cfg.run_name, str)
    assert cfg.exclude_title_regex == "relapse: RM"
    assert cfg.geo_accession == "GSE1"


def test_date_like_run_name_stays_a_string():
    cfg = load_config(overrides=["run_name=2025-01-01"])
    assert cfg.run_name == "2025-01-01" and not isinstance(cfg.run_name, datetime.date)
    assert cfg.run_dir.endswith(os.path.join("runs", "2025-01-01"))


def test_null_and_numbers_and_bools():
    cfg = load_config(overrides=["sample_limit=null", "max_epochs=3", "lr=0.5", "demo_mode=true", "use_morpho=no"])
    assert cfg.sample_limit is None
    assert cfg.max_epochs == 3 and isinstance(cfg.max_epochs, int)
    assert cfg.lr == 0.5
    assert cfg.demo_mode is True and cfg.use_morpho is False
    assert load_config(overrides=["sample_limit=4"]).sample_limit == 4


def test_tuple_fields_accept_yaml_list_and_comma_list():
    a = load_config(overrides=["cv_variants=[full,shuffled]"])
    b = load_config(overrides=["cv_variants=full,shuffled"])
    assert a.cv_variants == ("full", "shuffled") == b.cv_variants
    assert load_config(overrides=["seeds=1,2,3"]).seeds == (1, 2, 3)
    assert load_config(overrides=["cv_variants=full"]).cv_variants == ("full",)


def test_path_fields_expand_home():
    cfg = load_config(overrides=["root=~/mm_ws", "clinical_csv=~/clin.csv"])
    assert cfg.root == os.path.expanduser("~/mm_ws")
    assert cfg.clinical_csv == os.path.expanduser("~/clin.csv")
    assert load_config(overrides=["clinical_csv=null"]).clinical_csv is None


def test_unknown_key_raises():
    with pytest.raises(ValueError):
        load_config(overrides=["not_a_field=1"])
    with pytest.raises(ValueError):
        load_config(overrides=["no_equals_sign"])


def test_bad_bool_raises():
    with pytest.raises(ValueError):
        load_config(overrides=["demo_mode=maybe"])


def test_yaml_file_plus_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("run_name: fromfile\nseeds: [4, 5]\nsample_limit: 2\n")
    cfg = load_config(str(p), overrides=["sample_limit=9"])
    assert cfg.run_name == "fromfile" and cfg.seeds == (4, 5) and cfg.sample_limit == 9


def test_from_json_round_trip_drops_unknown_and_resets_device(tmp_path):
    cfg = load_config(overrides=["run_name=rt", "cv_variants=full,shuffled", "device=cuda"])
    path = tmp_path / "config.json"
    save_config(cfg, str(path))
    raw = json.load(open(path))
    raw["legacy_key"] = 1                       # a key from another version of the code
    json.dump(raw, open(path, "w"))
    back = Config.from_json(str(path))
    assert back.run_name == "rt" and back.cv_variants == ("full", "shuffled")
    assert back.device == "auto"
    assert not hasattr(back, "legacy_key")
