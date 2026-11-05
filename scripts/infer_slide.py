#!/usr/bin/env python
"""Morphology-only inference on one slide with a finished run's final model.

    python scripts/infer_slide.py --run-dir workspace/runs/gse299193_full --image slide.ome.tif --out slide_programs.csv
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mmprog.config import Config, resolve_device  # noqa: E402
from mmprog.logging_utils import banner, say  # noqa: E402
from mmprog.paths import RunDir  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--level", type=int, default=None, help="pyramid level to read (default: the run's img_level)")
    args = p.parse_args(argv)

    from mmprog.data.images import load_image
    from mmprog.inference.export import export_qupath, export_xenium_explorer
    from mmprog.inference.final_fit import load_final_model
    from mmprog.inference.virtual_map import virtual_program_map

    banner("mmprog single-slide inference")
    cfg = Config.from_json(os.path.join(args.run_dir, "config.json"))
    cfg.root, cfg.run_name = os.path.dirname(os.path.dirname(args.run_dir)), os.path.basename(args.run_dir)
    run = RunDir(cfg)
    cfg.device = "auto"            # the saved device belongs to the machine that trained, not this one
    device = resolve_device(cfg)
    meta = json.load(open(run.art("targets")))
    manifest = json.load(open(run.art("manifest")))
    n_samples = len({m["sample"] for m in manifest})
    model, fs, selector, _ = load_final_model(run, cfg, device)
    level = cfg.img_level if args.level is None else args.level
    say(f"reading {args.image} at pyramid level {level}")
    img = load_image(args.image, level, z_project=cfg.z_project, stain_mode=cfg.stain_mode)
    vmap = virtual_program_map(img, model, fs, selector, cfg, device, meta["targets"], n_samples)
    vmap.to_csv(args.out, index=False)
    export_qupath(vmap, os.path.splitext(args.out)[0] + "_qupath.tsv", cfg.um_per_px)
    export_xenium_explorer(vmap, os.path.splitext(args.out)[0] + "_xenium_explorer.csv", cfg.um_per_px)
    say(f"wrote {len(vmap)} nuclei with {len(meta['targets'])} program predictions to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
