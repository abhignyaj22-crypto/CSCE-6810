"""Encoder registry: every morphology encoder the pipeline can load, with its provenance.

The facts in ``ENCODERS`` (hub id, gating, licence, input size, embedding dimension and
the paper the model card cites) were copied from the model cards listed in
docs/ARCHITECTURE_V3.md section 9. Nothing is added from memory. The registry refuses
to load an encoder whose licence forbids commercial use when ``cfg.commercial_use`` is
true, and prints the card of whatever it loads so the provenance is in the run log.

Every encoder is frozen and returns two things: a pooled embedding (``forward``) and a
spatial feature map (``forward_features``) that FeatureExtraction pools to the g x g
grid tokens of the unified model.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Tuple

from ...logging_utils import say, warn

ENCODERS: Dict[str, Dict[str, Any]] = {
    "resnet50": dict(
        loader="timm", hub_id="resnet50", pretrained=True, gated=False, commercial=True,
        licence="Apache-2.0 (timm weights: see timm model card)", input_px=224, dim=2048,
        feature_map="(B, 2048, 7, 7) from timm forward_features", source="timm resnet50 (ImageNet)",
        paper="n/a (ImageNet-pretrained ResNet-50)", mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    ),
    "resnet50_random": dict(
        loader="timm", hub_id="resnet50", pretrained=False, gated=False, commercial=True,
        licence="Apache-2.0 (timm, random init)", input_px=224, dim=2048,
        feature_map="(B, 2048, 7, 7)", source="timm resnet50 random weights (control)", paper="n/a",
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    ),
    "tiny_test": dict(
        loader="tiny", hub_id=None, pretrained=False, gated=False, commercial=True,
        licence="n/a (test-only network)", input_px=224, dim=64, feature_map="(B, 64, 8, 8)",
        source="features/encoder._tiny_test_encoder", paper="n/a",
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    ),
    "uni": dict(
        loader="timm", hub_id="hf-hub:MahmoodLab/uni", pretrained=True, gated=True, commercial=False,
        licence="CC-BY-NC-ND-4.0", input_px=224, dim=1024, feature_map="ViT-L/16 patch tokens (14 x 14)",
        timm_kwargs=dict(init_values=1e-5, dynamic_img_size=True),
        source="https://huggingface.co/MahmoodLab/UNI",
        paper="Chen et al., Towards a general-purpose foundation model for computational pathology, "
              "Nature Medicine 2024, doi 10.1038/s41591-024-02857-3",
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    ),
    "virchow2": dict(
        loader="timm", hub_id="hf-hub:paige-ai/Virchow2", pretrained=True, gated=True, commercial=False,
        licence="CC-BY-NC-ND-4.0", input_px=224, dim=1280,
        feature_map="ViT-H/14 patch tokens (16 x 16); class token 1280-d, class + mean patch token = 2560-d",
        timm_kwargs="mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU",
        source="https://huggingface.co/paige-ai/Virchow2",
        paper="Zimmermann et al., Virchow2: scaling self-supervised mixed magnification models in pathology, "
              "arXiv 2408.00738, 2024 (no journal venue on the card)",
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    ),
    "hoptimus0": dict(
        loader="timm", hub_id="hf-hub:bioptimus/H-optimus-0", pretrained=True, gated=True, commercial=True,
        licence="Apache-2.0", input_px=224, dim=1536, feature_map="ViT-g/14 patch tokens (16 x 16); 224 px at 0.5 um/px",
        timm_kwargs=dict(init_values=1e-5, dynamic_img_size=False),
        source="https://huggingface.co/bioptimus/H-optimus-0",
        paper="software citation only (GitHub, 2024); no peer-reviewed paper on the card",
        mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517),
    ),
    "phikon_v2": dict(
        loader="transformers", hub_id="owkin/phikon-v2", pretrained=True, gated=False, commercial=False,
        licence="Owkin non-commercial licence", input_px=224, dim=1024, feature_map="ViT-L/16 patch tokens (14 x 14)",
        source="https://huggingface.co/owkin/phikon-v2",
        paper="Filiot et al., Phikon-v2, a large and public feature extractor for biomarker prediction, "
              "arXiv 2409.09173, 2024 (no journal venue on the card)",
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    ),
}


def _weight_hash(module) -> str:
    """Hash of the first parameter tensor: a cheap provenance stamp for the weights actually used."""
    import numpy as np
    for p in module.parameters():
        arr = p.detach().float().cpu().numpy().ravel()[:100000]
        return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]
    return "no-parameters"


class _TransformersWrapper:
    """Give a transformers ViT the two-method interface the pipeline uses."""

    def __init__(self, model, patch_grid: int):
        import torch.nn as nn
        self.model = model
        self.patch_grid = patch_grid
        self.num_features = model.config.hidden_size
        self._nn = nn

    def __call__(self, x):
        return self.model(pixel_values=x).last_hidden_state[:, 0]

    def forward_features(self, x):
        tokens = self.model(pixel_values=x).last_hidden_state[:, 1:]        # drop CLS
        b, n, d = tokens.shape
        g = int(round(n ** 0.5))
        return tokens.transpose(1, 2).reshape(b, d, g, g)

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()


class _VitFeatureMap:
    """Wrap a timm ViT so forward_features returns a (B, D, g, g) map of patch tokens."""

    def __init__(self, model):
        self.model = model
        self.num_features = model.num_features

    def __call__(self, x):
        return self.model(x)

    def forward_features(self, x):
        tokens = self.model.forward_features(x)                             # (B, 1 + n, D) for ViTs
        n_prefix = getattr(self.model, "num_prefix_tokens", 1)
        patch = tokens[:, n_prefix:]
        b, n, d = patch.shape
        g = int(round(n ** 0.5))
        return patch.transpose(1, 2).reshape(b, d, g, g)

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()


def _tiny_with_feature_map():
    """Test encoder that also exposes a (B, 64, 8, 8) spatial map.

    Weights come from a fixed seed so that training and deployment, which construct the encoder
    separately, see the same network; without this the deployed features would differ from the
    training features and the virtual map would not reproduce the training predictions.
    """
    import torch
    import torch.nn as nn

    class TinyEncoder(nn.Module):
        num_features = 64

        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(3, 16, 5, stride=4), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            )
            self.pool = nn.AdaptiveAvgPool2d(8)

        def forward_features(self, x):
            return self.pool(self.body(x))                                   # (B, 64, 8, 8)

        def forward(self, x):
            return self.forward_features(x).mean(dim=(2, 3))

    state = torch.random.get_rng_state()
    torch.manual_seed(20260101)
    try:
        enc = TinyEncoder()
    finally:
        torch.random.set_rng_state(state)
    return enc


def check_licence(name: str, cfg) -> Dict[str, Any]:
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder '{name}'; registered: {sorted(ENCODERS)}")
    card = dict(ENCODERS[name])
    card["name"] = name
    if getattr(cfg, "commercial_use", False) and not card["commercial"]:
        raise PermissionError(
            f"encoder '{name}' is licensed {card['licence']}, which forbids commercial use, and "
            f"commercial_use is true; choose resnet50 or hoptimus0, or set commercial_use false for research use")
    return card


def get_encoder(name: str, device: str, cfg) -> Tuple[Any, Dict[str, Any]]:
    """Load a frozen encoder and return it with its provenance card."""
    import torch

    card = check_licence(name, cfg)
    token = os.environ.get(getattr(cfg, "encoder_token_env", "HF_TOKEN"))
    if card["gated"] and not token:
        warn(f"encoder '{name}' is gated on the Hugging Face Hub and no token was found in "
             f"${getattr(cfg, 'encoder_token_env', 'HF_TOKEN')}; the download will fail unless you are logged in")
    if card["loader"] == "tiny":
        model = _tiny_with_feature_map()
    elif card["loader"] == "timm":
        import timm
        kwargs: Dict[str, Any] = {}
        if name == "virchow2":
            kwargs = dict(mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)
        elif isinstance(card.get("timm_kwargs"), dict):
            kwargs = dict(card["timm_kwargs"])
        if token:
            os.environ.setdefault("HF_TOKEN", token)
        model = timm.create_model(card["hub_id"], pretrained=card["pretrained"], num_classes=0, **kwargs)
        if not name.startswith("resnet"):
            model = _VitFeatureMap(model)
    elif card["loader"] == "transformers":
        from transformers import AutoModel
        model = _TransformersWrapper(AutoModel.from_pretrained(card["hub_id"], token=token), patch_grid=14)
    else:
        raise ValueError(card["loader"])

    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    card["weight_hash"] = _weight_hash(model)
    card["device"] = device
    if getattr(cfg, "channels_last", False) and device == "cuda" and name.startswith("resnet"):
        model = model.to(memory_format=torch.channels_last)
        card["channels_last"] = True
    if getattr(cfg, "compile_encoder", False) and device == "cuda":
        model = torch.compile(model)
        card["compiled"] = True
    say(f"encoder {name}: dim={card['dim']} input={card['input_px']}px licence='{card['licence']}' "
        f"gated={card['gated']} commercial={card['commercial']} source={card['source']}")
    say(f"encoder paper: {card['paper']}", indent=1)
    say(f"encoder weight hash: {card['weight_hash']}; frozen; device {device}", indent=1)
    return model, card
