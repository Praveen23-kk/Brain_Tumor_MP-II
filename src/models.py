"""Week 5: the six CNN models of the project, built with timm.

Used by src/train.py and notebooks/04_training.ipynb.
"""
import os
from pathlib import Path

# Keep downloaded model weights on D: (inside the project), same as src/splits.py.
# Must be set BEFORE timm / huggingface_hub are imported.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))

import timm    # noqa: E402  (import after setting the cache folders)
import torch   # noqa: E402

# Per-model settings for the 4 GB GTX 1650 Ti.
# batch    = starting batch size (halved automatically on CUDA out-of-memory)
# img_size = input size the network expects (inception_v3 was pre-trained on 299 x 299)
# heavy    = big baseline: backbone frozen for the first FREEZE_EPOCHS epochs, smaller learning rate
# amp      = train with mixed precision (float16). efficientnet_b0 is False: float16 gave NaN in epoch 1 in
#            2 of its 4 Week 6 runs on this GTX 1650 Ti, so it always trains in float32 (no NaN, consistent runs).
MODELS = {
    "mobilenetv3_small_100": {"batch": 48, "img_size": 224, "heavy": False, "amp": True},
    "efficientnet_b0":       {"batch": 32, "img_size": 224, "heavy": False, "amp": False},
    "resnet50":              {"batch": 24, "img_size": 224, "heavy": True,  "amp": True},
    "inception_v3":          {"batch": 24, "img_size": 299, "heavy": True,  "amp": True},
    "densenet121":           {"batch": 16, "img_size": 224, "heavy": True,  "amp": True},
    "vgg16":                 {"batch": 16, "img_size": 224, "heavy": True,  "amp": True},
}
TIMM_VERSION = timm.__version__
FREEZE_EPOCHS = 2      # heavy models: train only the new classifier for this many epochs first


def build_model(name, num_classes=4, pretrained=True):
    """ImageNet-pretrained timm model with a new classifier of num_classes outputs, in channels_last layout."""
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    return model.to(memory_format=torch.channels_last)


def set_backbone_trainable(model, trainable):
    """Freeze (trainable=False) or unfreeze every layer except the classifier.
    The classifier always stays trainable, because it is new and starts from random weights."""
    for p in model.parameters():
        p.requires_grad = trainable
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def count_params(model):
    """Total number of parameters (weights) in the model."""
    return sum(p.numel() for p in model.parameters())
