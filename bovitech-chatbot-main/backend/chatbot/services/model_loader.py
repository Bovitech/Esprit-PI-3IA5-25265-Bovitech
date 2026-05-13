"""
model_loader.py — charge le modèle EfficientNet-B3 une seule fois au démarrage.

Le modèle est lourd (~43 MB) et lent à charger.
Ce singleton garantit qu'il n'est jamais chargé deux fois,
même sous requêtes concurrentes.
"""
import logging
import threading

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

_model      = None
_class_names = None
_model_lock  = threading.Lock()

IMAGE_SIZE = 224
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]


def _build_model(num_classes: int) -> nn.Module:
    """Reconstruit l'architecture EfficientNet-B3 avec la même tête que l'entraînement."""
    net = models.efficientnet_b3(weights=None)
    in_features = net.classifier[1].in_features          # 1536
    net.classifier[1] = nn.Linear(in_features, num_classes)
    return net


def get_model() -> tuple[nn.Module, list[str]]:
    """
    Retourne (model, class_names).
    Charge depuis settings.SKIN_MODEL_PATH si pas encore en mémoire.
    Thread-safe.
    """
    global _model, _class_names

    if _model is not None:
        return _model, _class_names

    with _model_lock:
        if _model is not None:
            return _model, _class_names

        from django.conf import settings

        path = settings.SKIN_MODEL_PATH
        logger.info("Loading skin model from %s ...", path)

        ckpt         = torch.load(path, map_location="cpu", weights_only=False)
        class_names  = ckpt["class_names"]           # ['Dermatophilosis', 'Healthy', 'Lumpy', 'Pediculosis', 'Ringworm']
        state_dict   = ckpt["state_dict"]

        net = _build_model(num_classes=len(class_names))
        net.load_state_dict(state_dict)
        net.eval()

        _model       = net
        _class_names = class_names
        logger.info("Skin model loaded — classes: %s", class_names)

    return _model, _class_names