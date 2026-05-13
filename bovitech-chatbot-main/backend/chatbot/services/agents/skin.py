"""
SkinAgent — analyse une image de peau bovine avec EfficientNet-B3.

Retourne un dict avec :
  predicted_class : str
  confidence      : float (0-1)
  probabilities   : dict[str, float]
  level           : "high" | "medium" | "low"
  is_healthy      : bool
  lang            : str (pour que views.py puisse logger)
"""
import io
import logging

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from chatbot.services.model_loader import IMAGE_SIZE, MEAN, STD, get_model

logger = logging.getLogger(__name__)

# Seuils de confiance
HIGH_THRESHOLD   = 0.80
MEDIUM_THRESHOLD = 0.50

# Prétraitement identique à l'entraînement
_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Descriptions bilingues par maladie
_DESCRIPTIONS = {
    "fr": {
        "Dermatophilosis": "dermatophilose (infection bactérienne de la peau)",
        "Healthy":         "peau saine — aucun signe visible de maladie",
        "Lumpy":           "dermatose nodulaire contagieuse (Lumpy Skin Disease)",
        "Pediculosis":     "pédiculose (infestation par des poux)",
        "Ringworm":        "teigne (infection fongique de la peau)",
    },
    "ar": {
        "Dermatophilosis": "الجلدية الشعية (عدوى بكتيرية جلدية)",
        "Healthy":         "جلد سليم — لا توجد علامات مرضية ظاهرة",
        "Lumpy":           "مرض الجلد العقدي (Lumpy Skin Disease)",
        "Pediculosis":     "القمل (إصابة بالقمل)",
        "Ringworm":        "السعفة (عدوى فطرية جلدية)",
    },
}


class SkinAgent:

    def run(self, image_bytes: bytes, lang: str) -> dict | None:
        """
        Analyse l'image et retourne le résultat de classification.
        Retourne None si l'image est invalide ou illisible.
        """
        # Charger et valider l'image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, Exception) as exc:
            logger.warning("SkinAgent: invalid image — %s", exc)
            return None

        # Prétraitement
        tensor = _transform(image).unsqueeze(0)   # (1, 3, 224, 224)

        # Inférence
        model, class_names = get_model()
        with torch.no_grad():
            logits = model(tensor)                 # (1, 5)
            probs  = F.softmax(logits, dim=1)[0]   # (5,)

        # Résultat
        confidence, idx     = probs.max(dim=0)
        predicted_class     = class_names[idx.item()]
        confidence_val      = confidence.item()

        prob_dict = {cls: round(probs[i].item(), 4) for i, cls in enumerate(class_names)}

        # Niveau de confiance
        if confidence_val >= HIGH_THRESHOLD:
            level = "high"
        elif confidence_val >= MEDIUM_THRESHOLD:
            level = "medium"
        else:
            level = "low"

        logger.info(
            "action=skin predicted=%s confidence=%.3f level=%s lang=%s",
            predicted_class, confidence_val, level, lang,
        )

        return {
            "predicted_class": predicted_class,
            "confidence":      round(confidence_val, 4),
            "probabilities":   prob_dict,
            "level":           level,
            "is_healthy":      predicted_class == "Healthy",
            "description":     _DESCRIPTIONS.get(lang, _DESCRIPTIONS["fr"]).get(predicted_class, predicted_class),
        }