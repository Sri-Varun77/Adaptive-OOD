from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

MODEL_NAMES = ("resnet18", "mobilenet_v3_small")
IMAGE_SIZE = 160
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGENETTE_NAMES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def create_original_model(name, pretrained=True):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model: {name}")
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
    return model


def get_feature_modules(model, name):
    if name == "resnet18":
        return {"layer_x": model.layer2, "layer_y": model.layer3}
    if name == "mobilenet_v3_small":
        return {"layer_x": model.features[4], "layer_y": model.features[8]}
    raise ValueError(f"Unsupported model: {name}")


class FeatureExtractor:
    def __init__(self, model, name):
        self.model = model
        self.captured = {}
        self.handles = []
        for key, module in get_feature_modules(model, name).items():
            self.handles.append(module.register_forward_hook(self._make_hook(key)))

    def _make_hook(self, key):
        def hook(module, inputs, output):
            self.captured[key] = output
        return hook

    def __call__(self, images):
        self.captured.clear()
        logits = self.model(images)
        if "layer_x" not in self.captured or "layer_y" not in self.captured:
            raise RuntimeError("Intermediate feature hooks did not execute")
        input_embedding = F.adaptive_avg_pool2d(images, (16, 16)).flatten(1)
        layer_x = F.adaptive_avg_pool2d(self.captured["layer_x"], 1).flatten(1)
        layer_y = F.adaptive_avg_pool2d(self.captured["layer_y"], 1).flatten(1)
        return torch.cat((input_embedding, logits, layer_x, layer_y), dim=1), logits

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class AuxiliaryDetector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, values):
        return self.network(values).squeeze(1)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def class_names_from_synsets(synsets):
    return [IMAGENETTE_NAMES.get(name, name) for name in synsets]
