import tempfile
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from models import AuxiliaryDetector, FeatureExtractor, MODEL_NAMES, create_original_model, parameter_count
from train_two_models import train_auxiliary


def make_dataset(seed):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(8, 3, 160, 160, generator=generator)
    labels = torch.randint(0, 10, (8,), generator=generator)
    return TensorDataset(images, labels)


def main():
    device = torch.device("cpu")
    args = Namespace(
        batch_size=4,
        workers=0,
        aux_batch_size=4,
        aux_lr=0.001,
        aux_epochs=1,
        seed=42,
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        for index, name in enumerate(MODEL_NAMES):
            model = create_original_model(name, pretrained=False).to(device).eval()
            extractor = FeatureExtractor(model, name)
            batch = torch.randn(2, 3, 160, 160)
            with torch.no_grad():
                features, logits = extractor(batch)
            extractor.close()
            if logits.shape != (2, 10):
                raise AssertionError(f"{name}: expected ten logits, got {tuple(logits.shape)}")
            if features.ndim != 2 or features.shape[0] != 2:
                raise AssertionError(f"{name}: invalid feature tensor shape {tuple(features.shape)}")
            detector = AuxiliaryDetector(features.shape[1]).to(device)
            labels = torch.tensor([1.0, 0.0])
            optimizer = torch.optim.AdamW(detector.parameters(), lr=0.001)
            loss_function = nn.BCEWithLogitsLoss()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(detector(features), labels)
            loss.backward()
            optimizer.step()

            id_train = make_dataset(10 + index)
            id_validation = make_dataset(20 + index)
            id_test = make_dataset(30 + index)
            ood_train = make_dataset(40 + index)
            ood_validation = make_dataset(50 + index)
            ood_test = make_dataset(60 + index)
            metrics = train_auxiliary(
                model,
                name,
                id_train,
                id_validation,
                id_test,
                ood_train,
                ood_validation,
                ood_test,
                [f"class_{class_id}" for class_id in range(10)],
                args,
                device,
                Path(temporary_directory),
            )
            if not (0.0 <= metrics["test_auroc"] <= 1.0):
                raise AssertionError(f"{name}: invalid AUROC")
            print(
                f"PASS {name}: parameters={parameter_count(model):,}, "
                f"logits={tuple(logits.shape)}, features={tuple(features.shape)}, "
                f"aux_AUROC={metrics['test_auroc']:.4f}, one_step_loss={loss.item():.4f}"
            )
    print("Both model and auxiliary-training smoke tests passed")


if __name__ == "__main__":
    main()
