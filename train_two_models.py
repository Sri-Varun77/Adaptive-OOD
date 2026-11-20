import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms

from data_pipeline import (
    build_id_datasets,
    build_ood_datasets,
    make_loader,
    set_seed,
    split_id_validation_indices,
    split_classifier_indices,
)
from models import (
    AuxiliaryDetector,
    FeatureExtractor,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_NAMES,
    class_names_from_synsets,
    create_original_model,
    parameter_count,
)


def train_original_model(name, train_dataset, validation_dataset, classes, args, device, output_dir):
    try:
        model = create_original_model(name, pretrained=not args.no_pretrained).to(device)
    except Exception as exc:
        raise RuntimeError(
            f"Could not initialize pretrained weights for {name}. Check internet access or rerun with --no-pretrained. Original error: {exc}"
        ) from exc
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, workers=args.workers)
    validation_loader = make_loader(validation_dataset, args.batch_size, shuffle=False, workers=args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.backbone_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.backbone_epochs, 1))
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_state = None
    history = []

    for epoch in range(args.backbone_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        train_accuracy = correct / max(total, 1)
        validation_accuracy, validation_loss = evaluate_classifier(model, validation_loader, criterion, device)
        history.append({
            "epoch": epoch + 1,
            "train_loss": running_loss / max(total, 1),
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        })
        print(
            f"{name} backbone epoch {epoch + 1}/{args.backbone_epochs} "
            f"train_acc={train_accuracy:.4f} val_acc={validation_accuracy:.4f}",
            flush=True,
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f"No classifier checkpoint was produced for {name}")
    model.load_state_dict(best_state)
    model.eval()
    model_path = output_dir / f"{name}_original.pt"
    torch.save({
        "model_name": name,
        "model_state": model.state_dict(),
        "classes": list(classes),
        "class_names": class_names_from_synsets(classes),
        "output_count": len(classes),
        "input_size": 160,
        "pretrained_initialization": "torchvision ImageNet weights, fine-tuned to 10 Imagenette logits",
        "history": history,
    }, model_path)
    return model, best_accuracy, history


def evaluate_classifier(model, loader, criterion, device):
    model.eval()
    losses = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            losses += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1), losses / max(total, 1)


def extract_features(model, name, dataset, batch_size, device, workers, compute_correct=True):
    loader = make_loader(dataset, batch_size, shuffle=False, workers=workers)
    extractor = FeatureExtractor(model, name)
    feature_parts = []
    class_labels = []
    prediction_correct = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            features, logits = extractor(images)
            feature_parts.append(features.cpu())
            class_labels.append(labels.cpu())
            if compute_correct:
                prediction_correct.append((logits.argmax(1).cpu() == labels.cpu()).to(torch.int64))
            else:
                prediction_correct.append(torch.zeros(labels.size(0), dtype=torch.int64))
    extractor.close()
    if not feature_parts:
        raise RuntimeError("Feature extraction received an empty dataset")
    return torch.cat(feature_parts), torch.cat(class_labels), torch.cat(prediction_correct)


def choose_threshold(labels, probabilities):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities)
    valid = np.isfinite(thresholds)
    if not np.any(valid):
        return 0.5
    scores = true_positive_rate[valid] - false_positive_rate[valid]
    return float(thresholds[valid][int(np.argmax(scores))])


def balanced_indices(labels, target_count, seed):
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    if len(classes) == 0 or target_count <= 0:
        return np.asarray([], dtype=np.int64)
    per_class = max(1, target_count // len(classes))
    selected = []
    available = []
    for class_id in classes:
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        selected.extend(class_indices[:per_class].tolist())
        available.extend(class_indices[per_class:].tolist())
    if len(selected) > target_count:
        rng.shuffle(selected)
        selected = selected[:target_count]
    elif len(selected) < target_count and available:
        rng.shuffle(available)
        selected.extend(available[:target_count - len(selected)])
    return np.asarray(selected, dtype=np.int64)


def train_auxiliary(model, name, id_train_dataset, id_val_dataset, id_test_dataset,
                    ood_train_dataset, ood_val_dataset, ood_test_dataset,
                    class_names, args, device, output_dir):
    print(f"{name}: extracting ID and OOD features", flush=True)
    id_train_features, id_train_classes, _ = extract_features(model, name, id_train_dataset, args.batch_size, device, args.workers)
    id_val_features, id_val_classes, id_val_correct = extract_features(model, name, id_val_dataset, args.batch_size, device, args.workers)
    id_test_features, id_test_classes, id_test_correct = extract_features(model, name, id_test_dataset, args.batch_size, device, args.workers)
    ood_train_features, ood_train_classes, _ = extract_features(model, name, ood_train_dataset, args.batch_size, device, args.workers, compute_correct=False)
    ood_val_features, ood_val_classes, _ = extract_features(model, name, ood_val_dataset, args.batch_size, device, args.workers, compute_correct=False)
    ood_test_features, ood_test_classes, _ = extract_features(model, name, ood_test_dataset, args.batch_size, device, args.workers, compute_correct=False)

    train_count = min(id_train_features.size(0), ood_train_features.size(0))
    validation_count = min(id_val_features.size(0), ood_val_features.size(0))
    if min(train_count, validation_count) < 2:
        raise RuntimeError(f"Not enough samples for the auxiliary detector: train={train_count}, val={validation_count}")

    rng = np.random.default_rng(args.seed)
    id_train_order = balanced_indices(id_train_classes.numpy(), train_count, args.seed)
    if len(id_train_order) < train_count:
        raise RuntimeError("Unable to select enough class-balanced Imagenette ID training images")
    ood_train_order = balanced_indices(ood_train_classes.numpy(), train_count, args.seed + 1)
    if len(ood_train_order) < train_count:
        raise RuntimeError("Unable to select enough class-balanced ImageNet-R OOD training images")
    train_x = torch.cat((id_train_features[id_train_order], ood_train_features[ood_train_order]), dim=0)
    train_y = torch.cat((torch.ones(train_count), torch.zeros(train_count)), dim=0)
    train_order = torch.randperm(train_y.numel(), generator=torch.Generator().manual_seed(args.seed))
    train_x = train_x[train_order]
    train_y = train_y[train_order]

    id_val_order = balanced_indices(id_val_classes.numpy(), validation_count, args.seed + 2)
    ood_val_order = balanced_indices(ood_val_classes.numpy(), validation_count, args.seed + 3)
    validation_x = torch.cat((id_val_features[id_val_order], ood_val_features[ood_val_order]), dim=0)
    validation_y = torch.cat((torch.ones(validation_count), torch.zeros(validation_count)), dim=0).numpy()

    mean = train_x.mean(dim=0)
    standard_deviation = train_x.std(dim=0, unbiased=False).clamp_min(1e-5)
    train_x = (train_x - mean) / standard_deviation
    validation_x = (validation_x - mean) / standard_deviation

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.aux_batch_size, shuffle=True, num_workers=0)
    detector = AuxiliaryDetector(train_x.size(1)).to(device)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=args.aux_lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.aux_epochs):
        detector.train()
        total_loss = 0.0
        total = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = detector(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
        detector.eval()
        with torch.no_grad():
            validation_logits = detector(validation_x.to(device)).cpu()
            validation_probabilities = torch.sigmoid(validation_logits).numpy()
        validation_predictions = (validation_probabilities >= 0.5).astype(np.int64)
        validation_accuracy = accuracy_score(validation_y, validation_predictions)
        print(
            f"{name} auxiliary epoch {epoch + 1}/{args.aux_epochs} "
            f"loss={total_loss / max(total, 1):.4f} val_acc@0.5={validation_accuracy:.4f}",
            flush=True,
        )

    detector.eval()
    with torch.no_grad():
        validation_probabilities = torch.sigmoid(detector(validation_x.to(device))).cpu().numpy()
    threshold = choose_threshold(validation_y, validation_probabilities)

    test_x = torch.cat((id_test_features, ood_test_features), dim=0)
    test_y = np.concatenate((np.ones(id_test_features.size(0)), np.zeros(ood_test_features.size(0)))).astype(np.int64)
    test_x = (test_x - mean) / standard_deviation
    with torch.no_grad():
        test_probabilities = torch.sigmoid(detector(test_x.to(device))).cpu().numpy()
    test_predictions = (test_probabilities >= threshold).astype(np.int64)
    id_count = id_test_features.size(0)
    ood_count = ood_test_features.size(0)
    test_auc = float(roc_auc_score(test_y, test_probabilities))
    metrics = {
        "model": name,
        "id_class_count": len(class_names),
        "id_test_images": int(id_count),
        "ood_test_images": int(ood_count),
        "threshold": threshold,
        "test_auroc": test_auc,
        "test_accuracy": float(accuracy_score(test_y, test_predictions)),
        "id_acceptance_rate": float((test_probabilities[:id_count] >= threshold).mean()),
        "ood_rejection_rate": float((test_probabilities[id_count:] < threshold).mean()),
        "original_classifier_accuracy_on_id_test": float(id_test_correct.float().mean().item()),
        "ood_training_classes": len(getattr(ood_train_dataset, "classes", [])),
        "ood_validation_classes": len(getattr(ood_val_dataset, "classes", [])),
        "ood_test_classes": len(getattr(ood_test_dataset, "classes", [])),
        "label_rule": "1 if ground-truth synset is one of the ten Imagenette output classes; otherwise 0",
    }

    checkpoint_path = output_dir / f"{name}_auxiliary.pt"
    torch.save({
        "model_name": name,
        "detector_state": detector.cpu().state_dict(),
        "feature_mean": mean,
        "feature_std": standard_deviation,
        "threshold": threshold,
        "feature_dim": int(train_x.size(1)),
        "classes": list(class_names),
        "id_dataset": "Imagenette2-160 ten-class training/validation vocabulary",
        "ood_dataset": "ImageNet-R classes outside the ten supported Imagenette synsets",
        "ood_train_classes": list(getattr(ood_train_dataset, "classes", [])),
        "ood_validation_classes": list(getattr(ood_val_dataset, "classes", [])),
        "ood_test_classes": list(getattr(ood_test_dataset, "classes", [])),
        "positive_label": 1,
        "negative_label": 0,
        "metrics": metrics,
    }, checkpoint_path)
    return metrics


def write_report(metrics, output_dir):
    report_path = output_dir / "results.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    csv_path = output_dir / "results.csv"
    if metrics:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            writer.writerows(metrics)


def run(args):
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}", flush=True)
    print("Fixed models: ResNet-18 and MobileNetV3-Small", flush=True)
    print("ID: ground-truth synset is in the ten output classes. OOD: ImageNet-R ground-truth synset is outside those classes. Original predictions are never used as labels.", flush=True)

    id_train_aug, id_train_eval, id_validation = build_id_datasets(data_root)
    classifier_train_indices, classifier_validation_indices = split_classifier_indices(id_train_eval, args.seed)
    classifier_train_dataset = Subset(id_train_aug, classifier_train_indices)
    classifier_validation_dataset = Subset(id_train_eval, classifier_validation_indices)
    ood_train, ood_validation, ood_test = build_ood_datasets(data_root, id_train_aug.classes, args.seed)
    id_train_indices, id_validation_indices, id_test_indices = split_id_validation_indices(id_validation, args.seed)
    id_binary_train = Subset(id_validation, id_train_indices)
    id_binary_validation = Subset(id_validation, id_validation_indices)
    id_binary_test = Subset(id_validation, id_test_indices)
    class_names = list(id_train_aug.classes)

    metrics_all = []

    for model_index, model_name in enumerate(MODEL_NAMES, start=1):
        print("#" * 64, flush=True)
        print(f"MODEL {model_index}/2: {model_name}", flush=True)
        print("#" * 64, flush=True)
        start_time = time.time()
        model, best_accuracy, history = train_original_model(
            model_name,
            classifier_train_dataset,
            classifier_validation_dataset,
            class_names,
            args,
            device,
            output_dir,
        )
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        print(f"Original model parameters: {parameter_count(model):,}", flush=True)
        print(f"Original classifier validation accuracy: {best_accuracy:.4f}", flush=True)

        metrics = train_auxiliary(
            model,
            model_name,
            id_binary_train,
            id_binary_validation,
            id_binary_test,
            ood_train,
            ood_validation,
            ood_test,
            class_names,
            args,
            device,
            output_dir,
        )
        metrics["elapsed_seconds"] = round(time.time() - start_time, 2)
        metrics_all.append(metrics)
        print(json.dumps(metrics, indent=2), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_report(metrics_all, output_dir)
    print(f"Training completed. Results: {output_dir / 'results.json'}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--backbone-epochs", type=int, default=5)
    parser.add_argument("--aux-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--aux-batch-size", type=int, default=128)
    parser.add_argument("--backbone-lr", type=float, default=0.0002)
    parser.add_argument("--aux-lr", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
