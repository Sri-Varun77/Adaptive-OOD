import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets.utils import download_and_extract_archive

from models import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, IMAGENETTE_NAMES

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENET_R_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
IMAGENET_R_MD5 = "a61312130a589d0ca1a8fca1f2bd3337"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms(training=False):
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        steps.extend([transforms.RandomHorizontalFlip(), transforms.RandomRotation(8)])
    steps.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(steps)


def ensure_imagenette(root):
    root = Path(root)
    extracted = root / "imagenette2-160"
    if (extracted / "train").is_dir() and (extracted / "val").is_dir():
        return extracted
    root.mkdir(parents=True, exist_ok=True)
    try:
        download_and_extract_archive(IMAGENETTE_URL, str(root), filename="imagenette2-160.tgz")
    except Exception as exc:
        raise RuntimeError(
            "Could not download Imagenette2-160. Check internet access and rerun. "
            f"Source: {IMAGENETTE_URL}. Error: {exc}"
        ) from exc
    if not (extracted / "train").is_dir() or not (extracted / "val").is_dir():
        raise FileNotFoundError(f"Imagenette archive was extracted but expected folders are missing: {extracted}")
    return extracted


def validate_imagenette_classes(dataset):
    found = set(dataset.classes)
    expected = set(IMAGENETTE_NAMES.keys())
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise RuntimeError(f"Imagenette class folders do not match the expected ten classes. Missing={missing}, extra={extra}")


def build_id_datasets(data_root):
    imagenette_root = ensure_imagenette(Path(data_root) / "imagenette")
    train_aug = datasets.ImageFolder(imagenette_root / "train", transform=get_transforms(True))
    train_eval = datasets.ImageFolder(imagenette_root / "train", transform=get_transforms(False))
    validation = datasets.ImageFolder(imagenette_root / "val", transform=get_transforms(False))
    validate_imagenette_classes(train_aug)
    validate_imagenette_classes(validation)
    if train_aug.class_to_idx != validation.class_to_idx:
        raise RuntimeError("Imagenette training and validation class mappings differ")
    return train_aug, train_eval, validation


def find_imagenet_r_folder(root):
    root = Path(root)
    candidates = [root / "imagenet-r", root / "imagenet-r" / "imagenet-r", root]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        class_directories = [
            child for child in candidate.iterdir()
            if child.is_dir() and re.fullmatch(r"n\d{8}", child.name)
        ]
        if len(class_directories) >= 20:
            return candidate
    return None


def ensure_imagenet_r(root):
    root = Path(root)
    folder = find_imagenet_r_folder(root)
    if folder is not None:
        return folder
    root.mkdir(parents=True, exist_ok=True)
    try:
        download_and_extract_archive(
            IMAGENET_R_URL,
            str(root),
            filename="imagenet-r.tar",
            md5=IMAGENET_R_MD5,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not download ImageNet-R automatically. The archive is about 30,000 labelled images "
            "and requires several GB of free space. Check internet access and rerun. "
            f"Source: {IMAGENET_R_URL}. Error: {exc}"
        ) from exc
    folder = find_imagenet_r_folder(root)
    if folder is None:
        raise FileNotFoundError(f"ImageNet-R was downloaded but its synset folders were not found under {root}")
    return folder


class FilteredImageNetR(Dataset):
    def __init__(self, folder, synsets, transform, max_per_class, split, seed=42):
        self.folder = Path(folder)
        self.transform = transform
        self.split = split
        self.samples = []
        self.classes = list(synsets)
        self.class_to_idx = {synset: index for index, synset in enumerate(self.classes)}
        for synset in self.classes:
            class_folder = self.folder / synset
            files = sorted(
                path for path in class_folder.rglob("*")
                if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
            )
            if max_per_class is not None:
                class_seed = seed + sum(ord(character) for character in synset)
                class_rng = random.Random(class_seed)
                class_rng.shuffle(files)
                files = files[:max_per_class]
            self.samples.extend((str(path), synset) for path in files)
        if not self.samples:
            raise RuntimeError(f"ImageNet-R {split} split contains no images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, synset = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_idx[synset]


def build_ood_datasets(data_root, supported_synsets, seed=42):
    folder = ensure_imagenet_r(Path(data_root) / "imagenet-r")
    all_synsets = sorted(
        child.name for child in folder.iterdir()
        if child.is_dir() and re.fullmatch(r"n\d{8}", child.name)
    )
    supported = set(supported_synsets)
    overlap = sorted(set(all_synsets) & supported)
    ood_synsets = [synset for synset in all_synsets if synset not in supported]
    if len(ood_synsets) < 20:
        raise RuntimeError(
            f"Only {len(ood_synsets)} ImageNet-R classes remain after excluding supported Imagenette classes. "
            f"Excluded overlap: {overlap}"
        )

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(ood_synsets, dtype=object)
    rng.shuffle(shuffled)
    train_end = int(len(shuffled) * 0.70)
    validation_end = int(len(shuffled) * 0.85)
    split_classes = {
        "train": sorted(shuffled[:train_end].tolist()),
        "val": sorted(shuffled[train_end:validation_end].tolist()),
        "test": sorted(shuffled[validation_end:].tolist()),
    }
    if not split_classes["train"] or not split_classes["val"] or not split_classes["test"]:
        raise RuntimeError("ImageNet-R does not have enough distinct classes for train/validation/test splits")

    datasets_by_split = {
        "train": FilteredImageNetR(folder, split_classes["train"], get_transforms(False), 60, "train", seed),
        "val": FilteredImageNetR(folder, split_classes["val"], get_transforms(False), 30, "val", seed + 1),
        "test": FilteredImageNetR(folder, split_classes["test"], get_transforms(False), 30, "test", seed + 2),
    }
    manifest = {
        "dataset": "ImageNet-R",
        "supported_synsets_excluded_from_ood": overlap,
        "ood_train_synsets": split_classes["train"],
        "ood_validation_synsets": split_classes["val"],
        "ood_test_synsets": split_classes["test"],
        "images_per_class_cap": {"train": 60, "val": 30, "test": 30},
        "train_images": len(datasets_by_split["train"]),
        "validation_images": len(datasets_by_split["val"]),
        "test_images": len(datasets_by_split["test"]),
    }
    manifest_path = Path(data_root) / "imagenet_r_membership_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(
        f"ImageNet-R OOD classes: {len(ood_synsets)}; excluded supported synsets: {len(overlap)}; "
        f"images train/val/test: {len(datasets_by_split['train'])}/"
        f"{len(datasets_by_split['val'])}/{len(datasets_by_split['test'])}",
        flush=True,
    )
    return datasets_by_split["train"], datasets_by_split["val"], datasets_by_split["test"]


def make_loader(dataset, batch_size, shuffle=False, workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def split_id_validation_indices(dataset, seed=42):
    labels = np.asarray(dataset.targets, dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []
    test_indices = []
    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        first = int(len(class_indices) * 0.50)
        second = int(len(class_indices) * 0.75)
        train_indices.extend(class_indices[:first].tolist())
        validation_indices.extend(class_indices[first:second].tolist())
        test_indices.extend(class_indices[second:].tolist())
    return train_indices, validation_indices, test_indices


def split_classifier_indices(dataset, seed=42, train_fraction=0.9):
    labels = np.asarray(dataset.targets, dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []
    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        boundary = int(len(class_indices) * train_fraction)
        train_indices.extend(class_indices[:boundary].tolist())
        validation_indices.extend(class_indices[boundary:].tolist())
    return train_indices, validation_indices
