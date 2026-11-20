# Two-Fixed-Model Adaptive Class-Membership Detector

This project automatically trains and evaluates exactly two original classifiers:

1. ResNet-18
2. MobileNetV3-Small

It then freezes each classifier and trains a separate auxiliary neural network for each one. The app tests every uploaded image against both models; the user does not select an architecture.

## Why the earlier run failed

The earlier project required the entire ImageNet-1K dataset at `data/imagenet1k`. That dataset was not present, so training stopped with `FileNotFoundError`. This version does not require the ImageNet-1K dataset directory, does not use CIFAR-10, and does not expect you to prepare an ImageNet folder manually.

## Fixed dataset and class vocabulary

To make the ground-truth rule exact and reproducible, each original classifier is fine-tuned to have exactly ten output logits corresponding to the ten labelled Imagenette classes. The pretrained ImageNet weights are only an initialization; the final classifier head is replaced with a ten-class head and fine-tuned on Imagenette.

**ID / label 1:** Every Imagenette image whose ground-truth directory label is one of the ten output classes. The original classifier may predict the wrong one of the ten classes; that image still receives ID label 1.

**OOD / label 0:** Images from ImageNet-R whose ground-truth WordNet synset is not one of the ten Imagenette output synsets. The original classifier's prediction is ignored when assigning the auxiliary label.

The ten supported output classes are tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, and parachute.

| Image ground truth | Original prediction | Auxiliary target |
|---|---|---:|
| Supported Imagenette class | Correct class | 1 / YES |
| Supported Imagenette class | Wrong Imagenette class | 1 / YES |
| ImageNet-R category outside the ten output synsets | Any of the ten logits | 0 / NO |

The target is based on ground-truth dataset membership, never on `predicted_class == ground_truth_class` or the classifier's confidence.

## Data downloads

The code downloads the public datasets automatically on the first run:

- Imagenette2-160: original classifier training, ID auxiliary training/validation/test. [Dataset source](https://github.com/fastai/imagenette).
- ImageNet-R: OOD auxiliary training/validation/test. [Official dataset and download](https://github.com/hendrycks/imagenet-r). The code excludes any ImageNet-R synset that is one of the ten supported Imagenette synsets, then separates the remaining OOD synsets into disjoint train, validation, and test class groups. It samples up to 60 images per OOD training class and 30 per validation/test class, using class-balanced selection for auxiliary training.
- Torchvision pretrained weights for ResNet-18 and MobileNetV3-Small: initialization of the two original classifiers.

No CIFAR-10 or full ImageNet-1K dataset is used. An internet connection and several GB of free disk space are required for ImageNet-R. The official ImageNet-R collection contains 30,000 renditions across 200 ImageNet classes; the code filters out supported classes before training. Data are cached under `data/`; subsequent runs reuse the downloads.

## Install and run on Windows

Use Python 3.10 or 3.11. In PowerShell, open this folder and run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_two_models.py
python app.py
```

Open `http://127.0.0.1:5000` and upload an image. The app returns a result for both fixed models, the auxiliary ID score, the calibrated decision threshold, and the original classifier's top-three logits/classes.

You can also run `run_windows.bat` to install dependencies, train both models, and start the app.

## Training options

The default run uses five epochs for each original classifier and eight epochs for each auxiliary detector:

```powershell
python train_two_models.py
```

For a shorter smoke-sized training run on a machine with a GPU or for debugging:

```powershell
python train_two_models.py --backbone-epochs 1 --aux-epochs 1 --batch-size 32
```

If the pretrained model weights cannot be downloaded but the two image datasets are already present locally, initialize the original models from scratch:

```powershell
python train_two_models.py --no-pretrained --backbone-epochs 20
```

A from-scratch run generally needs more epochs and may produce lower classifier accuracy than transfer learning.

## What the auxiliary network receives

For each image, the feature vector contains:

- A pooled representation of the normalized input image.
- All ten original classifier logits.
- Global-average-pooled activations from two intermediate layers.

For ResNet-18, the selected layers are `layer2` and `layer3`. For MobileNetV3-Small, they are `features[4]` and `features[8]`. The original classifier is frozen while the auxiliary network is trained with `BCEWithLogitsLoss`.

The OOD training split is balanced across its unseen classes; the validation set is also sampled across classes. The validation set is used to select a threshold that maximizes Youden's J statistic. The test split is kept separate and reports AUROC, accuracy, ID acceptance, OOD rejection, and original-classifier accuracy on ID images.

## Test the installed model code without downloading datasets

```powershell
python smoke_test.py
```

This checks both architectures, their ten-logit classifier heads, both intermediate feature hooks, and an auxiliary-network optimizer step. It is an offline code-path test, not a substitute for training on the real datasets.

## Output files

```text
outputs/
  resnet18_original.pt
  resnet18_auxiliary.pt
  mobilenet_v3_small_original.pt
  mobilenet_v3_small_auxiliary.pt
  results.json
  results.csv
```

## Important scope and limitations

The ten output classes are the ten Imagenette classes because these two fixed models are fine-tuned with ten-logit heads. This deliberately avoids the previous missing ImageNet-1K folder and makes the supported class set explicit. It is not a detector for the full 1,000-class ImageNet vocabulary.

ImageNet-R provides labelled renditions from ImageNet categories. The code excludes the exact ten supported WordNet synsets and assigns OOD labels only to the remaining synsets. It also separates OOD train, validation, and test by class, so the detector is tested on unknown classes not seen during auxiliary training. A different dog breed, for example, is OOD when its exact synset is not one of the ten output classes. If you intend superclass-level membership (for example, every dog breed counts as Dog), that requires a separate WordNet ancestor mapping and broader exclusion rule.

A fixed classifier always produces logits for a valid image. The auxiliary YES/NO output estimates whether the image belongs to the classifier's supported ground-truth class set; it does not switch logits off and cannot guarantee perfect rejection of every unseen image.
