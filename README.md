# Trustworthy Brain Tumor Classification from MRI Using Leakage-Aware and Explainable Deep Learning

Major Project-II, Computer Engineering (AI), Marwadi University
**Team:** K Praveen Kumar, Ishika Kaur, Shreesha | **Guide:** Akshara Prachi

4-class classification of brain MRI slices: **glioma, meningioma, notumor, pituitary**.

## Why this project
The popular Kaggle *Brain Tumor MRI Dataset* contains many duplicate and near-duplicate images, and many of them
appear in both the Training and Testing folders. Models trained and tested on the original folders can look far
better than they really are. This project:

1. **Audits duplicates** (MD5 exact copies, perceptual-hash near copies, deep-feature look-alikes) and builds a
   **leakage-free split**. Every model is compared on the original split and on the clean split.
2. Compares **lightweight models** (mobilenetv3_small_100, efficientnet_b0) with **heavy baselines** (vgg16,
   resnet50, densenet121, inception_v3) on accuracy, parameters, size and CPU time per image.
3. Checks **explainability** with Grad-CAM and an *inside-brain ratio* (share of the heatmap inside an Otsu brain
   mask). *(planned)*
4. Adds **calibration and uncertainty**: temperature scaling, ECE, MC dropout, and a "refer to radiologist" flag
   below 0.70 confidence. *(planned)*

## Dataset
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by M. Nickparvar
(Kaggle). The dataset is **not included** in this repository. Download it and place it like this:

```
data/
  v1/   Training/{glioma,meningioma,notumor,pituitary}   Testing/{...}   # Kaggle version 1: 7,023 images
  v2/   Training/{...}                                    Testing/{...}   # latest version: 7,200 images
```
`data/` is treated as read-only by all code.

## Repository layout
```
src/
  data.py      file discovery, preprocessing (brain crop + pad + resize 224), image cache, Dataset, transforms
  audit.py     duplicate audit: MD5, pHash, GPU pairwise Hamming distances, union-find duplicate groups
  splits.py    ResNet-50 look-alike features, duplicate clusters, original split and leakage-free clean split
  models.py    the 6 timm models with per-model batch size, image size and precision settings
  train.py     training pipeline (one run per model x split), run summaries, health checks, plots
  evaluate.py  test evaluation: metrics, temperature scaling, ECE, MC dropout, referral rule, latency
  gradcam.py   Grad-CAM of the predicted class, brain mask (Otsu + largest blob + filled holes), inside-brain ratio
  report.py    the six final report figures (colour-blind-safe palette)
notebooks/
  00_data_check.ipynb       counts, versions, corrupt files, image sizes and colour modes
  01_preprocessing.ipynb    crop/resize cache, augmentation, DataLoader speed
  02_duplicate_audit.ipynb  exact and near duplicates, train-test leakage
  03_splits.ipynb           look-alikes, clusters, original and clean splits
  04_training.ipynb         training of all models (settings cell at the top)
  05_heavy_training.ipynb   resnet50 and vgg16 baselines
  06_evaluation.ipynb       test metrics, calibration, MC dropout, referral, size and latency
  07_gradcam.ipynb          Grad-CAM on every test image, inside-brain ratio, galleries
  08_results_plots.ipynb    final figures (outputs/plots/final) and plain-English summary
PROGRESS.md                 week-by-week log of what was done and every result
outputs/                    generated: cache, audit files, splits, runs, plots (not in git)
```

## Setup
Developed on Windows 11, Python 3.12.5, NVIDIA GTX 1650 Ti (4 GB).
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt  # PyTorch 2.14.1 + CUDA 12.6, timm 1.0.30, ...
```
Run the notebooks in order (00 to 08). Each one writes its results to `outputs/`. Pretrained weights are downloaded
once into `.cache/` inside the project.

## Training recipe (`src/train.py`)
- timm ImageNet-pretrained backbone with a new 4-class head, channels_last memory layout, seed 42
- AdamW, 1 warm-up epoch, then cosine learning-rate decay; label smoothing 0.1; inverse-frequency class weights
- early stopping on validation macro-F1 (patience 5); only the best checkpoint is kept
- mixed precision (float16) with a NaN guard; efficientnet_b0 trains in float32 (float16 gave NaN on this GPU)
- out-of-memory retry: halve the batch and double gradient accumulation
- every run writes `outputs/runs/<split>/<model>/` with `best.pt`, `history.csv` and `config.json`, and finished
  runs are skipped, so training can be stopped and resumed

## Results so far
All numbers come from the notebooks; see `PROGRESS.md` for details.

**Duplicate audit.** v1 has 503 exact-duplicate pairs, v2 has 227. Share of original Testing images with a twin
in Training (any link: exact, near or look-alike): **v1 70.2%, v2 63.1%**; clean split: **0.0%** for both.

**Splits (train / val / test images):**

| | original | clean |
|---|---|---|
| v1 | 4,855 / 857 / 1,311 | 2,426 / 520 / 520 |
| v2 | 4,760 / 840 / 1,600 | 2,508 / 537 / 538 |

**Light models: validation results** (the test sets have not been used yet):

| Split | Model | Val acc | Val macro-F1 | Epochs | Stopped by |
|---|---|---|---|---|---|
| v1 original | mobilenetv3_small_100 | 0.9918 | 0.9915 | 18 | early stopping |
| v1 original | efficientnet_b0 | 0.9860 | 0.9856 | 9 | early stopping |
| v1 clean | mobilenetv3_small_100 | 0.9808 | 0.9809 | 20 | max epochs |
| v1 clean | efficientnet_b0 | 0.9423 | 0.9394 | 3 | val acc >= 0.92 |
| v2 original | mobilenetv3_small_100 | 0.9738 | 0.9736 | 4 | val acc >= 0.92 |
| v2 original | efficientnet_b0 | 0.9750 | 0.9750 | 2 | val acc >= 0.92 |
| v2 clean | mobilenetv3_small_100 | 0.9683 | 0.9639 | 18 | early stopping |
| v2 clean | efficientnet_b0 | 0.9479 | 0.9432 | 3 | val acc >= 0.92 |

To save time, four runs were stopped as soon as validation accuracy reached 0.92. Their scores are not comparable
with the runs that used normal early stopping. Validation scores on the original split are expected to be inflated,
because its validation images come from the duplicate-filled Training folder. The honest comparison is the test set
evaluation, which comes next.

## Status
- [x] Week 1: data check
- [x] Week 2: preprocessing and image cache
- [x] Week 3: duplicate audit (MD5, pHash)
- [x] Week 4: look-alike audit, clusters, original and clean splits
- [x] Week 5: training pipeline
- [x] Week 6: light models on original and clean splits
- [x] Week 7: heavy baselines resnet50 and vgg16 on v1 (densenet121 and inception_v3 not trained)
- [x] Week 8: test-set evaluation, parameters, size, CPU/GPU ms per image, temperature scaling, ECE, MC dropout, referral flag
- [x] Week 9: Grad-CAM and inside-brain ratio, final figures, results summary
- [ ] Fair original-vs-clean comparison (equal training budgets) and report writing
