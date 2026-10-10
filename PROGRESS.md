# Progress Log

## Session 2026-10-08
### Done
- Project skeleton: CLAUDE.md, .gitignore, empty src/, notebooks/, outputs/, .venv (Python 3.12.5).
- Checked data: data/v2 present (Training 4 x 1400 = 5,600; Testing 4 x 400 = 1,600).
- GPU check: GTX 1650 Ti, 4 GB, driver supports CUDA 13.4 -> chose PyTorch cu126 build.
- Started package install (keeping everything on D:, no pip cache on C:), stopped at ~490 MB of
  ~2.5 GB because internet was too slow (~47 KB/s). Partial download deleted; nothing installed yet.

### Results
- None yet (no code written or run).

### Issues
- data/v1 is EMPTY -> download the original 7,023-image dataset into data/v1/Training and data/v1/Testing.
- .venv has no packages yet (torch missing).

### Next step
(Superseded by session 2026-10-10 below.) On fast internet, run in PowerShell from D:\Brain-Tumor (keeps everything off C:):
```
New-Item -ItemType Directory -Force D:\Brain-Tumor\.piptmp
$env:TMP="D:\Brain-Tumor\.piptmp"; $env:TEMP="D:\Brain-Tumor\.piptmp"
.venv\Scripts\python.exe -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python.exe -m pip install --no-cache-dir timm opencv-python imagehash scikit-learn pandas matplotlib grad-cam tqdm
Remove-Item -Recurse -Force D:\Brain-Tumor\.piptmp
```
Then: verify torch.cuda.is_available(), then plan the data pipeline + duplicate audit.

## Session 2026-10-10
### Done
- Downloaded original dataset (Kaggle masoudnickparvar/brain-tumor-mri-dataset, **version 1**, 2021-09-24,
  156 MB zip, integrity-checked) and extracted into data/v1 (zip kept in .downloads/v1 as backup).
- Installed packages into .venv, everything on D: (no pip cache, temp on D:). Exact versions in requirements.txt.
  torch 2.14.1+cu126, torchvision 0.29.1+cu126, timm 1.0.30, grad-cam 1.5.7, opencv 5.0.0,
  scikit-learn 1.9.1, pandas 3.0.6, imagehash 4.3.2, matplotlib 3.11.2, tqdm, ipykernel.
- GPU check passed: CUDA available, GTX 1650 Ti 4.3 GB, cuDNN 9.10; AMP + channels_last forward
  of mobilenetv3_small_100 on GPU OK.

### Results (counted by script)
| Split | glioma | meningioma | notumor | pituitary | Total |
|---|---|---|---|---|---|
| v1 Training | 1321 | 1339 | 1595 | 1457 | 5712 |
| v1 Testing  | 300  | 306  | 405  | 300  | 1311 |
| v2 Training | 1400 | 1400 | 1400 | 1400 | 5600 |
| v2 Testing  | 400  | 400  | 400  | 400  | 1600 |

### Next step
Plan (needs OK) for: src/ data utilities (seed, 224 px cache in outputs/), then the duplicate audit
(MD5 / pHash / deep-feature) on v1 and the leakage-free split.

## Session 2026-10-10 (Week 1: data check)
### Done
- `src/data.py` (shared helpers: version/class folder discovery (case-insensitive), file table, counts,
  v1/v2 identification, image integrity check, sample grid, plots) + `src/__init__.py`.
- `notebooks/00_data_check.ipynb`: built and executed end-to-end with nbclient (0 errors), saved with outputs.
- Plots in `outputs/plots/week1/`: class_counts.png, samples_v1.png, samples_v2.png, image_sizes.png.
- Added nbformat + nbclient (to run notebooks from the terminal); requirements.txt refreshed.

### Results (all from the notebook run)
- Versions found: v1 = 7,023 (5,712 train / 1,311 test), v2 = 7,200 (5,600 / 1,600). Both match the releases exactly.
- v1 per class: glioma 1,621, meningioma 1,645, notumor 2,000, pituitary 1,757 (unbalanced). v2: 1,800 each.
- v2 vs v1: glioma +179, meningioma +155, notumor -200, pituitary +43, so v2 is re-curated, not a subset of v1.
- Corrupt files: 0 of 14,223. All named .jpg; 9 are really PNG (5 v1, 4 v2), all in notumor.
- Colour modes: v1 3,093 L / 3,926 RGB / 3 RGBA / 1 P; v2 3,067 L / 4,129 RGB / 3 RGBA / 1 P.
  Mode is class-linked (v1 glioma 1,426 L vs 195 RGB; notumor 1,967 RGB vs 29 L).
- Sizes: 512x512 = 67.5% (v1), 69.6% (v2); width 150-1,920 px. Share of 512x512 per class:
  v1 glioma 100.0 / meningioma 84.3 / notumor 1.0 / pituitary 97.6 %; v2 95.8 / 86.1 / 1.1 / 95.6 %.
  So notumor comes from a different source (source-bias shortcut risk).
- v1 sample grid shows a visually identical meningioma pair (duplicate hint).

### Decisions
- Convert every image to 3-channel RGB; resize once to 224 (299 for inception_v3) and cache in outputs/.

### Next step
Week 2: plan (needs OK) the duplicate audit on v1 (MD5 / pHash / deep-feature, within and across
Training/Testing) and the leakage-free split.
