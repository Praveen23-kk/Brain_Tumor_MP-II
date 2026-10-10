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

## Session 2026-10-10 (Week 2: preprocessing)
### Done
- `src/data.py`: brain_mask, crop_brain (Otsu + largest contour, 2% margin, fallback), pad_resize (pad to square,
  no stretching), preprocess, make_index_csv, build_cache (resumable), train/eval transforms, denormalize,
  MRIDataset (CSV path/label/split), time_loader, plots.
- `notebooks/01_preprocessing.ipynb` executed end-to-end (0 errors). Plots in outputs/plots/week2/.
- CSVs in outputs/splits/: v1_original, v2_original, v1_cache224, v2_cache224.

### Results (from the notebook run)
- Crop fallback: v1 10/7,023 (0.14%), v2 14/7,200 (0.19%); mostly notumor.
- Cache: v1 181.1 MB, v2 185.7 MB (224x224 grey PNG).
- Item: (3,224,224) float32, channels identical, mean -1.379 (black background, expected).
- DataLoader, 500 images (pass 2 images/s): raw w0 124.7, raw w2 198.5, cached w0 205.4, cached w2 298.9.
  Worker start-up on Windows is about 30 s (pass 1). Cache speed-up 1.7x.
- Decision: cached data + num_workers=2 + persistent_workers=True.

## Session 2026-10-10 (Week 3: duplicate audit)
### Done
- `src/audit.py`: MD5 + 64-bit pHash, GPU pairwise Hamming (matrix product, 2,048-row chunks), check against
  imagehash, pair table, threshold summary, test twins per class, union-find duplicate groups, plots.
- `notebooks/02_duplicate_audit.ipynb` executed (0 errors). Results in outputs/audit/v1 and outputs/audit/v2.

### Results (from the notebook run)
- GPU check: 0 mismatches vs imagehash (1,000 pairs per version); a 304-image brute-force CPU check matched too.
  v2: 25.9 M pairs in 0.065 s, peak VRAM 143 MB.
- Exact (MD5) pairs: v1 503 (6,726 distinct of 7,023), v2 227 (7,013 distinct of 7,200).
- Every pHash has exactly 32 one-bits, so distances are always even (T=3 = T=2, T=5 = T=4).
- Test images with a Training twin: v1 41.7 / 55.2 / 87.3 %, v2 31.6 / 44.6 / 78.0 % at T = 3 / 5 / 8.
- At T=5 per class (v1 | v2): glioma 22.7 | 21.5, meningioma 51.3 | 43.3, notumor 95.1 | 88.3, pituitary 38.0 | 25.5 %.
- Label-conflict pairs: v1 2 / 33 / 1,850, v2 11 / 47 / 1,992 at T = 3 / 5 / 8.
- Groups at T=5: v1 4,516 groups (542 span train and test), v2 4,755 (520 span). Saved as groups_T5.csv.

### Decisions
- Threshold T = 5. Train-test counts have a valley at d=4 (v1: 945, 707, 625, then 842, 1,580, 4,599),
  pairs with d <= 4 look like the same image, d = 8 are different scans, and label conflicts explode at T = 8.

### Next step
Check the label-conflict pairs by eye, add the deep-feature look-alike check, then build the leakage-free split
from groups_T5 (whole groups in one split, stratified by class, seed 42).
