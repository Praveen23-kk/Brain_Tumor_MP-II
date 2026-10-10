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

## Session 2026-10-10 (Week 4: look-alikes, clusters, splits)
### Done
- `src/splits.py`: ResNet-50 features (float32, weights cached in D:/Brain-Tumor/.cache), GPU cosine pairs,
  clusters (union-find over exact + pHash<=5 + cosine>=0.95), conflict report, original split, clean split
  (with an assert that no cluster crosses splits), split summary, leak table, plots.
- `src/audit.py`: union-find moved into a shared `union_find()` (same groups as before, checked).
- `src/data.py`: `MRIDataset(split=None)` reads all rows.
- `notebooks/03_splits.ipynb` executed (0 errors). Split CSVs: outputs/splits/{v1,v2}_split_{original,clean}.csv.

### Results (from the notebook run)
- Look-alike pairs (cos >= 0.95): v1 5,464, v2 5,024. Random-pair cosine: mean 0.63, 99th percentile 0.83.
  All 8 eye-checked pairs at 0.95-0.96 (not found by pHash) are the same scan.
- Clusters: v1 3,483 (largest 47 = 0.67%), v2 3,607 (largest 47). Conflict clusters: v1 17 (125 images), v2 24 (175).
  They contain true label errors (same image, two classes) and chained different scans.
- Test leakage, % of original Testing (exact / near / look-alike / any): v1 7.9 / 55.2 / 57.1 / 70.2,
  v2 0.0 / 44.6 / 50.8 / 63.1. Clean split test: 0.0 for both.
- Splits (train / val / test): v1 original 4,855 / 857 / 1,311; v1 clean 2,426 / 520 / 520;
  v2 original 4,760 / 840 / 1,600; v2 clean 2,508 / 537 / 538.
- The clean split is unbalanced: notumor about 11% (most notumor files are copies).

### Issues / decisions
- HARDWARE: AMP float16 on this GTX 1650 Ti gave NaN outputs (cuDNN) for batches of 48+ (resnet50).
  Batches of 16 and 32 were fine. Training must check for NaN/Inf losses and fall back to float32 if needed.
- The clean split needs class weights plus per-class / balanced metrics.

### Next step
Plan the training pipeline (smoke test of 1 epoch first) and train the light models on both splits.

## Session 2026-10-10 (Week 5: training pipeline)
### Done
- `src/models.py`: the 6 timm models with per-model batch / image size / heavy flag, `build_model` (4 classes,
  channels_last), `set_backbone_trainable` (heavy models: backbone frozen for 2 epochs), `count_params`.
- `src/train.py`: `run(model, split_name, split_csv)` -> outputs/runs/<split>/<model>/{best.pt, history.csv, config.json}.
  AdamW, 1 warm-up epoch + cosine LR (per step), label smoothing 0.1, inverse-frequency class weights (train part only),
  AMP + channels_last, early stopping on val macro-F1 (patience 5, max 20), NaN guard (falls back to float32),
  OOM retry (halve batch, double grad accumulation), skip finished runs, crash-safe file writes (temp file + rename),
  `plot_history`. The test part of the splits is NOT used.
- `notebooks/04_training.ipynb` executed (0 errors): smoke test, mobilenetv3_small_100 on v1_clean and v2_clean, curves.
  Plots: outputs/plots/week5/curves_{v1,v2}_clean_mobilenetv3_small_100.png.

### Results (from the run outputs; validation only)
| Run | Epochs | Best epoch | Best val macro-F1 | Val acc (best) | Peak GPU MB | Train time s |
|---|---|---|---|---|---|---|
| v1_clean / mobilenetv3_small_100 | 20 | 18 | 0.9809 | 0.9808 | 690.9 | 674.7 |
| v2_clean / mobilenetv3_small_100 | 18 (early stop) | 13 | 0.9639 | 0.9683 | 453.5 | 535.3 |
- 1.52 M parameters, batch 48, no OOM, AMP never had to be switched off (float16 gave no NaN).
- Smoke test (50 images, 1 epoch): all 3 files written, finite loss, second call skipped. 121 s, mostly one-off cost.
- Steady epoch time 26-27 s. The first epoch costs 77-160 s: Windows worker start-up plus cuDNN benchmark
  autotuning (30-90 s per new batch shape, measured) -> training loader uses drop_last.
- Epoch 1-3 validation is low (v1 epoch 1: everything predicted as notumor). Checked: float16 = float32 predictions;
  the cause is BatchNorm running statistics lagging behind fast-changing weights (same epoch-2 weights: 0.40 val acc
  with running stats vs 0.85 with batch stats). It disappears by epoch 5.

### Issues / notes
- Hugging Face downloads failed (401) while `~/.cache/huggingface/token` (stale token) was used: timm must be imported
  only via src.models, which points HF_HOME to D:/Brain-Tumor/.cache first.
- The "lr_scheduler.step() before optimizer.step()" warning is harmless: GradScaler skips the very first step while it
  calibrates its float16 scale.

### Next step
Train efficientnet_b0 and the heavy baselines on v1/v2 clean AND original splits (same pipeline), then evaluate
every best.pt once on its test split (accuracy, macro-F1, confusion matrix) and compare original vs clean.

## Session 2026-10-10/11 (Week 6: light models on original and clean splits)
### Done
- `src/train.py`: `run_summary` (val scores at the best epoch + time + memory), `health_check` (NaN, overfitting:
  train-val acc gap > 0.10 at best epoch, or final val loss > 25% above its minimum while train loss falls),
  `plot_all_histories` (all runs in one figure), optional `target_acc` stop (records `stopped_by` in config.json).
- `notebooks/04_training.ipynb` (Weeks 5-6): settings cell (versions, split types, models, per-model epochs,
  patience, target accuracy), efficientnet_b0 smoke test, training loop with a summary line + health check per run,
  combined curves (outputs/plots/week6/all_runs_curves.png), summary table. Executed end-to-end, 0 errors.

### Protocol changes made during the week (to save time; decided by me)
- efficientnet_b0 is GPU-bound (GPU 99%, CPU 12%, measured): ~4.7 min/epoch on the original split. Max epochs set
  to 12 for efficientnet (cosine schedule built for 12, so LR still decays to 0).
- v2 mobilenet max epochs 12 (only v2_original used it; v2_clean mobilenet had finished in Week 5 with 20).
- Last 4 runs (v1_clean effnet, v2_original mobilenet + effnet, v2_clean effnet) stop as soon as val acc >= 0.92.
  So their val scores are NOT comparable with the 4 runs that used normal early stopping (those reached 0.98-0.99).

### Results (from the run outputs; validation only, test untouched)
| Split | Model | Val acc | Val macro-F1 | Epochs (best) | Max ep. | Stopped by | Time min | Peak GPU MB | Float16 |
|---|---|---|---|---|---|---|---|---|---|
| v1_original | mobilenetv3_small_100 | 0.9918 | 0.9915 | 18 (13) | 20 | patience | 18.0 | 707 | yes |
| v1_original | efficientnet_b0 | 0.9860 | 0.9856 | 9 (4) | 12 | patience | 44.5 | 2,237 | yes |
| v1_clean | mobilenetv3_small_100 | 0.9808 | 0.9809 | 20 (18) | 20 | max epochs | 11.2 | 691 | yes |
| v1_clean | efficientnet_b0 | 0.9423 | 0.9394 | 3 (3) | 12 | target 0.92 | 10.8 | 3,082 | NaN at epoch 1 -> float32 |
| v2_original | mobilenetv3_small_100 | 0.9738 | 0.9736 | 4 (4) | 12 | target 0.92 | 5.3 | 707 | yes |
| v2_original | efficientnet_b0 | 0.9750 | 0.9750 | 2 (2) | 12 | target 0.92 | 11.2 | 2,795 | NaN at epoch 1 -> float32 |
| v2_clean | mobilenetv3_small_100 | 0.9683 | 0.9639 | 18 (13) | 20 | patience | 8.9 | 454 | yes |
| v2_clean | efficientnet_b0 | 0.9479 | 0.9432 | 3 (3) | 12 | target 0.92 | 8.0 | 1,462 | yes |
- With the same stopping rule, original > clean on validation (mobilenet v1: 0.9915 vs 0.9809). Expected: the
  original split's val images come from the duplicate-filled Training folder. Test evaluation will show the real gap.
- Overfitting: none flagged; train and val curves stay close in every run.
- Epoch times: mobilenet ~52 s (original) / ~26 s (clean); efficientnet ~279 s (original) / ~143 s (clean, float16)
  vs ~167-212 s (clean, float32).

### Issues
- **float16 NaN in 2 of 4 efficientnet runs**, both at epoch 1 (v1_clean, v2_original). The NaN guard switched those
  runs to float32 and training continued with finite losses, so the results are valid, but those 2 runs were
  slower (~1.2-1.5x) and used more GPU memory. Their time and memory numbers are not comparable with the float16 runs.
  Not divergence (LR was still in warm-up, losses normal): same float16/cuDNN problem on this GTX 1650 Ti as Week 4.
  It is random from run to run because cudnn.benchmark picks the convolution algorithm by timing each time.
- Fix APPLIED (after this week's runs): per-model `amp` setting in `src/models.py`; efficientnet_b0 now always trains in
  float32 (`amp_at_start` is recorded in config.json). Checked with a 50-image, 1-epoch run: no NaN, finite loss,
  peak GPU 3,144 MB at batch 32 (close to the 4 GB limit; the OOM retry halves the batch if needed).
  The 4 existing efficientnet runs were NOT retrained (my decision): 2 used float16 throughout, 2 switched to
  float32 after epoch 1.
- The 4 target-stopped runs were NOT retrained either (my decision).

### Next step
Evaluate every best.pt once on its test split (accuracy, macro-F1, confusion matrix) and compare original vs clean,
keeping in mind which runs stopped at the 0.92 target.
