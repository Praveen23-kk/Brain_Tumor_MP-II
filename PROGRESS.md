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

## Session 2026-10-12 (Week 7: heavy baselines resnet50 + vgg16 on v1)
### Done
- `src/train.py`:
  - **Epoch-level resume**: `last.pt` (model, optimizer, LR schedule, GradScaler, AMP state, history, early-stop
    counters, shuffle generator + all RNG states) is saved after every epoch and deleted when the run finishes,
    so only best.pt is kept. `run()` continues an interrupted run with the batch it was using. config.json gets
    `resumed_at_epochs`. Test: killed a run after epoch 1 and restarted it -> "resuming from epoch 2", history 1-3 with no gap,
    LR schedule continued. (Not bit-identical: DataLoader workers restart with new augmentation seeds.)
  - Command-line entry: `python -u -m src.train <model> <split> --epochs 12 --target-acc 0.92` (background runs).
  - `probe()` (GPU memory + seconds per batch on random images, backbone unfrozen = worst case),
    `estimate_minutes()`, `runs_table()` (all runs, light + heavy), `to_markdown()`. `history.csv` now has an `accum` column.
- `src/models.py`: **resnet50 now trains in float32** (`amp=False`), see Issues.
- `notebooks/05_heavy_training.ipynb` (Week 7): probe, estimates vs actual time, smoke test, NaN evidence, training loop
  (finished runs skipped), checkpoint check, curves, full runs table. Executed end-to-end, 0 errors.
  Plots: outputs/plots/week7/heavy_runs_curves.png + curves_v1_*_{resnet50,vgg16}.png.

### Protocol (decided by me)
- v1 only (4 runs). Max 12 epochs, patience 5, stop at val acc >= 0.92 (same rule as the last 4 light runs).
- Heavy recipe as in CLAUDE.md: LR 3e-4, backbone frozen for epochs 1-2. Batch resnet50 24, vgg16 16 (no OOM).
- Exceptions: vgg16 v1_original used target 0.90, and I then **stopped it by hand after epoch 5** (val acc 0.897;
  `stopped_by = user_stop`). It was closed with a resume that trained 0 epochs, so it shows `resumed = True`.

### Probe (measured, random images, backbone unfrozen; outputs/runs/probe_week7.json)
| Model | Batch | Precision | s / train batch | s / frozen batch | s / eval batch | Peak alloc MB | Reserved MB |
|---|---|---|---|---|---|---|---|
| resnet50 | 24 | float16 | 2.7286 | 1.0809 | 0.3048 | 1762.0 | 1830.0 |
| vgg16 | 16 | float16 | 2.4595 | 0.7321 | 0.7295 | 2897.6 | 3212.0 |
| resnet50 | 24 | float32 | 2.2683 | 0.9403 | 0.1120 | 3021.7 | 3240.0 |
| vgg16 | 16 | float32 | 1.2493 | 0.4246 | 0.4243 | 3551.4 | 4532.0 |
| vgg16 | 8 | float32 | 0.4137 | 0.0997 | 0.0994 | 3256.9 | 3644.0 |
- float32 is FASTER than float16 on this GTX 1650 Ti (no tensor cores).
- vgg16 float32 at batch 16 reserves 4,532 MB > 4,096 MB: Windows spills into shared RAM instead of raising OOM.

### Estimated vs actual time (estimate from the probe, float16 settings, start-up not included)
| Model | Split | Est. min (3 ep.) | Est. min (12 ep.) | Epochs run | Actual min |
|---|---|---|---|---|---|
| resnet50 | v1_clean | 7.1 | 41.8 | 6 | 21.3 |
| resnet50 | v1_original | 14.2 | 83.5 | 3 | 16.1 |
| vgg16 | v1_clean | 11.1 | 70.4 | 10 | 31.9 |
| vgg16 | v1_original | 21.8 | 139.5 | 5 | 25.1 |
(resnet50 estimates are float32: the notebook uses the setting in src/models.py now. vgg16 fell back to float32 in
epoch 1 of both runs, so its real epochs were float32 at batch 16: ~196 s/epoch clean, ~387 s/epoch original.)

### Results: all training runs so far (validation only; test untouched; generated by `train.runs_table()`)
| split | model | params_M | val_acc | val_macro_f1 | epochs (best) | max_epochs | stopped_by | target_acc | batch x accum | time_min | peak_gpu_mb | float16 | resumed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_clean | efficientnet_b0 | 4.01 | 0.9423 | 0.9394 | 3 (3) | 12 | target_acc | 0.92 | 32 x 1 | 10.8 | 3082.0 | NaN at epoch 1 -> float32 | False |
| v1_clean | mobilenetv3_small_100 | 1.52 | 0.9808 | 0.9809 | 20 (18) | 20 | max_epochs | - | 48 x 1 | 11.2 | 690.9 | yes | False |
| v1_clean | resnet50 | 23.52 | 0.9538 | 0.9517 | 6 (6) | 12 | target_acc | 0.92 | 24 x 1 | 21.3 | 3021.7 | NaN at epoch 2 -> float32 | False |
| v1_clean | vgg16 | 134.28 | 0.9212 | 0.9145 | 10 (10) | 12 | target_acc | 0.92 | 16 x 1 | 31.9 | 3550.6 | NaN at epoch 1 -> float32 | False |
| v1_original | efficientnet_b0 | 4.01 | 0.986 | 0.9856 | 9 (4) | 12 | patience | - | 32 x 1 | 44.5 | 2236.7 | yes | False |
| v1_original | mobilenetv3_small_100 | 1.52 | 0.9918 | 0.9915 | 18 (13) | 20 | patience | - | 48 x 1 | 18.0 | 707.1 | yes | False |
| v1_original | resnet50 | 23.52 | 0.9218 | 0.92 | 3 (3) | 12 | target_acc | 0.92 | 24 x 1 | 16.1 | 3022.3 | no (float32) | False |
| v1_original | vgg16 | 134.28 | 0.8973 | 0.8944 | 5 (5) | 12 | user_stop | 0.9 | 16 x 1 | 25.1 | 3550.6 | NaN at epoch 1 -> float32 | True |
| v2_clean | efficientnet_b0 | 4.01 | 0.9479 | 0.9432 | 3 (3) | 12 | target_acc | 0.92 | 32 x 1 | 8.0 | 1461.8 | yes | False |
| v2_clean | mobilenetv3_small_100 | 1.52 | 0.9683 | 0.9639 | 18 (13) | 20 | patience | - | 48 x 1 | 8.9 | 453.5 | yes | False |
| v2_original | efficientnet_b0 | 4.01 | 0.975 | 0.975 | 2 (2) | 12 | target_acc | 0.92 | 32 x 1 | 11.2 | 2794.8 | NaN at epoch 1 -> float32 | False |
| v2_original | mobilenetv3_small_100 | 1.52 | 0.9738 | 0.9736 | 4 (4) | 12 | target_acc | 0.92 | 48 x 1 | 5.3 | 707.1 | yes | False |
- Heavy vs light on validation (v1_clean): resnet50 0.9517, vgg16 0.9145 vs mobilenet 0.9809, efficientnet 0.9394
  macro-F1. **Not a fair ranking yet**: every heavy run was still improving when it stopped (target / user stop),
  while mobilenet ran 18-20 epochs. The test evaluation is the real comparison.
- Overfitting: none (health_check warnings are only the float16 -> float32 switches). All 4 best.pt checked: 0 NaN/Inf
  tensors, no last.pt left.

### Issues
- **float16 NaN hit every heavy run** (resnet50: smoke, v1_clean epoch 2, v1_original epoch 1; vgg16: smoke, both runs at
  epoch 1, resume test at epoch 2). **NaN-guard gap found**: the float16 forward pass that makes the NaN has ALREADY written
  NaN into the BatchNorm running statistics before the guard switches to float32. The redo in float32 does not undo that, so
  eval is NaN forever. This happened in resnet50 smoke (84 tensors) and the first resnet50 v1_original run (78 tensors, val loss NaN at
  epoch 1, val acc 0.231). That run was stopped and kept in outputs/runs/failed/v1_original_resnet50_fp16_nan/.
  resnet50 v1_clean was hit at epoch 2 but its stats stayed finite (checked) and was kept.
  vgg16 has no BatchNorm, so its fall-back is safe. The 8 light checkpoints were checked: 0 non-finite tensors.
- Fix (my decision): resnet50 `amp=False`; v1_original re-run from scratch in float32. vgg16 left in float16 as configured
  (in practice it ran float32 after epoch 1). The guard itself is NOT fixed; only mobilenet still uses float16.

### Next step
Evaluate every best.pt once on its test split (accuracy, macro-F1, confusion matrix, params, MB, CPU ms/image) and compare
original vs clean and light vs heavy, keeping in mind which runs stopped early (target_acc / user_stop).

## Session 2026-10-13 (Week 8: test evaluation, calibration, MC dropout)
### Done
- `src/evaluate.py`: reloads `best.pt` of a run and evaluates it on that run's own test split: accuracy, balanced accuracy,
  macro-F1, per-class sensitivity / precision, one-vs-rest AUC, confusion matrix; temperature scaling (T fitted on the
  validation logits only, test never used), ECE with 15 bins, reliability diagrams before / after; MC dropout (30 passes
  on the classifier head), predictive entropy, accuracy-vs-coverage; "refer to radiologist" at confidence < 0.70;
  size (MiB of best.pt), parameters, CPU and GPU latency (batch 1, 20 warm-up + 100 timed test images, float32).
- Writes `metrics.json` per run, adds `temperature` (and `mc_dropout` settings) to each run's `config.json`, plots to
  `outputs/plots/week8/`, and one table `outputs/results/week8_results.csv` (12 runs x 37 columns).
- `notebooks/06_evaluation.ipynb` (06, not 05: `05_heavy_training` already exists). Built and executed end-to-end with
  nbclient (0 errors), saved with outputs.
- Smoke test first on v1_clean / mobilenetv3_small_100: recomputed val macro-F1 0.98092 = value in the training log.

### Results (test sets; generated by `src/evaluate.py`, table from the CSV)
| version | split | model | n_test | accuracy | balanced_acc | macro_f1 | macro_auc | epochs_run | stopped_by |
|---|---|---|---|---|---|---|---|---|---|
| v1 | clean | efficientnet_b0 | 520 | 0.9538 | 0.9588 | 0.9541 | 0.9958 | 3 | target_acc |
| v1 | clean | mobilenetv3_small_100 | 520 | 0.9788 | 0.9816 | 0.9756 | 0.9972 | 20 | max_epochs |
| v1 | clean | resnet50 | 520 | 0.9481 | 0.9451 | 0.9404 | 0.9928 | 6 | target_acc |
| v1 | clean | vgg16 | 520 | 0.9269 | 0.9236 | 0.9169 | 0.9849 | 10 | target_acc |
| v1 | original | efficientnet_b0 | 1311 | 0.9878 | 0.9868 | 0.9872 | 0.9996 | 9 | patience |
| v1 | original | mobilenetv3_small_100 | 1311 | 0.9908 | 0.9901 | 0.9905 | 0.9997 | 18 | patience |
| v1 | original | resnet50 | 1311 | 0.8848 | 0.8755 | 0.8815 | 0.981 | 3 | target_acc |
| v1 | original | vgg16 | 1311 | 0.9184 | 0.9131 | 0.9134 | 0.992 | 5 | user_stop |
| v2 | clean | efficientnet_b0 | 538 | 0.9164 | 0.9262 | 0.9132 | 0.9866 | 3 | target_acc |
| v2 | clean | mobilenetv3_small_100 | 538 | 0.948 | 0.9526 | 0.944 | 0.9923 | 18 | patience |
| v2 | original | efficientnet_b0 | 1600 | 0.9406 | 0.9406 | 0.9398 | 0.9877 | 2 | target_acc |
| v2 | original | mobilenetv3_small_100 | 1600 | 0.9344 | 0.9344 | 0.9334 | 0.9903 | 4 | target_acc |

Calibration and referral (T = fitted temperature; refer rule: temperature-scaled confidence < 0.70):
| version | split | model | temperature | ece_before | ece_after | refer_rate | acc_non_referred | mc_acc | mc_mean_entropy |
|---|---|---|---|---|---|---|---|---|---|
| v1 | clean | efficientnet_b0 | 0.6501 | 0.0906 | 0.022 | 0.0404 | 0.9679 | 0.9615 | 0.4634 |
| v1 | clean | mobilenetv3_small_100 | 0.5371 | 0.11 | 0.014 | 0.0288 | 0.9861 | 0.9788 | 0.7732 |
| v1 | clean | resnet50 | 0.592 | 0.0998 | 0.0141 | 0.0423 | 0.9639 | 0.9481 | 0.1843 |
| v1 | clean | vgg16 | 0.7332 | 0.0921 | 0.0312 | 0.05 | 0.9413 | 0.9269 | 0.2893 |
| v1 | original | efficientnet_b0 | 0.4775 | 0.0782 | 0.0065 | 0.0076 | 0.99 | 0.9863 | 0.2292 |
| v1 | original | mobilenetv3_small_100 | 0.4772 | 0.074 | 0.0059 | 0.0031 | 0.9916 | 0.9916 | 0.6817 |
| v1 | original | resnet50 | 0.6086 | 0.0663 | 0.0385 | 0.1198 | 0.9289 | 0.8841 | 0.2545 |
| v1 | original | vgg16 | 0.6866 | 0.0898 | 0.0282 | 0.0824 | 0.9501 | 0.9199 | 0.3031 |
| v2 | clean | efficientnet_b0 | 0.5353 | 0.0909 | 0.032 | 0.0669 | 0.9442 | 0.9126 | 0.4399 |
| v2 | clean | mobilenetv3_small_100 | 0.579 | 0.0914 | 0.0276 | 0.026 | 0.9542 | 0.9461 | 0.8133 |
| v2 | original | efficientnet_b0 | 0.4497 | 0.0554 | 0.036 | 0.0312 | 0.9581 | 0.9394 | 0.2139 |
| v2 | original | mobilenetv3_small_100 | 0.4814 | 0.0655 | 0.0332 | 0.0388 | 0.9564 | 0.9344 | 0.6834 |

Efficiency (mean of the runs of each model; latency is nearly the same for every run of a model):
| model | params_M | size_mb | cpu_ms | gpu_ms |
|---|---|---|---|---|
| efficientnet_b0 | 4.01 | 15.59 | 23.14 | 12.77 |
| mobilenetv3_small_100 | 1.52 | 5.93 | 9.8 | 10.03 |
| resnet50 | 23.52 | 90.01 | 59.22 | 9.06 |
| vgg16 | 134.28 | 512.24 | 261.15 | 20.3 |

### Decisions
- MC dropout is **post-hoc**: no model was trained with dropout (all timm heads have p = 0), so dropout (p = 0.3, fixed
  before seeing any result, not tuned) is applied to the feature vector entering the last Linear layer; the backbone runs
  once and 30 masks are applied to the head. Each pass divides its logits by the fitted T. (My first version used raw
  logits; on mobilenet that gave very flat probabilities, so T was added before the full run.)
- Main referral rule uses the deterministic temperature-scaled confidence; the MC-probability version is reported next to it
  (`refer_rate_mc`, `acc_non_referred_mc` in the CSV).
- Evaluation is float32 for every model. Validation macro-F1 recomputed here equals the training log exactly for most runs and
  differs by at most 0.003 for runs that trained with float16 autocast; tolerance in the notebook check is 0.005.

### Findings (read from the tables, single seed)
- All 12 runs have T < 1 (0.45-0.73): under-confident (label smoothing). Temperature scaling lowered ECE in all 12 runs
  (0.055-0.110 before, 0.006-0.039 after).
- Referral at < 0.70 refers 0.3-12.0% of scans; accuracy on the non-referred scans is above overall accuracy in all 12 runs.
- Mean MC entropy is higher on wrong than on correct scans in all 12 runs; MC accuracy is within 0.8 points of the
  deterministic accuracy. Mobilenet is far more sensitive to head dropout (entropy on correct scans 0.65-0.81) than resnet50 / vgg16
  (0.16-0.27), so its MC-probability referral rate is large (up to 44%).
- On v1 clean (test macro-F1): mobilenet 0.9756, efficientnet 0.9541, resnet50 0.9404, vgg16 0.9169, at 1.5 / 4.0 / 23.5 / 134 M
  parameters and 8 / 24 / 59 / 283 ms per image on the CPU. The heavy runs stopped early (target_acc), so this is not a
  fully fair ranking.
- Original vs clean is **confounded by early stopping** (resnet50 original: 3 epochs, vgg16 original: 5 epochs, hand-stopped; v2 originals:
  2 and 4 epochs). The only clean pair (both ran to patience / epoch limit) is v1 mobilenet: macro-F1 0.9905 original vs 0.9756 clean.

### Issues
- Test sets are small (520-538 images on the clean split; 1 image = 0.19 points). One seed. Differences of about 1 point are not meaningful.
- Original-split validation images come from the Training folder and may have near-duplicates in train, so their T is less trustworthy.
- Older configs have no `stopped_by`; the results table infers it the same way as `train.runs_table()`.

### Next step
Week 9: Grad-CAM with the inside-brain ratio on correct and wrong test scans. Possible follow-ups: retrain the early-stopped
heavy runs with an equal budget for a fair original-vs-clean comparison, and score the original-trained models on the clean test set.

## Session 2026-10-13 (Week 9: Grad-CAM inside-brain ratio, final figures)
### Done
- `src/gradcam.py`: Grad-CAM of the PREDICTED class for every test image (pytorch-grad-cam 1.5.7, float32, target layer = last
  7 x 7 spatial layer: MobileNetV3 `blocks[-1]`, EfficientNet `bn2`, ResNet-50 `layer4[-1]`, VGG16 `features[-1]`). Brain mask =
  Otsu -> largest connected component -> fill holes. Inside-brain ratio = heat inside the mask / all heat, also compared with the
  chance level (mask share of the picture). 11,600 images over the 12 runs, no NaN, Grad-CAM class = Week-8 prediction for every image,
  per-run row counts and accuracy equal the Week-8 values. Per-run CSVs in `outputs/gradcam/`, all in `outputs/results/week9_gradcam.csv`.
- `notebooks/07_gradcam.ipynb` (named 07, not 06: `06_evaluation` exists): mask check, smoke test, tables per run / class / correct vs wrong,
  sensitivity table without flagged masks, galleries of 12 random test images (6 wrong) for MobileNetV3-S, ResNet-50 and VGG16.
- `src/report.py` + `notebooks/08_results_plots.ipynb`: the six final figures in `outputs/plots/final/` (150 dpi, Okabe-Ito palette,
  hollow/filled markers and hatches so colour is never the only cue): 1 original vs clean, 2 size vs macro-F1 (log x), 3 per-class sensitivity,
  4 ECE before/after, 5 inside-brain ratio per model, 6 leakage per dataset version. The plain-English summary is saved in
  `outputs/results/week9_summary.md`. Both notebooks executed with nbclient (0 errors).

### Results (from `outputs/results/week9_gradcam.csv`; chance = average mask share of the picture)
| run | n | ratio_mean | ratio_median | chance | over_chance_median | tiny_masks | empty_cams |
|---|---|---|---|---|---|---|---|
| v1_clean_efficientnet_b0 | 520 | 0.76 | 0.798 | 0.579 | 1.332 | 10 | 0 |
| v1_clean_mobilenetv3_small_100 | 520 | 0.686 | 0.726 | 0.579 | 1.22 | 10 | 0 |
| v1_clean_resnet50 | 520 | 0.899 | 0.934 | 0.579 | 1.543 | 10 | 0 |
| v1_clean_vgg16 | 520 | 0.353 | 0.315 | 0.579 | 0.523 | 10 | 2 |
| v1_original_efficientnet_b0 | 1311 | 0.701 | 0.822 | 0.572 | 1.347 | 33 | 0 |
| v1_original_mobilenetv3_small_100 | 1311 | 0.638 | 0.657 | 0.572 | 1.145 | 33 | 0 |
| v1_original_resnet50 | 1311 | 0.849 | 0.88 | 0.572 | 1.486 | 33 | 0 |
| v1_original_vgg16 | 1311 | 0.531 | 0.534 | 0.572 | 0.906 | 33 | 0 |
| v2_clean_efficientnet_b0 | 538 | 0.783 | 0.823 | 0.578 | 1.36 | 9 | 0 |
| v2_clean_mobilenetv3_small_100 | 538 | 0.7 | 0.749 | 0.578 | 1.259 | 9 | 0 |
| v2_original_efficientnet_b0 | 1600 | 0.652 | 0.736 | 0.575 | 1.233 | 38 | 0 |
| v2_original_mobilenetv3_small_100 | 1600 | 0.749 | 0.782 | 0.575 | 1.295 | 38 | 0 |

Correct vs wrong predictions (mean ratio; the wrong groups are small):
| run | n_correct | ratio_mean_correct | n_wrong | ratio_mean_wrong |
|---|---|---|---|---|
| v1_clean_efficientnet_b0 | 496 | 0.762 | 24 | 0.73 |
| v1_clean_mobilenetv3_small_100 | 509 | 0.686 | 11 | 0.717 |
| v1_clean_resnet50 | 493 | 0.903 | 27 | 0.838 |
| v1_clean_vgg16 | 482 | 0.353 | 38 | 0.355 |
| v1_original_efficientnet_b0 | 1295 | 0.7 | 16 | 0.727 |
| v1_original_mobilenetv3_small_100 | 1299 | 0.637 | 12 | 0.749 |
| v1_original_resnet50 | 1160 | 0.854 | 151 | 0.805 |
| v1_original_vgg16 | 1204 | 0.528 | 107 | 0.567 |
| v2_clean_efficientnet_b0 | 493 | 0.786 | 45 | 0.756 |
| v2_clean_mobilenetv3_small_100 | 510 | 0.705 | 28 | 0.616 |
| v2_original_efficientnet_b0 | 1505 | 0.653 | 95 | 0.632 |
| v2_original_mobilenetv3_small_100 | 1495 | 0.754 | 105 | 0.689 |

### Decisions
- Ratio uses heat INTENSITY (not a thresholded area); the chance level is shown next to it because the brain covers only about 58% of the picture.
- Masks covering < 25% of the picture are flagged (`tiny_mask`, 267 of 11,600 images); removing them changes any per-run mean ratio by at most 0.015.
- No retraining before the figures (my decision): figure 1 marks early-stopped runs with a star instead of fixing the unequal budgets.
- Gallery images are drawn at random (seed 42), not hand-picked. No significance tests (small wrong-groups, simple code); n is always shown.

### Plain-English summary (copied from `outputs/results/week9_summary.md`)
Everything below is read from the tables and figures above. One seed per model, small test sets (clean split: 520 and 538 images, so one image is about 0.19 points), so differences of about one point should not be over-read.

**1. Leakage in the original split is large (figure 6).** In the original Testing folder, 7.9% of the v1 test images (0% for v2) are exact copies of a training image, 55.2% (v1) and 44.6% (v2) have a near-duplicate in training, and 70.2% (v1) and 63.1% (v2) belong to a cluster that also appears in training. After our clean split this is 0% by construction.

**2. Our main hypothesis, "the clean split gives lower scores", is not confirmed by these runs (figure 1).** The original-split score is higher than the clean one in 3 of the 6 model-version pairs and lower in the other 3. The pairs are not comparable because training stopped early in most of them: for example ResNet-50 (3 epochs) and VGG16 (5 epochs, stopped by hand) on the original split scored *below* their clean runs (accuracy 0.885 vs 0.948 and 0.918 vs 0.927), which reflects under-training, not leakage. The only pair where both runs trained to patience or the epoch limit is v1 MobileNetV3-S: accuracy 0.9908 (original) vs 0.9788 (clean), macro-F1 0.9905 vs 0.9756, a drop of about 1.2 and 1.5 points. That is consistent with some inflation, but it is a single pair. A fair test needs equal training budgets, which we did not run.

**3. Light models are not worse than the heavy baselines in our runs (figure 2).** On the v1 clean split the test macro-F1 is 0.9756 (MobileNetV3-S, 5.9 MB), 0.9541 (EfficientNet-B0, 15.6 MB), 0.9404 (ResNet-50, 90 MB) and 0.9169 (VGG16, 512 MB). The heavy runs also stopped early, so we cannot say they would not catch up with more training. What the numbers do show is the cost: VGG16 needed about 283 ms per image on this CPU against about 8 ms for MobileNetV3-S.

**4. Weak spot by class (figure 3).** On the v1 clean split meningioma has the lowest sensitivity for all four models (0.957 MobileNetV3-S, 0.936 EfficientNet-B0, 0.922 ResNet-50, 0.879 VGG16). Notumor sensitivity is at least 0.917 for every model on that split, but note that it has only 60 test images.

**5. Calibration (figure 4).** All 12 runs were under-confident (T between 0.45 and 0.73, likely a side effect of label smoothing). Temperature scaling lowered the ECE in every run, from 0.055-0.110 to 0.006-0.039. Referring scans below 0.70 confidence sent 0.3-12% of scans to a radiologist, and accuracy on the scans not referred was higher than overall accuracy in all 12 runs.

**6. Grad-CAM (figure 5).** The inside-brain ratio is highest for ResNet-50 (mean 0.899 on v1 clean), then EfficientNet-B0, then MobileNetV3-S, and lowest for VGG16 (0.353 on v1 clean, below the 0.58 a uniform heat map would give; its heat sits on the skull rim and picture borders). This ordering does not follow model size or accuracy, so the idea that lightweight models look at the brain more than heavy ones is not supported. The ratio did not separate correct from wrong predictions (wrong was lower in 7 of 12 runs and higher in 5, with only 11-151 wrong images per run). The "brain mask" is really a head-tissue mask (it includes skull and scalp and fails on very bright tumours), so a high ratio does not prove the model looks at the tumour.

**What we can and cannot claim.** We can claim: the original split leaks heavily; temperature scaling fixes under-confidence in every run; the lightweight models match the heavy ones at a small fraction of the size and CPU time; and VGG16's Grad-CAM heat lies more on the picture edge than on the brain (a warning sign; we did not test causally whether it relies on the background). We cannot claim: that leakage inflates accuracy by a specific amount (only one fair pair), that heavy models are worse in general (they stopped early), or that Grad-CAM proves what any network uses.


### Issues
- The "brain mask" is really a head-tissue mask (skull, scalp, neck included) and keeps only the tumour on very bright images; so a high ratio is not proof of looking at the tumour.
- The 7 x 7 Grad-CAM map is coarse. VGG16's low ratio (heat on skull rim / picture borders) was seen on a second layer (14 x 14, 16 images) but not tested causally.
- Original vs clean (figure 1) is confounded by early stopping; only v1 MobileNetV3-S is a fair pair. Single seed.
- A first smoke-test script hung on Windows because it had no `if __name__ == "__main__"` guard (DataLoader workers re-import the script); fixed in the script, not in `src/`.

### Next step
Write the report with these figures. Optional follow-ups: equal-budget retraining of the early-stopped runs (the only way to get a fair
leakage estimate), a causal check of VGG16 (mask out the background and re-score), densenet121 / inception_v3 which have no runs yet.

## Session 2026-10-21 (Week 10: Streamlit demo app)

### Done
- Installed `streamlit` 1.66.0 into the venv and added it (with its 14 new dependencies) to `requirements.txt`; torch, numpy and pandas were not changed.
- `app/inference.py`: CPU-only helpers that reuse `src/` (no Streamlit code): `src.data.crop_brain` + `pad_resize` + `eval_transforms` for preprocessing, `src.evaluate.load_run` / `softmax` / `REFER_BELOW`, `src.gradcam.make_cam` / `compute_cams` / `inside_mask` / `inside_ratio`. Class order, normalisation, image size and temperature are read from `config.json`.
- `app/app.py`: page with disclaimer (top and sidebar), file uploader, 4 clickable samples, predicted class, calibrated confidence bars for all 4 classes, Grad-CAM overlay with brain-mask outline and inside-brain ratio, red "Refer to radiologist" banner when confidence < 0.70, and a warning when no head region is found.
- `app/make_samples.py`: copies `best.pt`, `config.json`, `metrics.json` of `outputs/runs/v2_clean/mobilenetv3_small_100` into `app/model/` and the first clean-split test image of each class into `app/samples/` (`data/` only read).
- README: layout, "Demo app" section, status line.

### Results (all from the verification runs, CPU only, `CUDA_VISIBLE_DEVICES=""`)
| Check | Result |
|---|---|
| App preprocessing vs cached training PNG, 4 samples | pixel-identical in 4 of 4 |
| App probabilities vs `softmax(test_outputs.npz logits / T)` | max abs difference 7.2e-7 |
| Headless `AppTest`, click each of the 4 samples | no exception; 4 progress bars each time |
| `streamlit run app/app.py --server.headless true` | `/_stcore/health` and `/` returned HTTP 200 |

| Sample (clean-split test row) | Source file | True | Predicted | Calibrated conf | Referred |
|---|---|---|---|---|---|
| glioma (row 6) | data/v2/Testing/glioma/Te-gl_45.jpg | glioma | glioma | 0.9671 | no |
| meningioma (row 0) | data/v2/Training/meningioma/Tr-me_922.jpg | meningioma | glioma | 0.8742 | no |
| notumor (row 1) | data/v2/Testing/notumor/Te-no_150.jpg | notumor | notumor | 0.6083 | yes (red banner) |
| pituitary (row 2) | data/v2/Training/pituitary/Tr-pi_1129.jpg | pituitary | pituitary | 0.9929 | no |

Inside-brain ratios of these four: 0.823, 0.828, 0.682, 0.946. Time on this laptop CPU (2 threads): about 11-23 ms for the prediction and 47-72 ms for Grad-CAM per image after warm-up; the first call took about 1.1 s and 1.0 s.

### Decisions
- Model = `mobilenetv3_small_100` on the clean split: best of the two light models there by accuracy, macro-F1 and ECE in `outputs/results/week8_results.csv` (efficientnet_b0 stopped early at the 0.92 val-accuracy target, so that gap is not an equal-budget comparison).
- `best.pt` (6 MB) is committed under `app/model/` so the app runs from a fresh clone; `outputs/` stays out of git.
- Samples are the first test row per class of `outputs/splits/v2_split_clean.csv`, not hand-picked. Two of them come from the original `Training/` folder because the clean split reshuffles images; they are still held-out for this model.
- MC dropout is not in the app (not requested; the referral rule uses the temperature-scaled confidence as in Week 8).
- A model lock around Grad-CAM, because the hooks sit on the model that Streamlit shares between sessions.

### Issues
- The model is confidently wrong on pictures that are not brain MRIs: a black image was classified pituitary at 0.952 and uniform noise notumor at 0.998, so neither was referred. Both trigger the "no clear head region" warning in the app (crop fallback), which is the only guard. The referral rule only catches uncertainty, not out-of-distribution input.
- The meningioma sample is predicted as glioma at 0.874 confidence and is not referred (shown with "WRONG prediction" in the app).
- Not tested: uploading a file through the browser widget (`AppTest` cannot upload) and looking at the page in a browser; only the sample path and the server start were run.
- Single run on a 4 GB laptop: the Grad-CAM first-call delay is about 1 s.

### Next step
Look at the page in a browser and try a real upload; then write the report. Optional: an out-of-distribution check (for example MC-dropout entropy or a simple "is this an MRI" filter), and the equal-budget retraining listed in Week 9.
