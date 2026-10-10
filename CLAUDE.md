# Project: Trustworthy Brain Tumor Classification from MRI Using Leakage-Aware and Explainable Deep Learning
Team: K Praveen Kumar, Ishika Kaur, Shreesha. Guide: Akshara Prachi. Marwadi University, CE-AI, Major Project-II.
Task: 4-class MRI classification: glioma, meningioma, notumor, pituitary.
Novelty:
1. Duplicate audit (MD5 exact, pHash near-copy, deep-feature look-alike) + leakage-free split;
   every model compared on the original split vs the clean split.
2. Lightweight models (mobilenetv3_small_100, efficientnet_b0) vs heavy baselines
   (vgg16, resnet50, densenet121, inception_v3); report params, MB, CPU ms/image.
3. Grad-CAM with an "inside-brain ratio" (share of heatmap inside an Otsu brain mask).
4. Temperature scaling, ECE, MC dropout, and a "refer to radiologist" flag below 0.70 confidence.
Dataset: Brain Tumor MRI Dataset by M. Nickparvar (Kaggle), already downloaded into ./data/
(data/v1 = 7,023 images with duplicates, data/v2 = ~7,200 cleaned; each has Training/ and Testing/).
Hardware: NVIDIA GTX 1650 Ti, 4 GB VRAM -> mixed precision, channels_last, small batches,
retry with half batch on CUDA out-of-memory.
Rules:
- ./data/ is read-only. Never modify or delete it.
- Never invent or hard-code results; every number must come from code that ran.
- Shared code goes in src/, notebooks in notebooks/, all outputs in outputs/.
- Seed 42 everywhere. Simple, well-commented code: I must explain every line in my viva.
- Show me a plan and wait for my OK before big changes.
- Update PROGRESS.md at the end of every session (what was done, results, next step).

## Training efficiency (4 GB VRAM) - use epochs efficiently, don't overload the laptop
- Mixed precision (AMP) + channels_last; `torch.backends.cudnn.benchmark = True`.
- Image size 224 (inception_v3 needs 299 -> smaller batch). Start batch 32 for light models,
  16 for heavy ones; on CUDA OOM halve the batch and use gradient accumulation to keep the effective batch.
- Max ~25 epochs with early stopping (patience 5 on val loss) and ReduceLROnPlateau.
  Heavy models: freeze the backbone for the first 2-3 epochs, then fine-tune.
- Pre-resize images once into a cache under outputs/ so epochs don't re-decode full-size JPGs.
- DataLoader: num_workers=2, pin_memory=True, persistent_workers=True (keeps CPU/RAM load low).
- Save only the best checkpoint + a per-epoch CSV log; torch.cuda.empty_cache() between models;
  one training run at a time; support resuming from a checkpoint.
- Smoke-test every new pipeline with 1 epoch on a small subset before a full run.
