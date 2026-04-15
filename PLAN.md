# Technical Implementation Plan
## Notebook: `GGSN_Patryk_Chamera_Zad2.ipynb`

## Notebook style rules (use during implementation)
- Write the notebook in the same style as `CNN-lecture` materials.
- Keep markdown short, practical, and student-like.
- Do not add unnecessary "professional" boilerplate.
- Import libraries only when they are needed in the current part.
- Define hyperparameters progressively (when a model/training step is introduced), not all at the beginning.
- Use direct dataset paths (`Training`, `Testing`) because folder structure is known.
- Keep it simple, but still technically correct and complete for grading.

## 0. Project Goal and Success Criteria
Build, optimize, and compare CNN-based models for multiclass brain tumor MRI classification (4 classes), then justify conclusions using both quantitative metrics and explainability visualizations.

### Primary success criteria
- Achieve strong and stable test performance across all classes.
- Demonstrate measurable improvement from:
  1. baseline CNN -> CNN + augmentation/regularization,
  2. custom CNN -> transfer learning + fine-tuning.
- Provide explainability evidence (Grad-CAM) showing model attention aligns with tumor regions.

---

## Phase 1 - Basic setup and data check

### 1.1 Minimal imports for first cells
- Start with only basic libraries needed to inspect data:
  - `pathlib`, `pandas`, `matplotlib` (and optional `PIL` if image preview is needed).
- Add TensorFlow/Keras imports later when model-building starts.

### 1.2 Dataset folders
- Use direct paths:
  - `TRAIN_DIR = Path("Training")`
  - `TEST_DIR = Path("Testing")`
- Check if both folders exist.

### 1.3 Quick validation
- Validate train/test directories and class folders.
- Print sample file counts per class (train + test).
- If class names differ, show warning and inspect manually.
- Keep checks simple and readable.

---

## Phase 2 - Data Engineering and Exploratory Analysis (EDA)

### 2.1 Dataset integrity checks
- Confirm:
  - class names are consistent,
  - no empty folders,
  - image files readable.
- Log any corrupted files and optionally skip/remove them.

### 2.2 Data leakage safeguards
- Verify no file overlap between train and test (by filename hash/signature if needed).
- Keep test set untouched until final evaluation.
- Create validation split only from training data.

### 2.3 Visual EDA
- Plot class distribution bars (train/test).
- Display random grids (e.g., 5-8 images per class).
- Inspect resolution statistics (min/max/median dimensions).
- Comment on visible tumor morphology and inter-class variation.

### 2.4 Preprocessing strategy definition
- Resize all images to fixed shape (`IMG_SIZE`).
- Normalize pixel values to `[0,1]` or use model-specific preprocessing function.
- Optionally apply margin removal/cropping for black borders if beneficial.
- Decide whether to use RGB consistently (3 channels).

---

## Phase 3 - Data Pipelines and Input Generators

### 3.1 Baseline input pipeline (no augmentation)
- Build `tf.data` or Keras generator pipeline:
  - training subset,
  - validation subset (from train split),
  - test set (separate fixed dataset).
- Use one-hot labels for multiclass softmax output.

### 3.2 Augmented input pipeline
- Create train-only augmentation block:
  - random rotation,
  - width/height shift,
  - zoom,
  - horizontal flip (and vertical flip only if medically justified),
  - slight brightness/contrast jitter.
- Ensure validation/test pipelines are non-augmented.

### 3.3 Performance settings
- Cache/prefetch for faster throughput.
- Shuffle only training data.
- Keep deterministic order for evaluation datasets.

---

## Phase 4 - Baseline CNN from Scratch

### 4.1 Architecture (initial reference model)
- Sequential or Functional CNN:
  - 3-4 convolution blocks (`Conv2D + ReLU + MaxPool`),
  - optional batch normalization,
  - flatten/global pooling,
  - dense head with dropout,
  - final `Dense(4, activation="softmax")`.

### 4.2 Training setup
- Loss: `categorical_crossentropy`.
- Optimizer: `Adam` (initial LR e.g., `1e-3`).
- Metrics: `accuracy`, plus custom F1 (optional during training).
- Callbacks:
  - `EarlyStopping` (monitor `val_loss`, restore best weights),
  - `ReduceLROnPlateau`,
  - `ModelCheckpoint`.

### 4.3 Baseline evaluation
- Plot training/validation loss and accuracy curves.
- Evaluate on validation and test.
- Generate:
  - confusion matrix,
  - classification report (precision/recall/F1 per class),
  - macro and weighted F1 summary.
- Document overfitting/underfitting signals.

---

## Phase 5 - CNN Optimization with Augmentation and Regularization

### 5.1 Improvements over baseline
- Retrain CNN with augmented pipeline.
- Add/adjust:
  - dropout rates (e.g., 0.3-0.5),
  - batch normalization placement,
  - L2 regularization (kernel weight decay).

### 5.2 Hyperparameter experiments (compact grid)
Run a controlled set of experiments (small but meaningful):
- Learning rates: `1e-3`, `3e-4`, `1e-4`.
- Dropout: `0.3`, `0.5`.
- Batch size: `16`, `32`.
- Optional optimizer comparison: `Adam` vs `RMSprop`.

### 5.3 Comparison outputs
- Build a results table with:
  - model variant,
  - best validation accuracy,
  - test accuracy,
  - macro F1,
  - training time/epoch count.
- Select best custom CNN configuration for next comparison stage.

---

## Phase 6 - Transfer Learning (Feature Extraction)

### 6.1 Backbone selection
Choose at least one strong pre-trained model (recommended):
- `ResNet50` (primary),
- optional second backbone for comparison (`VGG16` or `EfficientNetB0`).

### 6.2 Feature extraction model
- Load ImageNet weights with `include_top=False`.
- Freeze entire backbone.
- Add custom classification head:
  - `GlobalAveragePooling2D`,
  - dense layer(s) + dropout,
  - softmax output (4 classes).
- Use model-specific preprocessing function.

### 6.3 Train frozen model
- Train head-only model until convergence.
- Track same metrics and plots as in previous phases.
- Save best checkpoint.

---

## Phase 7 - Fine-Tuning Strategy

### 7.1 Unfreezing protocol
- Unfreeze top N layers (e.g., last 20-40% of backbone).
- Keep early layers frozen to preserve generic visual features.

### 7.2 Fine-tuning optimization
- Lower learning rate (e.g., `1e-5` to `1e-6`).
- Continue training from best frozen checkpoint.
- Use early stopping and LR scheduling to prevent catastrophic forgetting.

### 7.3 Final transfer model evaluation
- Evaluate on test set with full report:
  - confusion matrix,
  - precision/recall/F1 per class,
  - macro/weighted averages,
  - ROC-AUC (one-vs-rest, optional but valuable).

---

## Phase 8 - Explainability and Representation Visualization

### 8.1 Grad-CAM analysis
- Generate Grad-CAM heatmaps for:
  - correctly classified examples (all classes),
  - selected misclassified examples.
- Overlay heatmaps on original MRIs.
- Compare localization quality baseline vs transfer model.

### 8.2 Filter/feature-map visualization (optional but recommended)
- Visualize intermediate activations for representative samples.
- Demonstrate how deeper layers capture tumor-specific structures.

### 8.3 Clinical interpretability discussion
- Evaluate whether highlighted areas correspond to plausible lesion regions.
- Identify possible failure patterns (artifact focus, diffuse attention, bias to borders).

---

## Phase 9 - Consolidated Comparison and Conclusions

### 9.1 Unified model comparison section
Create a final summary table:
- Baseline CNN (no augmentation)
- Optimized CNN (augmentation + regularization)
- Transfer Learning (frozen)
- Transfer Learning (fine-tuned)

For each model include:
- test accuracy,
- macro F1,
- per-class recall,
- parameter count,
- training time,
- qualitative explainability quality (brief note).

### 9.2 Required visual outputs
- Learning curves (loss/accuracy) per major model.
- Confusion matrices (at least best baseline and best transfer model).
- 1-2 pages worth of Grad-CAM examples.

### 9.3 Final conclusions (academic framing)
Answer explicitly:
1. Which optimization methods gave the largest gain?
2. Did augmentation reduce overfitting and improve minority-like confusions?
3. Did transfer learning outperform custom CNNs?
4. Did fine-tuning improve further or risk overfitting?
5. Are model decisions visually aligned with tumor regions?

---

## Phase 10 - Notebook Structure and Deliverable Quality

### 10.1 Suggested notebook section order
1. Introduction and objective
2. Dataset and EDA
3. Preprocessing and pipeline
4. Baseline CNN
5. Augmentation/regularization experiments
6. Transfer learning
7. Fine-tuning
8. Evaluation and comparison
9. Grad-CAM explainability
10. Final conclusions

### 10.2 Quality checklist before submission
- Notebook runs top-to-bottom without manual intervention.
- Key hyperparameters are clearly listed when introduced in relevant sections.
- All figures have titles and axis labels.
- Every major experiment has a short interpretation paragraph.
- Final conclusions are evidence-based and refer to tables/plots.
- File name is exactly: `GGSN_Patryk_Chamera_Zad2.ipynb`.

---

## Recommended Metrics and Visualizations (Quick Reference)

### Quantitative metrics
- Accuracy (overall)
- Precision, Recall, F1-score (per class)
- Macro F1 (primary for balanced multiclass fairness)
- Weighted F1
- Optional: One-vs-rest ROC-AUC

### Visual diagnostics
- Loss/accuracy learning curves
- Confusion matrices
- Class-wise sample predictions (correct/incorrect)
- Grad-CAM overlays
- Optional intermediate feature map visualizations

---

## Minimal Experiment Matrix (Practical and Sufficient)
To keep scope manageable while meeting requirements:
1. Baseline CNN (no augmentation)
2. Baseline CNN + augmentation
3. Baseline CNN + augmentation + regularization tuning
4. Transfer model (frozen backbone)
5. Transfer model (fine-tuned top layers)

This matrix is compact, academically defensible, and directly aligned with the required optimization, augmentation, transfer learning, and explainability objectives.
