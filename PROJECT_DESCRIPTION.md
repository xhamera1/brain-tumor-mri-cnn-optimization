# Task 2: Optimization of Models Operating on Image Data

## Project Title
Brain Tumor MRI Multiclass Classification with CNN Optimization, Data Augmentation, and Transfer Learning

## Project Context
This mini-project is developed for the academic assignment **"Task 2: Optimization of models operating on image data"** (deadline: **April 20, 2026, 23:59**).  
The project addresses a computer vision classification problem in the medical imaging domain and follows course requirements related to:
- designing and comparing CNN-based solutions,
- applying optimization and regularization strategies,
- using pre-trained models and transfer learning,
- validating results with quantitative metrics and visual explainability methods.

## Notebook Writing Style (important)
Implementation in `GGSN_Patryk_Chamera_Zad2.ipynb` should follow the style used in `CNN-lecture` notebooks:
- short and practical markdown comments,
- student-like, direct explanations instead of overly formal text,
- no "enterprise/professional" boilerplate that is not needed for the current step,
- import/configure only what is currently used,
- keep the code fully correct and academically valid, but concise.

## Problem Statement
The objective is to perform **multiclass classification** of brain MRI scans into four diagnostic categories:
1. Glioma
2. Meningioma
3. Pituitary tumor
4. No tumor

The practical goal is to build models that generalize reliably while preserving interpretability, which is critical in high-stakes medical contexts.

## Dataset Description
- **Dataset:** Brain Tumor MRI Dataset (Masoud Nickparvar, Kaggle)  
  <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>
- **Total size:** 7,200 images
- **Classes:** 4 (balanced)
- **Split:**
  - Training: 1,400 images per class
  - Testing: 400 images per class
- **Key properties:**
  - No train-test overlap (no leakage by design)
  - Variable spatial resolution (requires resizing)
  - MRI-specific intensity characteristics (requires normalization)
  - Visually meaningful lesions (supports explainability using Grad-CAM)

This dataset is especially suitable for comparative modeling because class balance allows fair metric interpretation and reduces confounding effects caused by class imbalance.

## Methodological Scope
The project implements a progressive experimental pipeline, from simple to advanced models:

### 1) Data Engineering and Preprocessing
- Data loading and structure verification.
- Exploratory Data Analysis (EDA) at class and sample level.
- Standardization of input dimensions (image resizing).
- Pixel value normalization and optional margin/crop cleanup.
- Construction of train/validation/test workflows in simple, clear notebook cells.

### 2) Baseline CNN (From Scratch)
- Development of a compact custom CNN as a reference architecture.
- Training with standard optimization settings.
- Baseline analysis of underfitting/overfitting behavior using learning curves.

### 3) Data Augmentation and Regularization
- Augmentation policies such as rotation, translation, horizontal/vertical flip, zoom, and brightness changes.
- Regularization techniques including dropout, batch normalization, and early stopping.
- Comparative assessment of augmentation impact on generalization.

### 4) Transfer Learning and Fine-Tuning
- Integration of pre-trained CNN backbones (e.g., VGG16, ResNet50, EfficientNet).
- Two-stage training:
  - frozen feature extractor phase,
  - selective unfreezing and fine-tuning phase.
- Optimization with learning-rate scheduling and weight decay when appropriate.

### 5) Evaluation and Comparative Analysis
- Performance comparison across model families and training strategies.
- Metrics: accuracy, macro/weighted precision, recall, F1-score, confusion matrix, per-class behavior.
- Error analysis to identify common misclassifications and likely anatomical/visual causes.

### 6) Explainability and Visual Validation
- Grad-CAM visualization to verify attention localization over tumor regions.
- Optional filter/feature-map visualization to inspect representation learning.
- Clinical relevance discussion: whether model decisions align with radiologically meaningful regions.

## Expected Outcomes
By the end of the notebook implementation, the project is expected to provide:
- A robust empirical comparison between baseline CNN and transfer learning approaches.
- Evidence that augmentation and regularization improve generalization.
- Demonstration that pre-trained models provide superior feature extraction on limited medical imaging tasks.
- Explainability artifacts (Grad-CAM) showing that high-performing models focus on diagnostically relevant image regions rather than spurious patterns.
- Clear, evidence-based conclusions regarding architecture selection, optimization strategy, and trustworthiness for medical-image classification scenarios.

## Academic Deliverable
The final deliverable will be a complete Jupyter notebook: **`GGSN_Patryk_Chamera_Zad2.ipynb`**, built in alignment with lecture materials from the `CNN-lecture` directory and structured as a reproducible end-to-end machine learning experiment.
