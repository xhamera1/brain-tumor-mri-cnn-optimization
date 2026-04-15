# Brain Tumor MRI Multiclass Classification

## 1. Executive Summary
This project investigates the application of deep learning architectures for the automated classification of brain tumors using Magnetic Resonance Imaging (MRI). The primary objective was to evaluate the effectiveness of various optimization strategies, ranging from custom Convolutional Neural Networks (CNNs) to advanced Transfer Learning and Fine-Tuning techniques.

## 2. Dataset and Problem Specification
The study utilizes the **Brain Tumor MRI Dataset**, comprising 7,200 images across four distinct classes: **Glioma, Meningioma, Pituitary Tumor, and No Tumor**. 
* **Data Volume:** 5,600 training samples and 1,600 testing samples.
* **Class Distribution:** Perfectly balanced (1,400 train / 400 test per class).
* **Objective:** Achieve high diagnostic accuracy while ensuring model interpretability through gradient-based localization.

## 3. Methodological Framework
The experimental pipeline was executed through a multi-stage approach to identify the optimal configuration:
* **Baseline CNN:** Implementation of a shallow custom architecture to establish a performance floor and identify generalization gaps.
* **Regularization & Augmentation:** Integration of Batch Normalization, Dropout (0.5), and GPU-accelerated spatial transformations (rotation, zoom, flips) to mitigate initial overfitting.
* **Transfer Learning:** Utilization of the **ResNet50** architecture with weights pre-trained on ImageNet as a robust feature extractor.
* **Fine-Tuning:** Unfreezing the terminal convolutional blocks of ResNet50 and applying a localized gradient update with a reduced learning rate ($10^{-5}$).

## 4. Performance Analysis
The empirical results demonstrate a significant performance trajectory across model iterations:

| Architecture | Test Accuracy | Macro F1-Score | Status |
| :--- | :---: | :---: | :--- |
| **Baseline CNN** | 83.69% | 0.83 | High Overfitting |
| **Optimized CNN** | 77.44% | 0.76 | Underfitting/Capacity Limit |
| **Transfer Learning (Frozen)** | 87.63% | 0.87 | Stable Baseline |
| **Fine-Tuned ResNet50** | **93.06%** | **0.93** | **Optimal Solution** |

## 5. Interpretability and Explainability (Grad-CAM)
To validate the clinical relevance of the model's decision-making process, **Gradient-weighted Class Activation Mapping (Grad-CAM)** was implemented. The generated heatmaps successfully localized pathological regions, confirming that the network identifies tumors based on correct anatomical features rather than image artifacts or background noise.

## 6. Conclusions
The project highlights that while custom architectures are valuable for understanding CNN fundamentals, **Transfer Learning and Fine-Tuning** are essential for achieving high-fidelity results in specialized medical domains. The final model reached a **93.06% accuracy**, providing a robust and interpretable framework for automated neuro-radiological diagnostic support.
