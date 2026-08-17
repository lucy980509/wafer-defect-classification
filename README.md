# Wafer Defect Classification Using CNN

A deep learning project for classifying semiconductor wafer defect patterns using Convolutional Neural Networks (CNNs) and PyTorch.

## Overview

Semiconductor wafer inspection is an important step in identifying manufacturing defects and maintaining process quality. This project explores an image-based deep learning approach to automatically classify wafer maps into different defect categories.

The goal of this project is to build an end-to-end classification pipeline, from dataset preprocessing and class-imbalance handling to CNN training, model evaluation, and error analysis.

## Dataset

This project uses the **WM-811K Wafer Map Dataset**, which contains wafer maps representing different semiconductor defect patterns.

Only labeled defect samples were used for this classification task.

The following defect classes were included:

* Center
* Donut
* Edge-Loc
* Edge-Ring
* Loc
* Near-full
* Random
* Scratch

## Methodology

The project follows the following pipeline:

1. Load and preprocess wafer map data
2. Filter labeled defect samples
3. Encode categorical defect labels into numerical classes
4. Split the dataset into training and validation sets
5. Address class imbalance using class-weighted loss
6. Convert wafer maps into CNN-compatible image tensors
7. Train a CNN using PyTorch
8. Evaluate model performance using F1 score and classification metrics
9. Analyze misclassified wafer maps

### Model

A Convolutional Neural Network (CNN) was implemented using PyTorch.

The training pipeline uses:

* PyTorch
* Adam optimizer
* Weighted Cross-Entropy Loss
* ReLU activation
* Max Pooling
* Batch-based training
* GPU acceleration when available

## Evaluation

Because the dataset contains imbalanced defect classes, **weighted F1 score** was used alongside accuracy to evaluate model performance.

### Validation Results

| Metric      |  Score |
| ----------- | -----: |
| Accuracy    | 83.05% |
| Weighted F1 | 83.53% |
| Macro F1    | 77.73% |

The classification report also provides class-level precision, recall, and F1 scores to identify defect categories that are more difficult for the model to distinguish.

## Results Visualization

### Training History

The training and validation curves show how model performance changed across epochs.

![Training History](./training_history.png)

### Confusion Matrix

The confusion matrix provides a class-level view of the model's predictions and highlights which defect categories are most frequently confused with one another.

![Best Model Confusion Matrix](./best_model_confusion_matrix.png)

### Misclassified Samples

The following examples show wafer maps that were incorrectly classified by the best-performing model during validation.

![Misclassified Samples](./misclassified_samples.png)

## Error Analysis

The project includes:

* Confusion matrix analysis
* Classification reports
* Visualization of misclassified wafer maps
* Overall validation error analysis

The error analysis showed that performance varied considerably across defect categories. In particular, classes such as **Scratch** and **Loc** were more challenging to classify.

These results suggest that visually similar defect patterns and class imbalance remain important challenges for wafer defect classification.

## Technologies

* Python
* PyTorch
* NumPy
* pandas
* scikit-learn
* Matplotlib
* Seaborn
* Google Colab
* Kaggle API

## Requirements

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## Notebook

The complete implementation is available in the Jupyter/Google Colab notebook:

[`wafer_fault_classification.ipynb`](./wafer_fault_classification.ipynb)

The notebook contains the complete workflow, including:

* Dataset preparation
* Data preprocessing
* CNN model construction
* Model training
* Validation
* Classification evaluation
* Confusion matrix analysis
* Misclassified sample analysis

## Reproducibility

Kaggle API credentials are **not stored in this repository**.

To run the notebook in Google Colab, configure your own Kaggle credentials using **Colab Secrets**:

* `KAGGLE_USERNAME`
* `KAGGLE_KEY`

The dataset is downloaded automatically through the Kaggle API.

### Dataset Download

The notebook automatically downloads the WM-811K dataset using the Kaggle API and prepares the dataset for training.

Users should provide their own Kaggle API credentials rather than using credentials stored in the repository.

## Future Improvements

Potential improvements include:

* Applying data augmentation to improve generalization
* Experimenting with deeper CNN architectures
* Comparing different CNN architectures and hyperparameters
* Investigating alternative approaches for severe class imbalance
* Exploring transfer learning and more advanced computer vision models
* Evaluating the model on a separate test dataset
* Investigating more robust approaches for visually similar defect categories

## Project Goal

This project was developed to gain practical experience in applying deep learning and computer vision techniques to a semiconductor manufacturing problem.

Beyond classification performance, the project explores how machine learning can be applied to **automated semiconductor wafer inspection**.

It serves as a foundation for further exploration of **AI-driven semiconductor inspection, computer vision, and intelligent hardware systems**.

## Project Structure

```text
wafer-defect-classification/
│
├── README.md
├── wafer_fault_classification.ipynb
│
├── training_history.png
├── best_model_confusion_matrix.png
└── misclassified_samples.png
```
