# Wafer Defect Classification Using CNN

A deep learning project for classifying semiconductor wafer defect patterns using Convolutional Neural Networks (CNNs) and PyTorch.

## Overview

Semiconductor wafer inspection is an important step in identifying manufacturing defects and maintaining process quality. This project explores an image-based deep learning approach to automatically classify wafer maps into different defect categories.

The goal of this project is to build an end-to-end classification pipeline, from dataset preprocessing and class-imbalance handling to CNN training and model evaluation.

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

The classification report also provides class-level precision, recall, and F1 scores to identify classes that are more difficult for the model to distinguish.

## Error Analysis

The project includes:

* Confusion matrix analysis
* Classification reports
* Visualization of misclassified wafer maps
* Overall validation error analysis

The error analysis showed that performance varied considerably across defect categories. In particular, classes such as **Scratch** and **Loc** were more challenging to classify, suggesting that visually similar defect patterns and class imbalance remain important challenges.

## Results Visualization

The notebook includes visualizations for:

* Training and validation loss
* Training and validation F1 score
* Confusion matrix
* Correct and incorrect predictions
* Misclassified wafer samples

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

## Notebook

The complete implementation is available in the Jupyter/Google Colab notebook:

[`wafer_fault_classification.ipynb`](./wafer_fault_classification.ipynb)

The notebook contains the complete workflow, including data preparation, model training, validation, evaluation, and error analysis.

## Reproducibility

Kaggle API credentials are **not stored in this repository**.

To run the notebook in Google Colab, configure your own Kaggle credentials using Colab Secrets:

* `KAGGLE_USERNAME`
* `KAGGLE_KEY`

The dataset is downloaded automatically through the Kaggle API.

## Future Improvements

Potential improvements include:

* Applying data augmentation to improve generalization
* Experimenting with deeper CNN architectures
* Comparing CNN architectures and hyperparameters
* Investigating alternative approaches for severe class imbalance
* Exploring transfer learning and more advanced computer vision models
* Evaluating the model on a separate test dataset

## Project Goal

This project was developed to gain practical experience in applying deep learning and computer vision techniques to a semiconductor manufacturing problem.

It serves as a foundation for further exploration of **AI-driven semiconductor inspection and intelligent hardware systems**.
