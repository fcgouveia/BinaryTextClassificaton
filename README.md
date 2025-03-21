# BinaryTextClassificaton

A machine learning-based text classification system that categorizes text into "Type A" and "Not Type A" classes using multiple classifiers and selects the best performing one.

## Overview

This project implements a text classification pipeline that:
1. Loads text data from TSV files
2. Processes and vectorizes the text
3. Trains multiple classification models
4. Evaluates each model's performance
5. Selects the best performing classifier
6. Applies the best model to classify new texts
7. Saves the classification results

## Features

- Support for multiple classification algorithms:
  - Naive Bayes
  - Support Vector Machine (LinearSVC)
  - Random Forest
  - Logistic Regression
  - Gradient Boosting
- Automated model selection based on accuracy
- Performance evaluation with detailed metrics (precision, recall, F1-score)
- Simple interface for classifying new text data

## Requirements

```
numpy
pandas
scikit-learn
```

## Usage

### Data Format

The system expects TSV (Tab-Separated Values) files:

1. Training data (`trainset.tsv`):
   - Two columns: text content and label (0 for "Not Type A", 1 for "Type A")
   - No header row

2. Text to classify (`to_classify.tsv`):
   - Single column with text content
   - No header row

### Running the Code

```bash
python bin_text_classifier.py
```

The script will:
1. Train all classifiers
2. Output performance metrics for each
3. Select the best performing model
4. Classify the texts from `to_classify.tsv`
5. Save results to `classified_results.tsv`

### Output

The system generates:
1. Console output with training progress and evaluation metrics
2. A TSV file (`classified_results.tsv`) containing:
   - Original text
   - Classification result ("Type A" or "Not Type A")

## Performance Notes

- SVM and Logistic Regression typically achieve the highest accuracy (~98%)
- Training parameters have been optimized to avoid convergence warnings
- The SVM uses 10,000 max iterations to ensure convergence
- Logistic Regression uses the liblinear solver for better performance

## Customization

To adapt this system for your own classification needs:
1. Replace the input TSV files with your own data
2. Modify the `target_names` in the classification report if needed
3. Adjust the prediction function to match your classes
4. Tune the classifier parameters for your specific dataset
