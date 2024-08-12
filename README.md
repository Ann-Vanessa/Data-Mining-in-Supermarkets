# Data-Mining-in-Supermarkets
This project analyses customer buying patterns using various data mining techniques such as the apriori algorithm and NOVA classification

# Apriori Algorithm and Bidirectional LSTM Model

## Overview

This project demonstrates the implementation of two data mining techniques: the **Apriori Algorithm** for association rule mining and a **Bidirectional Long Short-Term Memory (BiLSTM) Model** for NOVA classification. The Apriori algorithm is applied for discovering frequent itemsets in transactional datasets, while the BiLSTM model is used for classifying foods under their NOVA classes.

## Features

### Apriori Algorithm
- **Association Rule Mining:** Extract frequent itemsets from a dataset of transactions and generate association rules.
- **Support, Confidence, and Lift:** Calculate these metrics to evaluate the strength of the association rules.
- **Customizable Parameters:** Set minimum support, confidence, and lift thresholds to filter the generated rules.

### Bidirectional LSTM Model
- **NOVA class Prediction:** Train a BiLSTM model to predict the NOVA classes, food classes and their subclasses.
- **Bidirectional Architecture:** Utilize both past and future contexts to improve prediction accuracy.
- **Customizable Network Architecture:** Adjust the number of LSTM layers, units per layer, dropout rates, etc.
- **Training and Evaluation:** Train the model on a dataset and evaluate its performance using common metrics like accuracy, loss, etc.

## Project Structure

```
├── apriori/
│   ├── apriori.py                # Implementation of the Apriori algorithm
│   └── datasets/
│       └── transactions.csv      # Sample dataset for Apriori
│
├── bilstm/
│   ├── model.py           # Implementation of the BiLSTM model
│   ├── train.py                  # Script to train the BiLSTM model
│
├── README.md                     # Project documentation
```

## Setup and Installation

### Prerequisites
- Google Colab account
- Dataset files uploaded to Google Drive or accessible from Colab

### Running the Project in Google Colab

1. **Open Google Colab:**
   - Go to [Google Colab](https://colab.research.google.com/) and sign in with your Google account.

2. **Set Up the Environment:**
   - Create a new Colab notebook or upload the provided `.ipynb` files if available.
   - Install the necessary dependencies:
     ```python
     !pip install numpy pandas scikit-learn
     ```

3. **Mount Google Drive:**
   - If your dataset or scripts are stored in Google Drive, mount your drive to access them:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```

4. **Load and Run the Apriori Algorithm:**
   - Load the Apriori script and dataset from your drive or directly in Colab:
     ```python
     !python /content/drive/MyDrive/path_to_your_script/apriori.py --min_support 0.2 --min_confidence 0.5
     ```

5. **Train the BiLSTM Model:**
   - Load and run the training script for the BiLSTM model:
     ```python
     !python /content/drive/MyDrive/path_to_your_script/train.py --epochs 50 --batch_size 32
     ```

6. **Evaluate the BiLSTM Model:**
   - After training, evaluate the model’s performance:
     ```use an inbuilt library to evaluate the the model's performance.
     ```

### Notebooks

- **Association Rule Mining Algorithm:**
  - You can create a notebook to run the Apriori algorithm step by step, including data preprocessing, running the algorithm, and analyzing the results.

- **NOVA classification model:**
  - A notebook demonstrating how to load data, preprocess it, train the BiLSTM model, and evaluate the results.

## Dependencies

The project uses the following Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow` or `keras` (for BiLSTM)

