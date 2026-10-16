# Loan Prediction Data Preparation

This project prepares loan application data for a machine learning model that predicts loan approval. It includes exploratory data analysis, cleaning, feature engineering, encoding, scaling, and train/test splitting. Preprocessing is fitted only on training data to avoid data leakage.

## Dataset

- **Name:** Loan Prediction Problem Dataset
- **Source:** https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
- **Size:** 614 rows and 13 original columns
- **Target:** Loan_Status — Y = approved, N = not approved
- **Problem type:** Supervised binary classification

## Project Steps

1. Explore the dataset using summaries and charts.
2. Check duplicates and missing values.
3. Remove the application ID column.
4. Create TotalIncome and LogTotalIncome features.
5. Separate inputs and target.
6. Split the data into 80% training and 20% testing.
7. Fill numeric gaps with the training median and apply StandardScaler.
8. Fill categorical gaps with the training mode and apply OneHotEncoder.
9. Check the processed outputs for missing values.

## How to Run

### Google Colab

1. Download the notebook from this repository.
2. Open Google Colab: https://colab.research.google.com/
3. Upload the notebook.
4. Select **Runtime → Run all**.

The notebook automatically downloads the dataset. If downloading fails, download the labeled training CSV from the dataset source, rename it `loan_prediction.csv`, and upload it to Colab before running the notebook.

### Local Jupyter Notebook

Install the required packages:

pip install pandas numpy matplotlib seaborn "scikit-learn>=1.2" jupyter

Start Jupyter:

jupyter notebook

Open the notebook and run all cells from top to bottom.

## Final Outputs

- `X_train`: cleaned, encoded, and scaled training inputs
- `X_test`: cleaned, encoded, and scaled test inputs
- `y_train`: training labels
- `y_test`: test labels

The split contains 491 training applications and 123 test applications. The number of processed input columns depends on the included features and encoded categories.

## Execution Note

Run every cell in order so that TotalIncome and LogTotalIncome remain included when creating the input features and splitting the data.

## Tools Used

Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

## Project Scope

This project covers data preparation. Model training and performance evaluation are future steps.
