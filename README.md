# Week 2 – Machine Learning Weekly Assessment

This project follows the four tasks in the supplied 3-page assessment PDF. Each task has its own folder and runnable `main.py`.

## Project structure

```text
Week2_Machine_Learning_Assessment/
├── data/                         # Shared loan applicant dataset
├── src/data_utils.py              # Dataset generator and shared pipeline
├── Task_1_Concepts_Splits_Leakage/
├── Task_2_Preprocessing_Features_Pipeline/
├── Task_3_Supervised_Unsupervised_Models/
├── Task_4_Evaluation_Tuning/
├── requirements.txt
└── README.md
```

## Dataset note
No trainer dataset was attached with the assessment PDF. Therefore, `data/loan_applicants.csv` is a reproducible synthetic loan-applicant dataset generated for this project. It contains a binary `default` target, a numeric `loan_amount` target for regression, numeric and categorical columns, missing values, and an application date. If your trainer provides a dataset, replace the CSV and update column names in `src/data_utils.py` as needed.

## Setup (Windows PowerShell)

```powershell
cd Week2_Machine_Learning_Assessment
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run each task

Run from the main project folder so imports work:

```powershell
python Task_1_Concepts_Splits_Leakage/main.py
python Task_2_Preprocessing_Features_Pipeline/main.py
python Task_3_Supervised_Unsupervised_Models/main.py
python Task_4_Evaluation_Tuning/main.py
```

Task 3 and Task 4 require XGBoost. Task 4 performs hyperparameter searches and can take several minutes depending on your computer.

## Outputs
Each task saves its charts and model artifacts in its own folder. Task 2 saves a fitted pipeline as `loan_default_pipeline.joblib`. The dataset is generated automatically if it is missing.

## Important
Metrics are produced by running the code; the expected values shown in the assessment PDF are examples, not guaranteed results. Keep the test set untouched until the final evaluation in Task 4. Commit source code and documentation, not virtual environments or secrets.
