# Employee Attrition Analysis

## Overview

A large Telecommunications company employs around 4000 employees. Every year, around 15% of their employees leave and need to be replaced. The management believes this attrition level to be a problem for the following reasons: 

- Project delays leading to a reputation loss amongst consumers and partners
- Big amount of resources is needed to maintain a large recruitment department 
- Work Productivity/Effectiveness is reduced due to the onboarding period for new staff

Hence, they contracted a workplace engineering & analytics firm to understand what factors are contributing to the high attrition, and what changes they should make to their workplace to support better retention. In addition, given limited resources, the company would like to know, which variable is the most important and needs to be addressed straight away.

## Table of Contents

- [Employee Attrition Analysis](#canonical-attrition-analysis)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)

## Project Structure

```
.
├── input_data
│   ├── general_data.csv
│   ├── employee_survey_data.csv
│   ├── manager_survey_data.csv
│   ├── in_time.csv
│   └── out_time.csv
├── jupyter_notebooks
│   ├── exploratory_analysis.ipynb
│   ├── training_model.ipynb
├── preprocessed_data
│   ├── clean_data01.csv
├── results
│   └── feature_importances.csv
├── preprocessing.py
├── train_evaluate_model.py
├── main.py
├── README.md
└── requirements.txt
```

## Installation

1. Clone the repository:

   ```bash
   git clone fcfuentes/employee_attrition_analysis
   ```

2. Create and activate a virtual environment (optional):

   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preprocessing**:

   Run the `preprocessing.py` script to preprocess the raw data and generate the cleaned dataset.

   ```bash
   python preprocessing.py
   ```

2. **Model Training and Evaluation**:

   Run the `main.py` script to train and evaluate the models.

   ```bash
   python main.py
   ```

## Data Preprocessing

The `preprocessing.py` script performs the following steps:

- Joins employee data from multiple CSV files.
- Creates new features based on time data (e.g., average entry time, exit time).
- Removes features that do not significantly contribute to the model's performance.
- Removes missing data.
- Saves the cleaned data to `data/clean_data.csv`.

## Model Training and Evaluation

The `main.py` script performs the following steps:

- Loads and preprocesses the data.
- Calculates Variance Inflation Factor (VIF) to remove multicollinear features.
- Splits the data into training and testing sets.
- Applies SMOTE to handle class imbalance.
- Trains multiple classification models.
- Evaluates the models using cross-validation and test data.
- Performs hyperparameter tuning for the best model.
- Saves and plots the feature importances.

## Results

The feature importances and model performance metrics are saved in the `results` directory. The top features contributing to attrition and the performance of the best model are highlighted.
