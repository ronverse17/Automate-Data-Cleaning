# Automated Data Cleaning Pipeline Project
## 📌 Project Overview
This is an automated data preprocessing pipeline built using Python. 
This class provides common data cleaning steps to prepare data for analysis or machine learning algorithms. This tool standardizes column names, handles missing values, detects outliers, optimizes categorical data types & generates a cleaning report.

## ✨ Features
- Standardizes column names to lower case (snake case)
- Detects and imputes missing values (median for numeric, mode for categorical by default)
- Standardizes string columns to lowercase with spaces removed if present at the beginning/end
- Identifies constant columns and high-cardinality features
- Detects potential numeric outliers using the IQR rule
- Converts low cardinality object columns to category dtype
- Generates a structured report summarizing cleaning actions

## ⚙️ Requirements
- Python
- pandas
- numpy

## Clone the repository
```bash
git clone https://github.com/ronverse17/Automate-Data-Cleaning.git
cd Automate-Data-Cleaning
```

## 🚀 Usage
For usage, refer to the test_file.ipynb.

## 📂 Files in this Repo
- data_cleaner.py → Contains the pipeline for cleaning the DataFrame
- test_data.csv → Dataset used for testing the pipeline
- test_file.ipynb → Jupyter Notebook containing demo & test results for the pipeline 
