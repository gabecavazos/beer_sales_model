# Beer Sales Model

## Overview

The goal of this project is to analyze a dataset containing information about the tax paid by bars on beer, wine, and liquor sales in Texas, and create a predictive model and sales strategy to sell beer to bars with growing beer ratios in San Antonio.

The original dataset is provided by the State of Texas Open Data Portal. Every establishment licensed to sell alcohol by the Texas Alcoholic Beverage Commission must publicly report their mixed beverage gross receipts monthly by law. This file contains a list of taxpayers required to report mixed beverage gross receipts tax reports under Tax Code Chapter 183, Subchapter B. The list provides taxpayer names, amounts reported, and other public information. The data is managed by the Texas Comptroller of Public Accounts, it originally contained around 3 million rows and 24 columns. It can be accessed here: https://data.texas.gov/dataset/Mixed-Beverage-Gross-Receipts/naix-2893

## Data Cleaning and Preprocessing

The raw dataset was cleaned and preprocessed to create a more structured and usable format. Columns were added, and calculated columns were derived from the original dataset.

## Exploratory Data Analysis (EDA)

EDA was performed on the cleaned dataset to understand the relationships between different features and the target variable (beer ratio growth). Visualizations were created to analyze the trends and patterns in the data.

## Feature Engineering

Additional features were created to capture more information from the dataset, such as lagged features, rolling averages, and interaction features.

## Model Selection and Evaluation

Three regression models were chosen for predicting beer ratio growth: Linear Regression, Ridge Regression, and Lasso Regression. The performance of each model was evaluated using the Root Mean Squared Error (RMSE) metric. A table was created to show the performance of each model on the train, validation, and test sets.

## Predictions and Sales Strategy

A DataFrame of predictions for each model was created, and a new column called `combined_growth_predictions` was added, which takes the predicted value closest to the `beer_growth_ratio`. This can be used to create a sales strategy targeting bars with growing beer ratios.

# Dependencies
Python 3.6 or later
pandas
numpy
scikit-learn
seaborn
matplotlib

# Recreation

to recreate, clone the repo and run final_report.ipynb sequentially.
