# Beer Sales Model

## Executive Summary

This project aims to create a predictive model and sales strategy for selling beer to bars in San Antonio, Texas, that are experiencing growth in their beer ratios. Using a dataset containing information about the tax paid by bars on beer, wine, and liquor sales in Texas, we will analyze the data, engineer features, and train regression models to make predictions on beer ratio growth. The results will help us identify bars with the highest potential for beer ratio growth, allowing for a targeted sales strategy.

## Initial Questions/Hypotheses

1. Are there specific trends or patterns in the data that indicate a bar's beer ratio growth?
2. Can we identify key features that are strong predictors of beer ratio growth?
3. Do different regression models have varying performance levels when predicting beer ratio growth?

## Project Goals

1. Clean, preprocess, and engineer features in the dataset to create a structured and usable format for modeling.
2. Perform exploratory data analysis to understand the relationships between different features and the target variable (beer ratio growth).
3. Train and evaluate multiple regression models to predict beer ratio growth, comparing their performance using the Root Mean Squared Error (RMSE) metric.
4. Develop a sales strategy targeting bars in San Antonio with growing beer ratios, utilizing the model's predictions.
5. Continuously improve the model with new data and explore additional data sources to enhance its predictive power.

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

## The Plan 

1. **Data preprocessing**: Clean and preprocess the data to ensure it is suitable for creating a predictive model.  This involved handling missing values, converting categorical variables from numerical format, and converting obligation_end_date to date time

2. **Feature engineering**: Create new features that might have a relationship with a bar's beer ratio growth. This may include lagged features to account for previous months' beer ratios, aggregated features like rolling averages, and interaction features that capture the relationship between different columns.

3. **Exploratory data analysis (EDA)**: Perform an EDA to understand the relationships between different features and the target variable (beer ratio growth). This will help identify potential predictors and understand the trends and patterns in the data.

4. **Model training and validation**: Split the data into training and validation sets. Train the selected model(s) on the training set and evaluate their performance on the validation set using appropriate metrics such as Root Mean Squared Error (RMSE). Fine-tune the model parameters to improve the model's performance.

5. **Model evaluation**: Use cross-validation techniques to get a more accurate estimate of the model's performance on unseen data. This will help identify potential overfitting or underfitting issues.

6. **Sales strategy**: Based on the predictions from the model, identify the bars with the highest potential for beer ratio growth. Prioritize these bars in the sales strategy, focusing on areas with the highest concentration of high-potential bars. Develop a targeted marketing plan to appeal to these bars, emphasizing the growing demand for beer and the benefits of partnering with your distribution company.

7. **Continuous improvement**: Regularly update the model with new data to ensure it stays relevant and accurate. Monitor the model's performance and adjust the sales strategy as needed. Explore additional data sources that might help improve the model's predictive power, such as external factors like local events, economic indicators, or demographic data. A brand could also examine the list and remove existing customers from the sales strategy. Additional data collected about the bars and GIS information may improve the model and understanding of drivers.

# Dependencies

Python 3.6 or later
pandas
numpy
scikit-learn
seaborn
matplotlib
*important* you will need to install lazy predict to have file run cleanly, you can run the following the cli
pip install lazypredict
here is a link to lazypredict's documentation: https://pypi.org/project/lazypredict/
# Recreation

To recreate, clone the repo and run final_report.ipynb sequentially.
