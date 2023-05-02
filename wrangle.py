import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler


def regression_errors(y, yhat):
    """
    Calculates regression errors and returns the following values:
    sum of squared errors (SSE)
    explained sum of squares (ESS)
    total sum of squares (TSS)
    mean squared error (MSE)
    root mean squared error (RMSE)
    """
    # Calculate SSE, ESS, and TSS
    SSE = np.sum((y - yhat) ** 2)
    ESS = np.sum((yhat - np.mean(y)) ** 2)
    TSS = SSE + ESS

    # Calculate MSE and RMSE
    n = len(y)
    MSE = SSE / n
    RMSE = np.sqrt(MSE)

    return SSE, ESS, TSS, MSE, RMSE


def baseline_mean_errors(y):
    """
    Calculates the errors for the baseline model and returns the following values:
    sum of squared errors (SSE)
    mean squared error (MSE)
    root mean squared error (RMSE)
    """
    # Calculate baseline prediction
    y_mean = np.mean(y)
    yhat_baseline = np.full_like(y, y_mean)

    # Calculate SSE, MSE, and RMSE
    SSE_bl = np.sum((y - yhat_baseline) ** 2)
    n = len(y)
    MSE_bl = SSE / n
    RMSE_bl = np.sqrt(MSE)

    return SSE_bl, MSE_bl, RMSE_bl


def better_than_baseline(y, yhat):
    """
    Checks if your model performs better than the baseline and returns a boolean value.
    """
    # Calculate errors for model and baseline
    SSE_model = np.sum((y - yhat) ** 2)
    SSE_baseline = np.sum((y - np.mean(y)) ** 2)

    # Calculate R-squared and RMSE for model and baseline
    r2_model = r2_score(y, yhat)
    r2_baseline = 0.0  # since baseline prediction is always the mean value
    rmse_model = mean_squared_error(y, yhat, squared=False)
    rmse_baseline = mean_squared_error(y, np.full_like(y, np.mean(y)), squared=False)

    # Check if model SSE is less than baseline SSE
    if SSE_model < SSE_baseline:
        print("Model outperformed the baseline.")
    else:
        print("Model did not outperform the baseline.")
    
    # Print R-squared and RMSE for model and baseline
    print(f"R-squared (model): {r2_model:.4f}")
    print(f"R-squared (baseline): {r2_baseline:.4f}")
    print(f"RMSE (model): {rmse_model:.4f}")
    print(f"RMSE (baseline): {rmse_baseline:.4f}")    
    
    
def plot_residuals(y, yhat):
    """
    Creates a residual plot using matplotlib.
    """
    # Calculate residuals
    residuals = y - yhat

    # Create residual plot
    plt.scatter(yhat, residuals)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()
    
    
def select_kbest(X, y, k):
    """
    Select the top k features from X based on their correlation with y using the f_regression method.
    """
    selector = SelectKBest(f_regression, k=k)  # Create a SelectKBest object with the f_regression method and k as input
    selector.fit(X, y)  # Fit the selector to the data
    mask = selector.get_support()  # Get a mask of the selected features
    selected_features = []  # Create an empty list to store the names of the selected features
    for bool, feature in zip(mask, X.columns):  # Loop through the mask and the columns of X
        if bool:  # If the feature is selected
            selected_features.append(feature)  # Add the name of the feature to the selected_features list
    return selected_features  # Return the list of selected features


def rfe(X, y, k):
    """
    Perform Recursive Feature Elimination to select the top k features from X based on their correlation with y.
    """
    estimator = LinearRegression()  # Create a LinearRegression object as the estimator
    selector = RFE(estimator, n_features_to_select=k)  # Create an RFE object with k as the number of features to select
    selector.fit(X, y)  # Fit the selector to the data
    mask = selector.support_  # Get a mask of the selected features
    selected_features = []  # Create an empty list to store the names of the selected features
    for bool, feature in zip(mask, X.columns):  # Loop through the mask and the columns of X
        if bool:  # If the feature is selected
            selected_features.append(feature)  # Add the name of the feature to the selected_features list
    return selected_features  # Return the list of selected features


def visualize_corr(df, sig_level=0.05, figsize=(10,8)):
    """
    Takes a Pandas dataframe and a significance level, and creates a heatmap of 
    statistically significant correlations between the variables.
    """
    # Create correlation matrix
    corr = df.corr()

    # Mask upper triangle of matrix (redundant information)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Get statistically significant correlations (p-value < sig_level)
    pvals = df.apply(lambda x: df.apply(lambda y: stats.pearsonr(x, y)[1]))
    sig = (pvals < sig_level).values
    corr_sig = corr.mask(~sig)

    # Set up plot
    plt.figure(figsize=figsize)
    sns.set(font_scale=1.2)
    sns.set_style("white")

    # Create heatmap with statistically significant correlations
    sns.heatmap(corr_sig, cmap='Purples', annot=True, fmt=".2f", mask=mask, vmin=-1, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title(f"Statistically Significant Correlations (p<{sig_level})")
    plt.show()
    
    
    
def print_corr_table(df, target):
    """
    Takes a Pandas dataframe and a target variable name, and prints a table of 
    correlation coefficients and p-values ordered by highest to lowest correlation coefficient.
    """
    # Create correlation matrix and extract correlation coefficients for target variable
    corr = df.corr()
    corr_target = corr[target]

    # Calculate P-values for all correlations
    pvals = df.apply(lambda x: df.apply(lambda y: stats.pearsonr(x, y)[1]))

    # Combine correlation coefficients and P-values into a single dataframe
    corr_table = pd.concat([corr_target, pvals[target]], axis=1)
    corr_table.columns = ["corr_coef", "p_value"]
    
    # Sort table by absolute correlation coefficient (in descending order)
    corr_table["abs_corr_coef"] = corr_table["corr_coef"].abs()
    corr_table = corr_table.sort_values("abs_corr_coef", ascending=False)
    corr_table = corr_table.drop(columns=["abs_corr_coef"])

    # Print table
    print("Correlation Coefficients and P-Values:")
    print(corr_table)
    
    
def show_distribution(dataframe, column_name):
    """
    Display a histogram showing the distribution of values in a dataframe column.
    
    Args:
    dataframe: pandas DataFrame containing the data
    column_name: string representing the name of the column whose distribution is to be displayed
    
    Returns:
    None
    """
    plt.hist(dataframe[column_name])
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.title('Distribution of ' + column_name)
    plt.show()
    
    
def convert_to_object(df, columns):
    """Converts the data types of a list of DataFrame columns to object.

    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        columns (list): A list of column names to convert.

    Returns:
        pandas.DataFrame: The modified DataFrame.
    """
    for col in columns:
        df[col] = df[col].astype('object')
    return df



    
    
#**************************************************Distributions*******************************************************

def get_hist(df):
    ''' Gets histographs of acquired continuous variables'''
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = [col for col in df.columns if col not in ['fips', 'year_built']]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=5)

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.tight_layout()

    plt.show()
        
        
def get_box(df):
    ''' Gets boxplots of acquired continuous variables'''
    
    # List of columns
    cols = ['liquor_receipts_in_usd', 'wine_receipts_in_usd',
       'beer_receipts_in_usd', 'total_receipts_in_usd', 'beer_ratio']

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]])

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()

    
def scale_data(train, 
               validate, 
               test, 
               columns_to_scale=['bedrooms', 'bathrooms', 'tax_value'],
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    scaler = MinMaxScaler()
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values, 
                                                  index = train.index)
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled
    
def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def get_sa_mixed_bev():
    return pd.read_csv('edited_sa_mixed_bev.csv')