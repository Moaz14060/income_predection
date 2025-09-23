import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def check_na (df: pd.DataFrame) -> pd.DataFrame:
    """
    Function that counts the number of NA values and their percentages. Diplay them as a table.

    Parameters:
        df: your data frame

    Return Value:
        A table containg the missing values' count and their percentages.
    """
    # Create a new df with "count" column changed to "missing#" 
    missing_df=df.isnull().sum().reset_index(name="missing#")
    # Create a new column and calculate the percentage of the NA
    missing_df["missing%"]=(missing_df["missing#"]/len(df))*100
    # Filter the table to only show columns with NA values only
    missing_df=missing_df[missing_df["missing#"] > 0]
    missing_df.rename({"index": "feature"}, axis=1, inplace=True)
    return missing_df

def categorical_visualization(df: pd.DataFrame, cate_col: list, num_col=3, figsize=(12, 5), rotation=45, nunique=5, edgecolor="blue"):
    """
    Function to plot categorical features using seaborn "countplot".

    Parameters:
        1. df: your data frame (pd.DataFrame)
        2. cate_col: a list of the categorical columns (list)
        3. num_col: number of the columns to be used in the subplot (int, defualt= 3)
        4. figsize: the size of the figure (tuple, defualt= (12, 5))
        5. rotation: the rotation of the "xticks" (int, default= 45)
        6. nunique: the number categories you want to define in order to use the rotation parameter (int, default= 5)
            so for example: if you put nunique= 6 then if the categories are greater then 6 the rotation will be used.
        7. edgecolor: the edgecolor of the bins (str, defualt= "blue")

    Return Value:
        Returns the countplots of the categroical features as subplots in one figure.
    """
    # Calculating the num of row based on the number of the categorical columns
    num_row=int(np.ceil(len(cate_col)/num_col))
    plt.figure(figsize=figsize)
    for i, col in enumerate(cate_col): # Get the index and the column
        plt.subplot(num_row, num_col, i+1)
        sns.countplot(data=df, x=df[col], edgecolor=edgecolor)
        # Remove any "_", replace it with " " and display it as title 
        plt.title(F"Distribution of {col}".replace("_", " ").title())
        plt.xlabel("")
        plt.ylabel("")
        # If the categories of the feature > parameter (nunique) then use the parameter (rotation) 
        # Else: don't use it
        if df[col].nunique() > nunique:
            plt.xticks(rotation=rotation)

    plt.suptitle("Categroical Feature Distribution")
    plt.tight_layout()
    
    
def numerical_visualization(df: pd.DataFrame, numerical_col: list, num_col=3, figsize=(12, 5), bins=30, edgecolor="blue", kde=False):
    """
    Function to plot numerical features using seaborn "histplot".

    Parameters:
        1. df: your data frame (pd.DataFrame)
        2. numerical_col: a list of the numerical columns (list)
        3. num_col: number of the columns to be used in the subplot (int, defualt= 3)
        4. figsize: the size of the figure (tuple, defualt= (12, 5))
        5. bins: the number of bins to use in the histgram (int, default= 30)
        6. edgecolor: the edgecolor of the bins (str, defualt= "blue")
        7. kde: if you want to show the Kernel Density Estimation line with the bins (bool, default= False)

    Return Value:
        Returns the histplots of the numerical features as subplots in one figure.
    """
    # Calculating the num of row based on the number of the numerical columns
    num_row=int(np.ceil(len(numerical_col)/num_col))
    plt.figure(figsize=figsize)

    for i, col in enumerate(numerical_col): # Get the index and the column
        plt.subplot(num_row, num_col, i+1)
        sns.histplot(data=df, x=df[col], bins=bins, edgecolor=edgecolor, kde=kde)
        # Remove any "_", replace it with " " and display it as title 
        plt.title(F"Distribution of {col}".replace("_", " ").title())
        plt.xlabel("")
        plt.ylabel("")
            
    plt.suptitle("Numerical Feature Distribution")
    plt.tight_layout()
    

def boxplot_check(df: pd.DataFrame, numerical_col: list , num_col=3, figsize=(15, 5), color="blue"):
    """
    Function to plot boxplots for numerical features.

    Parameters:
        1. df: your data frame
        2. Numerical_col: a list of the numerical columns (list)
        3. num_col: number of the columns to be used in the subplot (int, defualt= 3)
        4. figsize: the size of the figure (tuple, defualt= (12, 5))
        5. color: the color of the boxplots (str, default= "blue")

    Return Value:
        Returns the boxplots of the numerical features as subplots in one figure.
    """
    # Calculating the num of row based on the number of the numerical columns
    num_row=int(np.ceil(len(numerical_col)/num_col))
    plt.figure(figsize=figsize)

    for i, col in enumerate(numerical_col): # Get the index and the column
        plt.subplot(num_row, num_col, i+1)
        sns.boxplot(data=df, x=col, color=color)
        # Remove any "_", replace it with " " and display it as title 
        plt.title(f"Quartile of {col} feature".replace("_", " ").title())
        plt.xlabel("")
        plt.ylabel("")

    plt.suptitle("Outlier Detection")
    plt.tight_layout()
    