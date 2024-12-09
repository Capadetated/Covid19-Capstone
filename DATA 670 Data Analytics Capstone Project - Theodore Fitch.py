# -*- coding: utf-8 -*-
"""
METADATA
Written by Theodore Fitch
DATA 670 at UMGC

Python code written for capstone project predicting ICU admission and mortality in COVID-19 patient datasets.

Last updated: 10DEC24
"""


#######################################################
#//
## IMPORT NECESSARY LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


#######################################################
#//
## DATA LOADING

# Filepaths for the file locations
covid1_path = 'C:/Users/soari/Documents/Assignments/Data Analytics/UMGC/Fall 2024 Data 670/Data/Covid/Covid1 COVID-19 patient pre-condition dataset/Covid1.csv' 
covid2_path = 'C:/Users/soari/Documents/Assignments/Data Analytics/UMGC/Fall 2024 Data 670/Data/Covid/Covid2 COVID-19 Dataset/Covid2.csv'
metadata_path = 'C:/Users/soari/Documents/Assignments/Data Analytics/UMGC/Fall 2024 Data 670/Assignment 2/Metadata.xlsx'

# Load the datasets
covid1_df = pd.read_csv(covid1_path)
covid2_df = pd.read_csv(covid2_path)

#######################################################
#//
## EDA FOR COVID1 AND COVID2

## COVID1
# Function to transform date variables to include only month and year
def transform_dates_to_month_year(df, date_columns):
    for date_col in date_columns:
        if date_col in df.columns:
            # Convert to datetime format if not already
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            # Extract only the year and month in the format 'YYYY-MM'
            df[date_col] = df[date_col].dt.strftime('%Y-%m')
    return df

# Define the date columns
date_columns = ['entry_date', 'date_symptoms', 'date_died']

# Apply the transformation to the dataframe
covid1_df = transform_dates_to_month_year(covid1_df, date_columns)

# Show the first few rows to check the transformation
print(covid1_df[date_columns].head())


# Function to categorize date columns by month
def categorize_dates(df, date_columns):
    for date_col in date_columns:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')  # Convert to datetime
            df[f'{date_col}_MONTH'] = df[date_col].dt.to_period('M')  # Extract year-month period
    return df

# Function to combine all categories of the first variable into one bar
def combine_first_variable(df, first_var):
    df[first_var] = "All"  # Group all values under the label 'All'
    return df

# Function to perform EDA on categorical columns, and create bar charts for each category
def eda_for_categorical(df, dataset_name, date_columns):
    # Adjust the number of rows and columns based on the number of categorical variables
    categorical_columns = df.select_dtypes(include=['object', 'category', 'int64']).columns.tolist()
    categorical_columns = [col for col in categorical_columns if col != 'ID']  # Exclude ID column

    # Adjust figure size based on the number of categorical variables
    fig, axes = plt.subplots(len(categorical_columns) // 3 + 1, 3, figsize=(18, 20))  # More space for better readability
    fig.suptitle(f'EDA for {dataset_name}', fontsize=16)

    for i, column in enumerate(categorical_columns):
        ax = axes[i // 3, i % 3]  # Plotting in a matrix
        sns.countplot(data=df, x=column, color='#ADD8E6', ax=ax)  # Light blue color
        ax.set_title(f'{column} Distribution')
        
        # Completely remove labels and axis ticks for age and date columns
        if column == 'age' or column in [f'{col}_MONTH' for col in date_columns]:
            ax.set_xticks([])  # Remove x-axis ticks
            ax.set_yticks([])  # Remove y-axis ticks
            ax.set_ylabel('')  # Remove y-axis label
            ax.set_xlabel('')  # Remove x-axis label
        else:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")  # Keep labels for other variables

            # Adding labels only for non-age and non-date columns
            for p in ax.patches:
                ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 1), 
                            textcoords='offset points')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # More space between the subplots for better readability
    plt.show()

# Define the date columns
date_columns = ['ENTRY_DATE', 'DATE_SYMPTOMS', 'DATE_DIED']

# Categorize date columns by month
covid1_df = categorize_dates(covid1_df, date_columns)

# Combine all categories of the first variable into one bar
first_variable = covid1_df.columns[0]  # Get the first column 
covid1_df = combine_first_variable(covid1_df, first_variable)

# Add the new month columns to the list of categorical variables
for date_col in date_columns:
    if f'{date_col}_MONTH' in covid1_df.columns:
        covid1_df[f'{date_col}_MONTH'] = covid1_df[f'{date_col}_MONTH'].astype(str)

# Perform EDA on the dataset
eda_for_categorical(covid1_df, "Covid1 Dataset", date_columns)


##COVID2

# Define the date column
date_columns_covid2 = ['DATE_DIED']
# Function to transform date variables to include only month and year
def transform_dates_to_month_year(df, date_columns):
    for date_col in date_columns:
        if date_col in df.columns:
            # Convert to datetime format if not already
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            # Extract only the year and month in the format 'YYYY-MM'
            df[date_col] = df[date_col].dt.strftime('%Y-%m')
    return df

# Apply the transformation to the 'DATE_DIED' column
covid2_df = transform_dates_to_month_year(covid2_df, date_columns_covid2)

# Show the first few rows to check the transformation
print(covid2_df.head())


# Function to perform EDA on categorical columns, and create bar charts for each category
def eda_for_categorical_covid2(df, dataset_name):
    # Adjust the number of rows and columns based on the number of categorical variables, excluding 'DATE_DIED'
    categorical_columns = df.select_dtypes(include=['object', 'category', 'int64']).columns.tolist()
    
    # Adjust figure size based on the number of categorical variables
    fig, axes = plt.subplots(len(categorical_columns) // 3 + 1, 3, figsize=(18, 20))  # More space for better readability
    fig.suptitle(f'EDA for {dataset_name}', fontsize=16)

    for i, column in enumerate(categorical_columns):
        ax = axes[i // 3, i % 3]  # Plotting in a matrix
        sns.countplot(data=df, x=column, color='#ADD8E6', ax=ax)  # Light blue color
        ax.set_title(f'{column} Distribution')

        # Remove labels and axis ticks for the 'AGE' column
        if column == 'AGE':
            ax.set_xticks([])  # Remove x-axis ticks
            ax.set_yticks([])  # Remove y-axis ticks
            ax.set_ylabel('')  # Remove y-axis label
            ax.set_xlabel('')  # Remove x-axis label
        else:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")  # Keep labels for other variables

            # Adding labels for each bar (standard for all other variables)
            for p in ax.patches:
                ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 1), 
                            textcoords='offset points')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # More space between the subplots for better readability
    plt.show()

# Perform EDA on the Covid2 dataset, excluding 'DATE_DIED' and adjusting 'AGE'
eda_for_categorical_covid2(covid2_df, "Covid2 Dataset")

#######################################################
#//
## EDA OF DEATH/AGE
# Function to create the age histogram for deceased patients
def plot_age_histogram_deceased(df, date_col, age_col, dataset_name):
    # Filter the dataset for deceased patients (where 'DATE_DIED' or 'date_died' is not NaT)
    deceased_df = df[df[date_col].notna()]
    
    # Remove any NaN values from the age column before plotting
    age_data = deceased_df[age_col].dropna()  # Drop any NaN values in the age column

    # Ensure that age_data is clean and doesn't have NaN or non-integer values
    if len(age_data) == 0:
        print(f"No valid age data for deceased patients in {dataset_name}.")
        return
    
    # Plot histogram for Age with same styling and 10-year bins
    plt.figure(figsize=(10, 6))
    sns.distplot(age_data.astype(int), bins=range(0, int(age_data.max()) + 10, 10), kde=False, color='#ADD8E6')  # Light blue color
    plt.title(f'Age Distribution of Deceased Patients in {dataset_name} (10-Year Bins)')
    plt.xlabel('Age (Years)')
    plt.ylabel('Frequency')
    plt.show()

# Call the function for both Covid1 and Covid2 datasets

# For Covid1 dataset, use 'date_died' and 'age' as the column names
plot_age_histogram_deceased(covid1_df, 'date_died', 'age', "Covid1 Dataset")

# For Covid2 dataset, use 'DATE_DIED' and 'AGE' as the column names
plot_age_histogram_deceased(covid2_df, 'DATE_DIED', 'AGE', "Covid2 Dataset")


#######################################################
#//
## MISSING VALUE COUNT

# Function to count rows with missing values based on custom criteria
def count_missing_values(df, dataset_name):
    # Define the conditions for missing values for the specific columns
    missing_conditions = (
        (df['intubed'] == 99) | 
        (df['pneumonia'] == 99) | 
        (df['pregnancy'] == 98) | 
        (df['diabetes'] == 98) | 
        (df['copd'] == 98) | 
        (df['asthma'] == 98) | 
        (df['inmsupr'] == 98) | 
        (df['hypertension'] == 98) | 
        (df['other_disease'] == 98) | 
        (df['cardiovascular'] == 98) | 
        (df['obesity'] == 98) | 
        (df['renal_chronic'] == 98) | 
        (df['tobacco'] == 98) | 
        (df['icu'] == 99)
    )
    
    # Count the number of rows where the conditions for missing values are met
    missing_count = df[missing_conditions].shape[0]
    
    # Print the result for the dataset
    print(f"Total number of tuples with missing values in {dataset_name}: {missing_count}")

# Function to count rows with missing values based on custom criteria for Covid2
def count_missing_values_covid2(df, dataset_name):
    # Define the conditions for missing values for the specific columns in Covid2
    missing_conditions = (
        (df['INTUBED'] == 99) | 
        (df['PNEUMONIA'] == 99) | 
        (df['PREGNANT'] == 98) | 
        (df['DIABETES'] == 98) | 
        (df['COPD'] == 98) | 
        (df['ASTHMA'] == 98) | 
        (df['INMSUPR'] == 98) | 
        (df['HIPERTENSION'] == 98) | 
        (df['OTHER_DISEASE'] == 98) | 
        (df['CARDIOVASCULAR'] == 98) | 
        (df['OBESITY'] == 98) | 
        (df['RENAL_CHRONIC'] == 98) | 
        (df['TOBACCO'] == 98) | 
        (df['ICU'] == 99)
    )
    
    # Count the number of rows where the conditions for missing values are met
    missing_count = df[missing_conditions].shape[0]
    
    # Print the result for the dataset
    print(f"Total number of tuples with missing values in {dataset_name}: {missing_count}")
    
# Call the function for both Covid1 and Covid2 datasets

# For Covid1 dataset
count_missing_values(covid1_df, "Covid1 Dataset")

# For Covid2 dataset
count_missing_values_covid2(covid2_df, "Covid2 Dataset")


#######################################################
#//
## DATASET INTEGRATION


#//
## RELOAD DATASETS
# Filepaths for the file locations
covid1_path = 'C:/Users/soari/Documents/Assignments/Data Analytics/UMGC/Fall 2024 Data 670/Data/Covid/Covid1 COVID-19 patient pre-condition dataset/Covid1.csv' 
covid2_path = 'C:/Users/soari/Documents/Assignments/Data Analytics/UMGC/Fall 2024 Data 670/Data/Covid/Covid2 COVID-19 Dataset/Covid2.csv'
metadata_path = 'C:/Users/soari/Documents/Assignments/Data Analytics/UMGC/Fall 2024 Data 670/Assignment 2/Metadata.xlsx'

# Load the datasets
covid1_df = pd.read_csv(covid1_path)
covid2_df = pd.read_csv(covid2_path)

#//
## ALIGN COLUMN NAMES

# Create dictionaries for renaming columns in Covid1 and Covid2
rename_dict_covid1 = {
    'id': 'ID',
    'sex': 'SEX',
    'patient_type': 'OUTPATIENT',
    'entry_date': 'ENTRY_DATE',
    'date_symptoms': 'DATE_SYMPTOMS',
    'date_died': 'DATE_DIED',
    'intubed': 'INTUBED',
    'pneumonia': 'PNEUMONIA',
    'age': 'AGE',
    'pregnancy': 'PREGNANT',
    'diabetes': 'DIABETES',
    'copd': 'COPD',
    'asthma': 'ASTHMA',
    'inmsupr': 'INMSUPR',
    'hypertension': 'HYPERTENSION',
    'other_disease': 'OTHER_DISEASE',
    'cardiovascular': 'CARDIOVASCULAR',
    'obesity': 'OBESITY',
    'renal_chronic': 'RENAL_CHRONIC',
    'tobacco': 'TOBACCO',
    'contact_other_covid': 'CONTACT_OTHER_COVID',
    'covid_res': 'CLASSIFICATION_FINAL',
    'icu': 'ICU'
}

rename_dict_covid2 = {
    'USMER': 'USMER',
    'MEDICAL_UNIT': 'MEDICAL_UNIT',
    'SEX': 'SEX',
    'PATIENT_TYPE': 'OUTPATIENT',
    'DATE_DIED': 'DATE_DIED',
    'INTUBED': 'INTUBED',
    'PNEUMONIA': 'PNEUMONIA',
    'AGE': 'AGE',
    'PREGNANT': 'PREGNANT',
    'DIABETES': 'DIABETES',
    'COPD': 'COPD',
    'ASTHMA': 'ASTHMA',
    'INMSUPR': 'INMSUPR',
    'HIPERTENSION': 'HYPERTENSION',
    'OTHER_DISEASE': 'OTHER_DISEASE',
    'CARDIOVASCULAR': 'CARDIOVASCULAR',
    'OBESITY': 'OBESITY',
    'RENAL_CHRONIC': 'RENAL_CHRONIC',
    'TOBACCO': 'TOBACCO',
    'CLASIFFICATION_FINAL': 'CLASSIFICATION_FINAL',
    'ICU': 'ICU'
}

# Rename columns in both datasets
covid1_df_renamed = covid1_df.rename(columns=rename_dict_covid1)
covid2_df_renamed = covid2_df.rename(columns=rename_dict_covid2)

# Update the CLASIFFICATION_FINAL column
covid2_df_renamed['CLASSIFICATION_FINAL'] = covid2_df_renamed['CLASSIFICATION_FINAL'].apply(
    lambda x: 1 if x in [2, 3] else 2 if x >= 4 else x
)

# Display the updated dataframe
print(covid2_df_renamed['CLASSIFICATION_FINAL'].value_counts())

# Concatenate the DataFrames and fill missing columns with NaN
combined_df = pd.concat([covid1_df_renamed, covid2_df_renamed], ignore_index=True)

# Print a preview of the combined DataFrame
print("Combined DataFrame preview:")
print(combined_df.head())



#######################################################
#//
## CHECK DATA ENGINEERING QUALITY

# Step 1: Print the number of rows in individual DataFrames
print(f"Total number of rows in Covid1 DataFrame: {covid1_df_renamed.shape[0]}")
print(f"Total number of rows in Covid2 DataFrame: {covid2_df_renamed.shape[0]}")

# Step 2: Print the number of rows and columns in the combined DataFrame
print(f"\nTotal number of rows in the combined DataFrame: {combined_df.shape[0]}")
print(f"Total number of columns in the combined DataFrame: {combined_df.shape[1]}")

# Step 3: Validate that the total number of rows matches the expected sum
expected_total_rows = covid1_df_renamed.shape[0] + covid2_df_renamed.shape[0]
if combined_df.shape[0] == expected_total_rows:
    print("\nRow count verification passed: The combined DataFrame has the expected number of rows.")
else:
    print("\nRow count verification failed: The combined DataFrame does not have the expected number of rows.")
    print(f"Expected: {expected_total_rows}, Actual: {combined_df.shape[0]}")

# Step 4: Verify column consistency in the combined DataFrame
print("\nColumn names in the combined DataFrame:")
print(combined_df.columns)

# Step 5: Check data types consistency across the combined DataFrame
print("\nData types in the combined DataFrame:")
print(combined_df.dtypes)



#######################################################
#//
## CHECK FOR DUPLICATE ROWS

# Check for potential duplicate rows in the combined DataFrame
duplicate_rows = combined_df[combined_df.duplicated(keep='first')]

# Print the number of duplicate rows detected
num_duplicates = duplicate_rows.shape[0]
if num_duplicates > 0:
    print(f"Number of potential duplicate rows detected: {num_duplicates}")
    # Display a preview of the duplicate rows
    print("\nPreview of duplicate rows:")
    print(duplicate_rows.head())
else:
    print("No potential duplicate rows detected in the combined DataFrame.")


#//
# DROP VARIABLES NOT NEEDED
# Code to drop specified columns from the combined DataFrame
columns_to_drop = ['ID', 'ENTRY_DATE', 'USMER', 'MEDICAL_UNIT', 'DATE_SYMPTOMS', 'CONTACT_OTHER_COVID']

# Drop the specified columns from the combined DataFrame
combined_df_cleaned = combined_df.drop(columns=columns_to_drop)

# Print confirmation and show the updated DataFrame preview
print("Specified columns dropped successfully.")
print("Updated DataFrame preview:")
print(combined_df_cleaned.head())

# Print the shape of the updated DataFrame to verify the change
print(f"Total number of columns after dropping: {combined_df_cleaned.shape[1]}")




# Check for potential duplicate rows in the combined DataFrame
duplicate_rows2 = combined_df_cleaned[combined_df_cleaned.duplicated(keep='first')]

# Print the number of duplicate rows detected
num_duplicates = duplicate_rows2.shape[0]
if num_duplicates > 0:
    print(f"Number of potential duplicate rows detected: {num_duplicates}")
    print("\nPreview of duplicate rows:")
    print(duplicate_rows2.head())
else:
    print("No potential duplicate rows detected in the combined DataFrame.")

# Check the total number of rows in combined df
print(f"\nTotal number of rows in the combined DataFrame: {combined_df_cleaned.shape[0]}")

# Check the total number of unique rows in combined df
unique_rows_count = combined_df_cleaned.drop_duplicates().shape[0]
print(f"Number of unique rows in the combined DataFrame: {unique_rows_count}")

# It's apparent there are likely duplicates. The only viable solutions are moving forward with:
# 1. df with all entries
# 2. df with duplicate entries removed
# Make both to see. df with dupes removed is only ~200k entries. Covid2 has ~800k dupes. Covid1 ~90k likely due to date fields
# Move forward using option 1, df with all entries.

# Rename first df with all entries
combined_df_withdupes = combined_df_cleaned

"""
# Code kept for posterity in case it's needed - but duplicates are kept for analysis
# Remove all duplicate rows from the DataFrame
combined_df_deduped = combined_df_cleaned.drop_duplicates()

# Print confirmation and show the updated DataFrame preview
print("Duplicates removed successfully.")
print("Updated DataFrame preview:")
print(combined_df_deduped.head())

# Print the number of rows in the deduplicated DataFrame
print(f"\nTotal number of rows after removing duplicates: {combined_df_deduped.shape[0]}")


#//
## CHECK OG DATASETS COVID1 AND COVID2 FOR DUPES

# Drop the ID column from each DataFrame before checking for duplicates
covid1_df_no_id = covid1_df_renamed.drop(columns=['ID'], errors='ignore')

# Function to check the number of duplicates using the 'first' method
def check_duplicates(df, dataset_name):
    duplicate_rows_flagged = df[df.duplicated(keep='first')]
    num_duplicates = duplicate_rows_flagged.shape[0]
    print(f"Number of duplicate rows detected in {dataset_name} (keeping first instance): {num_duplicates}")
    # Display a preview of duplicate rows if any
    if num_duplicates > 0:
        print(f"\nPreview of duplicate rows in {dataset_name}:")
        print(duplicate_rows_flagged.head())
    else:
        print(f"No duplicates detected in {dataset_name}.")

# Check for duplicates in Covid1 DataFrame
check_duplicates(covid1_df_no_id, "Covid1 Dataset")

# Check for duplicates in Covid2 DataFrame
check_duplicates(covid2_df_renamed, "Covid2 Dataset")

"""

#######################################################
#//
## DELETE TUPLES WITH MISSING VALUES

# Remove rows with specific codes for missing values in the combined DataFrame
combined_df_cleaned_no_missing = combined_df_withdupes[
    ~(
        (combined_df_withdupes['INTUBED'] == 99) |
        (combined_df_withdupes['PNEUMONIA'] == 99) |
        (combined_df_withdupes['PREGNANT'] == 98) |
        (combined_df_withdupes['DIABETES'] == 98) |
        (combined_df_withdupes['COPD'] == 98) |
        (combined_df_withdupes['ASTHMA'] == 98) |
        (combined_df_withdupes['INMSUPR'] == 98) |
        (combined_df_withdupes['HYPERTENSION'] == 98) |
        (combined_df_withdupes['OTHER_DISEASE'] == 98) |
        (combined_df_withdupes['CARDIOVASCULAR'] == 98) |
        (combined_df_withdupes['OBESITY'] == 98) |
        (combined_df_withdupes['RENAL_CHRONIC'] == 98) |
        (combined_df_withdupes['TOBACCO'] == 98) |
        (combined_df_withdupes['ICU'] == 99)
    )
]

# Print confirmation and show the number of rows after removal
print("Rows with specified missing values removed successfully.")
print(f"Total number of rows before removal: {combined_df_withdupes.shape[0]}")
print(f"Total number of rows after removal: {combined_df_cleaned_no_missing.shape[0]}")
print("Preview of cleaned DataFrame:")
print(combined_df_cleaned_no_missing.head())


#######################################################
#//
## REPLACE 2 VALUES WITH 0 WHEN INDICATING NO/NONE/MALE/ETC


# Replace 2 with 0 in all categorical variables except specified ones
columns_to_exclude = ['DATE_DIED', 'AGE', 'CLASSIFICATION_FINAL']

# Identify columns to modify
columns_to_replace = [col for col in combined_df_cleaned_no_missing.columns if col not in columns_to_exclude]

# Replace 2 with 0 in the selected columns
combined_df_cleaned_no_missing[columns_to_replace] = combined_df_cleaned_no_missing[columns_to_replace].replace(2, 0)

# Print confirmation and show a preview of the modified DataFrame
print("Replaced 2 with 0 in specified categorical variables.")
print("Preview of the updated DataFrame:")
print(combined_df_cleaned_no_missing.head())

# Use this to check column quality after finding value error
#print(combined_df_cleaned_no_missing['CLASSIFICATION_FINAL'].value_counts())



#######################################################
#//
## STRIP ALL TUPLES CONTAINING AGE >100

# Identify tuples with an AGE value greater than 100
age_greater_than_100 = combined_df_cleaned_no_missing[combined_df_cleaned_no_missing['AGE'] > 100]

# Count the number of tuples to be removed
num_tuples_removed = age_greater_than_100.shape[0]

# Remove tuples with AGE > 100
combined_df_final = combined_df_cleaned_no_missing[combined_df_cleaned_no_missing['AGE'] <= 100]

# Print the number of tuples removed and show a preview of the cleaned DataFrame
print(f"Number of tuples with AGE > 100 removed: {num_tuples_removed}")
print("Preview of the updated DataFrame:")
print(combined_df_final.head())

#######################################################
#//
## UPDATING MORTALITY TO CATEGORICAL VARIABLE

# Create the MORTALITY variable: 1 if DATE_DIED is a valid date, 0 if it's '9999-99-99'
combined_df_final['MORTALITY'] = combined_df_final['DATE_DIED'].apply(lambda x: 1 if pd.notnull(x) and x != '9999-99-99' else 0)

# Drop the original DATE_DIED variable
combined_df_final = combined_df_final.drop(columns=['DATE_DIED'])

# Print confirmation and preview the updated DataFrame
print("MORTALITY variable created and DATE_DIED column dropped.")
print("Preview of the updated DataFrame:")
print(combined_df_final.head())
print(combined_df_final['MORTALITY'].head(10))



#######################################################
#//
## FEATURE ENGINEERING

# Replace the AGE column with AGE_GROUP
def create_age_group(age):
    if age <= 20:
        return "0-20"
    elif age <= 40:
        return "21-40"
    elif age <= 60:
        return "41-60"
    elif age <= 80:
        return "61-80"
    elif age <= 100:
        return "81-100"
    else:
        return "Unknown"

# Apply the age group function and replace the AGE column
combined_df_final['AGE_GROUP'] = combined_df_final['AGE'].apply(create_age_group)
combined_df_final = combined_df_final.drop(columns=['AGE'])

# Create the COMORBIDITY_COUNT variable
comorbidity_columns = ['DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HYPERTENSION', 'PNEUMONIA',
                       'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO']

# Sum the comorbidity indicators to create the COMORBIDITY_COUNT variable
combined_df_final['COMORBIDITY_COUNT'] = combined_df_final[comorbidity_columns].sum(axis=1)

# Print a preview of the updated DataFrame
print("Preview of the DataFrame with AGE_GROUP and COMORBIDITY_COUNT added:")
print(combined_df_final.head())
print(f"Range of values in COMORBIDITY_COUNT: {combined_df_final['COMORBIDITY_COUNT'].min()} to {combined_df_final['COMORBIDITY_COUNT'].max()}")



"""
# Quick code to spot check the df using a CSV file
# Path
output_file_path = 'C:/Users/soari/Documents/Assignments/Data Analytics/UMGC/Fall 2024 Data 670/Assignment 3/combined_df_cleaned.csv'

# Export the cleaned DataFrame to a CSV file
combined_df_final.to_csv(output_file_path, index=False)

# Print confirmation message
print(f"DataFrame successfully exported to {output_file_path}")
"""



#//
# EXPLORATORY DATA ANALYSIS POST-CLEANING


# Basic descriptive statistics for the merged df
print("Basic Descriptive Statistics:")
print(combined_df_final.describe())

# Visualize missing data as a heatmap (optional, if there are missing values)
plt.figure(figsize=(10, 6))
sns.heatmap(combined_df_final.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()



# VISUALIZATION 1: CORRELATION MATRIX
# NOTE: DO NOT USE AGE AS CATEGORICAL VARIABLE FOR THIS ONE. KEEP AS CONTINUOUS
# Correlation matrix for variables
plt.figure(figsize=(10, 8))
corr_matrix = combined_df_final.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of COVID-19 Patient Data', fontsize=16)
plt.xlabel('Features')
plt.ylabel('Features')
plt.show()


# Correlation matrix clustered
# Cluster the correlation matrix
cg = sns.clustermap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f', figsize=(10, 8))
plt.title('Clustered Correlation Matrix', fontsize=16)


# VISUALIZATION 2: BAR CHART EDA MATRIX

# Ensure seaborn has the correct method, otherwise use plt.hist for fallback.
def plot_column(data, column, ax):
    if data[column].dtype == 'object' or data[column].nunique() < 20:
        sns.countplot(data=data, x=column, ax=ax, palette='pastel')
    else:
        ax.hist(data[column].dropna(), bins=30, color='skyblue', edgecolor='black')
    ax.set_title(f'{column} Distribution')
    ax.set_xlabel(column)
    ax.set_ylabel('Count')

# List of columns to plot
columns_to_plot = combined_df_final.columns

# Set up grid for subplots
num_columns = 4
num_rows = (len(columns_to_plot) + num_columns - 1) // num_columns
fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, num_rows * 5))
fig.suptitle('EDA for Combined Dataset', fontsize=16)

for i, column in enumerate(columns_to_plot):
    ax = axes[i // num_columns, i % num_columns]
    plot_column(combined_df_final, column, ax)

# Hide unused subplots
for j in range(i + 1, num_rows * num_columns):
    fig.delaxes(axes[j // num_columns, j % num_columns])

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


# VISUALIZATION 3: VARIABLE WORTH
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

# Define the target variables and features
target_vars = ['ICU', 'MORTALITY']
features = combined_df_final.drop(target_vars, axis=1)

# Ensure all features are numeric or properly encoded
features_encoded = features.apply(LabelEncoder().fit_transform)

# Chi-Square test for each feature and ICU
chi_scores_icu, p_values_icu = chi2(features_encoded, combined_df_final['ICU'])

# Chi-Square test for each feature and MORTALITY
chi_scores_mortality, p_values_mortality = chi2(features_encoded, combined_df_final['MORTALITY'])

# Create dataframes to store results
icu_chi_results = pd.DataFrame({
    'Feature': features.columns,
    'Chi-Square Score': chi_scores_icu,
    'p-Value': p_values_icu
}).sort_values(by='Chi-Square Score', ascending=False)

mortality_chi_results = pd.DataFrame({
    'Feature': features.columns,
    'Chi-Square Score': chi_scores_mortality,
    'p-Value': p_values_mortality
}).sort_values(by='Chi-Square Score', ascending=False)

# Function to plot with p-values
def plot_with_pvalues(df, target_name, color):
    plt.figure(figsize=(12, 6))
    bars = plt.barh(df['Feature'].head(20), df['Chi-Square Score'].head(20), color=color)
    plt.gca().invert_yaxis()
    plt.title(f'Top Chi-Square Scores for {target_name}')
    plt.xlabel('Chi-Square Score')
    plt.ylabel('Feature')
    
    # Annotate p-values on bars
    for bar, p_val in zip(bars, df['p-Value'].head(20)):
        plt.text(
            bar.get_width() + 0.5,  # Place text slightly to the right of the bar
            bar.get_y() + bar.get_height() / 2,  # Center text on bar
            f'p={p_val:.3g}',  # Format p-value to 3 significant digits
            va='center',
            fontsize=10,
            color='black' if p_val < 0.05 else 'gray'  # Highlight significant p-values
        )
    plt.show()

# Plot for ICU
plot_with_pvalues(icu_chi_results, 'ICU', 'skyblue')

# Plot for Mortality
plot_with_pvalues(mortality_chi_results, 'Mortality', 'coral')



#######################################################
#//
# Recode the AGE_GROUP variable to make it ammenable to modeling
def recode_age_group(age_group):
    if age_group == '0-20':
        return 1
    elif age_group == '21-40':
        return 2
    elif age_group == '41-60':
        return 3
    elif age_group == '61-80':
        return 4
    elif age_group == '81-100':
        return 5
    else:
        return None  # Handle unexpected values, if any

# Apply the function to recode AGE_GROUP
if 'AGE_GROUP' in combined_df_final.columns:
    combined_df_final['AGE_GROUP'] = combined_df_final['AGE_GROUP'].apply(recode_age_group)

# Verify the transformation
print(combined_df_final['AGE_GROUP'].value_counts())


#######################################################
#// Transformations for Risk Matrix

# Make file path for csv for truncated df before modeling
output_file_path1 = 'C:/Users/soari/Documents/Assignments/Data Analytics/UMGC/Fall 2024 Data 670/Assignment 6/df_final.csv'

# Make file path for csv for full df before modeling
output_file_path2 = 'C:/Users/soari/Documents/Assignments/Data Analytics/UMGC/Fall 2024 Data 670/Assignment 6/df_final_full.csv'

# Export the cleaned DataFrame to a CSV file
df_first_10_rows = combined_df_final.head(10)
df_first_10_rows.to_csv(output_file_path1, index=False)

# Step 1: Copy df as back up
combined_df_RM = combined_df_final
combined_df_RM = combined_df_RM.replace(97, 0)

# Step 2: Compute the weighted sum (Risk Score)
combined_df_RM['Risk_Score'] = (
    combined_df_final['ASTHMA'] * 0.046171279 +
    combined_df_final['CARDIOVASCULAR'] * 0.001509685 +
    combined_df_final['CLASSIFICATION_FINAL'] * 0.002129031 +
    combined_df_final['COPD'] * 0.0021714 +
    combined_df_final['DIABETES'] * 0.002478661 +
    combined_df_final['HYPERTENSION'] * 0.013663514 +
    combined_df_final['ICU'] * 0.002057093 +
    combined_df_final['INMSUPR'] * 0.01669408 +
    combined_df_final['INTUBED'] * 0.804682841 +
    combined_df_final['OBESITY'] * 0.003663514 +
    combined_df_final['OTHER_DISEASE'] * 0.002471875 +
    combined_df_final['OUTPATIENT'] * 0.002574518 +
    combined_df_final['PNEUMONIA'] * 0.003914547 +
    combined_df_final['PREGNANT'] * 0.002641832 +
    combined_df_final['RENAL_CHRONIC'] * 0.074350044 +
    combined_df_final['SEX'] * 0.00313825 +
    combined_df_final['TOBACCO'] * 0.001746782 +
    combined_df_final['AGE_GROUP'] * 0.001960395 +
    combined_df_final['COMORBIDITY_COUNT'] * 0.015644174
)

# Step 3: Classify risk level based on the risk score
def classify_risk(score):
    if score > 0.7:
        return "High"
    elif score > 0.4:
        return "Medium"
    else:
        return "Low"

combined_df_RM['Risk_Label'] = combined_df_RM['Risk_Score'].apply(classify_risk)

combined_df_RM.to_csv(output_file_path1, index=False)


# Count rows where Risk_Label is "High" or "Medium" and mortality is 1
high_medium_and_mortality = combined_df_RM[
    ((combined_df_RM['Risk_Label'] == "High") | (combined_df_RM['Risk_Label'] == "Medium")) &
    (combined_df_RM['MORTALITY'] == 1)
].shape[0]

# Total number of rows in the dataframe
total_rows = combined_df_RM.shape[0]

# Calculate the percentage
if total_rows > 0:
    percentage = (high_medium_and_mortality / total_rows) * 100
    print(f"The percentage of rows with mortality=1 and Risk_Label='High' or 'Medium' is: {percentage:.2f}%")
else:
    print("The dataframe has no rows.")


#######################################################
#//
## DATA MODELING
# MORTALITY FIRST THEN ICU



#######################################################################
#######################################################################
#######################################################################
#######################################################################
# MORTALITY MODELING CODE TO RUN EACH INDIVIDUALLY
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.class_weight import compute_class_weight


# Define evaluation metrics
def evaluate_model(y_true, y_pred):
    """Evaluate model on specified metrics."""
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)  # Sensitivity is recall for the positive class
    return {"confusion_matrix": cm, "accuracy": accuracy, "f1": f1, "sensitivity": sensitivity}

# Common function to run k-fold cross-validation
def run_kfold_cv(model, X, y, k=5):
    """Perform K-fold cross-validation for a given model and dataset."""
    metrics = []
    kf = KFold(n_splits=k, shuffle=True, random_state=69)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        metrics.append(evaluate_model(y_test, y_pred))
    
    # Aggregate metrics
    final_metrics = {
        "confusion_matrix": sum(metric["confusion_matrix"] for metric in metrics),
        "accuracy": np.mean([metric["accuracy"] for metric in metrics]),
        "f1": np.mean([metric["f1"] for metric in metrics]),
        "sensitivity": np.mean([metric["sensitivity"] for metric in metrics]),
        "Target Variable": "Mortality"
    }
    return final_metrics

# Individual model functions
def decision_tree_model(X, y):
    model = DecisionTreeClassifier(class_weight="balanced", random_state=69)
    return run_kfold_cv(model, X, y)

def logistic_regression_model(X, y):
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=69)
    return run_kfold_cv(model, X, y)

def random_forest_model(X, y):
    model = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=69)
    return run_kfold_cv(model, X, y)

def gradient_boosting_model(X, y):
    model = GradientBoostingClassifier(random_state=69)
    return run_kfold_cv(model, X, y)

def naive_bayes_model(X, y):
    model = GaussianNB()  
    return run_kfold_cv(model, X, y)

# NB model adjusted with class weights
def naive_bayes_modelW(X, y):
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    priors = class_weights / class_weights.sum()  
    model = GaussianNB(priors=priors)
    return run_kfold_cv(model, X, y)

# Results dictionary to store results for all models
model_results = {}

# Workflow to run models
def run_decision_tree(X, y):
    model_results["Decision Tree"] = decision_tree_model(X, y)

def run_logistic_regression(X, y):
    model_results["Logistic Regression"] = logistic_regression_model(X, y)

def run_random_forest(X, y):
    model_results["Random Forest"] = random_forest_model(X, y)

def run_gradient_boosting(X, y):
    model_results["Gradient Boosting"] = gradient_boosting_model(X, y)

def run_naive_bayes(X, y):
    model_results["Naive Bayes"] = naive_bayes_model(X, y)

def run_naive_bayesW(X, y):
    model_results["Naive Bayes Weighted"] = naive_bayes_modelW(X, y)

# Function to compile results into a DataFrame
def compile_results():
    compiled_df = pd.DataFrame(model_results).T
    return compiled_df

# Example usage
X = combined_df_final.drop(columns=["MORTALITY"])
y = combined_df_final["MORTALITY"]

# Run individual models
run_decision_tree(X, y) #30 sec
run_logistic_regression(X, y) # 3min
run_random_forest(X, y) # 7 min
run_gradient_boosting(X, y) # 10 min
run_naive_bayes(X, y) # 3 sec
run_naive_bayesW(X, y) # 3 sec

# Compile results into a table
results_table = compile_results()
print(results_table)

# Code to download the df using a CSV file
# Path
output_file_path = 'C:/Users/soari/Documents/Assignments/Data Analytics/UMGC/Fall 2024 Data 670/Assignment 5/Mort_results.csv'

# Export the cleaned DataFrame to a CSV file
results_table.to_csv(output_file_path, index=False)

# Print confirmation message
print(f"DataFrame successfully exported to {output_file_path}")



#######################################################
#######################################################
# ICU MODELING CODE TO RUN EACH INDIVIDUALLY


# Common function to run k-fold cross-validation
def run_kfold_cv(model, X, y, k=5):
    """Perform K-fold cross-validation for a given model and dataset."""
    metrics = []
    kf = KFold(n_splits=k, shuffle=True, random_state=69)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        metrics.append(evaluate_model(y_test, y_pred))
    
    # Aggregate metrics
    final_metrics = {
        "confusion_matrix": sum(metric["confusion_matrix"] for metric in metrics),
        "accuracy": np.mean([metric["accuracy"] for metric in metrics]),
        "f1": np.mean([metric["f1"] for metric in metrics]),
        "sensitivity": np.mean([metric["sensitivity"] for metric in metrics]),
        "Target Variable": "ICU"
    }
    return final_metrics




# Update values of 97 to 0 in the 'ICU' column
combined_df_final['ICU'] = combined_df_final['ICU'].replace(97, 0)

# Verify the update
print("Updated values in 'ICU':")
print(combined_df_final['ICU'].value_counts())




# Example usage
X = combined_df_final.drop(columns=["MORTALITY", "ICU"])
y = combined_df_final["ICU"]

# Run individual models
run_decision_tree(X, y) #30 sec
run_logistic_regression(X, y) # 3min
run_random_forest(X, y) # 7 min
run_gradient_boosting(X, y) # 10 min
run_naive_bayes(X, y) # 3 sec
run_naive_bayesW(X, y) # 3 sec

# Compile results into a table
results_table = compile_results()
print(results_table)


# Code to download the df using a CSV file
# Path
output_file_path = 'C:/Users/soari/Documents/Assignments/Data Analytics/UMGC/Fall 2024 Data 670/Assignment 5/ICU_results.csv'

# Export the cleaned df to a CSV file
results_table.to_csv(output_file_path, index=False)

# Print confirmation message
print(f"DataFrame successfully exported to {output_file_path}")



#######################################################
# ATTEMPT TO HYPER OPTIMIZE

# DECISION TREE AD HOC WITH PARAMETER TUNING - WORKED!
#Accuracy: 0.8705, F1 Score: 0.5017, Sensitivity: 0.9473
#Best Parameters: {'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': 10}
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    make_scorer,
)

# Step 1: Load final dataframe and split data
X = combined_df_final.drop(columns=["MORTALITY"])  # Features
y_mortality = combined_df_final["MORTALITY"]       # Target variable

# Step 2: Simplify hyperparameter grid
param_grid = {
    "max_depth": [5, 10],
    "min_samples_split": [5],
    "min_samples_leaf": [2],
}

# Step 3: Define evaluation scorer (F1 score for the positive class)
f1_scorer = make_scorer(f1_score, pos_label=1)

# Step 4: Set up 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=69)

# Step 5: Initialize decision tree with class weights for imbalance
decision_tree = DecisionTreeClassifier(random_state=69, class_weight="balanced")

# Step 6: Perform randomized hyperparameter search
random_search = RandomizedSearchCV(
    decision_tree,
    param_distributions=param_grid,
    scoring=f1_scorer,
    cv=kf,
    n_jobs=1,  # Sequential processing to reduce resource usage
    random_state=69,
    n_iter=5,  # Small number of iterations for speed
)
random_search.fit(X, y_mortality)

# Step 7: Evaluate the best model
best_model = random_search.best_estimator_
y_pred = best_model.predict(X)

# Step 8: Compute metrics
accuracy = accuracy_score(y_mortality, y_pred)
f1 = f1_score(y_mortality, y_pred, pos_label=1)
sensitivity = recall_score(y_mortality, y_pred, pos_label=1)
conf_matrix = confusion_matrix(y_mortality, y_pred)

# Step 9: Prepare metrics for output
metrics_df = pd.DataFrame(
    {
        "Metric": ["Accuracy", "F1 Score", "Sensitivity", "Confusion Matrix"],
        "Value": [accuracy, f1, sensitivity, conf_matrix.tolist()],  # Confusion matrix as a list
    }
)


# Path
output_file_path_final = 'C:/Users/soari/Documents/Assignments/Data Analytics/UMGC/Fall 2024 Data 670/Assignment 5/DTmodel_metrics.csv'

# Step 10: Save metrics to CSV
metrics_df.to_csv(output_file_path_final, index=False)

# Print results for user
print(f"Metrics saved to 'DTmodel_metrics.csv'")
print(metrics_df)

#######################################################
#######################################################
#######################################################

#DT_M_F1
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    make_scorer,
)
from sklearn.utils import resample

# Step 1: Load and preprocess the data
X = combined_df_final.drop(columns=["MORTALITY"])
y_mortality = combined_df_final["MORTALITY"]       

# Step 2: Manual Oversampling to handle class imbalance
data = pd.concat([X, y_mortality], axis=1)
majority = data[data["MORTALITY"] == 0]
minority = data[data["MORTALITY"] == 1]
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
upsampled_data = pd.concat([majority, minority_upsampled])

X_resampled = upsampled_data.drop("MORTALITY", axis=1)
y_resampled = upsampled_data["MORTALITY"]

# Step 3: Expand the hyperparameter grid
param_grid = {
    "max_depth": [5, 10, 15, None],           # Control tree depth
    "min_samples_split": [2, 5, 10],          # Minimum samples to split a node
    "min_samples_leaf": [1, 2, 5],            # Minimum samples required at a leaf
    "criterion": ["gini", "entropy"],         # Splitting criteria
    "ccp_alpha": [0.0, 0.01, 0.1],            # Cost Complexity Pruning parameter
}

# Step 4: Define F1 scorer for optimization
f1_scorer = make_scorer(f1_score, pos_label=1)

# Step 5: Set up 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=69)

# Step 6: Initialize Decision Tree with class weights for imbalance
decision_tree = DecisionTreeClassifier(random_state=69, class_weight="balanced")

# Step 7: Perform randomized hyperparameter search
random_search = RandomizedSearchCV(
    decision_tree,
    param_distributions=param_grid,
    scoring=f1_scorer,  # Optimize for F1 score
    cv=kf,
    n_jobs=1,               # Sequential execution to avoid memory errors
    random_state=69,
    n_iter=20,              # Explore more combinations
    verbose=2,              # Log progress for debugging
)
random_search.fit(X_resampled, y_resampled)

# Step 8: Evaluate the best model
best_model = random_search.best_estimator_
y_pred = best_model.predict(X)

accuracy = accuracy_score(y_mortality, y_pred)
f1 = f1_score(y_mortality, y_pred, pos_label=1)
sensitivity = recall_score(y_mortality, y_pred, pos_label=1)
conf_matrix = confusion_matrix(y_mortality, y_pred)

# Step 9: Prepare metrics for output
metrics_df = pd.DataFrame(
    {
        "Metric": ["Accuracy", "F1 Score", "Sensitivity", "Confusion Matrix"],
        "Value": [accuracy, f1, sensitivity, conf_matrix.tolist()],  # Confusion matrix as a list
    }
)

# Step 10: Save metrics to CSV
output_file_path_final = 'C:/Users/soari/Documents/Assignments/Data Analytics/UMGC/Fall 2024 Data 670/Assignment 5/DTmodel_metrics_optimized_f1.csv'
metrics_df.to_csv(output_file_path_final, index=False)

# Print results for user
print(f"Metrics saved to 'DTmodel_metrics_optimized_f1.csv'")
print(metrics_df)


#########################################################
#// MODEL INTERPRETIBILITY

# 1. Feature Importance
# Extract and save feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Save feature importance to CSV
feature_importance_file = 'C:/Users/soari/Documents/Assignments/Data Analytics/UMGC/Fall 2024 Data 670/Assignment 6/DT_M_F1_FI.csv'
feature_importance.to_csv(feature_importance_file, index=False)

# Print feature importance for user
print("Feature Importance:")
print(feature_importance)


#######################################
# 2. Show Tree Rules
from sklearn.tree import export_text

tree_rules = export_text(best_model, feature_names=list(X.columns))
rules_file = 'C:/Users/soari/Documents/Assignments/Data Analytics/UMGC/Fall 2024 Data 670/Assignment 6/DT_M_F1_Tree_Rules.txt'

# Save rules to a file
with open(rules_file, 'w') as file:
    file.write(tree_rules)

# Print a sample of the rules for user
print("Decision Tree Rules:")
print(tree_rules[:1000])  # Display the first 1000 characters of the rules


#######################################
# 3. Visualize Decision Tree
#Show all rules
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(
    best_model,
    feature_names=list(X.columns),
    class_names=["Not Mortality", "Mortality"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Visualization")
plt.show()


# Adjust the max_depth parameter to show only the top levels of the tree
plt.figure(figsize=(15, 7))
plot_tree(
    best_model,
    feature_names=list(X.columns),
    class_names=["Not Mortality", "Mortality"],
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=2  # Show only the first two levels of the tree
)
plt.title("Simplified Decision Tree Visualization")
plt.show()


#######################################
# 4. Partial Dependence Plots
from sklearn.inspection import PartialDependenceDisplay

top_features = feature_importance.sort_values('Importance', ascending=False)['Feature'].head(3)
PartialDependenceDisplay.from_estimator(best_model, X, features=top_features.tolist())
plt.gcf().set_size_inches(12, 6)
plt.show()

