import pandas as pd


def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.
    """
    data = pd.read_csv(file_path)
    return data


def clean_data(data):
    """
    Clean and preprocess the data.
    - Handle missing values
    - Replace numerical values in the "Diagnosis" column with meaningful labels
    """
    # Handle missing values
    data = data.fillna(0)

    # Replace values in "Diagnosis" with labels
    diagnosis_mapping = {0: "Negative", 1: "Positive"}
    data['Diagnosis'] = data['Diagnosis'].map(diagnosis_mapping)

    return data
