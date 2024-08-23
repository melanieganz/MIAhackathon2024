import json
import pandas as pd
import yaml


def read_yaml_file(file_dir):
    with open(file_dir, 'r') as f:
        config = yaml.safe_load(f)
    return config


def read_csv_file(file_path):
    """
    Reads a CSV file using Pandas and returns a DataFrame.

    Parameters:
    file_path (str): The path to the CSV file to be read.

    Returns:
    DataFrame: The data from the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"No file found at {file_path}")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")


def save_to_csv(df, file_path, index=False):
    """
    Saves the given DataFrame to a CSV file.

    Parameters:
    df (DataFrame): The DataFrame to be saved.
    file_path (str): The path where the CSV file will be saved.
    index (bool): Whether to include the DataFrame index in the CSV. Defaults to False.

    Returns:
    None
    """
    try:
        df.to_csv(file_path, index=index)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")


def read_json_file(file_path):
    """
    Reads a JSON file and returns the data as a Python dictionary.

    Parameters:
    file_path (str): The path to the JSON file to be read.

    Returns:
    dict: The data contained in the JSON file.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"No file found at {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file {file_path}")


def save_as_json(data, file_path):
    """
    Saves the given data to a file in JSON format.

    Parameters:
    data (dict): The data to be saved to the file.
    file_path (str): The path where the JSON file will be saved.

    Returns:
    None
    """
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")