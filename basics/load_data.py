import pandas as pd
import logging
from sklearn import datasets as sklearn_datasets
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets


def load_data_from_local(
    path, split_files=False, test_size=0.2, val_size=0.25, random_state=42
):
    """
    Load tabular data as train, validation, and test sets.

    Parameters:
        path (str): Path to the CSV file or directory containing train, valid, test CSV files.
        split_files (bool): If False, assumes path is a single CSV file to be split.
                            If True, assumes path is a dictionary with keys 'train', 'validation', 'test'.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the training dataset to include in the validation split.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    if split_files:
        # Load from separate files
        data_files = {
            "train": f"{path}/train.csv",
            "validation": f"{path}/valid.csv",
            "test": f"{path}/test.csv",
        }
        dataset = load_dataset("csv", data_files=data_files)
    else:
        # Load a single file and split
        dataset = load_dataset("csv", data_files=path)
        train_val, test = train_test_split(
            dataset["train"], test_size=test_size, random_state=random_state
        )
        train, val = train_test_split(
            train_val, test_size=val_size, random_state=random_state
        )
        dataset = DatasetDict({"train": train, "validation": val, "test": test})

    # Assuming 'features' and 'labels' are column names; adjust as needed
    X_train = dataset["train"]["features"]
    y_train = dataset["train"]["labels"]
    X_val = dataset["validation"]["features"]
    y_val = dataset["validation"]["labels"]
    X_test = dataset["test"]["features"]
    y_test = dataset["test"]["labels"]

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_data_from_sklearn(dataset_name, test_size=0.2, val_size=0.20, random_state=42):
    try:
        load_func = getattr(sklearn_datasets, "load_" + dataset_name, None)
        if not load_func:
            fetch_func = getattr(sklearn_datasets, "fetch_" + dataset_name, None)
            if fetch_func:
                data = fetch_func()
            else:
                logging.error(f"No such dataset: {dataset_name}")
                return None, None, None, None, None, None
        else:
            data = load_func()

        # Extract data and target
        X, y = data.data, data.target

        # Splitting data into train+validation and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Splitting train+validation into train and validation sets
        val_test_size = val_size / (
            1 - test_size
        )  # Adjust validation size relative to the training set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_test_size, random_state=random_state
        )

        # Create Hugging Face datasets from the splits
        train_dataset = Dataset.from_dict({"features": X_train, "labels": y_train})
        val_dataset = Dataset.from_dict({"features": X_val, "labels": y_val})
        test_dataset = Dataset.from_dict({"features": X_test, "labels": y_test})

        # Combine into a DatasetDict
        dataset_dict = DatasetDict(
            {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
        )

        return dataset_dict

    except Exception as e:
        logging.error(f"An error occurred while loading the dataset: {e}")
        return None


def load_data_from_local(
    path, label_col, split_files=False, test_size=0.2, val_size=0.2, random_state=42
):
    """
    Load tabular data as train, validation, and test sets and return as DatasetDict

    Parameters:
        path (str): Path to the CSV file or directory containing train, valid, test CSV files.
        label_col (str): Name of the column to be used as labels.
        split_files (bool): If False, assumes path is a single CSV file to be split.
                            If True, assumes path contains separate CSV files for 'train', 'validation', 'test'.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the entire dataset to include in the validation split.
        random_state (int): Random state for reproducibility.

    Returns:
        DatasetDict: Dataset dictionary with splits containing features and labels.
    """
    if split_files:
        # Load from separate files
        data_files = {
            "train": f"{path}/train.csv",
            "validation": f"{path}/valid.csv",
            "test": f"{path}/test.csv",
        }
        dataset = load_dataset("csv", data_files=data_files)
    else:
        # Load a single file and split
        dataset = load_dataset("csv", data_files=path)
        total_size = len(dataset["train"])
        test_size_abs = int(total_size * test_size)
        val_size_abs = int(total_size * val_size)

        # Shuffle dataset for randomness
        dataset = dataset["train"].shuffle(seed=random_state)

        # Splitting the dataset into test, validation, and train by slicing
        test_dataset = dataset.select(range(test_size_abs))
        val_dataset = dataset.select(range(test_size_abs, test_size_abs + val_size_abs))
        train_dataset = dataset.select(range(test_size_abs + val_size_abs, total_size))

        # Recreate DatasetDict
        dataset = DatasetDict(
            {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
        )

    # Separate features and labels
    def extract_features_labels(example):
        labels = example[label_col]
        features = {key: value for key, value in example.items() if key != label_col}
        return {"features": features, "labels": labels}

    for split in dataset.keys():
        dataset[split] = dataset[split].map(extract_features_labels)

    return dataset


def load_sklearn_dataset(dataset_name, test_size=0.2, val_size=0.20):
    try:
        load_func = getattr(sklearn_datasets, "load_" + dataset_name, None)
        if load_func:
            data = load_func()
        else:
            fetch_func = getattr(sklearn_datasets, "fetch_" + dataset_name, None)
            if fetch_func:
                data = fetch_func()
            else:
                logging.error(f"No such dataset: {dataset_name}")
                return None, None, None, None, None, None

        # Extract data and target
        X, y = data.data, data.target

        # Splitting data into train+validation and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Splitting train+validation into train and validation sets
        val_test_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_test_size, random_state=42
        )

        return X_train, X_val, X_test, y_train, y_val, y_test
    except Exception as e:
        logging.error(f"An error occurred while loading the dataset: {e}")
        return None, None, None, None, None, None


def load_data_from_huggingface(
    dataset_name, split_names=None, test_size=0.2, val_size=0.20
):
    try:
        # Load the dataset from Hugging Face datasets
        dataset = load_dataset(dataset_name)

        # Check if the dataset has predefined splits
        if split_names:
            train_split, val_split, test_split = split_names
            train_data = dataset[train_split]
            val_data = dataset[val_split] if val_split in dataset else None
            test_data = dataset[test_split] if test_split in dataset else None
        else:
            # Assuming the dataset is only in one piece and needs splitting
            full_data = dataset["train"]
            # Convert to pandas DataFrame for easier handling
            df = full_data.to_pandas()

            # Split data into train+validation and test
            train_val_df, test_df = train_test_split(
                df, test_size=test_size, random_state=42
            )

            # Split train+validation into train and validation
            val_test_size = val_size / (
                1 - test_size
            )  # Adjust val size based on remaining data
            train_df, val_df = train_test_split(
                train_val_df, test_size=val_test_size, random_state=42
            )

            # Extract features and target from DataFrames if necessary
            X_train, y_train = train_df.drop("target", axis=1), train_df["target"]
            X_val, y_val = val_df.drop("target", axis=1), val_df["target"]
            X_test, y_test = test_df.drop("target", axis=1), test_df["target"]

            return X_train, X_val, X_test, y_train, y_val, y_test
        return None, None, None, None, None, None  # In case splits are not defined

    except Exception as e:
        logging.error(f"An error occurred while loading the dataset: {e}")
        return None, None, None, None, None, None
