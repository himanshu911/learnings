from sklearn import datasets
from sklearn.model_selection import train_test_split
import logging


def load_sklearn_dataset(dataset_name, test_size=0.2, val_size=0.20):
    try:
        load_func = getattr(datasets, "load_" + dataset_name, None)
        if load_func:
            data = load_func()
        else:
            fetch_func = getattr(datasets, "fetch_" + dataset_name, None)
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

