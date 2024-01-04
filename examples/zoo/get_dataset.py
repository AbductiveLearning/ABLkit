import numpy as np
import openml


# Function to load and preprocess the dataset
def load_and_preprocess_dataset(dataset_id):
    dataset = openml.datasets.get_dataset(
        dataset_id, download_data=True, download_qualities=False, download_features_meta_data=False
    )
    X, y, _, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
    # Convert data types
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)
    y = y.cat.codes.astype(int)
    X, y = X.to_numpy(), y.to_numpy()
    return X, y


# Function to split data (one shot)
def split_dataset(X, y, test_size=0.3):
    # For every class: 1 : (1-test_size)*(len-1) : test_size*(len-1)
    label_indices, unlabel_indices, test_indices = [], [], []
    for class_label in np.unique(y):
        idxs = np.where(y == class_label)[0]
        np.random.shuffle(idxs)
        n_train_unlabel = int((1 - test_size) * (len(idxs) - 1))
        label_indices.append(idxs[0])
        unlabel_indices.extend(idxs[1 : 1 + n_train_unlabel])
        test_indices.extend(idxs[1 + n_train_unlabel :])
    X_label, y_label = X[label_indices], y[label_indices]
    X_unlabel, y_unlabel = X[unlabel_indices], y[unlabel_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    return X_label, y_label, X_unlabel, y_unlabel, X_test, y_test
