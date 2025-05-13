import numpy as np
import pandas as pd


def one_hot_encode(X, cols=None):
    """One-hot encodes non-numerical columns in a DataFrame or numpy array.

    Drops the original columns after encoding.

    Args:
        X: (pandas.DataFrame or numpy.ndarray) - The data to be encoded.
        cols: (list), optional - The list of column indices to be encoded (default is None).
            If None, all non-numerical columns will be encoded.

    Returns:
        X: (pandas.DataFrame or numpy.ndarray) - The data with one-hot encoded columns.
    """
    is_dataframe = isinstance(X, pd.DataFrame)
    if not is_dataframe:
        X = pd.DataFrame(X)  # Convert to DataFrame if not already

    if cols is None:
        cols = _find_categorical_columns(X)
    if len(cols) == 0:
        return X

    new_columns = []
    for col in cols:  # For each column index
        unique_values = X.iloc[:, col].unique()  # Get the unique values in the column
        for value in unique_values:  # For each unique value, create a new binary column
            new_columns.append((X.iloc[:, col] == value).astype(int).rename(str(value)))

    X = pd.concat(
        [X.drop(X.columns[cols], axis=1)] + new_columns, axis=1
    )  # Drop the original columns and add new columns

    if not is_dataframe:
        return (
            X.values
        )  # Convert back to numpy array if it was originally a numpy array
    return X  # Else, return the DataFrame


def _find_categorical_columns(X):
    """Finds the indices of non-numerical columns in a DataFrame or numpy array.

    Args:
        X: (pandas.DataFrame or numpy.ndarray) - The data to be checked.

    Returns:
        categorical_cols: (list) - The list of indices of non-numerical columns.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)  # Convert to DataFrame if not already

    # For each column, try to convert it to numeric
    # If it fails, it is a categorical column
    categorical_cols = []
    for i in range(X.shape[1]):
        try:
            pd.to_numeric(X.iloc[:, i])
        except ValueError:
            categorical_cols.append(i)
    return categorical_cols  # Return the list of indices of non-numerical columns


def normalize(X, norm="l2"):
    """Normalizes the input data using the specified norm.

    Args:
        X: (numpy.ndarray) - The input data to be normalized.
        norm: (str), optional - The type of norm to use for normalization (default is 'l2').
            Options:
                - 'l2': L2 normalization (Euclidean norm).
                - 'l1': L1 normalization (Manhattan norm).
                - 'max': Max normalization (divides by the maximum absolute value).
                - 'minmax': Min-max normalization (scales to [0, 1]).

    Returns:
        X: (numpy.ndarray) - The normalized data.
    """
    # Ensure the data is in the correct shape and type
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(X, pd.Series):
        X = X.values.reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim > 2:
        raise ValueError("Input data must be 1D or 2D.")

    if norm == "l2":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        X = X / norms
    elif norm == "l1":
        norms = np.sum(np.abs(X), axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        X = X / norms
    elif norm == "max":
        max_values = np.max(np.abs(X), axis=1, keepdims=True)
        max_values[max_values == 0] = 1  # Avoid division by zero
        X = X / max_values
    elif norm == "minmax":
        min_values = np.min(X, axis=1, keepdims=True)
        max_values = np.max(X, axis=1, keepdims=True)
        range_values = max_values - min_values
        range_values[range_values == 0] = 1  # Avoid division by zero
        X = (X - min_values) / range_values
    else:
        raise ValueError(f"Unsupported norm: {norm}")
    return X


class Scaler:
    """A class for scaling data by standardization and normalization."""

    def __init__(self, method="standard"):
        """Initializes the scaler with the specified method.

        Args:
            method: (str) - The scaling method to use. Options are 'standard', 'minmax', or 'normalize'.
        """
        if method not in ["standard", "minmax", "normalize"]:
            raise ValueError(f"Unsupported method: {method}")

        self.method = method
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.norm = None

    def fit(self, X):
        """Fits the scaler to the data.

        Args:
            X: (numpy.ndarray) - The data to fit the scaler to.
        """
        if self.method == "standard":
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        elif self.method == "minmax":
            self.min = np.min(X, axis=0)
            self.max = np.max(X, axis=0)
        elif self.method == "normalize":
            norms = np.linalg.norm(X, axis=1)
            norms[norms == 0] = 1  # Avoid division by zero
            self.norm = norms
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def transform(self, X):
        """Transforms the data using the fitted scaler.

        Args:
            X: (numpy.ndarray) - The data to transform.

        Returns:
            X_transformed: (numpy.ndarray) - The transformed data.
        """
        if self.method == "standard":
            return (X - self.mean) / (self.std + 1e-8)
        elif self.method == "minmax":
            return (X - self.min) / (self.max - self.min + 1e-8)
        elif self.method == "normalize":
            return X / (self.norm[:, np.newaxis] + 1e-8)

    def fit_transform(self, X):
        """Fits the scaler to the data and then transforms it.

        Args:
            X: (numpy.ndarray) - The data to fit and transform.

        Returns:
            X_transformed: (numpy.ndarray) - The transformed data.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """Inverse transforms the data using the fitted scaler.

        Args:
            X: (numpy.ndarray) - The data to inverse transform.

        Returns:
            X_inverse: (numpy.ndarray) - The inverse transformed data.
        """
        if self.method == "standard":
            return X * self.std + self.mean
        elif self.method == "minmax":
            return X * (self.max - self.min) + self.min
        elif self.method == "normalize":
            return X * (self.norm[:, np.newaxis] + 1e-8)


# TODO: Add tests for encoder
class Encoder:
    """Custom encoder for transforming categorical labels into numerical representations.

    Supports Label Encoding and can be extended for Label Binarization.
    """

    def __init__(
        self, strategy="label_encode", handle_unknown="error", unknown_value=-1
    ):
        """Initialize the Encoder.

        Args:
            strategy (str, default="label_encode"): The encoding strategy.
                Currently supports:
                - "label_encode": Encode target labels with value between 0 and n_classes-1.
                Future options could include "label_binarize".
            handle_unknown (str, default='error'): How to handle unknown categories
                during transform (i.e., categories not seen during fit).
                - 'error': Raise a ValueError.
                - 'use_unknown_value': Encode them as `unknown_value`.
            unknown_value (int, default=-1): Value to use for unknown categories
                when `handle_unknown='use_unknown_value'`. Ignored if `handle_unknown='error'`.
        """
        if strategy not in ["label_encode"]:  # Add "label_binarize" when implemented
            raise ValueError(
                f"Strategy '{strategy}' is not supported. "
                "Currently supported: 'label_encode'."
            )
        if handle_unknown not in ["error", "use_unknown_value"]:
            raise ValueError("handle_unknown must be 'error' or 'use_unknown_value'.")

        self.strategy = strategy
        self.handle_unknown = handle_unknown
        self.unknown_value = int(unknown_value)  # Ensure it's an integer

        self.classes_ = None
        self._mapping = None  # For label_encode: maps category to int
        self._inverse_mapping = None  # For label_encode: maps int to category
        self.is_fitted_ = False

    def fit(self, y):
        """Fit the encoder to determine the mapping from categories to numbers.

        Args:
            y (array-like): The input array of labels to be encoded.
                Can be a list, NumPy array, or pandas Series.
        """
        if not isinstance(y, (list, np.ndarray, pd.Series)):
            raise TypeError(
                "Input y must be array-like (list, NumPy array, or pandas Series)."
            )

        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise ValueError("Input y must be a 1-dimensional array.")

        # Find unique classes, handling NaNs consistently
        # Using pandas for robust unique with NaN handling
        unique_classes_pd = pd.unique(y_arr)
        # Filter out NaNs from classes_ if present (LabelEncoder typically doesn't map NaNs)
        self.classes_ = np.array([cls for cls in unique_classes_pd if not pd.isna(cls)])
        self.classes_.sort()  # Ensure consistent mapping order

        if self.strategy == "label_encode":
            self._mapping = {cls: i for i, cls in enumerate(self.classes_)}
            self._inverse_mapping = {i: cls for cls, i in self._mapping.items()}
        # Add logic for "label_binarize" here in the future
        # For binarizer, self.classes_ would define the columns

        self.is_fitted_ = True
        return self

    def transform(self, y):
        """Transform labels to their numerical representation.

        Args:
            y (array-like): The input array of labels to be transformed.

        Returns:
            np.ndarray: The transformed numerical labels. For "label_encode",
                        this is a 1D array of integers.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Encoder has not been fitted. Call fit() before transform()."
            )
        if not isinstance(y, (list, np.ndarray, pd.Series)):
            raise TypeError(
                "Input y must be array-like (list, NumPy array, or pandas Series)."
            )

        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise ValueError("Input y must be a 1-dimensional array.")

        if self.strategy == "label_encode":
            transformed_y = np.full(y_arr.shape, self.unknown_value, dtype=int)
            for i, item in enumerate(y_arr):
                if pd.isna(item):  # How to handle NaNs in transform?
                    # Option 1: Treat as unknown
                    if self.handle_unknown == "error":
                        # Or decide to map NaNs consistently if they were in fit classes_
                        # For now, if NaN wasn't in self.classes_ (it's filtered), treat as unknown
                        raise ValueError(
                            "Encountered NaN during transform and NaNs were not explicitly handled as a class during fit."
                        )
                    transformed_y[i] = self.unknown_value
                    continue

                encoded_value = self._mapping.get(item)
                if encoded_value is not None:
                    transformed_y[i] = encoded_value
                else:  # Unknown category
                    if self.handle_unknown == "error":
                        raise ValueError(
                            f"Unknown category '{item}' encountered during transform."
                        )
                    # Already pre-filled with self.unknown_value, so no explicit assignment needed here
                    # but good to be aware.
            return transformed_y

        # Add logic for "label_binarize" transform here
        # elif self.strategy == "label_binarize":
        #     if len(self.classes_) == 2: # Special case for binary: output 1D array (0 or 1)
        #         # Positive class is typically self.classes_[1] after sorting
        #         positive_class = self.classes_[1]
        #         transformed_y = np.zeros(len(y_arr), dtype=int)
        #         for i, item in enumerate(y_arr):
        #             if pd.isna(item):
        #                 if self.handle_unknown == 'error': raise ValueError("NaN in input y for binarize")
        #                 # How to binarize NaN? Perhaps all zeros. Or error.
        #                 continue # Skips, effectively leaving it 0
        #             if item == positive_class:
        #                 transformed_y[i] = 1
        #             elif item not in self.classes_: # Unknown class
        #                 if self.handle_unknown == 'error': raise ValueError(f"Unknown category '{item}'")
        #                 # For binarizer, unknown usually means all zeros.
        #         return transformed_y
        #     else: # Multi-class binarization (One-Hot Encoding like)
        #         transformed_y = np.zeros((len(y_arr), len(self.classes_)), dtype=int)
        #         for i, item in enumerate(y_arr):
        #             if pd.isna(item):
        #                 if self.handle_unknown == 'error': raise ValueError("NaN in input y for binarize")
        #                 continue # Row of zeros for NaN
        #             try:
        #                 # Find index of the item in self.classes_
        #                 class_idx = np.where(self.classes_ == item)[0][0]
        #                 transformed_y[i, class_idx] = 1
        #             except IndexError: # Item not in self.classes_ (unknown)
        #                 if self.handle_unknown == 'error':
        #                     raise ValueError(f"Unknown category '{item}' encountered during transform.")
        #                 # For binarizer, unknown means a row of zeros, which is default.
        #         return transformed_y

        else:
            raise NotImplementedError(
                f"Transform for strategy '{self.strategy}' not implemented."
            )

    def fit_transform(self, y):
        """Fit the encoder and then transform the labels.

        Args:
            y (array-like): The input array of labels.

        Returns:
            np.ndarray: The transformed numerical labels.
        """
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y_transformed):
        """Transform numerical labels back to their original categories.

        Args:
            y_transformed (array-like): The numerical labels to be transformed back.
                                       Should be a 1D array of integers for "label_encode".

        Returns:
            np.ndarray: The original categorical labels.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Encoder has not been fitted. Call fit() before inverse_transform()."
            )
        if not isinstance(y_transformed, (list, np.ndarray, pd.Series)):
            raise TypeError("Input y_transformed must be array-like.")

        y_transformed_arr = np.asarray(y_transformed)
        if y_transformed_arr.ndim != 1:
            raise ValueError(
                "Input y_transformed must be a 1-dimensional array for label_encode inverse."
            )

        if self.strategy == "label_encode":
            original_labels = np.full(
                y_transformed_arr.shape, fill_value=None, dtype=object
            )  # Use object for mixed types
            for i, item_code in enumerate(y_transformed_arr):
                if (
                    item_code == self.unknown_value
                    and self.handle_unknown == "use_unknown_value"
                ):
                    # What to inverse_transform unknown_value to? Could be None, or a specific placeholder.
                    # Let's use None, or raise error if not 'use_unknown_value'
                    original_labels[i] = (
                        None  # Or a specific placeholder like "unknown_category"
                    )
                else:
                    original_category = self._inverse_mapping.get(item_code)
                    if original_category is not None:
                        original_labels[i] = original_category
                    else:
                        # This case implies an integer was passed that doesn't correspond to a known class
                        # or the unknown_value if handle_unknown was 'error'
                        raise ValueError(
                            f"Value '{item_code}' not found in inverse mapping. "
                            "It might be an unknown value not handled by 'use_unknown_value' "
                            "or an invalid code."
                        )
            return original_labels

        # Add logic for "label_binarize" inverse_transform here
        # elif self.strategy == "label_binarize":
        #     if y_transformed_arr.ndim == 1 and len(self.classes_) == 2: # Binary case
        #         # Assuming 0 maps to self.classes_[0] and 1 to self.classes_[1]
        #         original_labels = np.full(len(y_transformed_arr), fill_value=None, dtype=object)
        #         for i, val in enumerate(y_transformed_arr):
        #             if val == 0: original_labels[i] = self.classes_[0]
        #             elif val == 1: original_labels[i] = self.classes_[1]
        #             else: raise ValueError(f"Invalid value {val} in binary transformed data.")
        #         return original_labels
        #     elif y_transformed_arr.ndim == 2 and y_transformed_arr.shape[1] == len(self.classes_): # Multi-class
        #         original_labels = np.full(y_transformed_arr.shape[0], fill_value=None, dtype=object)
        #         for i in range(y_transformed_arr.shape[0]):
        #             hot_indices = np.where(y_transformed_arr[i, :] == 1)[0]
        #             if len(hot_indices) == 1:
        #                 original_labels[i] = self.classes_[hot_indices[0]]
        #             elif len(hot_indices) == 0: # All zeros, could be unknown or NaN original
        #                 # This depends on how transform handled unknowns/NaNs.
        #                 # For now, let's assume it was an unknown if handle_unknown='use_unknown_value'
        #                 # or if transform created all zeros for NaNs.
        #                 original_labels[i] = None # Or "unknown_category_from_binarized"
        #             else: # Multiple 1s - invalid binarized format
        #                 raise ValueError(f"Invalid binarized row at index {i}: more than one 'hot' value.")
        #         return original_labels
        #     else:
        #         raise ValueError("Shape of y_transformed is incompatible with fitted classes for binarizer.")
        else:
            raise NotImplementedError(
                f"Inverse transform for strategy '{self.strategy}' not implemented."
            )


# --- Placeholder for Label Binarizer Tests (when implemented) ---
# print("\n--- Label Binarizer Tests (Placeholder) ---")
# encoder_bin = Encoder(strategy="label_binarize")
# # Test binary case
# labels_bin1 = ['yes', 'no', 'yes', 'yes', 'no']
# encoder_bin.fit(labels_bin1)
# print("Fitted classes_ (bin1):", encoder_bin.classes_)
# transformed_bin1 = encoder_bin.transform(labels_bin1)
# print("Transformed (bin1):", transformed_bin1)
# print("Inverse transformed (bin1):", encoder_bin.inverse_transform(transformed_bin1))

# # Test multi-class case
# labels_bin2 = ['A', 'B', 'C', 'A', 'B']
# encoder_bin.fit(labels_bin2)
# print("\nFitted classes_ (bin2):", encoder_bin.classes_)
# transformed_bin2 = encoder_bin.transform(labels_bin2)
# print("Transformed (bin2):\n", transformed_bin2)
# print("Inverse transformed (bin2):", encoder_bin.inverse_transform(transformed_bin2))
