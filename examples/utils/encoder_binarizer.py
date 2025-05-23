import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.utils.dataPreprocessing import Encoder

# --- Label Binarizer ---
print("\n------ Label Binarizer Tests ------")
encoder_bin = Encoder(
    strategy="label_binarize", handle_unknown="use_unknown_value"
)  # Use this for unknown handling


# Example binary case
print("\n--- Example binary case ---")
labels_bin1 = ["yes", "no", "yes", "yes", "no", np.nan]
encoder_bin.fit(labels_bin1)
transformed_bin1 = encoder_bin.transform(labels_bin1)
inverse_transformed_bin1 = encoder_bin.inverse_transform(transformed_bin1)

print("Original labels_bin1:", labels_bin1)
print("Fitted classes_ (bin1):", encoder_bin.classes_)
print("Transformed (bin1):", transformed_bin1)
print("Inverse transformed (bin1):", inverse_transformed_bin1)


# Example multi-class case
print("\n--- Example multi-class case ---")
labels_bin2 = ["A", "B", "C", "A", "B", "D", np.nan]  # D is unknown
encoder_bin.fit(labels_bin2)  # Fit on A, B, C, D
transformed_bin2 = encoder_bin.transform(labels_bin2)
encoder_bin_unknown = Encoder(
    strategy="label_binarize", handle_unknown="use_unknown_value"
)
encoder_bin_unknown.fit(["A", "B", "C"])
transformed_bin_unknown = encoder_bin_unknown.transform(
    labels_bin2
)  # Transform original with D and NaN
inverse_transformed_bin_unknown = encoder_bin_unknown.inverse_transform(
    transformed_bin_unknown
)
print("Original labels_bin2:", labels_bin2)
print("Fitted classes_ (bin2):", encoder_bin.classes_)  # Should be ['A', 'B', 'C', 'D']
print("Transformed (bin2):\n", transformed_bin2)
print("Transformed (bin_unknown with D and NaN):\n", transformed_bin_unknown)
print("Fitted classes_ (bin_unknown - A,B,C):", encoder_bin_unknown.classes_)
print(
    "Inverse transformed (bin_unknown with D and NaN as None):",
    inverse_transformed_bin_unknown,
)


# Example handle_unknown='error' for binarizer
print("\n--- Example handle_unknown='error' for binarizer ---")
encoder_bin_error = Encoder(strategy="label_binarize", handle_unknown="error")
encoder_bin_error.fit(["A", "B"])
print("Fitted classes_ (bin_error - A,B):", encoder_bin_error.classes_)
try:
    encoder_bin_error.transform(["A", "C", "B"])
except ValueError as e:
    print(f"Caught expected error for binarizer unknown: {e}")
try:
    invalid_binarized = np.array([[1, 1], [0, 1]])  # Invalid row
    encoder_bin_error.inverse_transform(invalid_binarized)
except ValueError as e:
    print(f"Caught expected error for invalid binarized input: {e}")
