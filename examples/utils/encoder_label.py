import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.utils.dataPreprocessing import Encoder

# --- Test Label Encoding ---
print("--- Label Encoding Tests ---")
labels1 = ["cat", "dog", "mouse", "cat", "dog", np.nan, "fish"]
encoder1 = Encoder(
    strategy="label_encode", handle_unknown="use_unknown_value", unknown_value=-99
)

print("Original labels1:", labels1)
encoder1.fit(labels1)
print("Fitted classes_:", encoder1.classes_)
print("Mapping:", encoder1._mapping)

transformed1 = encoder1.transform(labels1)
print("Transformed labels1:", transformed1)

inverse_transformed1 = encoder1.inverse_transform(transformed1)
print("Inverse transformed labels1:", inverse_transformed1)

# Test with unknown value during transform
labels_to_transform = ["dog", "cat", "python", np.nan, "fish", "cat"]
print("\nLabels to transform (with unknown):", labels_to_transform)
transformed_unknown = encoder1.transform(labels_to_transform)
print("Transformed (with unknown as -99):", transformed_unknown)
inverse_transformed_unknown = encoder1.inverse_transform(transformed_unknown)
print(
    "Inverse transformed (with unknown as -99, becomes None):",
    inverse_transformed_unknown,
)


# Test handle_unknown='error'
encoder2 = Encoder(strategy="label_encode", handle_unknown="error")
labels2 = ["red", "green", "blue"]
encoder2.fit(labels2)
print("\nFitted classes_ (encoder2):", encoder2.classes_)
try:
    encoder2.transform(["red", "yellow", "blue"])
except ValueError as e:
    print(f"Caught expected error for unknown: {e}")
try:
    encoder2.inverse_transform([0, 1, 5])  # 5 is an unknown code
except ValueError as e:
    print(f"Caught expected error for unknown code: {e}")

# Test with purely numerical input (should still work, treating numbers as categories)
labels_num = [10, 20, 30, 10, 20, 20]
encoder_num = Encoder()
encoder_num.fit(labels_num)
print("\nFitted classes_ (numeric input):", encoder_num.classes_)
transformed_num = encoder_num.transform(labels_num)
print("Transformed (numeric input):", transformed_num)
print("Inverse (numeric input):", encoder_num.inverse_transform(transformed_num))
