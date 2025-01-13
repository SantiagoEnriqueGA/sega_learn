import pandas as pd

class DataPrep(object):
    """
    A class for preparing data for machine learning models.
    """

    def one_hot_encode(df, cols):
        """
        One-hot encodes non-numerical columns in a DataFrame.
        Drops the original columns after encoding.

        Parameters:
        - df (pandas.DataFrame): The DataFrame to be encoded.
        - cols (list): The list of column indices to be encoded.

        Returns:
        - df (pandas.DataFrame): The DataFrame with one-hot encoded columns.
        """
        for col in cols:                                # For each column index
            unique_values = df.iloc[:, col].unique()    # Get the unique values in the column

            for value in unique_values:                                 # For each unique value, create a new binary column
                df[str(value)] = (df.iloc[:, col] == value).astype(int) # Set the value to 1 if the original column is equal to the unique value

        df = df.drop(df.columns[cols], axis=1)          # Drop the original column

        return df

    def write_data(df, csv_file, print_path=False):
        """
        Writes the DataFrame to a CSV file.

        Parameters:
        - df (pandas.DataFrame): The DataFrame to be written.
        - csv_file (str): The path of the CSV file to write to.
        """
        df.to_csv(csv_file, index=False)            # Write the DataFrame to a CSV file
        if print_path: print("Prepared data written to", csv_file) # Print the path of the written file

    def prepare_data(csv_file, label_col_index, cols_to_encode=[], write_to_csv=True):
        """
        Prepares the data by loading a CSV file, one-hot encoding non-numerical columns,
        and optionally writing the prepared data to a new CSV file.

        Parameters:
        - csv_file (str): The path of the CSV file to load.
        - label_col_index (int): The index of the label column.
        - cols_to_encode (list): The list of column indices to one-hot encode. Default is an empty list.
        - write_to_csv (bool): Whether to write the prepared data to a new CSV file. Default is True.

        Returns:
        - df (pandas.DataFrame): The prepared DataFrame.
        - prepared_csv_file (str): The path of the prepared CSV file. If write_to_csv is False, returns "N/A".
        """
        df = pd.read_csv(csv_file)              # Load the CSV file
        
        label_col = df.columns[label_col_index]             # Get the label column name(s)
        df = DataPrep.one_hot_encode(df, cols_to_encode)    # One-hot encode the specified columns

        df = pd.concat([df.drop(label_col, axis=1), df[[label_col]]], axis=1)   # Move the label column from start to the end

        if write_to_csv:    # If write_to_csv is True
            prepared_csv_file = csv_file.replace('.csv', '_prepared.csv')   # Create a new file path for the prepared data
            DataPrep.write_data(df, prepared_csv_file)                      # Write the prepared data to a new CSV file
            return df, prepared_csv_file                                    # Return the prepared DataFrame and the path of the written file

        return df, "N/A"    # Else, return the prepared DataFrame and "N/A"
        
    def df_to_ndarray(df, y_col=0):
        """
        Converts a DataFrame to a NumPy array.
        
        Parameters:
        - df (pandas.DataFrame): The DataFrame to be converted.
        - y_col (int): The index of the label column. Default is 0.
        
        Returns:
        - X (numpy.ndarray): The feature columns as a NumPy array.
        - y (numpy.ndarray): The label column as a NumPy array.
        """
        y = df.iloc[:, y_col].values            # Get the label column as a NumPy array
        X = df.drop(df.columns[y_col], axis=1).values  # Get the feature columns as a NumPy array
        
        return X, y
    
    def k_split(X, y, k=5):
        """
        Splits the data into k folds for cross-validation.
        
        Parameters:
        - X (numpy.ndarray): The feature columns.
        - y (numpy.ndarray): The label column.
        - k (int): The number of folds. Default is 5.
        
        Returns:
        - X_folds (list): A list of k folds of feature columns.
        - y_folds (list): A list of k folds of label columns.
        """
        n_samples = len(y)                      # Get the number of samples
        fold_size = n_samples // k              # Calculate the fold size
        
        X_folds = []                            # Initialize a list to store the feature column folds
        y_folds = []                            # Initialize a list to store the label column folds
        
        for i in range(k):                      # For each fold
            start = i * fold_size                # Calculate the start index of the fold
            end = (i + 1) * fold_size            # Calculate the end index of the fold
            
            if i == k - 1:                       # If it is the last fold
                end = n_samples                  # Set the end index to the number of samples
                
            X_folds.append(X[start:end])         # Add the feature columns to the list of folds
            y_folds.append(y[start:end])         # Add the label columns to the list of folds
        
        return X_folds, y_folds