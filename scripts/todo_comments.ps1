# Find TODO comments in Python files, excluding paths containing "__archive"
$todoComments = Get-ChildItem -Recurse -Filter *.py | Where-Object {
    -not ($_.FullName -like "*_archive*") -and -not ($_.FullName -like "*.venv*")
} | ForEach-Object {
    $file = $_
    $lines = Get-Content $file.FullName
    $lines | ForEach-Object -Begin { $global:lineNumber = 0 } -Process {
        $lineNumber++
        $trimmedLine = $_.TrimStart()
        if ($trimmedLine -match 'TODO') {
            [PSCustomObject]@{
                FileName   = $file.FullName -replace ".*sega_learn", "sega_learn"
                LineNumber = $global:lineNumber
                Line       = $trimmedLine
            }
        }
    }
}

# Write TODO comments to the file
$todoComments | Tee-Object -FilePath "scripts/out/todo_comments.txt" | Format-Table -AutoSize

# Append the "Other" section to the file
# Add new TODOs here:
$otherTodos = @"
Planned/Ideas:
Time Series Analysis - ARIMA, Moving Average, Exponential Smoothing
Data Preprocessing - Normalization, Standardization, Missing Value Imputation, StandardScaler
OneClassSVM - fix/check implementation
Deep Learning Enhancements
    - Implement Neural Network Regressor
    - Implement Flatten Layer, Conv Layer:Base Done, Numba: TODO
    - Implement Recurrent Layers
    - Implement Data Preprocessing Layers
    - Live plotting of training and validation loss
Classification
    - Implement Logistic Regression
    - Implement SGDClassifier
    - Implement Gradient Boosted Classifier
GitHub Action to run tests?:
# filepath: .github/workflows/run-tests.yml
name: Run Tests

on:
  push:
    branches: [ "main" ] # Trigger on push to main branch
  pull_request:
    branches: [ "main" ] # Trigger on pull request to main branch

jobs:
  run-tests:
    name: Run Selected Tests
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          # Add any additional dependencies here, e.g., pytest
          # pip install pytest

      - name: Run selected tests
        run: python tests/run_selected_tests.py
"@

# Add the "Other" section to the file and console output
Add-Content -Path "scripts/out/todo_comments.txt" -Value $otherTodos
