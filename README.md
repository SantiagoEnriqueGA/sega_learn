# SEGA_LEARN

SEGA_LEARN is a custom implementation of machine learning algorithms mainly from the Python standard library and NumPy. It includes implementations of various machine learning algorithms, including clustering, linear models, neural networks, and trees. The project also includes scripts for testing, documentation generation, and other tasks.

The project is organized into several directories, each with its own purpose. The `SEGA_LEARN/` directory contains the main library code, while the `tests/` directory contains unit and performance tests. The `examples/` directory contains example usages of the library, and the `docs/` directory contains the generated documentation. The `scripts/` directory contains PowerShell scripts to help with various tasks.

This project was created with the goal of learning about machine learning and improving coding skills in Python programming. It is not intended for production use and should be used for educational purposes only (See performance tests). Many of the algorithms are not optimized for performance and may not be suitable for large datasets.

This project was heavily inspired by [scikit-learn](https://scikit-learn.org/stable/), and [pytorch](https://pytorch.org/).

## Navigation
<!-- Add Links to Other Sections Here! -->
- [Features](#features)
- [File Structure](#file-structure)
- [Usage Example](#usage-examples)
- [Scripts](#scripts)
- [Documentation](#documentation)
- [Tests](#tests)
- [Installation](#installation)

## Current Features
- **Clustering**: Implements clustering algorithms like DBSCAN and KMeans.
- **Linear Models**: Implements various linear models like Bayesian Regression, Lasso Regression, Linear Discriminant Analysis, Ordinary Least Squares, Passive Aggressive Regressor, Quadratic Discriminant Analysis, RANSAC Regression, and Ridge Regression.
- **Neural Networks**: Implements neural network components like optimizers (Adadelta, Adam, SGD), loss functions (BCEWithLogitsLoss, CrossEntropyLoss), and the NeuralNetwork class.
- **Trees**: Implements tree-based algorithms like Classifier Tree, Random Forest Classifier, Gradient Boosted Regressor, Random Forest Regressor, Regressor Tree, and utilities for computing variance, partitioning classes, and calculating information gain.
- **Utils**: Implements utility functions for data preparation and voting regressor.

## Planned Features - Future Work
- Implement dimensionality reduction algorithms like Principal Component Analysis (PCA), truncated Singular Value Decomposition (t-SVD)
- Implement OPTICS clustering algorithm
- Implement model selection algorithms like cross-validation, grid search, and random search
  - Implement novel model selection algorithms like Bayesian Optimization, Bayesian Model Averaging, and Bayesian Model Selection.

## File Structure
The project directory structure is as follows:

- **sega_learn/**: Contains the main library code.
- [`__init__.py`](sega_learn/__init__.py): Initializes the SEGA_LEARN package.
  - **clustering/**: Contains clustering algorithms.
    - [`__init__.py`](sega_learn/clustering/__init__.py): Initializes the clustering package.
    - [`clustering.py`](sega_learn/clustering/clustering.py): Implements DBSCAN and KMeans clustering algorithms.
  - **linear_models/**: Contains linear models.
    - [`__init__.py`](sega_learn/linear_models/__init__.py): Initializes the linear models package.
    - [`linearModels.py`](sega_learn/linear_models/linearModels.py): Implements various linear models.
    - [`discriminantAnalysis.py`](sega_learn/linear_models/discriminantAnalysis.py): Implements Linear and Quadratic Discriminant Analysis.
  - **neural_networks/**: Contains neural network components.
    - [`__init__.py`](sega_learn/neural_networks/__init__.py): Initializes the neural networks package.
    - [`optimizers.py`](sega_learn/neural_networks/optimizers.py): Implements various optimizers.
    - [`schedulers.py`](sega_learn/neural_networks/schedulers.py): Implements learning rate schedulers.
    - [`loss.py`](sega_learn/neural_networks/loss.py): Implements loss functions.
    - [`neuralNetwork.py`](sega_learn/neural_networks/neuralNetwork.py): Implements the NeuralNetwork class.
  - **trees/**: Contains tree-based algorithms.
    - [`__init__.py`](sega_learn/trees/__init__.py): Initializes the trees package.
    - [`treeClassifier.py`](sega_learn/trees/treeClassifier.py): Implements Classifier Tree.
    - [`treeRegressor.py`](sega_learn/trees/treeRegressor.py): Implements Regressor Tree.
    - [`randomForestClassifier.py`](sega_learn/trees/randomForestClassifier.py): Implements Random Forest Classifier.
    - [`randomForestRegressor.py`](sega_learn/trees/randomForestRegressor.py): Implements Random Forest Regressor.
    - [`gradientBoostedRegressor.py`](sega_learn/trees/gradientBoostedRegressor.py): Implements Gradient Boosted Regressor.
  - **utils/**: Contains utility functions.
    - [`__init__.py`](sega_learn/utils/__init__.py): Initializes the utils package.
    - [`dataPrep.py`](sega_learn/utils/dataPrep.py): Implements data preparation functions.
    - [`voting.py`](sega_learn/utils/voting.py): Implements voting regressor.
    - [`metrics.py`](sega_learn/utils/metrics.py): Implements evaluation metrics.
    - [`model_selection.py`](sega_learn/utils/model_selection.py): Implements model selection algorithms.
    - [`polynomialTransform.py`](sega_learn/utils/polynomialTransform.py): Implements polynomial transformation.

- **tests/**: Contains unit and performance tests for the database library.
  - **Core Tests**:
    - [`test_clustering.py`](tests/test_clustering.py): Tests DBSCAN and KMeans clustering algorithms.
    - [`test_linear_models.py`](tests/test_linear_models.py): Tests various linear models.
    - [`test_neural_networks.py`](tests/test_neural_networks.py): Tests neural network components.
    - [`test_trees_classifier.py`](tests/test_trees_classifier.py): Tests Classifier Trees.
    - [`test_trees_regressor.py`](tests/test_trees_regressor.py): Tests Regressor Trees.
    - [`test_utils.py`](tests/test_utils.py): Tests utility functions.
  - **Imports**:
    - [`test_clustering_imports.py`](tests/test_clustering_imports.py): Tests imports for clustering package.
    - [`test_linear_models_imports.py`](tests/test_linear_models_imports.py): Tests imports for linear models package.
    - [`test_neural_networks_imports.py`](tests/test_neural_networks_imports.py): Tests imports for neural networks package.
    - [`test_trees_imports.py`](tests/test_trees_imports.py): Tests imports for trees package.
    - [`test_imports.py`](tests/test_imports.py): Tests imports for the main package.
  - **Examples**:
    - [`test_clustering_examples.py`](tests/test_clustering_examples.py): Tests clustering examples.
    - [`test_linear_models_examples.py`](tests/test_linear_models_examples.py): Tests linear models examples.
    - [`test_neural_networks_examples.py`](tests/test_neural_networks_examples.py): Tests neural networks examples.
    - [`test_trees_examples.py`](tests/test_trees_examples.py): Tests tree-based algorithms examples.
    - [`test_utils_examples.py`](tests/test_utils_examples.py): Tests utility functions examples.
  - **Run All Tests**:
    - [`run_all_tests.py`](tests/run_all_tests.py): Runs all available tests.
  - **Test Utilities**:
    - [`utils.py`](tests/utils.py): Contains utility functions for testing.

- **examples/**: Example usages of the SEGA_LEARN library.
  - **clustering/**: Contains clustering algorithms.
    - [`kmeans.py`](examples/clustering/kmeans.py): Demonstrates KMeans.
    - [`dbscan.py`](examples/clustering/dbscan.py): Demonstrates DBSCAN.
    - [`dbscan_3d.py`](examples/clustering/dbscan_3d.py): Demonstrates DBSCAN with 3D data.
    - [`dbscan_3d_aimated.py`](examples/clustering/dbscan_3d_aimated.py): Demonstrates DBSCAN with 3D data and animated plot.
  - **linear_models/**: Contains linear models.
    - [`ols.py`](examples/linear_models/ols.py): Demonstrates Ordinary Least Squares.
    - [`ridge.py`](examples/linear_models/ridge.py): Demonstrates Ridge Regression.
    - [`lasso.py`](examples/linear_models/lasso.py): Demonstrates Lasso Regression.
    - [`bayes.py`](examples/linear_models/bayes.py): Demonstrates Bayesian Regression.
    - [`ransac.py`](examples/linear_models/ransac.py): Demonstrates RANSAC Regression.
    - [`passiveAggressive.py`](examples/linear_models/passiveAggressive.py): Demonstrates Passive Aggressive Regressor.
    - [`lda.py`](examples/linear_models/lda.py): Demonstrates Linear Discriminant Analysis.
    - [`qda.py`](examples/linear_models/qda.py): Demonstrates Quadratic Discriminant Analysis.
  - **neural_networks/**: Contains neural network components.
    - [`neuralNetwork.py`](examples/neural_networks/neuralNetwork.py): Demonstrates the NeuralNetwork class.
    - [`neuralNetwork_hyper.py`](examples/neural_networks/neuralNetwork_hyper.py): Demonstrates the NeuralNetwork class with hyper-parameter tuning.
  - **trees/**: Contains tree-based algorithms.
    - [`gradientBoostedRegressor.py`](examples/trees/gradientBoostedRegressor.py): Demonstrates Gradient Boosted Regressor.
    - [`randomForestClassifier.py`](examples/trees/randomForestClassifier.py): Demonstrates Random Forest Classifier.
    - [`randomForestRegressor.py`](examples/trees/randomForestRegressor.py): Demonstrates Random Forest Regressor.
    - [`regressorTree.py`](examples/trees/regressorTree.py): Demonstrates Regressor Tree.
  - **utils/**: Contains utility functions.
    - [`votingRegressor.py`](examples/utils/votingRegressor.py): Demonstrates Voting Regressor.
    - [`polynomialTransform.py`](examples/utils/polynomialTransform.py): Demonstrates Polynomial Transform.
    - [`gridSearchCV_bayes.py`](examples/utils/gridSearchCV_bayes.py): Demonstrates Grid Search Cross Validation with Bayesian Regression.
    - [`gridSearchCV_gbr.py`](examples/utils/gridSearchCV_gbr.py): Demonstrates Grid Search Cross Validation with Gradient Boosted Regressor.
    - [`gridSearchCV_passiveAggressive.py`](examples/utils/gridSearchCV_passiveAggressive.py): Demonstrates Grid Search Cross Validation with Passive Aggressive Regressor.
    - [`gridSearchCV_rfc.py`](examples/utils/gridSearchCV_rfc.py): Demonstrates Grid Search Cross Validation with Random Forest Classifier.
    - [`gridSearchCV_rfr.py`](examples/utils/gridSearchCV_rfr.py): Demonstrates Grid Search Cross Validation with Random Forest Regressor.
  
- **docs/**: Contains the generated documentation for the SEGA_LEARN library.
  - [`documentation.md`](docs/documentation.md): Contains the generated documentation for the SEGA_LEARN library.
  - **HTML Documentation**: Contains the generated HTML documentation for the SEGA_LEARN library.
    - [`SEGA_LEARN.html`](docs/SEGA_LEARN.html): Contains the main HTML documentation for the SEGA_LEARN library.
    - Other HTML documentation files: Contain the additional HTML documentation for the SEGA_LEARN library.

- **scripts/**: PowerShell scripts to help with various tasks.
  - [`_run_all_scripts.ps1`](scripts/_run_all_scripts.ps1): Runs all PowerShell scripts in the `scripts/` folder sequentially.
  - [`todo_comments.ps1`](scripts/todo_comments.ps1): Finds and lists all TODO comments in Python files.
  - [`count_lines.ps1`](scripts/count_lines.ps1): Counts the number of lines in each Python file.
  - [`comment_density.ps1`](scripts/comment_density.ps1): Calculates the comment density in Python files.
  - [`documentation_html.ps1`](scripts/documentation_html.ps1): Generates HTML documentation.
  - [`documentation_md.ps1`](scripts/documentation_md.ps1): Generates markdown documentation.
  - [`export_env.ps1`](scripts/export_env.ps1): Exports the conda environment to a YAML file.

## Usage Examples

### Clustering
- [`kmeans.py`](examples/clustering/kmeans.py): Demonstrates KMeans.
- [`dbscan.py`](examples/clustering/dbscan.py): Demonstrates DBSCAN.
- [`dbscan_3d.py`](examples/clustering/dbscan_3d.py): Demonstrates DBSCAN with 3D data.
- [`dbscan_3d_aimated.py`](examples/clustering/dbscan_3d_aimated.py): Demonstrates DBSCAN with 3D data and animated plot.

### Linear Models
- [`ols.py`](examples/linear_models/ols.py): Demonstrates Ordinary Least Squares.
- [`ridge.py`](examples/linear_models/ridge.py): Demonstrates Ridge Regression.
- [`lasso.py`](examples/linear_models/lasso.py): Demonstrates Lasso Regression.
- [`bayes.py`](examples/linear_models/bayes.py): Demonstrates Bayesian Regression.
- [`ransac.py`](examples/linear_models/ransac.py): Demonstrates RANSAC Regression.
- [`passiveAggressive.py`](examples/linear_models/passiveAggressive.py): Demonstrates Passive Aggressive Regressor.
- [`lda.py`](examples/linear_models/lda.py): Demonstrates Linear Discriminant Analysis.
- [`qda.py`](examples/linear_models/qda.py): Demonstrates Quadratic Discriminant Analysis.

### Neural Networks
- [`neuralNetwork.py`](examples/neural_networks/neuralNetwork.py): Demonstrates the NeuralNetwork class.
- [`neuralNetwork_hyper.py`](examples/neural_networks/neuralNetwork_hyper.py): Demonstrates the NeuralNetwork class with hyper-parameter tuning.

### Trees
- [`gradientBoostedRegressor.py`](examples/trees/gradientBoostedRegressor.py): Demonstrates Gradient Boosted Regressor.
- [`randomForestClassifier.py`](examples/trees/randomForestClassifier.py): Demonstrates Random Forest Classifier.
- [`randomForestRegressor.py`](examples/trees/randomForestRegressor.py): Demonstrates Random Forest Regressor.
- [`regressorTree.py`](examples/trees/regressorTree.py): Demonstrates Regressor Tree.

### Utils
- [`votingRegressor.py`](examples/utils/votingRegressor.py): Demonstrates Voting Regressor.
- [`polynomialTransform.py`](examples/utils/polynomialTransform.py): Demonstrates Polynomial Transform.
- [`gridSearchCV_bayes.py`](examples/utils/gridSearchCV_bayes.py): Demonstrates Grid Search Cross Validation with Bayesian Regression.
- [`gridSearchCV_gbr.py`](examples/utils/gridSearchCV_gbr.py): Demonstrates Grid Search Cross Validation with Gradient Boosted Regressor.
- [`gridSearchCV_passiveAggressive.py`](examples/utils/gridSearchCV_passiveAggressive.py): Demonstrates Grid Search Cross Validation with Passive Aggressive Regressor.
- [`gridSearchCV_rfc.py`](examples/utils/gridSearchCV_rfc.py): Demonstrates Grid Search Cross Validation with Random Forest Classifier.
- [`gridSearchCV_rfr.py`](examples/utils/gridSearchCV_rfr.py): Demonstrates Grid Search Cross Validation with Random Forest Regressor.


## Scripts
The following PowerShell scripts are included in the `scripts/` folder to help with various tasks:

- **_run_all_scripts.ps1**: Runs all PowerShell scripts in the `scripts/` folder sequentially.
- **todo_comments.ps1**: Finds and lists all TODO comments in Python files.
- **count_lines.ps1**: Counts the number of lines in each Python file, sorts the files by line count in descending order, and calculates the total number of lines.
- **comment_density.ps1**: Calculates the comment density (percentage of lines that are comments) in Python files.
- **documentation_html.ps1**: Generates HTML documentation for Python files in the `SEGA_LEARN/` folder, and moves the generated HTML files to the `docs/` folder.
- **documentation_md.ps1**: Generates markdown documentation for Python files in the `SEGA_LEARN/` folder.
- **export_env.ps1**: Exports the conda environment to a YAML file. Remove the prefix from the environment name to make it compatible with other systems.

## Documentation
### HTML Documentation
Pydoc documentation is generated from the PowerShell script `documentation_html.ps1`.  
To see live version: https://santiagoenriquega.github.io/sega_learn/sega_learn  

Self host documentation, run the following command in the terminal: `python -m pydoc -p 8080`  
Then open a web browser and navigate to http://localhost:8080/SEGA_LEARN.html

### Markdown Documentation
Pydoc Markdown is also availible and is generated from the PowerShell script `documentation_md.ps1`.  
The output file is located in [`docs/documentation.md`](docs/documentation.md)

## Tests
To run the tests, use the following command: `python -m unittest discover -s tests`  
Or run the all tests file: `python run_all_tests.py`

### Test Results
The following are the results of running the tests:
```sh
(sega_learn) PS .../sega_learn/tests/run_all_tests.py
Testing Imports - Clustering
..Testing Imports - Main Package
..Testing Imports - Linear Models
..Testing Imports - Neural Networks
..Testing Imports - Trees
..Testing DBSCAN
.....Testing KMeans
..........Testing Bayesian Regression Model
..........Testing Lasso Regression Model
.........Testing Linear Discriminant Analysis
.......Testing Ordinary Least Squares Model
.......Testing Passive Aggressive Regressor Model
..........Testing Quadratic Discriminant Analysis
.......Testing RANSAC Regression Model
...........Testing Ridge Regression Model
.........Testing the AdadeltaOptimizer class
..Testing the AdamOptimizer class
..Testing the BCEWithLogitsLoss class
..Testing the CrossEntropyLoss class
..Testing the NeuralNetwork class
......Testing the SGDOptimizer class
..Testing Classifier Tree
....Testing Classifier Tree Utility
.....Testing Random Forest Classifier
...........Testing Gradient Boosted Regressor
............Testing Random Forest Regressor
..............Testing Regressor Tree
.....Testing Regressor Tree Utility
.................Testing Data Prep
......Testing GridSearchCV
...........Testing Metrics
...............Testing Polynomial Transform
....Testing RandomSearchCV
.............Testing Voting Regressor
....Testing example file: dbscan.py
.Testing example file: dbscan_3d.py
.Testing example file: dbscan_3d_aimated.py
.Testing example file: kmeans.py
.Testing example file: bayes.py
.Testing example file: lasso.py
.Testing example file: lda.py
.Testing example file: ols.py
.Testing example file: passiveAggressive.py
.Testing example file: passiveAggressive_vis.py
.Testing example file: qda.py
.Testing example file: ransac.py
.Testing example file: ransac_vis.py
.Testing example file: ridge.py
.Testing example file: neuralNetwork.py
.Testing example file: neuralNetwork_hyper.py
Tuning Hyperparameters: 100%|██████████| 4/4 [00:05<00:00,  1.32s/it]
Tuning Hyperparameters: 100%|██████████| 4/4 [00:04<00:00,  1.03s/it]
.Testing example file: gradientBoostedRegressor.py
.Testing example file: randomForestClassifier.py
.Testing example file: randomForestRegressor.py
.Testing example file: gridSearchCV_bayes.py
.Testing example file: gridSearchCV_gbr.py
.Testing example file: gridSearchCV_passiveAggReg.py
.Testing example file: gridSearchCV_rfc.py
.Testing example file: gridSearchCV_rfr.py
.Testing example file: polynomialTransform.py
.Testing example file: randomSearchCV_bayes.py
.Testing example file: randomSearchCV_gbr.py
.Testing example file: randomSearchCV_passiveAggReg.py
.Testing example file: randomSearchCV_rfc.py
.Testing example file: randomSearchCV_rfr.py
.Testing example file: votingRegressor.py
.
----------------------------------------------------------------------
Ran 263 tests in 114.101s

OK
```

## Installation

To set up the project environment, you can use the provided `environment.yml` file to create a conda environment with all the necessary dependencies.

1. Open a terminal or command prompt.
2. Navigate to the directory where your repository is located.
3. Run the following command to create the conda environment: `conda env create -f environment.yml`  
4. Activate the newly created environment: `conda activate sega_learn`

