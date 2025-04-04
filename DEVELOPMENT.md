# Development Guide for sega_learn

Welcome to the `sega_learn` project! This guide provides instructions for setting up your development environment, running tests, ensuring code quality with `ruff`, and contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Cloning the Repository](#cloning-the-repository)
  - [Setting Up the Environment](#setting-up-the-environment)
  - [Installing Dependencies](#installing-dependencies)
- [Code Style, Linting, and Formatting](#code-style-linting-and-formatting)
  - [Ruff Configuration](#ruff-configuration)
  - [Running Ruff Locally](#running-ruff-locally)
  - [Using Pre-commit Hooks](#using-pre-commit-hooks)
- [Running Tests](#running-tests)
- [Building Documentation](#building-documentation)
- [Contributing Process](#contributing-process)
- [Project Structure](#project-structure)

## Getting Started

### Prerequisites

-   Git
-   Python (See `pyproject.toml` or `environment.yml` for the required version)
-   An environment manager like Conda, venv, or uv.

### Cloning the Repository

```bash
git clone https://github.com/SantiagoEnriqueGA/sega_learn.git
cd sega_learn
```

### Setting Up the Environment

It's highly recommended to use a virtual environment to manage project dependencies.

**Using `venv` (Built-in):**

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**Using `conda`:**

```bash
# Create environment from file (if environment.yml exists)
conda env create -f environment.yml
conda activate sega_learn # Or the name defined in the file

# Or create a new environment manually
conda create --name sega_learn python=3.9 # Adjust Python version if needed
conda activate sega_learn
```

**Using `uv`:**

```bash
# uv automatically creates and manages virtual environments
# Ensure uv is installed: https://github.com/astral-sh/uv
# Creating an environment is often implicit with install commands
uv venv # Explicitly create if needed
# Activate (usually not needed if using uv commands directly)
# On Windows: .\.venv\Scripts\activate
# On macOS/Linux: source .venv/bin/activate
```

### Installing Dependencies

Install the required packages, including development dependencies (like `ruff`, `pytest`, `pre-commit`). The project uses `pyproject.toml`, so you can install using `pip` or `uv` with the `dev` extra.

**Using `pip`:**

```bash
# Install the package in editable mode along with dev dependencies
pip install -e ".[dev]"
```

**Using `uv`:**

```bash
# Install the package in editable mode along with dev dependencies
uv pip install -e ".[dev]"

# Or sync the environment based on pyproject.toml
# uv sync --dev # (This might be the preferred uv way)
```

**Using `conda`:**

```bash
# Ensure the environment is activated
conda activate sega_learn # Or your environment name

# If dev dependencies are listed in environment.yml, they should be installed already.
# If not, install them manually via pip or conda after activation:
# pip install ruff pytest pre-commit # Add other dev tools as needed
```

## Code Style, Linting, and Formatting

This project uses **`ruff`** for enforcing code style (formatting) and identifying potential issues (linting). This ensures consistency and helps catch errors early.

### Ruff Configuration

Configuration is managed in the `pyproject.toml` file under the `[tool.ruff]` section. This includes:
*   Target Python version.
*   Files/directories to include and exclude.
*   Selected lint rules (`[tool.ruff.lint]`).
*   Formatter settings (`[tool.ruff.format]`).

Please familiarize yourself with the configuration. Specific lint rules can be ignored if necessary, but this should be discussed with the maintainers.

### Running Ruff Locally

Before committing changes, please run `ruff` to check and format your code:

1.  **Check for Lint Errors:**
    ```bash
    ruff check .
    ```
2.  **Automatically Fix Lint Errors (where possible):**
    ```bash
    ruff check --fix .
    ```
3.  **Format Code:**
    ```bash
    ruff format .
    ```
4.  **Verify Formatting (Dry Run):** Ensure no formatting changes are needed.
    ```bash
    ruff format --check .
    ```

Ideally, both `ruff check .` and `ruff format --check .` should report no issues before you commit.

### Using Pre-commit Hooks

To automate these checks before each commit, we use `pre-commit`.

1.  **Install Hooks (One-time setup per clone):** After setting up your environment and installing dependencies (including `pre-commit`), run:
    ```bash
    pre-commit install
    ```
    This installs the git hooks defined in `.pre-commit-config.yaml`.

2.  **Workflow:** Now, whenever you run `git commit`:
    *   `pre-commit` will automatically run the configured hooks (including `ruff check --fix` and `ruff format`) on the files you've staged (`git add ...`).
    *   If `ruff format` modifies files, the commit will be aborted. You'll need to `git add` the reformatted files and commit again.
    *   If `ruff check --fix` modifies files, the commit will be aborted (due to `--exit-non-zero-on-fix`). Review the fixes, `git add` them, and commit again.
    *   If any unfixable errors are found, the commit will be aborted, and the errors will be printed. Fix them manually, `git add`, and commit again.
    *   If all checks pass, your commit will proceed.

3.  **Ensure Pre-commit Hooks Match `pyproject.toml`:**
    The `ruff` pre-commit hooks are configured to use the settings defined in `pyproject.toml`. By default, `ruff` automatically detects and uses this file. However, the `.pre-commit-config.yaml` explicitly specifies the configuration file to ensure consistency:
    ```yaml
    - id: ruff
        name: Run Ruff Linter (with auto-fix)
        args: [--fix, --exit-non-zero-on-fix] # Uses pyproject.toml by default
    - id: ruff-format
        name: Run Ruff Formatter
        args: [--config=pyproject.toml] # Explicitly specify config file (optional)
    ```
    This ensures that the pre-commit hooks use the same linting and formatting rules as defined in `pyproject.toml`.

4.  **Test Pre-commit Hooks Manually:**
    You can manually test the pre-commit hooks on all files by running:
    ```bash
    pre-commit run --all-files
    ```
    This will apply the hooks to all files in the repository and ensure they behave as expected.

Using `pre-commit` ensures that your code adheres to the project's style and quality standards before committing, reducing the chance of CI failures.
## Running Tests

The project contains unit tests in the `tests/` directory. Use the provided scripts to run them:

```bash
# Run all tests (imports, standard tests, examples)
python tests/run_all_tests.py

# Run only standard tests (excluding examples)
python tests/run_selected_tests.py

# Run only example file checks
python tests/run_all_examples.py

# Or run specific test files (if needed)
# python tests/test_linear_model.py
```

Please ensure all relevant tests pass before submitting a pull request. Add new tests for any new features or bug fixes.

## Building Documentation

Documentation is located in the `docs/` directory and appears to be generated using scripts. Based on the `scripts/` directory contents:

```bash
# Navigate to the scripts directory
cd scripts

# Run the PowerShell script to generate Markdown documentation
./documentation_md.ps1

# Or run the script to generate HTML documentation
./documentation_html.ps1

cd ..
```
*(Note: Ensure you have the necessary tools, like `pydoc-markdown`, installed as required by these scripts. Check script contents for details.)*

The documentation is generated from the docstrings in the code, so keep the docstrings updated with any changes you make.

## Contributing Process

1.  **Fork the Repository:**
2.  **Create a Branch:** Create a descriptive branch name from the `main` branch (e.g., `git checkout -b feat/add-new-model`).
3.  **Make Changes:** Implement your feature or bug fix.
4.  **Code Quality:** Run `ruff format .` and `ruff check --fix .` locally. Ensure pre-commit hooks pass when you commit.
5.  **Add Tests:** Write unit tests for your changes in the `tests/` directory.
6.  **Run Tests:** Ensure all tests pass using `python tests/run_all_tests.py`.
7.  **Update Documentation:** If necessary, update relevant documentation in `docs/` or add docstrings.
8.  **Commit Changes:** Write clear and concise commit messages.
9.  **Push Branch:** Push your branch to your fork or the main repository.
10. **Create Pull Request:** Open a pull request against the `main` branch of the `sega_learn` repository. Provide a clear description of your changes. Ensure CI checks (including the `ruff` GitHub Action) pass.

## Project Structure

```
sega_learn/
|
├── sega_learn/      # Main library source code
│   ├── auto/               # Automated model selection
│   ├── clustering/         # Clustering algorithms
│   ├── linear_models/      # Linear models
│   ├── nearest_neighbors/  # K-Nearest Neighbors
│   ├── neural_networks/    # Neural Network components
│   ├── svm/                # Support Vector Machines
│   ├── trees/              # Tree-based models
│   └── utils/              # Utility functions (metrics, data prep, etc.)
|
├── examples/           # Usage examples for different modules
|
├── tests/              # Unit tests for the library code
|
├── tests_performance/  # Performance benchmark tests
|
├── docs/               # Documentation source files (e.g., Markdown)
|
├── scripts/            # Helper scripts (e.g., for building docs, environment setup)
|
├── .github/            # GitHub specific files (workflows for CI)
│   └── workflows/
│       └── lint-format.yml # Ruff CI check workflow
|
├── .gitignore              # Files/directories ignored by Git
├── .pre-commit-config.yaml # Configuration for pre-commit hooks
├── DEVELOPMENT.md          # This file: Guide for developers
├── environment.yml         # Conda environment definition
├── pyproject.toml          # Project metadata, build config, Ruff config
├── README.md               # Main project README for users
└── uv.lock                 # Lock file for uv
```

Thank you for contributing!
