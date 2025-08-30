# `README.md`

# A Digital Twin for "FX-constrained growth: Fundamentalists, chartists and the dynamic trade-multiplier"

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type Checking: mypy](https://img.shields.io/badge/type_checking-mypy-blue)](http://mypy-lang.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.02252-b31b1b.svg)](https://arxiv.org/abs/2508.02252)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/fx_constrained_growth)
[![Discipline](https://img.shields.io/badge/Discipline-Behavioral%20Finance%20%26%20Econometrics-blue)](https://github.com/chirindaopensource/fx_constrained_growth)
[![Research](https://img.shields.io/badge/Research-Macro--Financial%20Modeling-green)](https://github.com/chirindaopensource/fx_constrained_growth)
[![Methodology](https://img.shields.io/badge/Methodology-Heterogeneous%20Agent%20Model-orange)](https://github.com/chirindaopensource/fx_constrained_growth)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%23025596?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-1B62A8.svg?style=flat&logo=statsmodels&logoColor=white)](https://www.statsmodels.org/stable/index.html)
[![Joblib](https://img.shields.io/badge/Joblib-00A0B0.svg?style=flat)](https://joblib.readthedocs.io/en/latest/)

--

**Repository:** `https://github.com/chirindaopensource/fx_constrained_growth`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"FX-constrained growth: Fundamentalists, chartists and the dynamic trade-multiplier"** by:

*   Marwil J. Dávila-Fernández
*   Serena Sordi

The project provides a complete, end-to-end computational framework for creating a "digital twin" of the paper's findings. It delivers a modular, auditable, and extensible pipeline that replicates the study's entire workflow: from rigorous data validation and cleaning, through the complex Bayesian estimation of a time-varying parameter model, to the numerical simulation and analysis of the proposed heterogeneous agent model. The goal is to provide a transparent and robust toolkit for researchers in computational economics, behavioral finance, and development macroeconomics to replicate, validate, and build upon this important work.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: execute_digital_twin_replication](#key-callable-execute_digital_twin_replication)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the methodologies presented in the 2025 paper "FX-constrained growth: Fundamentalists, chartists and the dynamic trade-multiplier." The core of this repository is the iPython Notebook `fx_constrained_growth_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings, from initial data validation to the final execution of a full suite of robustness checks.

The paper addresses a critical gap in behavioral finance by building a model from the perspective of developing, FX-constrained economies. This codebase operationalizes that model, allowing users to:
-   Rigorously validate and cleanse the input macroeconomic and financial time-series data.
-   Execute a sophisticated Bayesian state-space model to estimate the time-varying trade multiplier.
-   Simulate the theoretical heterogeneous agent model (HAM) with fundamentalist, chartist, and trend-extrapolator agents.
-   Perform advanced numerical analysis of the model's nonlinear dynamics, including generating bifurcation diagrams and basins of attraction.
-   Quantitatively validate that the simulated model output reproduces the key "stylized facts" (e.g., fat-tailed distributions) observed in the empirical data.
-   Conduct a comprehensive suite of robustness checks to test the stability of the findings to changes in parameters, model specifications, and data samples.

## Theoretical Background

The implemented methods are grounded in development macroeconomics, behavioral finance, and nonlinear dynamics.

**1. The Dynamic Trade Multiplier (Thirlwall's Law):**
A core concept is the balance-of-payments-constrained growth rate, which posits that a country's long-run growth is determined by the growth of its exports relative to its income elasticity of demand for imports (`π`). The paper's key empirical task is to estimate a time-varying version of this elasticity.

$$
\Delta y_t^{BP} = \frac{\Delta z_t^T}{\pi_t}
$$

where `Δy_t^{BP}` is the trade-multiplier growth rate, `Δz_t^T` is the trend growth of exports, and `π_t` is the time-varying income elasticity of imports. This is estimated using a Bayesian state-space model.

**2. Heterogeneous Agent Model (HAM):**
The theoretical model populates the FX market with boundedly rational agents using simple heuristics:
-   **Fundamentalists:** Bet on the convergence of the exchange rate (`e`) to its fundamental value (`f`). Their demand is nonlinear: `Δd^F ∝ (f - e)³`.
-   **Chartists:** Bet on the persistence of deviations from the fundamental. Their demand is linear: `Δd^C ∝ (e - f)`.
-   **Trend-Extrapolators:** Bet on the continuation of recent trends: `Δd^E ∝ (e_{t-1} - e_{t-2})`.

**3. Market Clearing and System Dynamics:**
The paper's central innovation is to show that the trade multiplier emerges as a market-clearing condition in the FX market. The full dynamic system is a set of coupled nonlinear difference equations for the exchange rate (`e_t`) and output growth (`Δy_t`), which can produce complex dynamics, including multiple equilibria and chaos.

$$
e_t = F(e_{t-1}, \Delta y_{t-1}, \dots)
$$
$$
\Delta y_t = G(e_{t-1}, \Delta y_{t-1}, \dots)
$$

## Features

The provided iPython Notebook (`fx_constrained_growth_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Task-Based Architecture:** The entire pipeline is broken down into 7 distinct, modular tasks, from data validation to robustness analysis.
-   **Professional-Grade Data Validation:** A comprehensive validation suite ensures all inputs (data and configurations) conform to the required schema before execution.
-   **Auditable Data Cleansing:** A non-destructive cleansing process that handles missing values and outliers, returning a detailed log of all transformations.
-   **Advanced Bayesian Estimation:** A custom, multi-chain Gibbs Sampler with an embedded FFBS algorithm for time-varying parameter estimation.
-   **Rigorous MCMC Diagnostics:** Implements both Gelman-Rubin (`R-hat`) and Effective Sample Size (ESS) diagnostics to ensure chain convergence.
-   **Sophisticated Numerical Analysis:** A parallelized engine for generating bifurcation diagrams, basins of attraction, and quantifying chaos (LLE, Correlation Dimension).
-   **Comprehensive Robustness Suite:** A master orchestrator to systematically test the sensitivity of the results to changes in parameters (local and global), model specifications, and data samples.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Data Validation and Preparation (Task 1):** Ingests and rigorously validates all raw data and configuration files, then cleanses and transforms the data.
2.  **Empirical Stylized Facts (Task 2):** Establishes the non-normal, fat-tailed nature of empirical FX returns and growth deviations.
3.  **Theoretical Model Implementation (Task 3):** Provides the functional toolkit for the HAM.
4.  **Numerical Dynamics Analysis (Task 4):** Executes numerical experiments (bifurcations, basins of attraction, etc.).
5.  **Bayesian Econometric Estimation (Task 5):** Runs the full MCMC pipeline to estimate the time-varying trade multiplier.
6.  **Statistical Validation (Task 6):** Provides a general toolkit for distributional and time-series analysis, used for both empirical data and simulation output.
7.  **Orchestration and Robustness (Task 7):** Provides the master functions to run the entire pipeline and the full suite of robustness checks.

## Core Components (Notebook Structure)

The `fx_constrained_growth_draft.ipynb` notebook is structured as a logical pipeline with modular orchestrator functions for each of the 7 major tasks.

## Key Callable: execute_digital_twin_replication

The central function in this project is `execute_digital_twin_replication`. It orchestrates the entire analytical workflow, providing a single entry point for running the baseline study replication and the advanced robustness checks.

```python
def execute_digital_twin_replication(
    raw_macro_df: pd.DataFrame,
    raw_fx_df: pd.DataFrame,
    master_config: Dict[str, Any],
    analyses_to_run: List[str],
    output_filepath: str
) -> Dict[str, Any]:
    """
    Executes the complete, end-to-end digital twin research pipeline.
    """
    # ... (implementation is in the notebook)
```

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `scipy`, `statsmodels`, `joblib`, `tqdm`, `arch`, `ruptures`, `nolds`, `SALib`, `memory_profiler`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/fx_constrained_growth.git
    cd fx_constrained_growth
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy scipy statsmodels joblib tqdm arch ruptures nolds SALib memory_profiler ipython
    ```

## Input Data Structure

The pipeline requires two `pandas.DataFrame` objects with specific structures, which are rigorously validated by the first task.
1.  **`raw_macro_df`:** A DataFrame with a `MultiIndex` of `('country_iso', 'year')` and columns `['gdp_const_lcu', 'imports_const_lcu', 'exports_const_lcu', 'reer']`.
2.  **`raw_fx_df`:** A DataFrame with a `MultiIndex` of `('country_iso', 'date')` and a column `['fx_rate_usd']`.

A mock data generation script is provided in the main notebook to create valid example DataFrames for testing the pipeline.

## Usage

The `fx_constrained_growth_draft.ipynb` notebook provides a complete, step-by-step guide. The core workflow is:

1.  **Prepare Inputs:** Create your DataFrames or use the provided mock data generator. Define the `master_config` dictionary to control all aspects of the run.
2.  **Execute Pipeline:** Call the grand master orchestrator function.

    ```python
    # Define which major phases to run
    analyses_to_run = ['data_prep', 'empirical', 'theoretical', 'simulation_validation']

    # This single call runs the entire project.
    final_results = execute_digital_twin_replication(
        raw_macro_df=raw_macro_df,
        raw_fx_df=raw_fx_df,
        master_config=master_config,
        analyses_to_run=analyses_to_run,
        output_filepath="./project_outputs/final_results.joblib"
    )
    ```
3.  **Inspect Outputs:** Programmatically access any result from the returned dictionary. For example, to view the main empirical performance summary:
    ```python
    performance_summary = final_results['pipeline_run_results']['empirical_results'] \
        ['trade_multiplier_validation']['cross_country_summary']['performance_summary']
    print(performance_summary)
    ```

## Output Structure

The `execute_digital_twin_replication` function returns a single, comprehensive dictionary containing all generated artifacts, including:
-   `pipeline_run_results`: A dictionary containing all primary results (data prep reports, empirical analysis, theoretical simulations).
-   `performance_report`: A dictionary detailing the computational performance of the run.
-   `quality_assurance_report`: A high-level summary and cross-validation of key findings.
-   `robustness_analysis_report`: A dictionary containing the summary tables from each of the executed robustness checks.

The full results object is also saved to disk at the specified `output_filepath`.

## Project Structure

```
fx_constrained_growth/
│
├── fx_constrained_growth_draft.ipynb  # Main implementation notebook
├── requirements.txt                   # Python package dependencies
├── LICENSE                            # MIT license file
└── README.md                          # This documentation file
```

## Customization

The pipeline is highly customizable via the `master_config` dictionary. Users can easily modify all relevant parameters, such as MCMC settings, prior distributions, the definition of theoretical model scenarios, and the specific robustness checks to perform.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:

-   **Automated Report Generation:** Creating a function that takes the final `master_results` dictionary and generates a full PDF or HTML report summarizing the findings.
-   **Alternative Behavioral Heuristics:** Expanding the theoretical model to include other types of agent behavior (e.g., adaptive expectations, learning agents).
-   **Policy Experiments:** Using the validated model to conduct "what-if" scenario analysis, such as simulating the impact of capital controls or changes in export growth on an economy's stability.
-   **Integration with General Equilibrium Models:** Embedding the HAM FX market into a larger-scale DSGE or Agent-Based Model to explore deeper macroeconomic feedback loops.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{davila2025fx,
  title={{FX-constrained growth: Fundamentalists, chartists and the dynamic trade-multiplier}},
  author={D{\'a}vila-Fern{\'a}ndez, Marwil J and Sordi, Serena},
  journal={arXiv preprint arXiv:2508.02252},
  year={2025}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). A Digital Twin of "FX-constrained growth: Fundamentalists, chartists and the dynamic trade-multiplier". 
GitHub repository: https://github.com/chirindaopensource/fx_constrained_growth
```

## Acknowledgments

-   Credit to **Marwil J. Dávila-Fernández** and **Serena Sordi** for their insightful and clearly articulated research, which forms the entire basis for this computational replication.
-   This project stands on the shoulders of giants. Sincere thanks to the global open-source community and the developers of the scientific Python ecosystem, whose tireless work provides the foundational tools for modern computational science. Specifically, this implementation relies heavily on:
    -   **NumPy** and **SciPy** for foundational numerical computing and scientific algorithms.
    -   **Pandas** for its indispensable data structures and time-series manipulation capabilities.
    -   **Statsmodels** for its robust implementation of econometric methods, including time-series diagnostics and filtering.
    -   The **arch** library for its specialized, professional-grade tools for volatility modeling.
    -   **Joblib** for enabling efficient, straightforward parallel processing.
    -   **SALib** for providing a state-of-the-art framework for sensitivity analysis.
    -   The **ruptures** and **nolds** libraries for their powerful algorithms in change-point detection and nonlinear dynamics.
    -   The **Jupyter** and **IPython** projects for creating an unparalleled environment for interactive scientific development and literate programming.

--

*This README was generated based on the structure and content of `fx_constrained_growth_draft.ipynb` and follows best practices for research software documentation.*

