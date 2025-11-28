Infer-Pred
==========

This repository accompanies the paper:

***Environmental dynamics shape human learning: change points versus
random walks***\
by **Cédric Foucault, Lilian A. Weber, and Laurence Hunt**\
(Preprint: [biorxiv.org/content/10.1101/2025.11.26.690700](https://www.biorxiv.org/content/10.1101/2025.11.26.690700)).

The repository includes everything needed to reproduce the results
presented in the paper, and:
-   the analysis code used for all figures and statistics
-   the Bayesian modeling framework introduced in the paper
-   the full task code used in the experiments
-   the behavioral data from both experiments

If you use any part of this repository for your research, please cite
the paper.

## Repository Structure


    analysis/             Analysis scripts and pipeline
                          (with a master script `run_all_main_results.py`
                          to reproduce all results from the paper in one go)

    modeling_framework/   Bayesian modeling framework (code + demos)
                          (will be shared as a standalone repository with full
                          documentation and tutorial notebooks upon publication)

    tasks/                Browser-based behavioral task 

    data/                 Experimental datasets
                            - change-point-env: Experiment 1
                            - random-walk-env: Experiment 2
    
    results_paper/        Pre-generated results included for convenience

## Installation

### (Optional) Create a conda environment

    conda create -n infer-pred
    conda activate infer-pred

Python **3.9 or higher** is recommended (earlier versions may work but
were not tested).

### Install required Python packages

    pip install -r requirements.txt

## Reproducing the results

From the root of the repository, run:

    python analysis/run_all_main_results.py

This will:

-   run all analyses to generate all figures and tables reported in the paper\
-   save outputs into the `results/` directory

You can also run individual analysis scripts in `analysis/` to reproduce
specific results.

For convenience, precomputed results are provided in `results_paper/`
(output of `run_all_main_results.py`).

## Running the task locally

To run the web-based task locally:

### 1. Start a local HTTP server

    cd tasks
    python -m http.server

### 2. Open your browser and navigate to:

-   **Change-point environment (Experiment 1)**

        http://localhost:8000/infer-pred-task.html?version=cp

-   **Random-walk environment (Experiment 2)**

        http://localhost:8000/infer-pred-task.html?version=rw

### Optional URL parameters

-   `subjectId=XXX` --- participant ID
-   `nBlocks=K` --- number of blocks (e.g. `nBlocks=2`)
-   `skipInstructions` --- skips task instructions (useful for testing the task
directly or debugging)

Parameters are combined using `&`. For example:

    http://localhost:8000/infer-pred-task.html?version=cp&subjectId=47&nBlocks=3&skipInstructions

to perform the change-point experiment as subject '47' with 3 blocks and skip the instructions.

See `tasks/urlParams.js` for the full list of configurable parameters.

## Running the task

You can run the task locally in your web browser.

In one terminal tab:

```
cd tasks
python -m http.server
```

Then, open your web browser (e.g. Google Chrome, Firefox) and navigate to
```
http://localhost:8000/infer-pred-task.html?version=cp
```

To run the task/experiment in the change-point environment

```
http://localhost:8000/infer-pred-task.html?version=rw
```

To run the task/experiment in the random-walk environment.

In addition to the version parameter (`version=cp` or `rw`), other parameters you can change through the URL are:
- `subjectId` (this is used to identify the participant in the recorded data)
- `nBlocks` (changes the number of blocks needed to complete the experiment, e.g. `nBlocks=2` to perform only two blocks)
- `skipInstructions` (no value - this is used for debugging/testing the task, to skip the instructions and go directly to the task blocks)
- Refer to `tasks/urlParams.js` for a more complete list of parameters

Different parameters are delimited by `&` in the URL — for example, if you want to perform the change-point experiment as subject '47' with 3 blocks and skip the instructions, you would run `http://localhost:8000/infer-pred-task.html?version=cp&subjectId=47&nBlocks=3&skipInstructions`

## Modeling framework

Several example use cases and demo scripts illustrating how the modeling
framework can be applied to environments and scenarios beyond those explored
in the paper are provided in the `modeling_framework` directory (e.g.,
perceptual discrimination/evidence accumulation tasks, probabilistic
reversal learning and other probability-learning tasks).

You can try the existing demos with:
```
python modeling_framework/use_case_all.py
```

Upon publication, the modeling framework will be released as a standalone
repository with full documentation, example notebooks, and additional
use cases and environment templates.
