# Graph Construction for Electrical Power Transmissions Networks (GC4EPTN)
Source repo for *Data-Driven Graph Construction of Power Flow Graphs for Electric Power Transmission Networks*. 

For the [online appendix](https://github.com/RL-BCI-Lab/gc4eptn/blob/master/online-appendix.pdf) containing additional results, see the `online-appendix.pdf` file.

<!-- ```
@inproceedings{gc4eptn,
    title={Data-Driven Graph Construction of Power Flow Graphs for Electric Power Transmission Networks},
    author={},
    booktitle={International Conference on Machine Learning and Applications},
    year={2024},
}
``` -->

## Environment
To create a a conda environment, follow the below instructions.

1) To install the conda environment simply run the below command from the root of the repository:

    ```
    conda env create --file env.yaml
    ```

2) Once complete you can activate the conda environment using `conda activate gc4etpn` and you can deactivate using `conda deactivate` 

3) Next, you will need to install the `gc4eptn` repo. First make sure to have activated the `gc4eptn` conda environment. Then, run the following command from the root of the repository:

    ```
    pip install -e .
    ```

4) Finally, to run the GGM algorithms the required R (e.g., GGMncv) packages will need to be installed, do so by running the following command. You should be prompt to select and install server before the install begins.

    ```
    python scripts/install_r.py
    ```

## Structure

Below will be a brief description of how this repository is structured.

- `notebooks/`: Jupyter notebooks for running code and replicating paper results.
- `exps/`: Storage location for all results produced by notebooks.
- `gc4epttn.`: This contains all general code for the GGM algorithms, running experiments, plotting,
metrics, utilities, etc.
    - `gc4eptn.dataloaders`: Contains classes for loading the real-time data simulation (RTDS) and MATPOWER data. 
    - `gc4eptn.ggm`" Contains code for running GGM algorithms and their experiments.
    - `gc4eptn.pngs`" Contains code for running the power network graph score (PNGS) algorithm and experiments for running it with GGMs.
    - `gc4eptn.gsp`: Contains code for running a basic GPS for graph construction algorithm.
    - `gc4eptn.utils`: Contains utility code from plotting, to metrics, to normalization, and more.
- `datasets`: Contains all the datasets that can be loaded using gc4eptn. Currently only supports the RTDS Kundur’s two-area, four-machine system (as in paper) and two MATPOWER systems with no current data (case9 and case14).
- `scripts`: Scripts for managing experiments, installing R packages, and running MATPOWER.

## Running the Code

All code execution is done through Jupyter Notebooks. See the `notebooks/` directory for the set of notebooks which run various aspects of the code. A brief description of each notebook is given below.

- Primary Notebooks for Replicating Paper Results
    - `data-loading-rtds-v5`: Display examples of how to load the RTDS data.
    - `data-loading-matpower`: Displays examples of how to load the MATPOWER data.
    - `kernel-analysis`: Runs experiments that visualizes various different kernels. Useful to validate kernels before being used for GGM algorithms. 
    - `ggm`: Runs experiments for network graph prediction using various GGM algorithms. Useful for validating just the network graph estimation part before refinement with PNGS. 
    - `pngs-single-exp`: Runs a single experiment for the PNGS algorithm from network estimation using GGMs to graph refinement for flow graph estimation.
    - `pngs-multi-exp`: Runs a multiple experiments using various different parameters for the PNGS algorithm. Automated version of running different variations of experiments in `pngs-single-exp.ipynb`.
- Other Secondary Notebooks
    - `gsp`: Runs experiments for network graph prediction using a GSP algorithm.
    - `gsp-syn-test`: Tests the GSP algorithm using synthetic data.
    - `norm-tests`: Tests various normalization effects on RTDS and MATPOWER data. Specifically used to observer effect of feature-wise normalization. 
    - `pngs-synthetic-test`: Test PNGS algorithm assuming fully connected network graph estimation. Allows for testing graph refinement into flow graph when using worst case scenario for network estimation. 


