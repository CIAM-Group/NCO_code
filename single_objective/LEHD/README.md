# Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization

We have developed a model with a Light Encoder and a Heavy Decoder 
for learning to solve TSP and CVRP. 
Our paper has been accepted as a poster at NeurIPS 2023.

## Dependencies
```bash
Python=3.8.6
torch==1.12.1
numpy==1.23.3
matplotlib==3.5.2
tqdm==4.64.1
pytz==2022.1
```

We don't use any hard-to-install packages. 
If any package is missing, just install it following the prompts.

## Download the datasets
Here is the download link for the TSP/CVRP training/testing data:
```bash
https://www.aliyundrive.com/s/CgG4fxY8vWK
```
- For TSP, download the training/testing datasets and put them to the path <LEHD_main/TSP/data>.

- For CVRP, download the training/testing datasets and put them to the path <LEHD_main/CVRP/data>.
See <LEHD_main/CVRP/Transform_data/Format_of_CVRP_datatset.md> for more details about the format of the CVRP dataset.


## Implementation

This project's structure is clear, the codes are based on .py files, and they should be easy to read, understand, and run.


## Acknowledgements
LEHD's code implementation is based on the code of [POMO](https://github.com/yd-kwon/POMO/tree/master/NEW_py_ver).
Thanks to them.
