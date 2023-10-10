# Copyright (c) 2023 CIAM Group
**The code can only be used for non-commercial purposes. Please contact the authors if you want to use this code for business matters.**  
**If this repository is helpful for your research, please cite our paper:<br />**
*"Fu Luo, Xi Lin, Fei Liu, Qingfu Zhang, and Zhenkun Wang, Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization, The 37th Anniversary Conference on Neural Information Processing Systems, NeurIPS 2023" <br />*

**OR**

```
@inproceedings{luo2023neural,
  title={Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization},
  author={Fu Luo, Xi Lin, Fei Liu, Qingfu Zhang, Zhenkun Wang},
  booktitle={The 37th Anniversary Conference on Neural Information Processing Systems, NeurIPS 2023},
  year={2023}
}
```
****
This code can develop an NCO model with a Light Encoder and a Heavy Decoder for solving TSP and CVRP. 

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
The training and test datasets can be downloaded from Google Drive:
```bash
https://drive.google.com/drive/folders/1LptBUGVxQlCZeWVxmCzUOf9WPlsqOROR?usp=sharing
```
or  Baidu Cloud:
```bash
https://pan.baidu.com/s/12uxjol_5pAlnm0j4F6D_RQ?pwd=rzja
```
- For TSP, download the training/testing datasets and put them to the path <LEHD_main/TSP/data>.

- For CVRP, download the training/testing datasets and put them to the path <LEHD_main/CVRP/data>.
See <LEHD_main/CVRP/Transform_data/Format_of_CVRP_datatset.md> for more details about the format of the CVRP dataset.


## Implementation

This project's structure is clear, the codes are based on .py files, and they should be easy to read, understand, and run.


## Acknowledgements
LEHD's code implementation is based on the code of [POMO](https://github.com/yd-kwon/POMO/tree/master/NEW_py_ver).
Thanks to them.
