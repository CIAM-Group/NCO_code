# Copyright (c) 2023 CIAM Group
**The code can only be used for non-commercial purposes. Please contact the authors if you want to use this code for business matters.**  
**If this repository is helpful for your research, please cite our paper:<br />**
*"Zhenkun Wang,  Shunyu Yao, Genghui Li, Qingfu Zhang, Multi-objective Combinatorial Optimization Using A Single Deep Reinforcement Learning Model, IEEE Transactions on Cybernetics, in press, 2023."<br />*
**OR**
```
@article{wang2023multiobjective,
  title={Multiobjective Combinatorial Optimization Using a Single Deep Reinforcement Learning Model},
  author={Wang, Zhenkun and Yao, Shunyu and Li, Genghui and Zhang, Qingfu},
  journal={IEEE Transactions on Cybernetics},
  year={2023},
  publisher={IEEE}
}
```
****
This code is heavily based on the [attention-learn-to-route repository](https://github.com/wouterkool/attention-learn-to-route).

Note: All experiments (including training and testing) in this paper are conducted on 1 Tesla V100 GPU, except for MORAM, which is trained in parallel on 2 GPUs due to memory limitation. The statistical unit used for comparing training times is GPU hour.
