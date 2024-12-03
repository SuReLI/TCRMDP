# TCRMDP
Official implementation of Time-Constrained Robust MDP, NeurIPS 2024. 


## Installation
All the environment is available in the `RRLS` package. To install the package, run the following commands:
```bash
git clone https://github.com/SuReLI/RRLS
cd RRLS 
pip install -e .
```
Then go to TCRM folder and install the requirements:
```bash
cd TCRMDP
pip install -r requirements.txt
```

### How to run the code
You have multiple entry points to run the code. One per algorithm. For example,
to run the Stack TC m2t3d algorithm, you can run the following command:
```bash
python main_stacked_tc_m2td3.py --help
```
## Codebase
```bash
.
└── src 
    ├── evaluation.py # Evaluation script
    ├── __init__.py
    ├── m2td3 # This a the official implementation of M2TD3 from tanabe et al. 2022
    │   ├── agent_wrapper.py
    │   ├── algo.py
    │   ├── factory.py
    │   ├── __init__.py
    │   ├── trainer.py
    │   └── utils.py
    ├── main_dr.py # Entry point for the DR algorithm
    ├── main_m2td3.py # Entry point for the M2TD3 algorithm
    ├── main_oracle_tc_m2td3.py # Entry point for the Oracle TC M2TD3 algorithm
    ├── main_oracle_tc_rarl.py # Entry point for the Oracle TC RARL algorithm
    ├── main_rarl.py # Entry point for the RARL algorithm
    ├── main_stacked_tc_m2td3.py # Entry point for the Stacked TC M2TD3 algorithm
    ├── main_stacked_tc_rarl.py # Entry point for the Stacked TC RARL algorithm
    ├── main_tc_adversary.py # Entry point for train a TC adversary on a trained agent
    ├── main_vanilla.py # Entry point for the TD3 algorithm
    ├── main_vanilla_tc_m2td3.py # Entry point for the TC M2TD3 algorithm
    ├── main_vanilla_tc_rarl.py # Entry point for the TC RARL algorithm
    ├── mock_agent.py
    ├── scheduler.py # Fixed adversary scheduler in the paper
    ├── tc_mdp.py # This is where the magic happens, the implementation of the TCRMDP
    ├── td3
    │   ├── buffer.py
    │   ├── __init__.py
    │   ├── models.py
    │   ├── td3.py
    │   └── trainer.py
    └── utils.py
```
## Citation
```
@inproceedings{
zouitine2024timeconstrained,
title={Time-Constrained Robust {MDP}s},
author={Adil Zouitine and David Bertoin and Pierre Clavier and Matthieu Geist and Emmanuel Rachelson},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=NKpPnb3YNg}
}
```
