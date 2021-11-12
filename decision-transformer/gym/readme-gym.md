
## Course Project Info

## Scripts
To run experiments

Naive evaluation: 
``` bash
CUDA_VISIBLE_DEVICES=0 python experiment.py --env kitchen-complete --dataset complete --model_savepath /proj/vondrick2/james/robotics/kitchen_complete/ -w /proj/vondrick2/james/robotics/logs

CUDA_VISIBLE_DEVICES=1 python experiment.py --env kitchen-mixed --dataset mixed --model_savepath /proj/vondrick2/james/robotics/kitchen_mixed/ -w /proj/vondrick2/james/robotics/logs

CUDA_VISIBLE_DEVICES=2 python experiment.py --env kitchen-partial --dataset partial --model_savepath /proj/vondrick2/james/robotics/kitchen_partial/ -w /proj/vondrick2/james/robotics/logs
```
# OpenAI Gym

## Installation

Experiments require MuJoCo.
Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install.
Then, dependencies can be installed with the following command:

```
conda env create -f conda_env.yml
```

## Downloading datasets

Datasets are stored in the `data` directory.
Install the [D4RL repo](https://github.com/rail-berkeley/d4rl), following the instructions there.
Then, run the following script in order to download the datasets and save them in our format:

```
python download_d4rl_datasets.py
```

## Example usage

Experiments can be reproduced with the following:

```
python experiment.py --env hopper --dataset medium --model_type dt
```

Adding `-w True` will log results to Weights and Biases.
