# Code for Paper "Transformers can navigate mazes with Multi-Step Prediction"

[![arXiv](https://img.shields.io/badge/arXiv-2412.05117-red.svg)](https://arxiv.org/abs/2412.05117)
[![deploy](https://img.shields.io/badge/Website%20%20-8A2BE2)](https://facebookresearch.github.io/maze_navigation_MLMU/)

## Installation

1. Install PyTorch
2. `pip install pytest submitit hydra-core hydra-submitit-launcher loguru tqdm gitpython transformers lightning matplotlib datasets sortedcontainers maze-dataset pymongo numpy maze-dataset`
### (OPTIONAL) If you want to run A* mazes (from https://github.com/facebookresearch/searchformer/)

NOTE: This is optional. We do not use this.

3. Install mongodb
4. Download maze.gz and maze.vocabulary.gz from https://github.com/facebookresearch/searchformer/blob/main/doc/mongodb.md
5. add those to your mongodb   
`mongorestore --gzip --archive=maze.gz`  
`mongorestore --gzip --archive=maze.vocabulary.gz`

### (REQUIRED) Adjusting the paths

adjust locations: search for "TODO" and you will find them:
1. main.py --> code snapshot dir
2. train_defaults.yaml --> logs dir
3. train_defaults.yaml --> data dir

## Run next token (AR) Baseline
Locally

`python main.py -m mode=local model=gpt dataset=maze datamodule.grid_n=4`

* `use_wandb=False` or True to enable or disable debugging 

## Run MLM-U (optional)

NOTE: We do not use this. We compare to the next token baseline.

`python main.py -m mode=local model=past dataset=maze datamodule.grid_n=4`

PAST is an encoder-decoder model that runs best with mlm-u (model.train_mode=absorbing). GPT is the best model for AR (left to right next token prediction)


## Run with a reasoning architecture

In our work, we propose a reasoning architecture that is learned as an adapter on top of the pretrained AR baseline.

Hence, assume that you have already run the AR baseline which resulted in a .ckpt file. You need to specify the checkpoint path in the train_from_pretrained.yaml file.
The run:

`python main.py -cn train_from_pretrained -m mode=local model=gpt dataset=maze datamodule.grid_n=4 use_wandb=True`


## Code modifications for reasoning architecture

Compared to the original code which implemented the AR baseline and the MLM-U model, we made the following modifications:

1. Adjusted the gpt_wrapper.py file to inherit from a custom_gpt2.py implementation that adds a small adapter in each layer of the transformer.
2. Added planning.py file with several implementations of planner modules. We are interested in the `TokenPlanner` class, which takes care of calling the baseline AR model to predict continuations of the current prompt first, then plugs it into an adapter module to reduce the generated continuations to a single vector.
3. Modified the forward pass of the GPT2PL class in pl_model.py such that "plan" vectors are generated at random positions in the input first and then passed into the baseline AR model with added adapter modules.

TODOs:
- Currently, finetuning the plan adapter on top of the AR baseline doesn't meaningfully improve performance in terms of validation loss. We need to investigate why this is the case.
- Once validation loss improves meaningfully, modify the gpt_wrapper.py file such that it can take into account generated plans during inference. This is needed for accuracy evaluation.

## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
This project is Apache 2.0 licensed, as found in the LICENSE file.

The stargraph dataset has been adapted from https://github.com/gregorbachmann/Next-Token-Failures/
