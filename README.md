# Code for training MLM-U on mazes (put arxiv link and website here)

## Installation

1. Install PyTorch
2. `pip install pytest submitit hydra-core hydra-submitit-launcher loguru tqdm gitpython transformers lightning matplotlib datasets sortedcontainers maze-dataset pymongo numpy`
3. `pip install -e .`
### If you want to run A* mazes (from https://github.com/facebookresearch/searchformer/)
4. Install mongodb
5. Download maze.gz and maze.vocabulary.gz from https://github.com/facebookresearch/searchformer/blob/main/doc/mongodb.md
6. add those to your mongodb   
`mongorestore --gzip --archive=maze.gz`  
`mongorestore --gzip --archive=maze.vocabulary.gz`

adjust locations: search for "TODO" and you will find them:
1. main.py --> code snapshot dir
2. train_defaults.yaml --> logs dir
3. train_defaults.yaml --> data dir

## Run next token (AR) Baseline
Locally

`python main.py -m mode=local model=gpt dataset=maze datamodule.grid_n=4`

* `use_wandb=False` or True to enable or disable debugging 

## Run MLM-U 

`python main.py -m mode=local model=past dataset=maze datamodule.grid_n=4`

PAST is an encoder-decoder model that runs best with mlm-u (model.train_mode=absorbing). GPT is the best model for AR (left to right next token prediction)


## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
This project is Apache 2.0 licensed, as found in the LICENSE file.
