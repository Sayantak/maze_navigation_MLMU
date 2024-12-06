"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# %%
from recipe.analysis.run import Runs
import torch
from maze_dataset.plotting import MazePlot
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# put the path to a log_dir here
path = "PATH_TO_LOGS"
model_name = "mlmu" # or gpt


model, config = Runs.load_model(path=path, resume_index=-1)
model = model.eval()
ds20 = Runs.load_dataset(path=path)
ds10 = Runs.load_dataset(path=path, update_conf = {"datamodule" : {"grid_n" : 10, "tokenizer_grid_n": 20}})

# %%
# replicate the tokenization process to revert it
# from maze.py (find where as_tokens is used)
def maze_to_tokens(maze, grid_n):
  from maze_dataset.tokenization import MazeTokenizer, TokenizationMode
  maze_tokenizer = MazeTokenizer(
      tokenization_mode=TokenizationMode.AOTP_UT_rasterized,
      max_grid_size=max(100, grid_n),
  )
  return " ".join([word for word in maze.as_tokens(maze_tokenizer) if word != "<-->"])

def tokens_to_coord(coord_str):
  coord_str = coord_str.strip("()").split(",")
  return tuple(map(int, coord_str))

def edge_to_cntlist_idx(edge):
  # edge is a string like '(2,0) (2,1)'
  # we need to return an index to set to true in the connection list
  # the connection list has dimensions [2, dim, dim]
  # following this logic:
  # if there is a downward connection, [0, x, y] = True, where x y is from the top node
  # if there is a rightward connection, [1, x, y] = True, where x y is from the left node

  # 1. extract coordinates
  edge = edge.split(" ")
  coord1 = tokens_to_coord(edge[0])
  coord2 = tokens_to_coord(edge[1])
  
  #2. sort coordinates to make sure we have the top/left node first
  coords = sorted([coord1, coord2])
  #3. horizontal or vertical connection?
  if coords[0][0] == coords[1][0]:
    # vertical
    return (1,) + coords[0]
  else:
    # horizontal
    return (0,) + coords[0]

def extract_cnctlist_from_tokens(tokens, dim):
  import numpy as np

  start_str = "<ADJLIST_START> "
  idx_start = tokens.index(start_str)
  end_str = " ; <ADJLIST_END>"
  idx_end = tokens.index(end_str)
  tokens = tokens[idx_start + len(start_str):idx_end]
  adjlist = tokens.split(" ; ")
  edge_idxs = [edge_to_cntlist_idx(i) for i in adjlist]
  cnct_list = np.zeros((2, dim, dim), dtype=bool)
  for idx in edge_idxs:
    cnct_list[idx] = True
  return cnct_list

def extract_path_from_tokens(tokens):
  import numpy as np

  start_str = "<PATH_START> "
  idx_start = tokens.index(start_str)
  end_str = " <PATH_END>"
  try:
    idx_end = tokens.index(end_str)
  except ValueError:
    idx_end = len(tokens)
  tokens = tokens[idx_start + len(start_str):idx_end]
  path = tokens.split(" ")
  return np.array([tokens_to_coord(i) for i in path])

def extract_origin_from_tokens(tokens):
  start_str = "<ORIGIN_START> "
  idx_start = tokens.index(start_str)
  end_str = " <ORIGIN_END>"
  idx_end = tokens.index(end_str)
  tokens = tokens[idx_start + len(start_str):idx_end]
  return tokens_to_coord(tokens)

def extract_target_from_tokens(tokens):
  start_str = "<TARGET_START> "
  idx_start = tokens.index(start_str)
  end_str = " <TARGET_END>"
  idx_end = tokens.index(end_str)
  tokens = tokens[idx_start + len(start_str):idx_end]
  return tokens_to_coord(tokens)

def make_solvedmaze_from_tokens(tokens, dim):
  from maze_dataset.maze import SolvedMaze

  connection_list = extract_cnctlist_from_tokens(tokens, dim)
  solution = extract_path_from_tokens(tokens)
  origin = extract_origin_from_tokens(tokens)
  target = extract_target_from_tokens(tokens)
  return SolvedMaze(connection_list=connection_list, solution=solution, start_pos=origin, end_pos=target, allow_invalid=True)

def load_smaller_into_larger_maze(dssmall, dslarge):
    for batchsmall, batchlarge in zip(dssmall.val_dataloader()[0], dslarge.val_dataloader()[0]):
      decoded_larges = dslarge.tokenizer.batch_decode(batchlarge["input_ids"], dslarge.grid_n)
      decoded_smalls = dslarge.tokenizer.batch_decode(batchsmall["input_ids"], dssmall.grid_n)
      small_as_large_tokens = []
      for decoded_large, decoded_small in zip(decoded_larges, decoded_smalls):
        cncts_large = extract_cnctlist_from_tokens(decoded_large, dim=dslarge.grid_n)
        cncts_small = extract_cnctlist_from_tokens(decoded_small, dim=dssmall.grid_n)
        # insert small maze connections into larger one
        cncts_large[:,:cncts_small.shape[1],:cncts_small.shape[2]] = cncts_small
        # get solution and origin/target positions
        solution = extract_path_from_tokens(decoded_small)
        origin = extract_origin_from_tokens(decoded_small)
        target = extract_target_from_tokens(decoded_small)
        # create new maze
        from maze_dataset.maze import SolvedMaze
        maze_small_as_large = SolvedMaze(connection_list=cncts_large, solution=solution, start_pos=origin, end_pos=target, allow_invalid=True)
        # convert back to tokens
        small_as_large_tokens.append(maze_to_tokens(maze_small_as_large, dslarge.grid_n))
    
      # return batch of smaller mazes, but as if they were larger ones
      yield dslarge.collate_fn(dslarge.tokenizer.batch_encode(small_as_large_tokens))
      

def eval_model(model, loader, eval_ds, path=None):
  outs = []
  with torch.autocast('cuda', enabled = config.trainer.precision == "16-mixed"): # this makes a big difference!!
    for batch in tqdm(loader):
      batch = {k: v.to(model.device) for k, v in batch.items()}
      # check that this line works
      out = eval_ds.eval_fn(model.model, batch, mode=model.eval_mode, return_samples=True)
      for k in out.keys():
        if isinstance(out[k], torch.Tensor):
          out[k] = out[k].detach().cpu()
      outs.append(out)
  if path is not None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(outs, path)
  return outs

path = f"eval/{model_name}/10_via_20.pt"
eval_model(model, loader=ds10.val_dataloader()[0], eval_ds=ds20, path=path)
path = f"eval/{model_name}/10_via_20_embedded.pt"
eval_model(model, loader=load_smaller_into_larger_maze(ds10, ds20), eval_ds=ds20, path=path)
path = f"eval/{model_name}/20.pt"
eval_model(model, loader=ds20.val_dataloader()[0], eval_ds=ds20, path=path)
