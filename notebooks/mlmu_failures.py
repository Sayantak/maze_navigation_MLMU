"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# %%
from recipe.analysis.run import Runs
import torch

# put the path to a log_dir here
path = "PATH_TO_LOGS"
model_name = "mlmu"

model, config = Runs.load_model(path=path, resume_index=-1)
model = model.eval()
dataset = Runs.load_dataset(path=path)

# %%
# replicate the tokenization process to revert it
# from maze.py (find where as_tokens is used)
def maze_to_tokens(maze):
  from maze_dataset.tokenization import MazeTokenizer, TokenizationMode
  maze_tokenizer = MazeTokenizer(
      tokenization_mode=TokenizationMode.AOTP_UT_rasterized,
      max_grid_size=max(100, dataset.grid_n),
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

# %%
from maze_dataset.plotting import MazePlot
import matplotlib.pyplot as plt

n_plots = 5

for batch_i in range(5, 100):
  with torch.autocast('cuda', enabled = config.trainer.precision == "16-mixed"): # this makes a big difference!!
    for i, batch in enumerate(dataset.val_dataloader()[0]):
      if i < batch_i:
        continue
      batch = {k: v.to(model.device) for k, v in batch.items()}
      # check that this line works
      out = dataset.eval_fn(model.model, batch, mode=model.eval_mode, return_samples=True)
      print(out)
      break

  print("what the model sees:", out["_prefix_ids"])
  print("the associated attn:", out["_prefix_attn"])
  prefix_decoded = out["SAVE_truth"]
  print("Here it is for human eyes:", *prefix_decoded, sep="\n")

  ans_decoded = out["SAVE_generated"]
  tgt_decoded = out["SAVE_truth"]
  print("model generated outputs:", ans_decoded)

  # let's compare outputs and targets in human readable format
  print("zipped tgt and ans:", *map(lambda x: "\n".join(x), zip(tgt_decoded, ans_decoded)), sep="\n\n")

  for i in range(dataset.batch_size):
    if out["SAVE_per_sample_acc"][i] != 1 and n_plots > 0:
      tgt_maze = make_solvedmaze_from_tokens(tgt_decoded[i], dataset.grid_n)
      ans_maze = make_solvedmaze_from_tokens(ans_decoded[i], dataset.grid_n)
      plot = MazePlot(tgt_maze)
      plot.add_predicted_path(ans_maze.solution)
      plot.plot()
      # remove y ticks and x ticks
      plt.xticks([])
      plt.yticks([])
      plt.xlabel("")
      plt.ylabel("")
      plt.tight_layout()
      plt.savefig(f"figures/maze_{model_name}_errors_{n_plots}.pdf")
      plt.show()
      n_plots -= 1
  if n_plots == 0:
    # we made enough plots
    break

# %%
