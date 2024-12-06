"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# %%
# import omegaconf
from recipe.analysis.run import Runs
import torch

# put the path to a log_dir here
path = "PATH_TO_LOGS"

model, config = Runs.load_model(path=path, resume_index=-1)
model = model.eval()
dataset = Runs.load_dataset(path=path)

# %%
# make this into a grid representation
def get_walls(grid_tokens):
    walls = []
    for i,tok in enumerate(grid_tokens):
        if tok == "wall":
            # get the x, y coordinates
            x = int(grid_tokens[i+1])
            y = int(grid_tokens[i+2])
            walls.append((x, y))
    return walls

def get_node(grid_tokens, name):
    for i,tok in enumerate(grid_tokens):
        if tok == name:
            # get the x, y coordinates
            x = int(grid_tokens[i+1])
            y = int(grid_tokens[i+2])
            return (x, y)
    raise ValueError(f"No {name} found")

def get_plan(grid_tokens):
    plan = []
    for i,tok in enumerate(grid_tokens):
      try:
        if tok == "plan":
            # get the x, y coordinates
            x = int(grid_tokens[i+1])
            y = int(grid_tokens[i+2])
            plan.append((x, y))
      except ValueError:
        continue
      except IndexError:
        continue
    return plan

def plot_maze(grid_tokens, grid_size, solution=None, identifier=None):
    import matplotlib.pyplot as plt
    walls = get_walls(grid_tokens)
    plan = get_plan(grid_tokens)
    start = get_node(grid_tokens, "start")
    goal = get_node(grid_tokens, "goal")
    plt.figure(figsize=(5, 5))
    if grid_size == 10:
        marker_size = 730
        spine_width = 6
    elif grid_size == 20:
        marker_size = 182
        spine_width = 12
    elif grid_size == 30:
        marker_size = 79
        spine_width = 16

    plt.scatter(*start, c="r", s=marker_size/6)
    plt.plot(*goal, "x", c="r", markersize=marker_size**.5/2)
    
    for wall in walls:
        plt.scatter(*wall, c="k", s=marker_size, marker="s")
    plt.plot(*zip(*plan), '--', c="r")
    if solution is not None:
        x = torch.tensor([p[0] for p in solution_path])
        y = torch.tensor([p[1] for p in solution_path])
        # make it with a matplotlib quiver
        plt.quiver(
            x[:-1],
            y[:-1],
            x[1:] - x[:-1],
            y[1:] - y[:-1],
            scale_units="xy",
            angles="xy",
            scale=1,
            color="orange",
            width=0.015 
        )
        
    # for s,e in zip(plan[:-1], plan[1:]):
    #     arrow = FancyArrowPatch(s, e, arrowstyle='-|>', mutation_scale=mutation_scale, color="b")
    #     plt.gca().add_patch(arrow)
        # plt.arrow(*s, e[0]-s[0], e[1]-s[1], head_width=0.5, head_length=0.5, fc='b', ec='b')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks([])
    plt.yticks([])
    for spine in plt.gca().spines.values():
      spine.set_linewidth(spine_width)

    plt.savefig(f"figures/{grid_size}x{grid_size}_astarmaze_failure_{identifier}.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

# %%
total_plots = 8
with torch.autocast('cuda', enabled = config.trainer.precision == "16-mixed"): # this makes a big difference!!
  for batch in dataset.val_dataloader()[0]:
    batch = {k: v.to(model.device) for k, v in batch.items()}
    # check that this line works
    out = dataset.eval_fn(model.model, batch, mode=model.eval_mode, return_samples=True)
    for i, per_sample_acc in enumerate(out["SAVE_per_sample_acc"]):
      print(per_sample_acc)
      if per_sample_acc < 1:
        grid_repr = dataset.tokenizer.decode(batch["input_ids"][i].tolist())
        solution_path = get_plan(out["SAVE_generated"][i])
        plot_maze(grid_repr, dataset.grid_n, solution_path, total_plots)
        total_plots -= 1
        if total_plots <= 0:
          break
    else:
      continue
    break

# %%
