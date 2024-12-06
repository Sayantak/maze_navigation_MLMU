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

model_name = "mlmu"


#WARNING: these are created by mlmu_ood_gridn.py

path = f"../eval/{model_name}/10_via_20.pt"
maze10via20 = torch.load(path)
path = f"../eval/{model_name}/10_via_20_embedded.pt"
maze10via20emb = torch.load(path)
path = f"../eval/{model_name}/20.pt"
maze20 = torch.load(path)


# %%

import numpy as np
for ds, name in [(maze20, "20x20"), (maze10via20, "10x10 via 20x20 tokenizer"), (maze10via20emb, "10x10 embedded in 20x20 maze")]:
  accs = np.mean([out["metrics"]["val/acc"] for out in ds])
  full_accs = np.mean([out["metrics"]["val/full_acc"] for out in ds])
  print(f"{model_name} {name}: token acc {accs:.2f}, full path acc {full_accs:.2f}")


# %%
