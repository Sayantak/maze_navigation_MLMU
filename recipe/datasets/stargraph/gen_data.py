"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from ...utils import get_data_root
import os


def star_graph(degSource, pathLen, numNodes, reverse=False):
    source = np.random.randint(0, numNodes, 1)[0]
    goal = np.random.randint(0, numNodes, 1)[0]
    while goal == source:
        goal = np.random.randint(0, numNodes, 1)[0]

    path = [source]
    edge_list = []

    # Choose random nodes along the path
    for _ in range(pathLen - 2):
        node = np.random.randint(0, numNodes, 1)[0]
        while node in path or node == goal:
            node = np.random.randint(0, numNodes, 1)[0]
        path.append(node)

    path.append(goal)
    # Connect the path
    for i in range(len(path) - 1):
        edge_list.append([path[i], path[i + 1]])

    remaining_nodes = []
    for i in range(numNodes):
        if i not in path:
            remaining_nodes.append(i)

    i = 0
    deg_nodes = set()
    while i < degSource - 1:
        # Add neighbour to source
        node = source
        next_node = np.random.randint(0, numNodes, 1)[0]
        l = 1
        while l < pathLen:
            if next_node not in deg_nodes and next_node not in path:
                edge_list.append([node, next_node])
                deg_nodes.add(next_node)
                node = next_node
                l += 1
            next_node = np.random.randint(0, numNodes, 1)[0]

        i += 1

    random.shuffle(edge_list)
    if reverse:
        path = path[::-1]

    return path, edge_list, source, goal


def generate_and_save(n_train, n_test, degSource, pathLen, numNodes, reverse=False):
    """
    Generate a list of train and testing graphs and save them for reproducibility
    """
    data_root = get_data_root()
    file = open(
        os.path.join(
            data_root,
            "graphs",
            "deg_"
            + str(degSource)
            + "_path_"
            + str(pathLen)
            + "_nodes_"
            + str(numNodes)
            + "_train_"
            + str(n_train)
            + ".txt",
        ),
        "w",
    )

    for i in range(n_train):
        path, edge_list, start, goal = star_graph(
            degSource, pathLen, numNodes, reverse=reverse
        )
        path_str = ""
        for node in path:
            path_str += str(node) + ","
        path_str = path_str[:-1]

        edge_str = ""
        for e in edge_list:
            edge_str += str(e[0]) + "," + str(e[1]) + "|"
        edge_str = edge_str[:-1]
        edge_str += "/" + str(start) + "," + str(goal) + "="

        out = edge_str + path_str
        file.write(out + "\n")
    file.close()

    file = open(
        os.path.join(
            data_root,
            "graphs",
            "deg_"
            + str(degSource)
            + "_path_"
            + str(pathLen)
            + "_nodes_"
            + str(numNodes)
            + "_test_"
            + str(n_test)
            + ".txt",
        ),
        "w",
    )

    for i in range(n_test):
        path, edge_list, start, goal = star_graph(
            degSource, pathLen, numNodes, reverse=reverse
        )
        path_str = ""
        for node in path:
            path_str += str(node) + ","
        path_str = path_str[:-1]

        edge_str = ""
        for e in edge_list:
            edge_str += str(e[0]) + "," + str(e[1]) + "|"
        edge_str = edge_str[:-1]
        edge_str += "/" + str(start) + "," + str(goal) + "="

        out = edge_str + path_str
        file.write(out + "\n")

    file.close()


def prefix_target_list(filename=None, reverse=False):
    """
    Load graphs and split them into prefix and target and return the list
    """
    data_list = []
    with open(filename, "r") as f:
        lines = f.readlines()
    for line in lines:
        prefix = line.strip().split("=")[0] + "="
        target = line.strip().split("=")[1]
        if reverse:
            target = ",".join(target.split(",")[::-1])
        data_list.append((prefix, target))

    return data_list


class Graphs(Dataset):
    def __init__(
        self,
        tokenizer,
        n_samples,
        data_path,
        device,
        eval=False,
        teacherless_token=None,
        reverse=False,
    ):
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.device = device
        self.eval_mode = eval
        self.data_path = data_path
        self.teacherless_token = teacherless_token
        self.reverse = reverse

        self.data_file = prefix_target_list(self.data_path, reverse=reverse)[:n_samples]
        self.tokenized, self.num_prefix_tokens, self.num_target_tokens = (
            tokenizer.tokenize(self.data_file)
        )

        self.num_tokens = self.num_prefix_tokens + self.num_target_tokens

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        if self.eval_mode:
            # In eval mode return the entire sequence
            return self.tokenized[idx].to(self.device)

        # Create inputs
        x = self.tokenized[idx][:-1].clone()
        if self.teacherless_token is not None:
            x[self.num_prefix_tokens :] = self.teacherless_token
            x = x.to(self.device)
        # Create targets in the form [-1, ..., -1, 4, 7, 9, 2, ...] where we replace the prefix tokens by -1 so that
        # we can skip their gradient calculation in the loss (double-check if that's correct)
        y = torch.cat(
            [
                -torch.ones((self.num_prefix_tokens - 1,)),
                self.tokenized[idx][self.num_prefix_tokens :].clone(),
            ]
        )

        return x.to(self.device), y.long().to(self.device)

    def eval(self):
        # Switch to "eval" mode when generating sequences without teacher-forcing
        self.eval_mode = True

    def train(self):
        # Switch back to "train" mode for teacher-forcing
        self.eval_mode = False


def get_edge_list(x, num_nodes, path_len):
    """
    Given the tokenised input for the Transformer, map back to the edge_list
    """
    edge_list = []
    pair = []
    x = x.squeeze().cpu().numpy()

    for i, n in enumerate(x):
        if n in range(num_nodes):
            pair.append(n)
        if len(pair) == 2:
            edge_list.append(pair)
            pair = []
        if n == num_nodes + 2:
            break

    start = x[i + 1]
    goal = x[i + 2]
    path = [x[i + j] for j in range(4, 4 + path_len)]

    return edge_list, start, goal, path


def get_edge_list_byte(x, num_nodes, path_len, decode):
    """
    Given the tokenised input for the Transformer, map back to the edge_list
    """
    edge_list = []
    x = list(x.squeeze().cpu().numpy())
    dec = [decode([val]) for val in x]
    edge = []
    for i, val in enumerate(dec):
        if val not in [",", "|", "=", "->"]:
            edge.append(val)
        if len(edge) == 2:
            edge_list.append(edge)
            edge = []

        if val == "->":
            break
    i += 2
    start = dec[i + 1]
    goal = dec[i - 1]
    path = [dec[i + 3 + 2 * j] for j in range(0, path_len - 2)]

    return edge_list, start, goal, path


def get_dataset(args, tokenizer, device):
    if args.teacherless and tokenizer.name == "numeral":
        teacherless_token = tokenizer.encode("$")[0]
    elif args.teacherless:
        teacherless_token = tokenizer.encode("$")[0]
    else:
        teacherless_token = None

    if args.dataset == "chess":
        raise NotImplementedError

    elif args.dataset == "graph":
        data_path = os.path.join(
            get_data_root(),
            "graphs/deg_"
            + str(args.deg)
            + "_path_"
            + str(args.path_len)
            + "_nodes_"
            + str(args.num_nodes),
        )
        train_path, test_path = (
            data_path + "_train_200000.txt",
            data_path + "_test_20000.txt",
        )
        train_data = Graphs(
            tokenizer=tokenizer,
            n_samples=args.n_train,
            data_path=train_path,
            device=device,
            teacherless_token=teacherless_token,
            reverse=args.reverse,
        )
        test_data = Graphs(
            tokenizer=tokenizer,
            n_samples=args.n_test,
            data_path=test_path,
            device=device,
            teacherless_token=teacherless_token,
            reverse=args.reverse,
        )

    return train_data, test_data


import torch
from transformers import AutoTokenizer

numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


class NumeralTokenizer:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        # Define encoder and decoder as a dictionary
        self.encoder = {str(i): i for i in range(num_nodes)}
        self.encoder["|"] = num_nodes
        self.encoder["="] = num_nodes + 1
        self.encoder["/"] = num_nodes + 2
        self.encoder["$"] = num_nodes + 3

        self.decoder = {i: i for i in range(num_nodes)}
        self.decoder[num_nodes] = "|"
        self.decoder[num_nodes + 1] = "="
        self.decoder[num_nodes + 2] = "/"
        self.decoder[num_nodes + 3] = "$"
        self.decoder[-1] = ":"

    def encode(self, x):
        out = []
        i = 0
        while i < len(x):
            if x[i] == ",":
                i += 1
                continue
            s = ""
            j = 0
            while i + j < len(x) and x[i + j] in numbers:
                s += x[i + j]
                j += 1
            if s == "":
                s = x[i]
                i += 1
            else:
                i += j
            out.append(self.encoder[s])

        return out

    def decode(self, x):
        return [self.decoder[i] for i in x]


class Tokenizer:
    def __init__(self, encoder, decoder, vocab_size, name=None):
        self.encode = encoder
        self.decode = decoder
        self.vocab_size = vocab_size
        self.name = name

    def tokenize(self, data_list):
        """
        Takes a list of prefix-target pairs, tokenizes and concatenates them
        """
        out = []
        prefix_len = len(self.encode(data_list[0][0]))
        target_len = len(self.encode(data_list[0][1]))
        same_len = True

        for prefix, target in data_list:
            prefix = torch.tensor(self.encode(prefix))
            target = torch.tensor(self.encode(target))
            if not (len(prefix) == prefix_len and len(target) == target_len):
                same_len = False
            seq = torch.concatenate([prefix, target], dim=-1).long()
            out.append(seq)

        # Check if all prefixes and all targets have the same length
        if not same_len:
            print("Not all prefixes or targets have the same length!!")
        else:
            print("Equal sequence lengths!")

        return out, prefix_len, target_len


def get_tokenizer(args):
    if args.model == "gpt":
        t = NumeralTokenizer(args.num_nodes)
        tokenizer = Tokenizer(
            encoder=t.encode,
            decoder=t.decode,
            vocab_size=args.num_nodes + 4,
            name="numeral",
        )
    elif args.model.startswith("gpt2"):
        t = AutoTokenizer.from_pretrained("gpt2")
        tokenizer = Tokenizer(
            encoder=t.encode, decoder=t.decode, vocab_size=50257, name="gpt2"
        )
    elif args.model.startswith("pythia"):
        t = AutoTokenizer.from_pretrained("EleutherAI/" + args.model)
        tokenizer = Tokenizer(
            encoder=t.encode, decoder=t.decode, vocab_size=50304, name="gpt2"
        )
    elif args.model.startswith("phi"):
        t = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        tokenizer = Tokenizer(
            encoder=t.encode, decoder=t.decode, vocab_size=51200, name="phi"
        )

    return tokenizer


if __name__ == "__main__":
    import types

    # Create graphs and save
    n_train = 200000
    n_test = 20000
    deg = 2
    path_len = 5
    num_nodes = 50
    reverse = False
    generate_and_save(
        n_train=n_train,
        n_test=n_test,
        degSource=deg,
        pathLen=path_len,
        numNodes=num_nodes,
        reverse=reverse,
    )

    # Load data
    device = "cpu"
    args = types.SimpleNamespace(model="gpt", num_nodes=num_nodes)
    args.dataset = "graph"
    args.deg = deg
    args.path_len = path_len
    args.n_train = n_train
    args.n_test = n_test
    args.reverse = False
    args.teacherless = False

    args.dollar = 11
    tokenizer = get_tokenizer(args)
    trainset, testset = get_dataset(args, tokenizer, device)
    print(trainset.num_tokens)
    trainset.__getitem__(10)
    trainset.eval()
    trainset.__getitem__(10)

    import matplotlib.pyplot as plt
    import networkx as nx

    path, edge_list, start, goal = star_graph(deg, path_len, num_nodes, reverse=reverse)
    print(len(edge_list))
    print(path)
    print(edge_list)
    print("Start:", start, "Goal:", goal)
    G = nx.Graph()
    G.add_edges_from(edge_list)
    nx.draw(G, with_labels=True)
    plt.show()
