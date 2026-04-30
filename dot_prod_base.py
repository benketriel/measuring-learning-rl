"""
Signal Evaluation under Controlled Difficulty and Noise
=======================================================

Supplementary code for "Measuring Learning Progress via Gradient-Momentum
Correlation". This script implements the controlled experiments described in
Section 3.1 of the paper.

A dynamics model (classifier) learns to predict image labels while a policy
network (actor) selects which class/group to sample from. Six intrinsic-reward
signals are compared: Uniform, Curiosity, GMC, NormAll, NormLast, and
DeltaLoss. Two experimental conditions test (1) emergent curriculum learning
under varied task difficulty ("Curriculum") and (2) noise robustness under
aleatoric label noise ("Noise").

Usage:
    python dot_prod_base.py                  # Run training experiments
    python dot_prod_base.py generate_graphs  # Generate graphs from logs

Requires: torch, torchvision, backpack-for-pytorch, numpy, matplotlib.
"""

import sys
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
import time
from functools import reduce
from backpack import backpack, extend
from backpack.extensions import BatchGrad
import os

# ============================================================================
# Constants and Algorithm Identifiers
# ============================================================================
NOISE = "noise"
CURRICULUM = "curriculum"

MNIST_NAME = "mnist"
CIFAR_NAME = "cifar"

UNIFORM = "uniform"
CURIOSITY = "curiosity"
DOTP = "dotprod"
GRADNORMALL = "gradnormall"
GRADNORMLAST = "gradnormlast"
LOSSDELTA = "lossdelta"
DOTP_OUTER = "dotprodouter"
DOTP_COS = "dotprodcos"


# Display names for algorithms and conditions (used in graphs and tables)
_nice_name = {
    NOISE: "Noise",
    CURRICULUM: "Curriculum",
    MNIST_NAME: "MNIST",
    CIFAR_NAME: "CIFAR-10",
    UNIFORM: "Uniform",
    CURIOSITY: "Curiosity",
    DOTP: "GMC",
    GRADNORMALL: "NormAll",
    GRADNORMLAST: "NormLast",
    LOSSDELTA: "DeltaLoss",

    DOTP_OUTER: "DotProduct",
    DOTP_COS: "Cosine",

    "train": "Train",
    "test": "Test",
}


# ============================================================================
# Experiment Configuration
# ============================================================================
# To reproduce the paper results, run all combinations:
#   problems:   [NOISE, CURRICULUM]
#   datasets:   [MNIST_NAME, CIFAR_NAME]
#   algorithms: [UNIFORM, CURIOSITY, DOTP, GRADNORMLAST, GRADNORMALL, LOSSDELTA]
#   seeds:      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# The ablation study (Appendix C) additionally uses DOTP_OUTER and DOTP_COS.
# Adjust the active configuration below to run a subset.
EXPERIMENT_CONFIG = {
    "problems": [NOISE, CURRICULUM],
    
    "datasets": [MNIST_NAME, CIFAR_NAME],

    "algorithms": [UNIFORM, CURIOSITY, DOTP, GRADNORMLAST, GRADNORMALL, LOSSDELTA],
    # "algorithms": [DOTP, DOTP_OUTER, DOTP_COS],
    
    # "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    # "seeds": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "seeds": [n for n in range(20)],
}

# Create logs directory if it doesn't exist
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
GRAPH_DIR = "./graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)


class SimpleNN(nn.Module):
    """Fully-connected network with two hidden layers (256 units each, ReLU).

    Used both as the dynamics model (classifier predicting image labels) and
    as the policy network (actor selecting which group to sample).
    Layers are wrapped with backpack's extend() to enable per-sample
    gradient computation via BatchGrad.
    """

    def __init__(self, inputs, outputs):
        super(SimpleNN, self).__init__()
        self.inputs = inputs
        self.fc1 = extend(nn.Linear(inputs, 256))
        self.fc2 = extend(nn.Linear(256, 256))
        self.fc3 = extend(nn.Linear(256, outputs))

    def forward(self, x):
        x = x.view(-1, self.inputs)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Trainer(object):
    """Runs one (problem, dataset, algorithm, seed) experiment.

    The trainer manages the dynamics model (classifier), the policy network
    (actor), per-class data loaders, and logging. The run() method trains
    for a fixed number of epochs, logging per-class losses and sample counts.
    """

    def __init__(self, problem, dataset, algorithm, seed):
        """
        Args:
            problem:   'noise' (labels re-randomized each draw) or
                       'curriculum' (labels permanently scrambled once).
            dataset:   'mnist' or 'cifar'.
            algorithm: one of 'uniform', 'curiosity', 'dotprod' (GMC),
                       'dotprodouter', 'dotprodcos', 'gradnormall',
                       'gradnormlast', 'lossdelta'.
            seed:      random seed for reproducibility.
        """
        self.problem = problem
        self.dataset = dataset
        self.algorithm = algorithm
        self.seed = seed
        
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Log file setup
        self.log_filename = os.path.join(LOG_DIR, f"{problem}_{dataset}_{algorithm}_{seed}.log")
        self.log_file = open(self.log_filename, 'x')  # Exclusive lock
        
        # Validate algorithm
        if algorithm not in EXPERIMENT_CONFIG["algorithms"]:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Dataset loading
        if dataset == CIFAR_NAME:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.trainset = CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        elif dataset == MNIST_NAME:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            self.trainset = MNIST(root='./data', train=True, download=True, transform=self.transform)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        # Group the 10 dataset classes into 4 groups of increasing size.
        # Group 0 has 1 class, Group 1 has 2, Group 2 has 3, Group 3 has 4.
        # In the Noise condition this creates noise levels of 0%, 50%, 67%, 75%.
        # In the Curriculum condition this creates tasks of varying difficulty.
        self.class_ranges = {
            0: [0],
            1: [1, 2],
            2: [3, 4, 5],
            3: [6, 7, 8, 9],
        }

        # In Curriculum: permanently reassign each sample's label to a random
        # label within its group (one-time scramble, fully learnable).
        # In Noise: map labels to their group index (labels are re-randomized
        # at draw time in the training loop, simulating aleatoric noise).
        SCRAMBLE_ONCE = (problem == CURRICULUM)
        if SCRAMBLE_ONCE:
            self.grouped_labels = torch.tensor([self.class_ranges[self.map_to_new_class(label)][np.random.choice(range(len(self.class_ranges[self.map_to_new_class(label)])))] for label in self.trainset.targets])
        else:
            self.grouped_labels = torch.tensor([self.map_to_new_class(label) for label in self.trainset.targets])
        self.trainset.targets = self.grouped_labels

        self.num_classes = sum([len(x) for x in self.class_ranges.values()])
        if SCRAMBLE_ONCE:
            self.class_ranges = {i: [i] for i in range(self.num_classes)}
        self.num_grouped_classes = sum([1 for x in self.class_ranges.values()])

        # Create dataloaders for the new grouped dataset
        self.grouped_class_datasets = {i: torch.utils.data.Subset(self.trainset, (self.grouped_labels == i).nonzero().squeeze(1).numpy()) for i in range(self.num_grouped_classes)}
        self.grouped_class_dataloaders = {i: torch.utils.data.DataLoader(self.grouped_class_datasets[i], batch_size=1, shuffle=True, drop_last=True) for i in range(self.num_grouped_classes)}
        self.grouped_class_iterators = {i: iter(self.grouped_class_dataloaders[i]) for i in range(self.num_grouped_classes)}

        # Create the network, criterion and optimizer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_size = reduce(lambda x, y: x * y, self.trainset.data.shape[1:])
        self.net = SimpleNN(inputs=input_size, outputs=self.num_classes).to(self.device)
        self.actor = SimpleNN(inputs=self.num_grouped_classes, outputs=self.num_grouped_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        # Dynamics model optimizer (standard Adam defaults)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        # Policy optimizer: lower LR and higher beta_1 for stable learning
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.00001, betas=(0.99, 0.999))
        self.actor_update_frequency = 1

        # GMC momentum decay parameters (both set to 0.999 for longer
        # averaging windows than Adam's defaults; see Section 2 and Table 2)
        self.beta_dot = (0.999, 0.999)

        # Entropy regularization weight (lambda_ent in Equation 3)
        self.ENTROPY_WEIGHT = 0.05

        # Training parameters
        if problem == CURRICULUM and dataset == MNIST_NAME:
            self.num_epochs = 500
        else:
            self.num_epochs = 100
        self.batch_size = 256
        self.num_batches_per_epoch = len(self.trainset) // self.batch_size
        self.num_test_batches_per_epoch = 100

        # GMC internal state (momentum and second moment per parameter group)
        self.gca_state = {}
        
        # LOSSDELTA state: rolling window of (loss_sum, count) per class
        # for computing empirical loss improvement as an intrinsic signal
        self.lossdelta_state = {i: [] for i in range(self.num_grouped_classes)}
        
        # Write CSV header
        self._write_log_header()
    
    def _write_log_header(self):
        """Write CSV header for the log file."""
        headers = ["epoch"]
        for cls in range(self.num_grouped_classes):
            headers.extend([
                f"class_{cls}_count_train",
                f"class_{cls}_loss_train",
                f"class_{cls}_count_test",
                f"class_{cls}_loss_test"
            ])
        self.log_file.write(",".join(headers) + "\n")
        self.log_file.flush()
    
    def log_epoch_data(self, epoch, train_counts, train_losses, test_counts, test_losses):
        """Log all data for an epoch as a single CSV row."""
        row = [str(epoch)]
        for cls in range(self.num_grouped_classes):
            row.append(str(train_counts.get(cls, 0)))
            row.append(str(train_losses.get(cls, 0.0)))
            row.append(str(test_counts.get(cls, 0)))
            row.append(str(test_losses.get(cls, 0.0)))
        self.log_file.write(",".join(row) + "\n")
        self.log_file.flush()

    def _update_lossdelta_state(self, epoch, class_counts, class_loss_sums):
        """Update LOSSDELTA state after each epoch."""
        for cls in range(self.num_grouped_classes):
            count = class_counts[cls]
            loss_sum = class_loss_sums[cls]
            
            # Only advance if previous epoch had samples (or if this is first epoch)
            if epoch == 0 or self.lossdelta_state[cls][0][1] > 0:
                # Add new epoch data at front
                self.lossdelta_state[cls].insert(0, (loss_sum, count))
                # Keep only last 3 epochs
                if len(self.lossdelta_state[cls]) > 3:
                    self.lossdelta_state[cls].pop()
    
    def _compute_lossdelta_advantages(self, epoch, grouped_labels):
        """Compute advantages based on LOSSDELTA."""
        batch_size = len(grouped_labels)
        advantages = torch.zeros(batch_size, device=self.device)
        
        # For epochs 0-1, advantages are 0
        if epoch < 2:
            return advantages
        
        # For epoch 2+, compute advantage per class
        for cls in range(self.num_grouped_classes):
            loss_sum_n_minus_1, count_n_minus_1 = self.lossdelta_state[cls][0]  # This epoch
            loss_sum_n_minus_2, count_n_minus_2 = self.lossdelta_state[cls][1]  # Last epoch

            if count_n_minus_1 == 0:
                loss_sum_n_minus_1, count_n_minus_1 = self.lossdelta_state[cls][1]  # Last epoch
                loss_sum_n_minus_2, count_n_minus_2 = self.lossdelta_state[cls][2]  # The one before that
            
            avg_loss_n_minus_1 = loss_sum_n_minus_1 / count_n_minus_1
            avg_loss_n_minus_2 = loss_sum_n_minus_2 / count_n_minus_2
            
            advantage = abs(avg_loss_n_minus_1 - avg_loss_n_minus_2) / (count_n_minus_2 + count_n_minus_1)
            
            mask = grouped_labels == cls
            advantages[mask] = advantage
        
        return advantages

    def run(self):
        for epoch in range(self.num_epochs):
            class_loss_sums = {i: 0.0 for i in range(self.num_grouped_classes)}
            class_counts = {i: 0 for i in range(self.num_grouped_classes)}

            batch_counter = 0

            for _ in range(self.num_batches_per_epoch):

                inputs_list, labels_list, grouped_labels_list = [], [], []

                # Use actor to produce a sampling distribution over groups
                # (Uniform has no actor; it samples groups uniformly at random)
                if self.algorithm != UNIFORM:
                    pi = torch.softmax(self.actor(torch.rand((self.batch_size, self.num_grouped_classes)).to(device=self.device)), dim=1)
                    p = pi.cpu().detach().numpy()

                # For each data point in the batch, sample a class and retrieve one data point from it
                for bi in range(self.batch_size):
                    if self.algorithm != UNIFORM:
                        chosen_class = np.random.choice(range(self.num_grouped_classes), p=p[bi])
                    else:
                        chosen_class = np.random.choice(range(self.num_grouped_classes))

                    try:
                        input_sample, label_sample = next(self.grouped_class_iterators[chosen_class])
                    except StopIteration:
                        # Restart the iterator for the grouped class if we've used up all its data
                        self.grouped_class_iterators[chosen_class] = iter(self.grouped_class_dataloaders[chosen_class])
                        input_sample, label_sample = next(self.grouped_class_iterators[chosen_class])

                    # Label the sample with the chosen_class
                    label_sample = label_sample * 0 + self.class_ranges[chosen_class][np.random.choice(range(len(self.class_ranges[chosen_class])))]
                    chosen_class = label_sample * 0 + chosen_class

                    inputs_list.append(input_sample)
                    labels_list.append(label_sample)
                    grouped_labels_list.append(chosen_class)

                inputs = torch.cat(inputs_list).to(self.device)
                labels = torch.cat(labels_list).to(self.device)
                grouped_labels = torch.cat(grouped_labels_list).to(self.device)

                inputs, labels, grouped_labels = inputs.to(self.device), labels.to(self.device), grouped_labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                losses = self.criterion(outputs, labels)
                overall_loss = losses.mean()
                with backpack(BatchGrad()):
                    overall_loss.backward()
                
                self.optimizer.step()

                # Update actor for non-UNIFORM algorithms
                batch_counter += 1
                if self.algorithm != UNIFORM and (batch_counter % self.actor_update_frequency == 0):
                    self.actor_optimizer.zero_grad()
                    gathered_probs = torch.gather(pi, 1, grouped_labels.view(-1, 1)).squeeze()
                    
                    # Compute intrinsic reward (advantage) based on algorithm
                    if self.algorithm == DOTP:
                        # GMC: per-parameter |gradient * momentum / second_moment| (Equation 2)
                        advantages = self._extract_gmc(self.optimizer, "0").flatten().detach()

                    elif self.algorithm == DOTP_OUTER:
                        # Dot Product variant: outer absolute value (Appendix C)
                        advantages = self._extract_gmc(self.optimizer, "0", inner_abs=False).flatten().detach() * 5

                    elif self.algorithm == DOTP_COS:
                        # Cosine similarity variant (Appendix C)
                        advantages = self._extract_gmc(self.optimizer, "0", inner_abs=False, cosine=True).flatten().detach() * 50

                    elif self.algorithm == CURIOSITY:
                        # Curiosity: raw prediction loss as intrinsic reward
                        advantages = losses.detach()

                    elif self.algorithm == GRADNORMALL:
                        # NormAll: gradient norm across all layers
                        advantages = []
                        for i in range(self.batch_size):
                            grad_norm = 0.0
                            for param in self.net.parameters():
                                if hasattr(param, 'grad_batch'):
                                    g = param.grad_batch[i].detach()
                                    grad_norm += torch.sum(g ** 2).item()
                            advantages.append(grad_norm)
                        # Scale up to match other magnitudes (else entropy term dominates)
                        advantages = torch.tensor(advantages, device=self.device) * 10000

                    elif self.algorithm == GRADNORMLAST:
                        # NormLast: gradient norm of the output layer only
                        advantages = []
                        for i in range(self.batch_size):
                            grad_norm = 0.0
                            last_layer = list(self.net.parameters())[-1]
                            if hasattr(last_layer, 'grad_batch'):
                                g = last_layer.grad_batch[i].detach()
                                grad_norm += torch.sum(g ** 2).item()
                            advantages.append(grad_norm)
                        # Scale up to match other magnitudes (else entropy term dominates)
                        advantages = torch.tensor(advantages, device=self.device) * 100000

                    elif self.algorithm == LOSSDELTA:
                        # DeltaLoss: per-group empirical loss improvement
                        # Scale up to match other magnitudes (else entropy term dominates)
                        advantages = self._compute_lossdelta_advantages(epoch, grouped_labels) * 100000

                    else:
                        raise ValueError(f"Unknown algorithm for actor update: {self.algorithm}")
                    
                    # Policy gradient update with entropy regularization (Equation 3)
                    actor_loss = torch.mean(-torch.log(gathered_probs) * advantages)
                    actor_entropy = torch.mean(-torch.sum(torch.log(pi + 1e-8) * pi, dim=1))
                    actor_loss -= self.ENTROPY_WEIGHT * actor_entropy
                    actor_loss.backward()
                    self.actor_optimizer.step()

                for j, label in enumerate(labels):
                    group_label = self.map_to_new_class(label.item())
                    class_loss_sums[group_label] += losses[j].item()
                    class_counts[group_label] += 1

            # Compute per-class train losses
            train_losses = {}
            for cls in range(self.num_grouped_classes):
                train_losses[cls] = class_loss_sums[cls] / max(1, class_counts[cls])
            
            # Update LOSSDELTA state for next epoch
            if self.algorithm == LOSSDELTA:
                self._update_lossdelta_state(epoch, class_counts, class_loss_sums)

            # Test phase: evaluate classifier on uniformly-sampled data
            test_class_loss_sums = {i: 0.0 for i in range(self.num_grouped_classes)}
            test_class_counts = {i: 0 for i in range(self.num_grouped_classes)}

            for _ in range(self.num_test_batches_per_epoch):
                inputs_list, labels_list = [], []

                for bi in range(self.batch_size):
                    chosen_class = np.random.choice(range(self.num_grouped_classes))
                    try:
                        input_sample, label_sample = next(self.grouped_class_iterators[chosen_class])
                    except StopIteration:
                        self.grouped_class_iterators[chosen_class] = iter(self.grouped_class_dataloaders[chosen_class])
                        input_sample, label_sample = next(self.grouped_class_iterators[chosen_class])

                    label_sample = label_sample * 0 + self.class_ranges[chosen_class][np.random.choice(range(len(self.class_ranges[chosen_class])))]

                    inputs_list.append(input_sample)
                    labels_list.append(label_sample)

                inputs = torch.cat(inputs_list).to(self.device)
                labels = torch.cat(labels_list).to(self.device)

                outputs = self.net(inputs)
                losses = self.criterion(outputs, labels)

                for j, label in enumerate(labels):
                    group_label = self.map_to_new_class(label.item())
                    test_class_loss_sums[group_label] += losses[j].item()
                    test_class_counts[group_label] += 1

            # Compute per-class test losses
            test_losses = {}
            for cls in range(self.num_grouped_classes):
                test_losses[cls] = test_class_loss_sums[cls] / max(1, test_class_counts[cls])
            
            # Log all epoch data
            self.log_epoch_data(epoch, class_counts, train_losses, test_class_counts, test_losses)
    
    def close(self):
        """Close the log file."""
        self.log_file.close()

    def map_to_new_class(self, original_label):
        for new_class, original_classes in self.class_ranges.items():
            if original_label in original_classes:
                return new_class

    def _extract_gmc(self, optimizer, label, inner_abs=True, cosine=False):
        """Compute per-sample Gradient-Momentum Correlation (Equation 2).

        Maintains exponential moving averages of the mean gradient (momentum m)
        and its second moment (v) across batches. For each sample, computes the
        correlation between its individual gradient and the normalized momentum
        m / v, aggregated across all model parameters.

        Args:
            optimizer:  the optimizer whose param_groups contain the model params.
            label:      string key for tracking momentum state across calls.
            inner_abs:  if True, use |g_i| * |m_i/v_i| per parameter (GMC,
                        preferred: captures multi-directional relevance).
                        if False, use |sum_i g_i * m_i/v_i| (Dot Product
                        variant: single dominant direction).
            cosine:     if True, use cosine similarity instead (Appendix C).

        Returns:
            Tensor of shape (batch_size, 1) with per-sample GMC scores.
        """
        if label not in self.gca_state:
            self.gca_state[label] = {}
        state = self.gca_state[label]
        if 'step' not in state:
            state['step'] = 0
        step = state['step']

        b0, b1 = self.beta_dot

        dot_prod = None
        total_params = 0
        label_offset = 0
        for group in optimizer.param_groups:
            for param in group['params']:
                if not hasattr(param, 'grad_batch'):
                    # For example this happens in LayerNorm
                    continue
                
                g = param.grad_batch.detach()
                batch_size = g.shape[0]
                bias_grad = len(g.shape) == 2
                g = g.reshape(batch_size, -1)
                n_params = g.shape[1]

                if label_offset not in state:
                    state[label_offset] = {}
                g_state = state[label_offset]
                label_offset += 1

                if 'm' not in g_state:
                    g_state['m'] = torch.zeros((n_params,), device=self.device)
                    g_state['v'] = torch.zeros((n_params,), device=self.device)
                m = g_state['m']
                v = g_state['v']

                if step > 0:
                    m = m / (1 - b0 ** step)
                    v = v / (1 - b1 ** step)

                normalized_momentum = m / torch.clamp(v, min=1e-8)

                if cosine:
                    if inner_abs:
                        print("Warning: cosine similarity with inner_abs=True is not a meaningful metric, as the cosine similarity already captures directionality. Ignoring inner_abs and using outer abs.")
                    dp = torch.sum(g * normalized_momentum, dim=1) / (torch.norm(g, dim=1) * torch.norm(normalized_momentum) + 1e-8)
                    dp = dp.abs().unsqueeze(-1)
                elif inner_abs:
                    # GMC (default): per-parameter absolute correlation (Eq. 2)
                    dp = (g.abs() @ normalized_momentum.abs()).unsqueeze(-1)
                else:
                    # Dot Product variant: signed sum, then outer absolute value
                    dp = (g @ normalized_momentum).abs().unsqueeze(-1)
                if dot_prod is None:
                    dot_prod = dp
                else:
                    dot_prod.add_(dp)
                total_params += n_params

                g = g.mean(dim=0)
                g_state['m'].mul_(b0).add_(g * (1 - b0))
                g_state['v'].mul_(b1).add_((g ** 2) * (1 - b1))

        dot_prod = dot_prod / np.sqrt(total_params)

        state['step'] += 1

        return dot_prod


def _run_experiment(problem, dataset, algorithm, seed):
    """Run a single experiment in an isolated process."""
    trainer = None
    try:
        trainer = Trainer(problem, dataset, algorithm, seed)
        trainer.run()

        print(f"Completed: {problem}_{dataset}_{algorithm}_{seed}")
        print(f"Log saved to: {trainer.log_filename}\n")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        if trainer:
            trainer.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def run_all_experiments():
    ctx = multiprocessing.get_context("spawn")
    for seed in EXPERIMENT_CONFIG["seeds"]:
        for problem in EXPERIMENT_CONFIG["problems"]:
            for algorithm in EXPERIMENT_CONFIG["algorithms"]:
                for dataset in EXPERIMENT_CONFIG["datasets"]:
                    print(f"Running: {problem}_{dataset}_{algorithm}_{seed}")

                    proc = ctx.Process(
                        target=_run_experiment,
                        args=(problem, dataset, algorithm, seed),
                    )
                    proc.start()
                    proc.join()

                    if proc.exitcode != 0:
                        raise RuntimeError(f"Experiment {problem}_{dataset}_{algorithm}_{seed} failed with exit code {proc.exitcode}")

# ============================================================================
# Graph Generation Functions
# ============================================================================

def load_log_file(filename, max_rows):
    """Load a log CSV file and return a dictionary with class data."""
    rows = []

    with open(filename, 'r') as f:
        header = f.readline().strip().split(',')

        for line_idx, line in enumerate(f):
            values = line.strip().split(',')
            epoch = int(values[0])
            row_data = {}
            for i, col_name in enumerate(header[1:], 1):
                row_data[col_name] = float(values[i])

            rows.append((epoch, row_data))
            if len(rows) >= max_rows:
                break

    if len(rows) != max_rows:
        for _ in range(10):
            print(f"***** Warning: Log file {filename} has not enough rows. *****")
        raise ValueError(f"Expected {max_rows} rows in log file {filename}, but got {len(rows)}")

    # Smooth very long runs by averaging blocks of rows
    if len(rows) > 100:
        merged_rows = []
        merge_size = 5

        for start in range(0, len(rows), merge_size):
            chunk = rows[start:start + merge_size]
            if not chunk:
                continue

            # Average epoch to keep spacing roughly aligned across seeds
            merged_epoch = int(np.mean([r[0] for r in chunk]))

            merged_data = {}
            for key in chunk[0][1].keys():
                merged_data[key] = float(np.mean([r[1][key] for r in chunk]))

            merged_rows.append((merged_epoch, merged_data))

        rows = merged_rows

    return {epoch: row for epoch, row in rows}

def aggregate_logs_by_seed(problem, dataset, algorithm):
    """
    Aggregate log data across multiple seeds.
    Returns: (epochs, aggregated_data_dict)
    aggregated_data_dict[metric_name] = {'mean': array, 'std': array}
    """
    # Determine max_epochs for this (problem, dataset)
    if problem == CURRICULUM and dataset == MNIST_NAME:
        max_epochs = 400
    else:
        max_epochs = 100

    all_data = []
    for seed in EXPERIMENT_CONFIG["seeds"]:
        filename = os.path.join(LOG_DIR, f"{problem}_{dataset}_{algorithm}_{seed}.log")
        if os.path.exists(filename):
            data = load_log_file(filename, max_epochs)
            all_data.append(data)

    if not all_data:
        return None, None

    # Get epochs from first seed
    epochs = sorted(all_data[0].keys())

    # Aggregate metrics
    aggregated = {}

    # Get all metric names from first epoch
    first_epoch_keys = all_data[0][epochs[0]].keys()

    for metric_name in first_epoch_keys:
        values_across_seeds = []
        for seed_idx, data in enumerate(all_data):
            metric_values = [data[epoch][metric_name] for epoch in epochs]
            values_across_seeds.append(metric_values)

        values_array = np.array(values_across_seeds)  # shape: (num_seeds, num_epochs)
        n_seeds = values_array.shape[0]
        std = np.std(values_array, axis=0, ddof=1)  # sample std
        t_crit = scipy_stats.t.ppf(0.975, df=n_seeds - 1)  # 95% CI, t-distribution
        aggregated[metric_name] = {
            'mean': np.mean(values_array, axis=0),
            'std': t_crit * std / np.sqrt(n_seeds)  # 95% CI half-width
        }

    return epochs, aggregated

def extract_class_data(aggregated_data, num_classes, metric_type, train_or_test):
    """
    Extract per-class data from aggregated data.
    metric_type: 'loss' or 'count'
    train_or_test: 'train' or 'test'
    Returns: dict {class_id: {'mean': array, 'std': array}}
    """
    class_data = {}
    for cls in range(num_classes):
        key = f"class_{cls}_{metric_type}_{train_or_test}"
        if key in aggregated_data:
            class_data[cls] = aggregated_data[key]
    
    return class_data

def compute_average_loss(loss_data, count_data, num_classes, epochs):
    """
    Compute weighted average loss across all classes.
    Returns: {'mean': array, 'std': array}
    """
    # Compute weighted average: sum(loss * count) / sum(count)
    total_loss = np.zeros((len(epochs),))
    total_count = np.zeros((len(epochs),))
    total_loss_sq = np.zeros((len(epochs),))
    
    for cls in range(num_classes):
        if cls in loss_data and cls in count_data:
            loss_mean = loss_data[cls]['mean']
            loss_std = loss_data[cls]['std']
            count_mean = count_data[cls]['mean']
            
            total_loss += loss_mean * count_mean
            total_count += count_mean
            # Approximate std propagation
            total_loss_sq += (loss_std * count_mean) ** 2
    
    avg_loss_mean = total_loss / np.maximum(total_count, 1e-8)
    avg_loss_std = np.sqrt(total_loss_sq) / np.maximum(total_count, 1e-8)
    
    return {'mean': avg_loss_mean, 'std': avg_loss_std}

def compute_geometric_mean_loss(loss_data, num_classes, epochs):
    """
    Compute geometric mean of losses across classes (only for epochs with positive values).
    Returns: {'mean': array, 'std': array}
    """
    log_losses = []
    
    for cls in range(num_classes):
        if cls in loss_data:
            loss_mean = loss_data[cls]['mean']
            # Avoid log of zero
            loss_safe = np.maximum(loss_mean, 1e-8)
            log_losses.append(np.log(loss_safe))
    
    if not log_losses:
        return {'mean': np.zeros(len(epochs)), 'std': np.zeros(len(epochs))}
    
    log_losses = np.array(log_losses)  # shape: (num_classes, num_epochs)
    mean_log_loss = np.mean(log_losses, axis=0)
    std_log_loss = np.std(log_losses, axis=0)
    
    # Convert back to loss space
    geom_mean = np.exp(mean_log_loss)
    # Approximate std in loss space
    geom_std = geom_mean * std_log_loss
    
    return {'mean': geom_mean, 'std': geom_std}

def plot_loss_and_count_grid(problem, dataset, algorithms, train_or_test, num_classes, merged_classes=None):
    """
    Create a 2x3 grid of plots (loss and count) for 3 algorithms.
    
    Args:
        problem: 'noise' or 'curriculum'
        dataset: 'mnist' or 'cifar'
        algorithms: list of 3 algorithm names
        train_or_test: 'train' or 'test'
        num_classes: number of classes (before merging if applicable)
        merged_classes: dict mapping original class to merged class (for regrouping)
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex=True)
    
    # Color palette for classes
    colors = plt.cm.tab10(np.linspace(0, 1, max(num_classes, 10)))
    
    # Determine effective number of classes for plotting
    plot_num_classes = len(set(merged_classes.values())) if merged_classes else num_classes
    label_name = 'Group' if merged_classes else 'Class'
    
    for algo_idx, algorithm in enumerate(algorithms):
        epochs, aggregated = aggregate_logs_by_seed(problem, dataset, algorithm)
        
        if epochs is None:
            print(f"Warning: No data found for {problem}_{dataset}_{algorithm}")
            continue
        
        # Extract loss and count data
        loss_data = extract_class_data(aggregated, num_classes, 'loss', train_or_test)
        count_data = extract_class_data(aggregated, num_classes, 'count', train_or_test)
        
        # Merge classes if needed
        if merged_classes:
            # Use weighted merge for loss data to properly account for group sizes
            loss_data = merge_class_data_weighted(loss_data, count_data, merged_classes)
            # Use sum merge for count data to get total samples in merged group
            count_data = merge_class_data_sum(count_data, merged_classes)
        
        # --- Plot loss (top row) ---
        ax_loss = axes[0, algo_idx]
        
        for cls in range(plot_num_classes):
            if cls in loss_data:
                mean = loss_data[cls]['mean']
                std = loss_data[cls]['std']
                ax_loss.plot(epochs, mean, label=f'{label_name} {cls}', color=colors[cls], linewidth=2)
                ax_loss.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=colors[cls])
        
        # Add average loss (dashed black line)
        avg_loss = compute_average_loss(loss_data, count_data, plot_num_classes, epochs)
        ax_loss.plot(epochs, avg_loss['mean'], 'k--', label='Avg Loss', linewidth=2)
        ax_loss.fill_between(epochs, avg_loss['mean'] - avg_loss['std'], avg_loss['mean'] + avg_loss['std'], alpha=0.1, color='black')
        
        ax_loss.set_title(f'{_nice_name[algorithm]}')
        ax_loss.set_ylabel(f'{_nice_name[train_or_test]} Loss')
        ax_loss.legend(loc='best', fontsize=8)
        ax_loss.grid(True, alpha=0.3)
        
        # --- Plot count (bottom row) ---
        ax_count = axes[1, algo_idx]
        
        for cls in range(plot_num_classes):
            if cls in count_data:
                mean = count_data[cls]['mean']
                std = count_data[cls]['std']
                ax_count.plot(epochs, mean, label=f'{label_name} {cls}', color=colors[cls], linewidth=2)
                ax_count.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=colors[cls])
        
        ax_count.set_title(f'{_nice_name[algorithm]}')
        ax_count.set_xlabel('Epoch')
        ax_count.set_ylabel(f'{_nice_name[train_or_test]} Count')
        ax_count.legend(loc='best', fontsize=8)
        ax_count.grid(True, alpha=0.3)
    
    # Share y-axis within rows
    for row in range(2):
        y_min, y_max = float('inf'), float('-inf')
        for col in range(3):
            y_lim = axes[row, col].get_ylim()
            y_min = min(y_min, y_lim[0])
            y_max = max(y_max, y_lim[1])
        for col in range(3):
            axes[row, col].set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Add space at top for title
    return fig

def plot_loss_and_count_merged(problem, dataset, algorithms, num_classes, merged_classes=None, normalize_counts=False):
    """
    Create a 2x3 merged grid of plots combining test loss (top) and train count (bottom).
    
    Args:
        problem: 'noise' or 'curriculum'
        dataset: 'mnist' or 'cifar'
        algorithms: list of 3 algorithm names
        num_classes: number of classes (before merging if applicable)
        merged_classes: dict mapping original class to merged class (for regrouping)
        normalize_counts: if True, divide counts by group size (for average per class in group)
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex=True)
    
    # Color palette for classes
    colors = plt.cm.tab10(np.linspace(0, 1, max(num_classes, 10)))
    
    # Determine effective number of classes for plotting
    plot_num_classes = len(set(merged_classes.values())) if merged_classes else num_classes
    
    for algo_idx, algorithm in enumerate(algorithms):
        epochs, aggregated = aggregate_logs_by_seed(problem, dataset, algorithm)
        
        if epochs is None:
            print(f"Warning: No data found for {problem}_{dataset}_{algorithm}")
            continue
        
        # Extract test loss data
        test_loss_data = extract_class_data(aggregated, num_classes, 'loss', 'test')
        test_count_data = extract_class_data(aggregated, num_classes, 'count', 'test')
        
        # Extract train count data
        train_count_data = extract_class_data(aggregated, num_classes, 'count', 'train')
        
        # Merge classes if needed
        if merged_classes:
            # Use weighted merge for loss data to properly account for group sizes
            test_loss_data = merge_class_data_weighted(test_loss_data, test_count_data, merged_classes)
            # Use sum merge for count data to get total samples in merged groups
            test_count_data = merge_class_data_sum(test_count_data, merged_classes)
            # For train counts, use average if normalize_counts is True, otherwise sum
            if normalize_counts:
                train_count_data = merge_class_data_average(train_count_data, merged_classes)
            else:
                train_count_data = merge_class_data_sum(train_count_data, merged_classes)
        
        # --- Plot test loss (top row) ---
        ax_loss = axes[0, algo_idx]
        
        for cls in range(plot_num_classes):
            if cls in test_loss_data:
                mean = test_loss_data[cls]['mean']
                std = test_loss_data[cls]['std']
                ax_loss.plot(epochs, mean, label=f'Group {cls}', color=colors[cls], linewidth=2)
                ax_loss.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=colors[cls])
        
        # Add average loss (dashed black line)
        avg_loss = compute_average_loss(test_loss_data, test_count_data, plot_num_classes, epochs)
        ax_loss.plot(epochs, avg_loss['mean'], 'k--', label='Avg Loss', linewidth=2)
        ax_loss.fill_between(epochs, avg_loss['mean'] - avg_loss['std'], avg_loss['mean'] + avg_loss['std'], alpha=0.1, color='black')
        
        ax_loss.set_title(f'{_nice_name[algorithm]}')
        ax_loss.set_ylabel('Test Loss')
        ax_loss.legend(loc='best', fontsize=8)
        ax_loss.grid(True, alpha=0.3)
        
        # --- Plot train count (bottom row) ---
        ax_count = axes[1, algo_idx]
        
        for cls in range(plot_num_classes):
            if cls in train_count_data:
                mean = train_count_data[cls]['mean']
                std = train_count_data[cls]['std']
                ax_count.plot(epochs, mean, label=f'Group {cls}', color=colors[cls], linewidth=2)
                ax_count.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=colors[cls])
        
        ax_count.set_title(f'{_nice_name[algorithm]}')
        ax_count.set_xlabel('Epoch')
        ax_count.set_ylabel('Train Count')
        ax_count.legend(loc='best', fontsize=8)
        ax_count.grid(True, alpha=0.3)
    
    # Share y-axis within rows
    for row in range(2):
        y_min, y_max = float('inf'), float('-inf')
        for col in range(3):
            y_lim = axes[row, col].get_ylim()
            y_min = min(y_min, y_lim[0])
            y_max = max(y_max, y_lim[1])
        for col in range(3):
            axes[row, col].set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Add space at top for title
    return fig

def _group_classes_by_merge_mapping(class_data, merged_classes):
    """Helper to group original classes by their merged class mapping."""
    merged_classes_groups = {}
    for orig_cls in class_data.keys():
        merged_cls = merged_classes.get(orig_cls, orig_cls)
        if merged_cls not in merged_classes_groups:
            merged_classes_groups[merged_cls] = []
        merged_classes_groups[merged_cls].append(orig_cls)
    return merged_classes_groups

def merge_class_data_sum(class_data, merged_classes):
    """
    Merge per-class count data by summing.
    Used for count data where we want to combine all samples from merged classes.
    
    Args:
        class_data: dict {class_id: {'mean': array, 'std': array}}
        merged_classes: dict mapping original class to merged class
    
    Returns:
        merged_data: dict {merged_class: {'mean': array, 'std': array}}
    """
    merged_classes_groups = _group_classes_by_merge_mapping(class_data, merged_classes)
    
    merged_data = {}
    for merged_cls, original_classes in merged_classes_groups.items():
        mean_sum = None
        std_sq_sum = None
        
        for orig_cls in original_classes:
            if orig_cls in class_data:
                data = class_data[orig_cls]
                if mean_sum is None:
                    mean_sum = np.copy(data['mean'])
                    std_sq_sum = np.copy(data['std'] ** 2)
                else:
                    mean_sum += data['mean']
                    std_sq_sum += data['std'] ** 2
        
        if mean_sum is not None:
            merged_data[merged_cls] = {
                'mean': mean_sum,
                'std': np.sqrt(std_sq_sum)
            }
    
    return merged_data

def merge_class_data_average(class_data, merged_classes):
    """
    Merge per-class count data by averaging (sum divided by number of classes in group).
    Used for count data where we want to show average samples per class within merged groups.
    
    Args:
        class_data: dict {class_id: {'mean': array, 'std': array}}
        merged_classes: dict mapping original class to merged class
    
    Returns:
        merged_data: dict {merged_class: {'mean': array, 'std': array}}
    """
    merged_classes_groups = _group_classes_by_merge_mapping(class_data, merged_classes)
    
    merged_data = {}
    for merged_cls, original_classes in merged_classes_groups.items():
        mean_sum = None
        std_sq_sum = None
        
        for orig_cls in original_classes:
            if orig_cls in class_data:
                data = class_data[orig_cls]
                if mean_sum is None:
                    mean_sum = np.copy(data['mean'])
                    std_sq_sum = np.copy(data['std'] ** 2)
                else:
                    mean_sum += data['mean']
                    std_sq_sum += data['std'] ** 2
        
        if mean_sum is not None:
            group_size = len(original_classes)
            merged_data[merged_cls] = {
                'mean': mean_sum / group_size,
                'std': np.sqrt(std_sq_sum) / group_size
            }
    
    return merged_data

def merge_class_data_weighted(loss_data, count_data, merged_classes):
    """
    Merge per-class loss data by weighted averaging.
    Each class is weighted by its count, so the merged loss properly reflects
    the average loss across all samples in the merged group.
    
    Args:
        loss_data: dict {class_id: {'mean': array, 'std': array}}
        count_data: dict {class_id: {'mean': array, 'std': array}}
        merged_classes: dict mapping original class to merged class
    
    Returns:
        merged_loss_data: dict {merged_class: {'mean': array, 'std': array}}
    """
    merged_classes_groups = _group_classes_by_merge_mapping(loss_data, merged_classes)
    
    merged_loss_data = {}
    for merged_cls, original_classes in merged_classes_groups.items():
        weighted_loss_sum = None
        weighted_loss_sq_sum = None
        total_count = None
        
        for orig_cls in original_classes:
            if orig_cls in loss_data and orig_cls in count_data:
                loss = loss_data[orig_cls]['mean']
                loss_std = loss_data[orig_cls]['std']
                count = count_data[orig_cls]['mean']
                
                if weighted_loss_sum is None:
                    weighted_loss_sum = loss * count
                    weighted_loss_sq_sum = (loss_std * count) ** 2
                    total_count = np.copy(count)
                else:
                    weighted_loss_sum += loss * count
                    weighted_loss_sq_sum += (loss_std * count) ** 2
                    total_count += count
        
        if total_count is not None:
            merged_loss_data[merged_cls] = {
                'mean': weighted_loss_sum / np.maximum(total_count, 1e-8),
                'std': np.sqrt(weighted_loss_sq_sum) / np.maximum(total_count, 1e-8)
            }
    
    return merged_loss_data

def plot_average_loss_comparison(problem, dataset, algorithms, train_or_test, num_classes):
    """
    Create a single plot comparing mean and geometric mean losses across algorithms.
    
    Args:
        problem: 'noise' or 'curriculum'
        dataset: 'mnist' or 'cifar'
        algorithms: list of algorithm names
        train_or_test: 'train' or 'test'
        num_classes: number of classes
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Color palette for algorithms
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    
    for algo_idx, algorithm in enumerate(algorithms):
        epochs, aggregated = aggregate_logs_by_seed(problem, dataset, algorithm)
        
        if epochs is None:
            print(f"Warning: No data found for {problem}_{dataset}_{algorithm}")
            continue
        
        # Extract loss and count data
        loss_data = extract_class_data(aggregated, num_classes, 'loss', train_or_test)
        count_data = extract_class_data(aggregated, num_classes, 'count', train_or_test)
        
        # Compute average loss
        avg_loss = compute_average_loss(loss_data, count_data, num_classes, epochs)
        ax.plot(epochs, avg_loss['mean'], '-', 
                label=f'{_nice_name[algorithm]}', 
                color=colors[algo_idx], linewidth=2)
        ax.fill_between(epochs, 
                        avg_loss['mean'] - avg_loss['std'], 
                        avg_loss['mean'] + avg_loss['std'], 
                        alpha=0.2, color=colors[algo_idx])
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(f'{_nice_name[train_or_test]} Loss', fontsize=12)
    # ax.set_title(f'{_nice_name[problem]} - {_nice_name[dataset]} - Overall {_nice_name[train_or_test]} Loss', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def generate_comparison_graphs(algorithms):
    
    # Mapping: in CURRICULUM, classes 0-9 become individual classes, but we regroup them
    # Original mapping was: {0: [0], 1: [1, 2], 2: [3, 4, 5], 3: [6, 7, 8, 9]}
    merged_map = {
        0: 0,  # class 0 -> group 0
        1: 1, 2: 1,  # classes 1,2 -> group 1
        3: 2, 4: 2, 5: 2,  # classes 3,4,5 -> group 2
        6: 3, 7: 3, 8: 3, 9: 3  # classes 6,7,8,9 -> group 3
    }
    
    # MERGED GRAPHS: Test Loss (top) + Train Count (bottom)
    print("Generating merged graphs...")
    
    # 1. NOISE + MNIST (merged)
    print("  1. NOISE + MNIST")
    try:
        fig = plot_loss_and_count_merged(NOISE, MNIST_NAME, algorithms, num_classes=4)
        # fig.suptitle(f'{_nice_name[NOISE]} - {_nice_name[MNIST_NAME]}')
        fig.savefig(os.path.join(GRAPH_DIR, 'graph_01_noise_mnist.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Error generating graph 1: {e}")
    
    # 2. NOISE + CIFAR (merged)
    print("  2. NOISE + CIFAR")
    try:
        fig = plot_loss_and_count_merged(NOISE, CIFAR_NAME, algorithms, num_classes=4)
        # fig.suptitle(f'{_nice_name[NOISE]} - {_nice_name[CIFAR_NAME]}')
        fig.savefig(os.path.join(GRAPH_DIR, 'graph_02_noise_cifar.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Error generating graph 2: {e}")
    
    # 3. CURRICULUM + MNIST (original grouping, merged)
    print("  3. CURRICULUM + MNIST")
    try:
        fig = plot_loss_and_count_merged(CURRICULUM, MNIST_NAME, algorithms, num_classes=10)
        # fig.suptitle(f'{_nice_name[CURRICULUM]} - {_nice_name[MNIST_NAME]}')
        fig.savefig(os.path.join(GRAPH_DIR, 'graph_03_curriculum_mnist.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Error generating graph 3: {e}")
    
    # 4. CURRICULUM + CIFAR (original grouping, merged)
    print("  4. CURRICULUM + CIFAR")
    try:
        fig = plot_loss_and_count_merged(CURRICULUM, CIFAR_NAME, algorithms, num_classes=10)
        # fig.suptitle(f'{_nice_name[CURRICULUM]} - {_nice_name[CIFAR_NAME]}')
        fig.savefig(os.path.join(GRAPH_DIR, 'graph_04_curriculum_cifar.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Error generating graph 4: {e}")
    
    # 5. CURRICULUM + MNIST (regrouped, merged)
    print("  5. CURRICULUM + MNIST")
    try:
        fig = plot_loss_and_count_merged(CURRICULUM, MNIST_NAME, algorithms, num_classes=10, merged_classes=merged_map, normalize_counts=True)
        # fig.suptitle(f'{_nice_name[CURRICULUM]} - {_nice_name[MNIST_NAME]}')
        fig.savefig(os.path.join(GRAPH_DIR, 'graph_05_curriculum_mnist_regrouped.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Error generating graph 5: {e}")
    
    # 6. CURRICULUM + CIFAR (regrouped, merged)
    print("  6. CURRICULUM + CIFAR")
    try:
        fig = plot_loss_and_count_merged(CURRICULUM, CIFAR_NAME, algorithms, num_classes=10, merged_classes=merged_map, normalize_counts=True)
        # fig.suptitle(f'{_nice_name[CURRICULUM]} - {_nice_name[CIFAR_NAME]}')
        fig.savefig(os.path.join(GRAPH_DIR, 'graph_06_curriculum_cifar_regrouped.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Error generating graph 6: {e}")
    
def generate_mean_loss_graphs(algorithms):
    # Average loss comparison graphs (test only)
    print("\nGenerating average loss comparison graphs...")
    
    # 7. NOISE + MNIST + Test (avg comparison)
    print("  7. NOISE + MNIST + Test (avg comparison)")
    try:
        fig = plot_average_loss_comparison(NOISE, MNIST_NAME, algorithms, 'test', num_classes=4)
        fig.savefig(os.path.join(GRAPH_DIR, 'graph_07_noise_mnist_test_avg.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Error generating graph 7: {e}")
    
    # 8. NOISE + CIFAR + Test (avg comparison)
    print("  8. NOISE + CIFAR + Test (avg comparison)")
    try:
        fig = plot_average_loss_comparison(NOISE, CIFAR_NAME, algorithms, 'test', num_classes=4)
        fig.savefig(os.path.join(GRAPH_DIR, 'graph_08_noise_cifar_test_avg.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Error generating graph 8: {e}")
    
    # 9. CURRICULUM + MNIST + Test (avg comparison)
    print("  9. CURRICULUM + MNIST + Test (avg comparison)")
    try:
        fig = plot_average_loss_comparison(CURRICULUM, MNIST_NAME, algorithms, 'test', num_classes=10)
        fig.savefig(os.path.join(GRAPH_DIR, 'graph_09_curriculum_mnist_test_avg.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Error generating graph 9: {e}")
    
    # 10. CURRICULUM + CIFAR + Test (avg comparison)
    print("  10. CURRICULUM + CIFAR + Test (avg comparison)")
    try:
        fig = plot_average_loss_comparison(CURRICULUM, CIFAR_NAME, algorithms, 'test', num_classes=10)
        fig.savefig(os.path.join(GRAPH_DIR, 'graph_10_curriculum_cifar_test_avg.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Error generating graph 10: {e}")
    
    print("\nAll graphs generated and saved to ./graphs/")
    print("Total: 6 merged grid plots + 4 average loss comparison plots = 10 graphs")


def compute_total_loss_auc(problem, dataset, algorithm, num_classes, metric='test', merged_classes=None):
    """
    Compute the total AUC (sum of average loss across all epochs).
    This computes the AUC under the same average loss curve shown in graphs 1-6.
    This is a simple summation metric: lower is better.
    
    Args:
        problem: 'noise' or 'curriculum'
        dataset: 'mnist' or 'cifar'
        algorithm: algorithm name
        num_classes: number of classes (before merging if applicable)
        metric: 'train' or 'test'
        merged_classes: dict mapping original class to merged class (for regrouping)
    
    Returns:
        total_auc: sum of average loss across all epochs
    """
    # Aggregate data across all seeds
    epochs, aggregated = aggregate_logs_by_seed(problem, dataset, algorithm)
    
    if epochs is None:
        raise RuntimeError(f"No log data found for {algorithm}")
    
    loss_data = extract_class_data(aggregated, num_classes, 'loss', metric)
    count_data = extract_class_data(aggregated, num_classes, 'count', metric)
    
    # For CURRICULUM with merged_classes, we still want to compute the average over the ORIGINAL 10 classes
    # (not the merged groups), so we do NOT merge the data here
    # This ensures the AUC matches the average loss shown in graphs 3 and 4 (ground truth)
    # and also matches graphs 5 and 6 (which should have the same average, just visually grouped)
    
    # Compute the average loss across all original classes (same as the dashed black line in graphs)
    avg_loss = compute_average_loss(loss_data, count_data, num_classes, epochs)
    
    # Sum the average loss across all epochs to get the AUC
    total_auc = np.sum(avg_loss['mean'])
    
    return total_auc

def compute_total_loss_auc_per_seed(problem, dataset, algorithm, num_classes, metric='test', merged_classes=None):
    """
    Compute the total AUC per seed (for computing std across seeds).
    
    Args:
        problem: 'noise' or 'curriculum'
        dataset: 'mnist' or 'cifar'
        algorithm: algorithm name
        num_classes: number of classes (before merging if applicable)
        metric: 'train' or 'test'
        merged_classes: dict mapping original class to merged class (for regrouping)
    
    Returns:
        auc_per_seed: list of AUC values (one per seed)
    """
    if problem == CURRICULUM and dataset == MNIST_NAME:
        max_epochs = 400
    else:
        max_epochs = 100
    
    auc_per_seed = []
    for seed in EXPERIMENT_CONFIG["seeds"]:
        filename = os.path.join(LOG_DIR, f"{problem}_{dataset}_{algorithm}_{seed}.log")
        if not os.path.exists(filename):
            continue
        
        # Load data for this seed
        data = load_log_file(filename, max_epochs)
        epochs = sorted(data.keys())
        
        # Extract per-class loss and count data for this seed
        class_loss_sums = {cls: [] for cls in range(num_classes)}
        class_counts = {cls: [] for cls in range(num_classes)}
        
        for epoch in epochs:
            for cls in range(num_classes):
                loss_key = f"class_{cls}_loss_{metric}"
                count_key = f"class_{cls}_count_{metric}"
                if loss_key in data[epoch] and count_key in data[epoch]:
                    class_loss_sums[cls].append(data[epoch][loss_key])
                    class_counts[cls].append(data[epoch][count_key])
        
        # Compute weighted average loss per epoch
        avg_losses = []
        for epoch_idx in range(len(epochs)):
            total_loss = 0.0
            total_count = 0.0
            for cls in range(num_classes):
                if epoch_idx < len(class_loss_sums[cls]):
                    loss = class_loss_sums[cls][epoch_idx]
                    count = class_counts[cls][epoch_idx]
                    total_loss += loss * count
                    total_count += count
            if total_count > 0:
                avg_losses.append(total_loss / total_count)
            else:
                avg_losses.append(0.0)
        
        # Sum to get AUC
        auc = np.sum(avg_losses)
        auc_per_seed.append(auc)
    
    return auc_per_seed

def compute_auc_table(problem, dataset, num_classes, metric='test', merged_classes=None):
    """
    Compute table with all 6 algorithms showing total AUC of losses.
    
    Args:
        problem: 'noise' or 'curriculum'
        dataset: 'mnist' or 'cifar'
        num_classes: number of classes (before merging if applicable)
        metric: 'train' or 'test'
        merged_classes: dict mapping original class to merged class (for regrouping)
    
    Returns:
        results_dict: dictionary with algorithm results
    """
    algorithms = EXPERIMENT_CONFIG["algorithms"]
    
    results = {}
    
    for algorithm in algorithms:
        total_auc = compute_total_loss_auc(problem, dataset, algorithm, num_classes, metric, merged_classes)
        results[algorithm] = {
            'total_auc': total_auc
        }
    
    return results

def generate_auc_table_image(problem, dataset, num_classes, metric='test', merged_classes=None):
    """
    Generate a bar graph showing AUC comparisons across algorithms.
    
    Args:
        problem: 'noise' or 'curriculum'
        dataset: 'mnist' or 'cifar'
        num_classes: number of classes (before merging if applicable)
        metric: 'train' or 'test'
        merged_classes: dict mapping original class to merged class (for regrouping)
    """
    results = compute_auc_table(problem, dataset, num_classes, metric, merged_classes)
    
    algorithms = EXPERIMENT_CONFIG["algorithms"]
    algorithm_names = [_nice_name[algo] for algo in algorithms]
    auc_values = [results[algo]['total_auc'] for algo in algorithms]
    
    # Create bar graph
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    
    bars = ax.bar(algorithm_names, auc_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, auc_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Total Loss AUC ({_nice_name[metric]})', fontsize=12, fontweight='bold')
    ax.set_title(f'{_nice_name[problem]} - {_nice_name[dataset]} - Total Loss AUC ({_nice_name[metric]})', 
                 fontsize=14, weight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def generate_merged_auc_bar_graph(metric='test'):
    """
    Generate a single merged bar graph with all 4 problem+dataset combinations.
    Each algorithm gets 4 bars (one for each problem+dataset), normalized by uniform.
    Includes error bars showing 1 std across seeds.
    Uses split y-axis: bottom zoomed to 0.95-1.05, top extends to 1.1-1.6.
    
    Args:
        metric: 'train' or 'test'
    """
    # Mapping for CURRICULUM
    merged_map = {
        0: 0,  # class 0 -> group 0
        1: 1, 2: 1,  # classes 1,2 -> group 1
        3: 2, 4: 2, 5: 2,  # classes 3,4,5 -> group 2
        6: 3, 7: 3, 8: 3, 9: 3  # classes 6,7,8,9 -> group 3
    }
    
    # Collect AUC values for all 4 combinations
    combinations = [
        (CURRICULUM, MNIST_NAME, 10, merged_map, _nice_name[CURRICULUM] + '+' + _nice_name[MNIST_NAME]),
        (CURRICULUM, CIFAR_NAME, 10, merged_map, _nice_name[CURRICULUM] + '+' + _nice_name[CIFAR_NAME]),
        (NOISE, MNIST_NAME, 4, None, _nice_name[NOISE] + '+' + _nice_name[MNIST_NAME]),
        (NOISE, CIFAR_NAME, 4, None, _nice_name[NOISE] + '+' + _nice_name[CIFAR_NAME]),
    ]
    
    algorithms = EXPERIMENT_CONFIG["algorithms"]
    
    # Collect all AUC values per seed for each algorithm and combination
    auc_data_per_seed = {}  # {algorithm: [[seed_aucs_comb0], [seed_aucs_comb1], ...]}
    for algorithm in algorithms:
        auc_data_per_seed[algorithm] = []
        for problem, dataset, num_classes, merged_classes, label in combinations:
            seed_aucs = compute_total_loss_auc_per_seed(problem, dataset, algorithm, num_classes, metric, merged_classes)
            auc_data_per_seed[algorithm].append(seed_aucs)
    
    # Compute mean and 95% CI for each algorithm and combination
    auc_mean = {}  # {algorithm: [mean1, mean2, mean3, mean4]}
    auc_ci = {}    # {algorithm: [ci1, ci2, ci3, ci4]}  -- 95% CI half-widths
    for algorithm in algorithms:
        auc_mean[algorithm] = [np.mean(seed_aucs) for seed_aucs in auc_data_per_seed[algorithm]]
        auc_ci[algorithm] = []
        for seed_aucs in auc_data_per_seed[algorithm]:
            n = len(seed_aucs)
            std = np.std(seed_aucs, ddof=1)  # sample std
            t_crit = scipy_stats.t.ppf(0.975, df=n - 1)  # 95% CI, t-distribution
            auc_ci[algorithm].append(t_crit * std / np.sqrt(n))
    
    # Normalize by uniform's mean (so uniform mean is always 1.0)
    uniform_mean_values = auc_mean[UNIFORM]
    normalized_mean = {}
    normalized_ci = {}
    for algorithm in algorithms:
        normalized_mean[algorithm] = [auc_mean[algorithm][i] / uniform_mean_values[i] for i in range(4)]
        # Normalize CI as well (by dividing by uniform's mean, not by uniform's CI)
        normalized_ci[algorithm] = [auc_ci[algorithm][i] / uniform_mean_values[i] for i in range(4)]
    
    # Create split axis figure (top shows high values, bottom shows low values)
    # Note: subplots returns axes top-to-bottom, so first axis is physically on top
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, 
                                             gridspec_kw={'height_ratios': [1, 2], 'hspace': 0.05})
    
    x = np.arange(len(algorithms))
    width = 0.2  # width of each bar
    
    # Color palette for the 4 problem+dataset combinations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # blue, orange, green, red
    labels = [comb[4] for comb in combinations]
    
    # Plot bars for each problem+dataset combination on both axes
    for i, (problem, dataset, num_classes, merged_classes, label) in enumerate(combinations):
        values = [normalized_mean[algo][i] for algo in algorithms]
        errors = [normalized_ci[algo][i] for algo in algorithms]
        offset = (i - 1.5) * width
        
        # Top axis (1.1-1.6) - for outliers
        bars_top = ax_top.bar(x + offset, values, width, label=label, color=colors[i], 
                             alpha=0.85, edgecolor='black', linewidth=1.2, 
                             yerr=errors, capsize=3, error_kw={'linewidth': 1.5})
        
        # Bottom axis (0.95-1.05) - for normal range
        bars_bottom = ax_bottom.bar(x + offset, values, width, color=colors[i], 
                                    alpha=0.85, edgecolor='black', linewidth=1.2, 
                                    yerr=errors, capsize=3, error_kw={'linewidth': 1.5})
        
        # Add value labels (place on whichever axis shows the bar)
        for bar_bottom, bar_top, value, error in zip(bars_bottom, bars_top, values, errors):
            if value > 1.08:  # Bar is in top range (outlier)
                height = bar_top.get_height()
                ax_top.text(bar_top.get_x() + bar_top.get_width()/2., height + error,
                           f'{value:.3f}',
                           ha='center', va='bottom', fontsize=7.5, fontweight='bold')
            else:  # Bar is in bottom range (normal)
                height = bar_bottom.get_height()
                ax_bottom.text(bar_bottom.get_x() + bar_bottom.get_width()/2., height + error,
                              f'{value:.3f}',
                              ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    
    # Set y-axis limits for split view
    ax_top.set_ylim(1.1, 1.6)       # Top shows outliers
    ax_bottom.set_ylim(0.95, 1.05)  # Bottom shows normal range
    
    # Hide the spines between axes
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.xaxis.tick_top()
    ax_top.tick_params(labeltop=False)
    ax_bottom.xaxis.tick_bottom()
    
    # Add diagonal lines to indicate the break
    d = 0.015  # size of diagonal lines
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    
    # Formatting
    ax_bottom.set_xlabel('Algorithm', fontsize=13, fontweight='bold')
    # Place ylabel between both axes
    fig.text(0.04, 0.5, 'Normalized Loss AUC', va='center', rotation='vertical', 
             fontsize=13, fontweight='bold')
    
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels([_nice_name[algo] for algo in algorithms], fontsize=11, fontweight='bold')
    ax_top.legend(loc='upper right', fontsize=10, framealpha=0.95)
    
    ax_bottom.grid(True, alpha=0.3, axis='y')
    ax_top.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout manually to avoid tight_layout warning
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08)
    return fig

def print_auc_table_console(problem, dataset, num_classes, metric='test', merged_classes=None):
    """
    Print AUC table to console in formatted text.
    
    Args:
        problem: 'noise' or 'curriculum'
        dataset: 'mnist' or 'cifar'
        num_classes: number of classes (before merging if applicable)
        metric: 'train' or 'test'
        merged_classes: dict mapping original class to merged class (for regrouping)
    """
    results = compute_auc_table(problem, dataset, num_classes, metric, merged_classes)
    algorithms = EXPERIMENT_CONFIG["algorithms"]
    
    print(f"\n{'='*60}")
    print(f"Total Loss AUC: {_nice_name[problem]} - {_nice_name[dataset]} - {_nice_name[metric]}")
    print(f"{'='*60}")
    print(f"{'Algorithm':<15} {'Total Loss AUC':<25}")
    print(f"{'-'*60}")
    
    for algorithm in algorithms:
        total_auc = results[algorithm]['total_auc']
        print(f"{_nice_name[algorithm]:<15} {total_auc:<25.2f}")
    
    print(f"{'='*60}\n")

def generate_table():
    # Generate AUC tables
    print("\nGenerating Total Loss AUC tables...")

    # Mapping for CURRICULUM: regroup classes like in graph 5
    merged_map = {
        0: 0,  # class 0 -> group 0
        1: 1, 2: 1,  # classes 1,2 -> group 1
        3: 2, 4: 2, 5: 2,  # classes 3,4,5 -> group 2
        6: 3, 7: 3, 8: 3, 9: 3  # classes 6,7,8,9 -> group 3
    }

    # For NOISE problems
    print("\n--- NOISE Problem ---")
    try:
        print_auc_table_console(NOISE, MNIST_NAME, 4)
    except Exception as e:
        print(f"Error generating console table for NOISE + MNIST: {e}")

    try:
        print_auc_table_console(NOISE, CIFAR_NAME, 4)
    except Exception as e:
        print(f"Error generating console table for NOISE + CIFAR: {e}")

    # For CURRICULUM (10 classes, regrouped to 4)
    print("\n--- CURRICULUM Problem (regrouped to 4 groups) ---")
    try:
        print_auc_table_console(CURRICULUM, MNIST_NAME, 10, merged_classes=merged_map)
    except Exception as e:
        print(f"Error generating console table for CURRICULUM + MNIST: {e}")

    try:
        print_auc_table_console(CURRICULUM, CIFAR_NAME, 10, merged_classes=merged_map)
    except Exception as e:
        print(f"Error generating console table for CURRICULUM + CIFAR: {e}")
    
    # Generate merged bar graph
    print("\nGenerating merged bar graph...")
    try:
        fig = generate_merged_auc_bar_graph()
        fig.savefig(os.path.join(GRAPH_DIR, 'table_merged_auc_normalized.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Merged bar graph saved!")
    except Exception as e:
        print(f"Error generating merged bar graph: {e}")
    
    print("\nTotal Loss AUC analysis complete!")


def print_pairwise_ttest_table():
    """Print pairwise Welch's t-tests comparing GMC (DOTP) against all other
    algorithms for each (problem, dataset) condition.  Reports t-statistic,
    p-value, and significance stars.
    """
    merged_map = {
        0: 0,
        1: 1, 2: 1,
        3: 2, 4: 2, 5: 2,
        6: 3, 7: 3, 8: 3, 9: 3,
    }
    combinations = [
        (CURRICULUM, MNIST_NAME, 10, merged_map),
        (CURRICULUM, CIFAR_NAME, 10, merged_map),
        (NOISE,      MNIST_NAME, 4,  None),
        (NOISE,      CIFAR_NAME, 4,  None),
    ]

    ref_algo = DOTP  # GMC
    other_algos = [a for a in EXPERIMENT_CONFIG["algorithms"] if a != ref_algo]

    print(f"\n{'='*80}")
    print(f"Pairwise Welch's t-tests: {_nice_name[ref_algo]} vs each baseline (AUC, metric=test)")
    print(f"{'='*80}")
    header = f"{'Condition':<28} {'Baseline':<12} {'t-stat':>8} {'p-value':>10} {'sig':>5}"
    print(header)
    print("-" * 80)

    for problem, dataset, num_classes, merged_classes in combinations:
        ref_aucs = compute_total_loss_auc_per_seed(
            problem, dataset, ref_algo, num_classes, 'test', merged_classes)
        if not ref_aucs:
            continue
        cond_label = f"{_nice_name[problem]}+{_nice_name[dataset]}"

        for other in other_algos:
            other_aucs = compute_total_loss_auc_per_seed(
                problem, dataset, other, num_classes, 'test', merged_classes)
            if not other_aucs:
                continue
            t_stat, p_val = scipy_stats.ttest_ind(
                ref_aucs, other_aucs, equal_var=False)
            if p_val < 0.001:
                stars = "***"
            elif p_val < 0.01:
                stars = "**"
            elif p_val < 0.05:
                stars = "*"
            else:
                stars = "ns"
            print(f"{cond_label:<28} {_nice_name[other]:<12} {t_stat:>8.3f} {p_val:>10.4f} {stars:>5}")

    print(f"{'='*80}")
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns not significant\n")


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        run_all_experiments()

    elif sys.argv[1] == "generate_graphs":
        algorithms = [UNIFORM, CURIOSITY, DOTP]
        # algorithms = [GRADNORMLAST, GRADNORMALL, LOSSDELTA]
        
        # Ablation variants (Appendix C)
        # algorithms = [DOTP, DOTP_OUTER, DOTP_COS]
        generate_comparison_graphs(algorithms)

        # All six methods for mean loss comparison
        algorithms = [UNIFORM, CURIOSITY, DOTP, GRADNORMLAST, GRADNORMALL, LOSSDELTA]
        generate_mean_loss_graphs(algorithms)

        # AUC summary table and bar graph
        generate_table()

        # Pairwise statistical tests
        print_pairwise_ttest_table()

    else:
        print("Unknown command:", sys.argv[1])
    

# Duration averages:

# Experiment: curriculum_cifar_curiosity
# Average duration: 0:31:26

# Experiment: curriculum_cifar_dotprod
# Average duration: 0:33:30

# Experiment: curriculum_cifar_gradnormall
# Average duration: 1:51:09

# Experiment: curriculum_cifar_gradnormlast
# Average duration: 0:42:01

# Experiment: curriculum_cifar_lossdelta
# Average duration: 0:31:25

# Experiment: curriculum_cifar_uniform
# Average duration: 0:29:53

# Experiment: curriculum_mnist_curiosity
# Average duration: 2:43:36

# Experiment: curriculum_mnist_dotprod
# Average duration: 2:46:05

# Experiment: curriculum_mnist_gradnormall
# Average duration: 8:29:59

# Experiment: curriculum_mnist_gradnormlast
# Average duration: 3:44:30

# Experiment: curriculum_mnist_lossdelta
# Average duration: 2:42:19

# Experiment: curriculum_mnist_uniform
# Average duration: 2:32:42

# Experiment: noise_cifar_curiosity
# Average duration: 0:31:21

# Experiment: noise_cifar_dotprod
# Average duration: 0:33:22

# Experiment: noise_cifar_gradnormall
# Average duration: 1:53:08

# Experiment: noise_cifar_gradnormlast
# Average duration: 0:45:17

# Experiment: noise_cifar_lossdelta
# Average duration: 0:31:22

# Experiment: noise_cifar_uniform
# Average duration: 0:29:52

# Experiment: noise_mnist_curiosity
# Average duration: 0:32:40

# Experiment: noise_mnist_dotprod
# Average duration: 0:33:20

# Experiment: noise_mnist_gradnormall
# Average duration: 2:15:55

# Experiment: noise_mnist_gradnormlast
# Average duration: 0:48:11

# Experiment: noise_mnist_lossdelta
# Average duration: 0:32:32

# Experiment: noise_mnist_uniform
# Average duration: 0:30:34

