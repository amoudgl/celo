"""
Collection of all the util methods used in this repo.
"""

import functools
import os
from collections import OrderedDict
from typing import TypeVar

import jax
import jax.numpy as jnp
import learned_optimization
import numpy as np
from absl import logging
from flax import serialization
from learned_optimization.optimizers import base as opt_base

T = TypeVar("T")


def pytree_to_ordered_dict(pytree):
    # since jax tree flatten order is deterministic, we return ordered dict
    key_tree = learned_optimization.tree_utils.map_named(lambda k, v: k, pytree)
    keys, _ = jax.tree_util.tree_flatten(key_tree)
    vals, _ = jax.tree_util.tree_flatten(pytree)
    return OrderedDict(zip(keys, vals))


def load_from_pretrained(input_tree, pretrained_tree):
    input_dict = pytree_to_ordered_dict(input_tree)
    pretrained_dict = pytree_to_ordered_dict(pretrained_tree)
    updated_vals = []
    not_found = []
    found = []
    # copy params from pretrained_dict which exist in param_dict
    for k, v in input_dict.items():
        val = None
        if k in pretrained_dict:
            val = pretrained_dict[k]
            found.append(k)
        else:
            val = v
            not_found.append(k)
        updated_vals.append(val)
    if len(not_found) > 0:
        logging.info(f"Params NOT found in checkpoint: {not_found}")
    logging.info(f"Params loaded from checkpoint: {found}")
    struct = jax.tree_util.tree_structure(input_tree)
    output_tree = struct.unflatten(updated_vals)
    return output_tree


def load_state(path: str, state: T) -> T:
    """Load a pytree state directly from a file.

    Args:
      path: path to load pytree state from.
      state: pytree whose structure should match that of the stucture saved in the
        path. The values of this pytree are not used.

    Returns:
      The restored pytree matching the pytree structure of state.
    """
    with open(path, "rb") as fp:
        state_new = serialization.from_bytes(state, fp.read())
    tree = jax.tree_util.tree_structure(state)
    leaves_new = jax.tree_util.tree_leaves(state_new)
    return jax.tree_util.tree_unflatten(tree, leaves_new)


def init_lopt_from_ckpt(lopt, path, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    theta = lopt.init(key)
    pretrained_params = load_state(path, theta)
    opt = lopt.opt_fn(pretrained_params)
    return opt


@functools.lru_cache(None)
def cached_jit(fn, *args, **kwargs):
    return jax.jit(fn, *args, **kwargs)


class WeightDecayWrapper(opt_base.Optimizer):
    """Weight decay wrapper to add weight decay to the optimizer."""
    def __init__(self, opt, weight_decay=0.0, add_to_loss=True):
        super().__init__()
        self.opt = opt
        self.weight_decay = weight_decay
        self.add_to_loss = add_to_loss

    def get_params(self, opt_state):
        return self.opt.get_params(opt_state)

    def set_params(self, state, params):
        return self.opt.set_params(state, params)

    def get_state(self, opt_state):
        return self.opt.get_state(opt_state)

    def init(self, params, model_state=None, **kwargs):
        return self.opt.init(params, model_state=model_state, **kwargs)

    def update(self, opt_state, grads, loss=None, model_state=None, **kwargs):
        ps = self.opt.get_params(opt_state)

        if self.add_to_loss:
            l2 = [jnp.sum(p**2) for p in jax.tree_util.tree_leaves(ps)]
            loss = loss + sum([x * self.weight_decay for x in l2])

        grad_l2 = jax.tree_util.tree_map(lambda p: self.weight_decay * p, ps)
        grads = jax.tree_util.tree_map(lambda g, g_l2: g + g_l2, grads, grad_l2)

        return self.opt.update(
            opt_state, grads, loss=loss, model_state=model_state, **kwargs)


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def get_train_dirs(experiment_root, exp_name):
    """
    Returns (root, log_dir) for training artifacts.
    experiment_root: root directory for all experiment artifacts
    exp_name: directory name for the training run (e.g. <exp_name>)
    root is the exp_name directory itself (no checkpoint/ subdir).
    """
    experiment_root = os.path.expanduser(experiment_root)
    root = os.path.join(experiment_root, "train", exp_name)
    log_dir = os.path.join(root, "logs")
    os.makedirs(root, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return root, log_dir


def get_test_dirs(experiment_root, exp_name, task, seed):
    """
    Returns (root, log_dir) for test artifacts.
    experiment_root: root directory for all experiment artifacts
    exp_name: directory name of the trained optimizer (e.g. <exp_name>)
    task: str
    seed: int or str
    """
    experiment_root = os.path.expanduser(experiment_root)
    root = os.path.join(experiment_root, "test", exp_name, task, f"seed_{seed}")
    os.makedirs(root, exist_ok=True)
    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return root, log_dir
