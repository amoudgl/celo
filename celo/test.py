"""
Script to test learned optimizer on a given task.
"""

import os
import pickle

import jax
import numpy as np
from absl import app, flags, logging
from learned_optimization import checkpoints, filesystem, summary
from learned_optimization.research.general_lopt.tasks.fast_mlp_diff_data import *
from learned_optimization.tasks.fixed.conv import *
from learned_optimization.tasks.fixed.es_wrapped import *
from learned_optimization.tasks.fixed.image_mlp import *
from learned_optimization.tasks.fixed.image_mlp_ae import *
from learned_optimization.tasks.fixed.lopt import *
from learned_optimization.tasks.fixed.mlp_mixer import *
from learned_optimization.tasks.fixed.resnet import *
from learned_optimization.tasks.fixed.rnn_lm import *
from learned_optimization.tasks.fixed.transformer_lm import *
from learned_optimization.tasks.fixed.vit import *
from learned_optimization.tasks.quadratics import *

from celo.eval_training import single_task_training_curves
from celo.factory import get_optimizer
from celo.utils import get_test_dirs
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")

# ----------------------------------------------------------------
# Meta-testing config
# ----------------------------------------------------------------
# fmt: off
flags.DEFINE_string("optimizer", None, "learned optimizer to meta-test")
flags.DEFINE_enum("optimizer_type", 'learned', ["learned", "handcrafted"], "if learned, checkpoint path will be used to load optimizer params")
flags.DEFINE_string("taskset", None, "taskset for meta-testing")
flags.DEFINE_string("task", None, "task for meta-testing")
flags.DEFINE_float("test_reparam", None, "reparam aug at test-time")
flags.DEFINE_integer("steps", None, "number of inner training steps")
flags.DEFINE_string("exp_id", None, "Experiment ID (artifact directory name) to evaluate. Required.")
flags.DEFINE_string("exp_name", None, "Experiment name, used to log on wandb. If none, exp_id is used.")
flags.DEFINE_string("ckpt_file", "theta.state", "name of checkpoint file")
flags.DEFINE_string("experiment_root", os.environ.get("EXPERIMENT_ROOT", "./experiments"), "Root directory for all experiment artifacts")
flags.DEFINE_integer("seed", 0, "seed id")
flags.DEFINE_bool("disable_wandb", False, "disable wandb logging e.g. for debugging")
flags.DEFINE_string("wandb_project", "celo-test", "wandb project name")
flags.DEFINE_string("wandb_entity", None, "wandb entity name")
flags.DEFINE_integer("metrics_every", 10, "compute additional validation metrics this frequently")
# fmt: on

flags.mark_flag_as_required("optimizer")
flags.mark_flag_as_required("steps")
flags.mark_flag_as_required("exp_id")
FLAGS = flags.FLAGS


# ----------------------------------------------------------------
# Meta-testing
# ----------------------------------------------------------------
def test(unused_argv):
    """Tests a pre-trained optimizer with single seed on a given task.
    Uses learned optimization repo method `single_task_training_curves`.

    Args:
        unused_argv: Unrecognized arguments by flags module (not used)
    """
    # load taskset
    task_name = FLAGS.task
    seed = FLAGS.seed

    # append slurm job id to config, if running on SLURM
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "nojid")
    config = FLAGS.flag_values_dict()
    config["slurm_job_id"] = slurm_job_id

    # load pretrained optimizer params
    task = eval(f"{task_name}()")
    if FLAGS.optimizer_type == "learned":
        # if learned optimizer is specified, pick params from checkpoint
        lopt = get_optimizer(FLAGS.optimizer)
        theta = lopt.init(jax.random.PRNGKey(0))
        ckpt_path = os.path.join(
            os.path.expanduser(FLAGS.experiment_root), "train", FLAGS.exp_id, FLAGS.ckpt_file
        )
        pretrained_params = checkpoints.load_state(ckpt_path, theta)
        opt = lopt.opt_fn(pretrained_params)
    else:
        opt = get_optimizer(FLAGS.optimizer)

    # setup directories
    save_dir, log_dir = get_test_dirs(FLAGS.experiment_root, FLAGS.exp_id, FLAGS.task, FLAGS.seed)
    key = jax.random.PRNGKey(seed)
    if FLAGS.disable_wandb:
        summary_writer = summary.PrintWriter()
    else:
        wandb_writer = summary.WandbWriter(
            project=FLAGS.wandb_project,
            entity=FLAGS.wandb_entity,
            logdir=log_dir,
            config=config,
            name=FLAGS.exp_id if FLAGS.exp_name is None else FLAGS.exp_name,
        )

        # Dump source code directory to wandb
        import wandb
        celo_dir = os.path.join(os.path.dirname(__file__), '..', 'celo')
        artifact = wandb.Artifact(name=f"celo-source-code-{FLAGS.exp_id}", type="source-code")
        artifact.add_dir(celo_dir)
        wandb.log_artifact(artifact)

        summary_writer = summary.MultiWriter(wandb_writer, summary.PrintWriter())

    # launch inner training
    results = single_task_training_curves(
        task=task,
        task_name=task_name,
        opt=opt,
        num_steps=FLAGS.steps,
        key=key,
        eval_every=10,
        eval_batches=5,
        last_eval_batches=10,
        eval_task=None,
        device=None,
        metrics_every=FLAGS.metrics_every,
        summary_writer=summary_writer,
    )

    is_nested_dict = lambda d: any(isinstance(v, dict) for v in d.values())
    assert is_nested_dict(results) == False, "npz save does not allow nested dictionaries"
    path = os.path.join(save_dir, f"metrics_unroll{FLAGS.steps}.npz")
    np.savez(path, **results)
    logging.info(f"Saved test metrics at: {path}")


if __name__ == "__main__":
    app.run(test)
