"""
Script to test learned optimizer on a given task.
"""

import os
import pickle

import jax
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

# ----------------------------------------------------------------
# Meta-testing config
# ----------------------------------------------------------------
jax.config.parse_flags_with_absl()
# fmt: off
flags.DEFINE_string("optimizer", None, "learned optimizer to meta-test")
flags.DEFINE_enum("optimizer_type", 'learned', ["learned", "handcrafted"], "if learned, checkpoint path will be used to load optimizer params")
flags.DEFINE_string("taskset", None, "taskset for meta-testing")
flags.DEFINE_string("task", None, "task for meta-testing")
flags.DEFINE_float("test_reparam", None, "reparam aug at test-time")
flags.DEFINE_integer("steps", None, "number of inner training steps")
flags.DEFINE_string("ckpt_dir", None, "directory to load checkpoint from")
flags.DEFINE_string("ckpt_file", "theta.state", "name of checkpoint file")
flags.DEFINE_string("test_log_dir", "logs/test", "common dir for all tensorboard logs")
flags.DEFINE_string("results_save_dir", "eval_results/", "common dir to dump meta-testing results")
flags.DEFINE_integer("seed", None, "seed id")
flags.DEFINE_integer("seeds", 3, "number of runs")
flags.DEFINE_string("name", None, "run name")
flags.DEFINE_integer("metrics_every", 10, "plot metrics this frequently")
# fmt: on

flags.mark_flag_as_required("optimizer")
flags.mark_flag_as_required("steps")
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

    # load pretrained optimizer params
    task = eval(f"{task_name}()")
    if FLAGS.optimizer_type == "learned":
        # if learned optimizer is specified, pick params from checkpoint
        lopt = get_optimizer(FLAGS.optimizer)
        theta = lopt.init(jax.random.PRNGKey(0))
        ckpt_path = os.path.join(FLAGS.ckpt_dir, FLAGS.ckpt_file)
        pretrained_params = checkpoints.load_state(ckpt_path, theta)
        opt = lopt.opt_fn(pretrained_params)
        ckpt_dirname = os.path.abspath(FLAGS.ckpt_dir).split("/")[-1]
        eval_name = f"ckpt:{ckpt_dirname}__task:{task_name}__steps:{FLAGS.steps}__seed:{seed}"
    else:
        opt = get_optimizer(FLAGS.optimizer)
        eval_name = f"ckpt:{FLAGS.optimizer}__task:{task_name}__steps:{FLAGS.steps}__seed:{seed}"

    # set task and seed
    logdir = os.path.join(FLAGS.test_log_dir, eval_name)
    save_dir = os.path.join(FLAGS.results_save_dir, eval_name)
    filesystem.make_dirs(save_dir)
    filesystem.make_dirs(logdir)
    key = jax.random.PRNGKey(seed)
    summary_writer = summary.MultiWriter(summary.PrintWriter(), summary.JaxboardWriter(logdir))

    # launch inner training
    results = single_task_training_curves(
        task=task,
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
    path = os.path.join(save_dir, "results.pkl")
    with open(path, "wb") as f:
        pickle.dump(results, f)
        logging.info(f"Saved results at: {path}")


if __name__ == "__main__":
    app.run(test)
