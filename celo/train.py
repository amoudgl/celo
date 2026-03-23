"""
Script to meta-train learned optimizers.
"""

import json
import os
import pickle
import shutil
import time

import jax
import numpy as np
import tqdm
from absl import app, flags, logging
from learned_optimization import checkpoints, filesystem, summary, tree_utils
from learned_optimization.optimizers import optax_opts

from celo import gradient_learner
from celo.factory import (
    get_augmented_task_family,
    get_grad_estimator,
    get_optimizer,
    get_task_family,
)
from celo.utils import get_train_dirs, load_from_pretrained
import jax.numpy as jnp
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")

# ----------------------------------------------------------------
# Meta-training config
# ----------------------------------------------------------------
# fmt: off
flags.DEFINE_string("optimizer", None, "learned optimizer to meta-train")
flags.DEFINE_string("task", None, "task for meta-training")
flags.DEFINE_string("trainer", None, "gradient estimator for meta-training")
flags.DEFINE_string("experiment_root", os.environ.get("EXPERIMENT_ROOT", "./experiments"), "Root directory for all experiment artifacts")
flags.DEFINE_string("init_from_ckpt", None, "path to serialized lopt checkpoint for initialization.")
flags.DEFINE_string("init_ckpt_optimizer_name", None, "checkpoint optimizer name to partially load params, skip if optimizer pytree matches checkpoint pytree.")
flags.DEFINE_bool("resume", False, "resume training from latest checkpoint using trainer saved state")
flags.DEFINE_bool("train_partial", False, "enables training subset of celo params while keeping rest frozen")
flags.DEFINE_string("exp_name", 'exp', "Experiment name, used in wandb. If not set, defaults to 'exp'.")
flags.DEFINE_string("exp_id", None, "Manually specify exp ID, used as a directory name for artifacts. If None, auto-generated from exp name, timestamp and job id from env (e.g. SLURM_JOB_ID) or 'local'.")
flags.DEFINE_float("outer_lr", 1e-4, "learning rate of meta-trainer")
flags.DEFINE_integer("outer_iterations", 100000, "number of meta-iterations")
flags.DEFINE_integer("ckpt_interval", 5000, "checkpoint save interval")
flags.DEFINE_integer("metrics_every", 50, "log interval of additional metrics and intermediate values in jit functions")
flags.DEFINE_integer("seed", 0, "training seed")
flags.DEFINE_string("aug", None, "task augmentation")
flags.DEFINE_bool("disable_wandb", False, "disable wandb logging e.g. for debugging")
flags.DEFINE_string("wandb_project", "ttt-train", "wandb project name")
flags.DEFINE_string("wandb_entity", "amoudgl", "wandb project name")
# flags.DEFINE_string("wandb_tags", None, "optionally specify wandb tags separated by comma like 'tag1,tag2,tag3'")
# fmt: on

flags.mark_flag_as_required("optimizer")
flags.mark_flag_as_required("task")
flags.mark_flag_as_required("trainer")
FLAGS = flags.FLAGS


def train(unused_argv):
    seed = FLAGS.seed
    key = jax.random.PRNGKey(seed)
    
    config = FLAGS.flag_values_dict()

    # setup log and checkpoint directories
    if FLAGS.exp_id:
        exp_id = FLAGS.exp_id
    else:
        # if exp_id is not set, generate a unique exp_id based on the timestamp
        # and hyperparameters
        timestamp = time.strftime("%y%m%d%H")
        exp_id = f"{FLAGS.exp_name}_{timestamp}_lr{FLAGS.outer_lr}_seed{FLAGS.seed}"
        config["exp_id"] = exp_id
    root, log_dir = get_train_dirs(FLAGS.experiment_root, exp_id)

    # initialize logger
    if FLAGS.disable_wandb:
        summary_writer = summary.PrintWriter()
    else:
        wandb_writer = summary.WandbWriter(
            project=FLAGS.wandb_project,
            entity=FLAGS.wandb_entity,
            logdir=log_dir,
            config=config,
            name=FLAGS.exp_name,
        )
        
        # Dump source code directory to wandb
        import wandb
        celo_dir = os.path.join(os.path.dirname(__file__), '..', 'celo')
        artifact = wandb.Artifact(name=f"celo-source-code-{exp_id}", type="source-code")
        artifact.add_dir(celo_dir)
        wandb.log_artifact(artifact)
        
        summary_writer = summary.MultiWriter(wandb_writer, summary.PrintWriter())

    # save config
    with open(os.path.join(root, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
        logging.info(json.dumps(config, ensure_ascii=False, indent=4))
        logging.info("Saved config at: {}".format(os.path.join(root, "config.json")))

    # setup optimizer and tasks for meta-training
    logging.info("--- Starting training setup ---")
    
    lopt = get_optimizer(FLAGS.optimizer)
    logging.info(f"Optimizer: {FLAGS.optimizer}")
    
    task_families = get_task_family(FLAGS.task)
    logging.info(f"Task families: {len(task_families)}")
    
    if FLAGS.aug:
        task_families = [get_augmented_task_family(FLAGS.aug, family) for family in task_families]
        logging.info(f"Augmentation: {FLAGS.aug}")

    # setup outer trainer
    logging.info("Setting up gradient estimators...")
    gradient_estimators = []
    for i, task_family in enumerate(task_families):
        logging.info(f"  Creating gradient estimator {i+1}/{len(task_families)}...")
        grad_est = get_grad_estimator(FLAGS.trainer, task_family, lopt)
        gradient_estimators.append(grad_est)
    
    theta_opt = optax_opts.AdamW(FLAGS.outer_lr)
    frozen_keys = lopt.get_frozen_param_keys() if FLAGS.train_partial else None
    logging.info(f"Outer optimizer: AdamW(lr={FLAGS.outer_lr}), frozen_keys={frozen_keys}")
    
    # Print param routing for dual optimizers (if supported)
    if hasattr(lopt, 'print_routing'):
        for task_family in task_families:
            logging.info(f"Parameter routing for task family: {task_family.name}")
            task = task_family.sample_task(key)
            sample_params = task.init(key)
            lopt.print_routing(sample_params)
    
    outer_trainer = gradient_learner.SingleMachineGradientLearner(
        lopt, gradient_estimators, theta_opt, frozen_keys
    )
    logging.info("Outer trainer created")

    # initialize learned optimizer params from checkpoint if specified
    if FLAGS.init_from_ckpt:
        ckpt_path = os.path.expanduser(FLAGS.init_from_ckpt)  # treat as direct path
        logging.info(f"Loading params from checkpoint: {ckpt_path}")
        lopt_tree = lopt.init(key)
        if FLAGS.init_ckpt_optimizer_name:
            ckpt_lopt = get_optimizer(FLAGS.init_ckpt_optimizer_name)
            ckpt_tree = ckpt_lopt.init(key)
        else:
            ckpt_tree = lopt_tree
        pretrained_tree = checkpoints.load_state(ckpt_path, ckpt_tree)
        init_tree = load_from_pretrained(lopt_tree, pretrained_tree)
        outer_trainer.gradient_learner._init_theta = init_tree
    
    logging.info("Initializing outer trainer state...")
    outer_trainer_state = outer_trainer.init(key)
    logging.info("Outer trainer state initialized")

    # initialize meta-training
    start_iteration = 0
    losses = []
    m = []
    m_xs = []
    is_nested_dict = lambda d: any(isinstance(v, dict) for v in d.values())

    # resume from checkpoint if specified
    if FLAGS.resume:
        ckpt_path = os.path.join(root, "trainer_latest.pkl")
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        outer_trainer_state = ckpt["state"]
        m = ckpt["metrics"]
        m_xs = ckpt["metrics_xs"]
        losses = ckpt["losses"]
        start_iteration = (
            outer_trainer_state.gradient_learner_state.theta_opt_state.iteration.item()
        )
        if start_iteration == FLAGS.outer_iterations:
            logging.info(
                "Resumed job has already completed training, change `--outer_iterations` flag to train for longer"
            )
            return
    summary_writer.scalar("seed", seed, step=start_iteration)

    logging.info("--- Starting meta-training loop ---")
    logging.info(f"  iters: {start_iteration} -> {FLAGS.outer_iterations}, lr: {FLAGS.outer_lr}, task: {FLAGS.task}")

    # meta-training loop
    for i in tqdm.trange(
        start_iteration,
        FLAGS.outer_iterations + 1,
        initial=start_iteration,
        total=FLAGS.outer_iterations + 1,
    ):
        with_m = True if i % FLAGS.metrics_every == 0 else False
        key1, key = jax.random.split(key)
        
        if i == start_iteration:
            logging.info(f"Starting iteration {i}...")
        
        outer_trainer_state, loss, metrics = outer_trainer.update(
            outer_trainer_state, key1, with_metrics=with_m
        )
        losses.append(loss)
        
        if i == start_iteration:
            logging.info(f"First iteration {i} done, loss={loss:.6f}")

        # log summaries to wandb
        if with_m:
            log_m = {}
            avg_loss = float(np.asarray(np.mean(losses)))
            summary_writer.scalar("average_meta_loss", avg_loss, step=i)
            log_m["average_meta_loss"] = avg_loss
            for k, v in metrics.items():
                agg_type, metric_name = k.split("||")
                if agg_type == "collect":
                    summary_writer.histogram(metric_name, v, step=i)
                else:
                    summary_writer.scalar(metric_name, v, step=i)
                    # Materialize to Python float to avoid retaining JAX arrays in m (host memory leak)
                    log_m[metric_name] = float(np.asarray(v))
            summary_writer.flush()
            m.append(log_m)
            m_xs.append(i)
            losses = []

        # save learned optimizer params periodically
        if i % FLAGS.ckpt_interval == 0:
            theta = outer_trainer.get_meta_params(outer_trainer_state)
            checkpoints.save_state(os.path.join(root, f"theta_{i}.state"), theta)
            logging.info(f"Saved learned optimizer params at: {os.path.join(root, f'theta_{i}.state')}")
            trainer_state = {
                "state": outer_trainer_state,
                "metrics": m,
                "metrics_xs": m_xs,
                "losses": losses,
            }
            with open(os.path.join(root, "trainer_latest.pkl"), "wb") as f:
                pickle.dump(trainer_state, f)
            logging.info(f"Saved trainer state at: {os.path.join(root, 'trainer_latest.pkl')}")

            # save logs
            stacked_metrics = tree_utils.tree_zip_onp(m)
            ret = {f"train/{k}": v for k, v in stacked_metrics.items()}
            ret["train/xs"] = np.asarray(m_xs)
            assert is_nested_dict(ret) == False, "npz save does not allow nested dictionaries"
            path = os.path.join(root, "metrics.npz")
            np.savez(path, **ret)
            logging.info(f"Saved training metrics at: {path}")

    # save final learned state
    path = os.path.join(root, "theta.state")
    theta = outer_trainer.get_meta_params(outer_trainer_state)
    checkpoints.save_state(path, theta)
    logging.info(f"Saved learned optimizer params at: {path}")
    trainer_state = {
        "state": outer_trainer_state,
        "metrics": m,
        "metrics_xs": m_xs,
        "losses": losses,
    }
    trainer_path = os.path.join(root, "trainer_latest.pkl")
    with open(trainer_path, "wb") as f:
        pickle.dump(trainer_state, f)
    logging.info(f"Saved trainer state at: {trainer_path}")

    # save logs
    stacked_metrics = tree_utils.tree_zip_onp(m)
    ret = {f"train/{k}": v for k, v in stacked_metrics.items()}
    ret["train/xs"] = np.asarray(m_xs)
    assert is_nested_dict(ret) == False, "npz save does not allow nested dictionaries"
    path = os.path.join(root, "metrics.npz")
    np.savez(path, **ret)
    logging.info(f"Saved training metrics at: {path}")
    logging.info("Finished training.")


if __name__ == "__main__":
    app.run(train)
