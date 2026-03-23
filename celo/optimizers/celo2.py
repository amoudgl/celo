# Portions of this code are adapted from Google's learned_optimization repository
# (https://github.com/google/learned_optimization), which is licensed under the
# Apache License, Version 2.0. You may obtain a copy of the License at:
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Original copyright (c) 2021 Google LLC.
# Modifications copyright (c) 2025 Abhinav Moudgil.

"""
Learned-optimization wrappers for Adam, Celo2Base, and Celo2.

Celo2 and Celo2Base wrappers delegate core update logic to the
self-contained optax transformation in celo2_optax.py.
"""

import functools
import re
from typing import Any, Optional

import chex
import flax
import gin
import jax
from jax import lax
import jax.numpy as jnp
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import common
from learned_optimization.optimizers import base as opt_base
import optax

from celo.optimizers.celo2_optax import (
    vec_bias_correction,
    Celo2Transformation,
    Celo2State,
)


# =============================================================================
# Adam: optax transformation + learned_optimization wrapper
# =============================================================================

@flax.struct.dataclass
class AdamState:
    """Optax state for Adam."""
    rms_rolling: chex.ArrayTree
    mom_rolling: chex.ArrayTree
    iteration: jnp.ndarray


class AdamTransformation:
    """Adam optimizer optax transformation."""

    def __init__(self, b1=0.9, b2=0.999, epsilon=1e-8, bias_correction=True):
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon
        self.bias_correction = bias_correction

    def accumulators_for_decays(self):
        mom_decay = jnp.asarray((self.b1,))
        rms_decay = jnp.asarray((self.b2,))
        return common.vec_rolling_mom(mom_decay), common.vec_rolling_rms(rms_decay)

    def _compute_step(self, m, rms):
        rsqrt = lax.rsqrt(rms + self.epsilon)
        norm_g = m * rsqrt
        return norm_g[..., 0]

    def init(self, params: chex.ArrayTree) -> AdamState:
        mom_roll, rms_roll = self.accumulators_for_decays()
        return AdamState(
            rms_rolling=rms_roll.init(params),
            mom_rolling=mom_roll.init(params),
            iteration=jnp.asarray(0, dtype=jnp.int32),
        )

    def update(
        self,
        grads: chex.ArrayTree,
        state: AdamState,
        params: Optional[chex.ArrayTree] = None,
    ) -> tuple:
        """Returns (step, new_state). Step is the raw Adam update direction."""
        iteration = optax.safe_increment(state.iteration)
        mom_roll, rms_roll = self.accumulators_for_decays()
        next_mom_rolling = mom_roll.update(state.mom_rolling, grads)
        next_rms_rolling = rms_roll.update(state.rms_rolling, grads)

        if self.bias_correction:
            m = vec_bias_correction(next_mom_rolling.m, (self.b1,), iteration)
            rms = vec_bias_correction(next_rms_rolling.rms, (self.b2,), iteration)
        else:
            m = next_mom_rolling.m
            rms = next_rms_rolling.rms

        step = jax.tree_util.tree_map(self._compute_step, m, rms)

        new_state = AdamState(
            mom_rolling=next_mom_rolling,
            rms_rolling=next_rms_rolling,
            iteration=iteration,
        )
        return step, new_state


def scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    epsilon: float = 1e-8,
    bias_correction: bool = True,
) -> optax.GradientTransformation:
    """Create an optax GradientTransformation for Adam."""
    transformation = AdamTransformation(
        b1=b1, b2=b2, epsilon=epsilon, bias_correction=bias_correction,
    )

    def init_fn(params):
        return transformation.init(params)

    def update_fn(updates, state, params=None, **kwargs):
        return transformation.update(updates, state, params)

    return optax.GradientTransformation(init_fn, update_fn)


@flax.struct.dataclass
class AdamLoptState:
    """Learned-optimization state for Adam."""
    params: chex.ArrayTree
    rms_rolling: chex.ArrayTree
    mom_rolling: chex.ArrayTree
    iteration: jnp.ndarray
    state: chex.ArrayTree
    step_mult: jnp.ndarray


@gin.configurable
class Adam(lopt_base.LearnedOptimizer):
    """Adam optimizer.

    Wraps AdamTransformation for learned_optimization compatibility.
    The Adam update logic lives in AdamTransformation.
    """

    def __init__(
        self,
        step_mult=0.001,
        weight_decay=0.0,
        b1=0.9,
        b2=0.999,
        epsilon=1e-8,
        bias_correction=True,
    ):
        super().__init__()
        self.step_mult = step_mult
        self.weight_decay = weight_decay
        self._transformation = AdamTransformation(
            b1=b1, b2=b2, epsilon=epsilon, bias_correction=bias_correction,
        )

    def init(self, key) -> lopt_base.MetaParams:
        return {}

    def opt_fn(self, theta=None, is_training=True) -> opt_base.Optimizer:
        parent = self
        transformation = self._transformation

        class _Opt(opt_base.Optimizer):
            def __init__(self):
                super().__init__()
                self.step_mult = parent.step_mult

            @functools.partial(jax.jit, static_argnums=(0,))
            def init(self, params: Any, model_state=None, num_steps=None, key=None) -> AdamLoptState:
                optax_state = transformation.init(params)
                return AdamLoptState(
                    params=params,
                    state=model_state,
                    rms_rolling=optax_state.rms_rolling,
                    mom_rolling=optax_state.mom_rolling,
                    iteration=optax_state.iteration,
                    step_mult=jnp.asarray(parent.step_mult, dtype=jnp.float32),
                )

            @functools.partial(jax.jit, static_argnums=(0,))
            def update(self, opt_state, grads, loss=None, model_state=None, is_valid=False, key=None) -> AdamLoptState:
                inner_state = AdamState(
                    rms_rolling=opt_state.rms_rolling,
                    mom_rolling=opt_state.mom_rolling,
                    iteration=opt_state.iteration,
                )
                step, new_inner = transformation.update(grads, inner_state, opt_state.params)

                apply_update = lambda p, s: p - opt_state.step_mult * s - parent.weight_decay * p * opt_state.step_mult
                next_params = jax.tree_util.tree_map(apply_update, opt_state.params, step)

                ss = AdamLoptState(
                    params=next_params,
                    state=model_state,
                    mom_rolling=new_inner.mom_rolling,
                    rms_rolling=new_inner.rms_rolling,
                    iteration=new_inner.iteration,
                    step_mult=opt_state.step_mult,
                )
                return tree_utils.match_type(ss, opt_state)

        return _Opt()


# =============================================================================
# Celo2Base: learned_optimization wrapper
# =============================================================================

@flax.struct.dataclass
class Celo2BaseLoptState:
    """Learned-optimization state for Celo2Base."""
    params: chex.ArrayTree
    rms_rolling: chex.ArrayTree
    mom_rolling: chex.ArrayTree
    fac_rolling: chex.ArrayTree
    iteration: jnp.ndarray
    state: chex.ArrayTree
    step_mult: jnp.ndarray


@gin.configurable
class Celo2Base(lopt_base.LearnedOptimizer):
    """Celo2-base uses learned MLP update rule for all parameters.

    Wraps Celo2Transformation for learned_optimization compatibility.
    The MLP definition and all update logic live in Celo2Transformation.
    """

    def __init__(
        self,
        ff_hidden_size=8,
        ff_hidden_layers=2,
        initial_momentum_decays=(0.9, 0.99, 0.999),
        initial_rms_decays=(0.95,),
        initial_adafactor_decays=(0.9, 0.99, 0.999),
        exp_mult=0.0,
        step_mult=0.001,
        rmsmult=1.0,
        with_g=True,
        with_m=True,
        with_rms=True,
        with_rms_norm_g=True,
        with_rsqrt_rms=True,
        with_p=True,
        with_fac_norm_g=True,
        with_fac_rms=True,
        with_fac_rsqrt=True,
        with_grad_clip_feat=True,
        with_fac_mom_mult=True,
        with_rms_only_norm_g=True,
        param_scale_mult=False,
        precondition_output=False,
        normalize_input=True,
        normalize_output=True,
        weight_decay=0.0,
        aggregate_mag=False,
        bias_correction=False,
        mlp_activation="relu",
        orthogonalize=False,
        ns_coeffs=(3.4445, -4.7750, 2.0315),
        ns_iters=5,
        ns_eps=1e-8,
    ):
        super().__init__()
        self.step_mult = step_mult
        self.weight_decay = weight_decay

        self._celo2d_config = dict(
            ff_hidden_size=ff_hidden_size,
            ff_hidden_layers=ff_hidden_layers,
            initial_momentum_decays=initial_momentum_decays,
            initial_rms_decays=initial_rms_decays,
            initial_adafactor_decays=initial_adafactor_decays,
            exp_mult=exp_mult,
            rmsmult=rmsmult,
            with_g=with_g,
            with_m=with_m,
            with_rms=with_rms,
            with_rms_norm_g=with_rms_norm_g,
            with_rsqrt_rms=with_rsqrt_rms,
            with_p=with_p,
            with_fac_norm_g=with_fac_norm_g,
            with_fac_rms=with_fac_rms,
            with_fac_rsqrt=with_fac_rsqrt,
            with_grad_clip_feat=with_grad_clip_feat,
            with_fac_mom_mult=with_fac_mom_mult,
            with_rms_only_norm_g=with_rms_only_norm_g,
            param_scale_mult=param_scale_mult,
            precondition_output=precondition_output,
            normalize_input=normalize_input,
            normalize_output=normalize_output,
            aggregate_mag=aggregate_mag,
            bias_correction=bias_correction,
            mlp_activation=mlp_activation,
            orthogonalize=orthogonalize,
            ns_coeffs=ns_coeffs,
            ns_iters=ns_iters,
            ns_eps=ns_eps,
        )

    def _make_transformation(self, theta=None):
        return Celo2Transformation(theta=theta, **self._celo2d_config)

    def init(self, key) -> lopt_base.MetaParams:
        return self._make_transformation().init_meta_params(key)

    def opt_fn(self, theta, is_training=True) -> opt_base.Optimizer:
        parent = self
        transformation = self._make_transformation(theta=theta)

        class _Opt(opt_base.Optimizer):
            def __init__(self):
                super().__init__()
                self.step_mult = parent.step_mult

            @functools.partial(jax.jit, static_argnums=(0,))
            def init(self, params: Any, model_state=None, num_steps=None, key=None) -> Celo2BaseLoptState:
                optax_state = transformation.init(params)
                return Celo2BaseLoptState(
                    params=params,
                    state=model_state,
                    rms_rolling=optax_state.rms_rolling,
                    mom_rolling=optax_state.mom_rolling,
                    fac_rolling=optax_state.fac_rolling,
                    iteration=optax_state.iteration,
                    step_mult=jnp.asarray(parent.step_mult, dtype=jnp.float32),
                )

            @functools.partial(jax.jit, static_argnums=(0,))
            def update(self, opt_state, grads, loss=None, model_state=None, is_valid=False, key=None) -> Celo2BaseLoptState:
                inner_state = Celo2State(
                    rms_rolling=opt_state.rms_rolling,
                    mom_rolling=opt_state.mom_rolling,
                    fac_rolling=opt_state.fac_rolling,
                    iteration=opt_state.iteration,
                )
                step, new_inner = transformation.update(grads, inner_state, opt_state.params)

                apply_update = lambda p, s: p - opt_state.step_mult * s - parent.weight_decay * p * opt_state.step_mult
                next_params = jax.tree_util.tree_map(apply_update, opt_state.params, step)

                ss = Celo2BaseLoptState(
                    params=next_params,
                    state=model_state,
                    mom_rolling=new_inner.mom_rolling,
                    rms_rolling=new_inner.rms_rolling,
                    fac_rolling=new_inner.fac_rolling,
                    iteration=new_inner.iteration,
                    step_mult=opt_state.step_mult,
                )
                return tree_utils.match_type(ss, opt_state)

        return _Opt()


# =============================================================================
# Celo2: Combined optimizer for meta-training
# =============================================================================

def _path_flat_str(path):
    """Flatten path to string like 'mlp/~/linear_0/b' for regex and display."""
    parts = [str(getattr(entry, "key", entry)) for entry in path]
    return "/".join(parts)


def _mask_1d_from_regex(params_tree, regex_1d):
    """Boolean mask per leaf: True if param path matches regex_1d (-> Adam).

    Paths are flattened like 'mlp/~/linear_0/b'. Use Python regex; e.g. paths
    ending with '/b' (biases): regex_1d=r'/b$'
    """
    if not regex_1d:
        flat_params, _ = jax.tree_util.tree_flatten(params_tree)
        return [False] * len(flat_params)
    pattern = re.compile(regex_1d)
    pairs = jax.tree_util.tree_leaves_with_path(params_tree)
    return [bool(pattern.search(_path_flat_str(p))) for p, _ in pairs]


def _split_by_regex(tree, params_tree, regex_1d):
    """Split pytree by regex: params matching regex_1d -> 1D (Adam), rest -> 2D (Celo2Base)."""
    flat_tree, treedef = jax.tree_util.tree_flatten(tree)
    mask_1d = _mask_1d_from_regex(params_tree, regex_1d)
    flat_1d = [x if m else None for x, m in zip(flat_tree, mask_1d)]
    flat_2d = [x if not m else None for x, m in zip(flat_tree, mask_1d)]
    return jax.tree_util.tree_unflatten(treedef, flat_1d), jax.tree_util.tree_unflatten(treedef, flat_2d)


def _merge_by_regex(tree_1d, tree_2d, params_tree, regex_1d):
    """Merge 1D and 2D pytrees back using same regex mask."""
    is_none = lambda x: x is None
    flat_1d, _ = jax.tree_util.tree_flatten(tree_1d, is_leaf=is_none)
    flat_2d, _ = jax.tree_util.tree_flatten(tree_2d, is_leaf=is_none)
    mask_1d = _mask_1d_from_regex(params_tree, regex_1d)
    flat_merged = [x1 if m else x2 for x1, x2, m in zip(flat_1d, flat_2d, mask_1d)]
    assert all(x is not None for x in flat_merged), "Merged params should not contain None"
    _, treedef = jax.tree_util.tree_flatten(params_tree)
    return jax.tree_util.tree_unflatten(treedef, flat_merged)


def print_param_routing(params, regex_1d):
    """Print which params go to Adam vs Celo2 by regex, using Rich tables."""
    from collections import Counter, defaultdict
    from rich.console import Console
    from rich.table import Table

    mask_1d = _mask_1d_from_regex(params, regex_1d)
    pairs = list(jax.tree_util.tree_leaves_with_path(params))

    def bytes_to_str(num_bytes: int) -> str:
        if num_bytes >= 1_000_000:
            return f"{num_bytes/1e6:.1f} MB"
        if num_bytes >= 1_000:
            return f"{num_bytes/1e3:.1f} KB"
        return f"{num_bytes} B"

    elems_by_label = Counter()
    bytes_by_label = defaultdict(int)
    console = Console()

    detail = Table(
        title=f"Parameter Routing (regex_1d={regex_1d!r})",
        title_justify="left",
        show_header=True,
        header_style="bold",
    )
    detail.add_column("Optimizer", style="bold cyan", no_wrap=True)
    detail.add_column("Path", style="dim", overflow="fold")
    detail.add_column("Shape", style="magenta", no_wrap=True)
    detail.add_column("Elems", justify="right")

    for (path, arr), is_1d in zip(pairs, mask_1d):
        label = "Adam" if is_1d else "Celo2"
        elems = int(arr.size)
        size_bytes = int(arr.size * arr.dtype.itemsize)
        path_str = _path_flat_str(path)
        row_style = "yellow" if label == "Adam" else None
        detail.add_row(label, path_str, str(tuple(arr.shape)), f"{elems:_}", style=row_style)
        elems_by_label[label] += elems
        bytes_by_label[label] += size_bytes

    console.print(detail)

    summary = Table(
        title="Parameter Totals by Optimizer",
        title_justify="left",
        show_header=True,
        header_style="bold",
    )
    summary.add_column("Optimizer", style="bold")
    summary.add_column("Elems", justify="right")
    summary.add_column("Percent", justify="right")
    summary.add_column("Size", justify="right")

    total_elems = sum(elems_by_label.values())
    for label in sorted(elems_by_label.keys()):
        elems = elems_by_label[label]
        percent = (100.0 * elems / total_elems) if total_elems else 0.0
        size_bytes = bytes_by_label[label]
        row_style = "yellow" if label == "Adam" else None
        summary.add_row(label, f"{elems:_}", f"{percent:.1f}%", bytes_to_str(size_bytes), style=row_style)

    console.print(summary)


@flax.struct.dataclass
class Celo2LoptState:
    """Learned-optimization state for Celo2 optimizer that uses
    Adam update for 1D params and Celo2 update for 2D+ params.
    """
    params: chex.ArrayTree
    state_1d: AdamState
    state_2d: Celo2State
    state: chex.ArrayTree
    step_mult: jnp.ndarray
    step_mult_1d: jnp.ndarray


@gin.configurable
class Celo2(lopt_base.LearnedOptimizer):
    """Celo2 optimizer that uses Adam update for 1D params (biases, embeddings, etc)
    and Celo2 learned update for 2D+ params (matrix parameters).
    """
    def __init__(
        self,
        regex_1d=r".*bias.*|.*embed.*",
        step_mult=0.001,
        weight_decay=0.0,
        adam_b1=0.9,
        adam_b2=0.999,
        adam_epsilon=1e-8,
        ff_hidden_size=8,
        ff_hidden_layers=2,
        initial_momentum_decays=(0.9, 0.99, 0.999),
        initial_rms_decays=(0.95,),
        initial_adafactor_decays=(0.9, 0.99, 0.999),
        exp_mult=0.0,
        rmsmult=1.0,
        with_g=True,
        with_m=True,
        with_rms=True,
        with_rms_norm_g=True,
        with_rsqrt_rms=True,
        with_p=True,
        with_fac_norm_g=True,
        with_fac_rms=True,
        with_fac_rsqrt=True,
        with_grad_clip_feat=True,
        with_fac_mom_mult=True,
        with_rms_only_norm_g=True,
        param_scale_mult=False,
        precondition_output=False,
        normalize_input=True,
        normalize_output=True,
        aggregate_mag=False,
        bias_correction=False,
        adam_bias_correction=False,
        mlp_activation="relu",
        orthogonalize=True,
        ns_coeffs=(3.4445, -4.7750, 2.0315),
        ns_iters=5,
        ns_eps=1e-8,
        mult_1d=20.0,
    ):
        super().__init__()
        self.step_mult = step_mult
        self.weight_decay = weight_decay
        self.regex_1d = regex_1d
        self.mult_1d = mult_1d

        self._adam_transform = AdamTransformation(
            b1=adam_b1, b2=adam_b2, epsilon=adam_epsilon,
            bias_correction=adam_bias_correction,
        )

        self._celo2_config = dict(
            ff_hidden_size=ff_hidden_size,
            ff_hidden_layers=ff_hidden_layers,
            initial_momentum_decays=initial_momentum_decays,
            initial_rms_decays=initial_rms_decays,
            initial_adafactor_decays=initial_adafactor_decays,
            exp_mult=exp_mult,
            rmsmult=rmsmult,
            with_g=with_g,
            with_m=with_m,
            with_rms=with_rms,
            with_rms_norm_g=with_rms_norm_g,
            with_rsqrt_rms=with_rsqrt_rms,
            with_p=with_p,
            with_fac_norm_g=with_fac_norm_g,
            with_fac_rms=with_fac_rms,
            with_fac_rsqrt=with_fac_rsqrt,
            with_grad_clip_feat=with_grad_clip_feat,
            with_fac_mom_mult=with_fac_mom_mult,
            with_rms_only_norm_g=with_rms_only_norm_g,
            param_scale_mult=param_scale_mult,
            precondition_output=precondition_output,
            normalize_input=normalize_input,
            normalize_output=normalize_output,
            aggregate_mag=aggregate_mag,
            bias_correction=bias_correction,
            mlp_activation=mlp_activation,
            orthogonalize=orthogonalize,
            ns_coeffs=ns_coeffs,
            ns_iters=ns_iters,
            ns_eps=ns_eps,
        )

    def init(self, key) -> lopt_base.MetaParams:
        return Celo2Transformation(**self._celo2_config).init_meta_params(key)

    def print_routing(self, params):
        print_param_routing(params, self.regex_1d)

    def opt_fn(self, theta, is_training=True) -> opt_base.Optimizer:
        parent = self
        adam_transform = parent._adam_transform
        celo2_transform = Celo2Transformation(theta=theta, **parent._celo2_config)
        regex_1d = parent.regex_1d

        class _Opt(opt_base.Optimizer):
            def __init__(self):
                super().__init__()

            @functools.partial(jax.jit, static_argnums=(0,))
            def init(self, params: Any, model_state=None, num_steps=None, key=None) -> Celo2LoptState:
                params_1d, params_2d = _split_by_regex(params, params, regex_1d)
                step_mult = jnp.asarray(parent.step_mult, dtype=jnp.float32)
                return Celo2LoptState(
                    params=params,
                    state_1d=adam_transform.init(params_1d),
                    state_2d=celo2_transform.init(params_2d),
                    state=model_state,
                    step_mult=step_mult,
                    step_mult_1d=step_mult * jnp.asarray(parent.mult_1d, dtype=step_mult.dtype),
                )

            @functools.partial(jax.jit, static_argnums=(0,))
            def update(self, opt_state, grads, loss=None, model_state=None, is_valid=False, key=None) -> Celo2LoptState:
                params_1d, params_2d = _split_by_regex(opt_state.params, opt_state.params, regex_1d)
                grads_1d, grads_2d = _split_by_regex(grads, opt_state.params, regex_1d)

                step_1d, new_state_1d = adam_transform.update(grads_1d, opt_state.state_1d, params_1d)
                step_2d, new_state_2d = celo2_transform.update(grads_2d, opt_state.state_2d, params_2d)

                wd = parent.weight_decay
                apply_1d = lambda p, s: p - opt_state.step_mult_1d * s - wd * p * opt_state.step_mult_1d
                apply_2d = lambda p, s: p - opt_state.step_mult * s - wd * p * opt_state.step_mult
                next_1d = jax.tree_util.tree_map(apply_1d, params_1d, step_1d)
                next_2d = jax.tree_util.tree_map(apply_2d, params_2d, step_2d)
                next_params = _merge_by_regex(next_1d, next_2d, opt_state.params, regex_1d)

                ss = Celo2LoptState(
                    params=next_params,
                    state_1d=new_state_1d,
                    state_2d=new_state_2d,
                    state=model_state,
                    step_mult=opt_state.step_mult,
                    step_mult_1d=opt_state.step_mult_1d,
                )
                return tree_utils.match_type(ss, opt_state)

        return _Opt()
