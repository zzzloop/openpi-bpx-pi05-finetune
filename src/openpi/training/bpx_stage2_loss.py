"""JAX objectives for Stage-2. All quantities use OpenPI's noise-at-t=1 convention."""

import dataclasses

import jax
import jax.numpy as jnp

from openpi.policies.bpx_constants import BPX_EFFECTIVE_DIM
from openpi.training.bpx_counterfactual import SEMANTIC_EDGES, SPATIAL_EDGES, SWAP_EDGES

LANGUAGE_DROPOUT = 0.15
DEFAULT_CFG_SCALE = 2.0
CFG_MIN_TIME = 0.7
LAMBDA_NC = 1.0
LAMBDA_ATTN = 0.1


def language_dropout(rng, observation, extras):
    drop = jax.random.bernoulli(rng, LANGUAGE_DROPOUT, (observation.state.shape[0], 1))
    return dataclasses.replace(
        observation,
        tokenized_prompt=jnp.where(drop, extras["null_prompt"], observation.tokenized_prompt),
        tokenized_prompt_mask=jnp.where(drop, extras["null_prompt_mask"], observation.tokenized_prompt_mask),
    ), jnp.mean(drop)


def difference_loss(velocity, actions, time):
    """Match pairwise conditional velocity differences to the demonstrated differences."""
    velocity = velocity[..., :BPX_EFFECTIVE_DIM].astype(jnp.float32)
    actions = actions[..., :BPX_EFFECTIVE_DIM].astype(jnp.float32)
    residual = velocity + actions  # shared noise cancels: v_i-v_j target is A_j-A_i.

    def pair_mean(edges):
        return jnp.mean(jnp.stack([jnp.mean(jnp.square(residual[i] - residual[j])) for i, j in edges]))

    return time * (pair_mean(SEMANTIC_EDGES) + pair_mean(SPATIAL_EDGES))


def attention_loss(raw, mask):
    # Smooth zeros before log/norm to keep gradients finite even when an edge is inactive.
    mass = raw.astype(jnp.float32) + 1e-8
    p = mass / jnp.sum(mass, axis=-1, keepdims=True)
    logp = jnp.log(p)
    unit = p * jax.lax.rsqrt(jnp.sum(jnp.square(p), axis=-1, keepdims=True))
    semantic = jnp.stack([jnp.sum(unit[i] * unit[j]) for i, j in SEMANTIC_EDGES])
    swap = jnp.stack([0.5 * jnp.sum((p[i] - p[j]) * (logp[i] - logp[j])) for i, j in SWAP_EDGES])

    def masked_mean(values, keep):
        return jnp.sum(jnp.where(keep, values, 0)) / jnp.maximum(jnp.sum(keep), 1)

    return masked_mean(semantic, mask[:2]), masked_mean(swap, mask[2:])


def objective(model, rng, observation, actions, extras, mode, *, aligned):
    """The auxiliary losses exist only on explicitly aligned Group-4 batches."""
    if mode == "ccfa_ours" and aligned != ("attention_edges" in extras):
        raise ValueError("Stage-2 sampler and optimizer disagree about whether the batch is an aligned quad")
    if mode != "ccfa_ours" or not aligned:
        metrics = {}
        if mode == "policy_cfg":
            dropout_rng = jax.random.fold_in(rng, 0x434647)
            observation, metrics["language_drop_rate"] = language_dropout(dropout_rng, observation, extras)
        chunk = model.compute_loss(rng, observation, actions, train=True, shared_augmentation=aligned)
        cfm = jnp.mean(chunk)
        if mode == "ccfa_ours":
            zero = jnp.zeros_like(cfm)
            metrics = {"cfm_loss": cfm, "nc_dfm_loss": zero, "attn_sem_loss": zero, "attn_swap_loss": zero}
        return chunk, cfm, metrics

    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    noise = jax.random.normal(noise_rng, actions.shape[1:])
    time = jax.random.beta(time_rng, 1.5, 1) * 0.999 + 0.001
    noisy = time * noise + (1 - time) * actions
    velocity, attention = model.counterfactual_velocity(
        preprocess_rng, observation, noisy, jnp.broadcast_to(time, (4,)), extras["task_token_mask"]
    )
    chunk = jnp.mean(jnp.square(velocity.astype(jnp.float32) - (noise - actions)), axis=-1)
    cfm = jnp.mean(chunk)
    nc = difference_loss(velocity, actions, time)
    sem, swap = attention_loss(attention, extras["attention_edges"][0])
    return chunk, cfm + LAMBDA_NC * nc + LAMBDA_ATTN * (sem + swap), {
        "cfm_loss": cfm, "nc_dfm_loss": nc, "attn_sem_loss": sem, "attn_swap_loss": swap,
    }
