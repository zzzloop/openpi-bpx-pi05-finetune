"""Synthetic checks for the server's existing OpenPI JAX environment."""

from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma
from openpi.training import bpx_counterfactual as cf
from openpi.training import bpx_stage2_loss as loss


def test_nc_dfm_sign_and_physical_dimension_mask():
    actions = jax.random.normal(jax.random.key(1), (4, 30, 32))
    shared_noise = jax.random.normal(jax.random.key(2), (30, 32))
    expected = shared_noise - actions
    padded_difference = expected.at[0, :, 18:].add(100)
    np.testing.assert_allclose(loss.difference_loss(padded_difference, actions, 0.9), 0, atol=1e-10)
    assert float(loss.difference_loss(-expected, actions, 0.9)) > 0.1


def test_attention_masks_leave_no_gradient_for_disabled_edges():
    raw = jnp.array([[1., 0., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]])
    mask = jnp.array([True, False, False, True])
    sem, swap = loss.attention_loss(raw, mask)
    np.testing.assert_allclose([sem, swap], 0, atol=1e-6)
    disabled = jnp.zeros(4, dtype=bool)
    grad = jax.grad(lambda value: sum(loss.attention_loss(value, disabled)))(raw)
    np.testing.assert_array_equal(grad, jnp.zeros_like(grad))


def test_complete_quad_uses_one_noise_and_time():
    actions = jax.random.normal(jax.random.key(2), (4, 30, 32))

    class Oracle:
        def counterfactual_velocity(self, _rng, _obs, noisy, time, _mask):
            np.testing.assert_array_equal(time, jnp.repeat(time[0], 4))
            time = time[:, None, None]
            recovered_noise = (noisy - (1 - time) * actions) / time
            np.testing.assert_allclose(recovered_noise, jnp.repeat(recovered_noise[:1], 4, axis=0), atol=1e-6)
            return recovered_noise - actions, jnp.ones((4, 4))

    _, _, metrics = loss.objective(
        Oracle(), jax.random.key(3), None, actions,
        {"task_token_mask": None, "attention_edges": jnp.zeros((4, 4), dtype=bool)}, "ccfa_ours",
    )
    np.testing.assert_allclose([metrics["cfm_loss"], metrics["nc_dfm_loss"]], 0, atol=1e-9)


def test_probe_scan_matches_legacy_forward_and_is_differentiable(monkeypatch):
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 32)
    expert = gemma.get_config("dummy")
    llm = gemma.Module([expert, expert], embed_dtype="float32", adarms=True)
    _, variables = llm.init_with_output(jax.random.key(0), [False, True], method="init")
    prefix = jax.random.normal(jax.random.key(3), (4, 8, expert.width))
    suffix = jax.random.normal(jax.random.key(4), (4, 3, expert.width))
    positions = jnp.broadcast_to(jnp.arange(11), (4, 11))
    mask = jnp.ones((4, 11, 11), dtype=bool)
    adarms = [None, jnp.ones((4, expert.width))]
    probe = (jnp.ones((4, 3), dtype=bool), jnp.ones((4, 4), dtype=bool))
    legacy, _ = llm.apply(variables, [prefix, suffix], positions, mask, adarms)
    outputs, _, attention = llm.apply(variables, [prefix, suffix], positions, mask, adarms, attention_probe=probe)
    for a, b in zip(legacy, outputs, strict=True):
        np.testing.assert_array_equal(a, b)
    assert attention.shape == (4, 4)

    def objective(params):
        _, _, attention = llm.apply({"params": params}, [prefix, suffix], positions, mask, adarms,
                                    attention_probe=probe)
        return jnp.sum(attention[:, 0])

    grads = traverse_util.flatten_dict(jax.jit(jax.grad(objective))(variables["params"]))
    assert any(np.linalg.norm(np.asarray(v)) > 0 for path, v in grads.items()
               if cf.is_vlm_parameter(("PaliGemma", "llm", *path)) and "attn" in path)
