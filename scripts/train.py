import dataclasses
import functools
import hashlib
import json
import logging
import pathlib
import platform
import shutil
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
from openpi.training import bpx_counterfactual as _bpx_cf
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.monitoring as _monitoring
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.training.validation as _validation
import openpi.training.validation_split as _validation_split
import openpi.training.checkpoint_selection as _selection
import openpi.training.selection_probes as _selection_probes


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        if _bpx_cf.CONFIGS.get(config.name) != "partial_ft":
            params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions] | tuple[_model.Observation, _model.Actions, dict[str, at.Array]],
    *,
    stage2_aligned: bool = False,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()
    stage2 = _bpx_cf.CONFIGS.get(config.name)

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        if stage2 is None:
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            total, extra_metrics = jnp.mean(chunked_loss), {}
        else:
            from openpi.training.bpx_stage2_loss import objective

            chunked_loss, total, extra_metrics = objective(
                model, rng, observation, actions, batch[2], stage2, aligned=stage2_aligned,
            )
        horizon_reduce_axes = tuple(range(chunked_loss.ndim - 1))
        return total, {
            "loss_std": jnp.std(chunked_loss),
            "loss_min": jnp.min(chunked_loss),
            "loss_max": jnp.max(chunked_loss),
            "loss_by_horizon": jnp.mean(chunked_loss, axis=horizon_reduce_axes),
            **extra_metrics,
        }

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch[:2]

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, loss_metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        **loss_metrics,
        "grad_norm": optax.global_norm(grads),
        "update_norm": optax.global_norm(updates),
        "param_norm": optax.global_norm(kernel_params),
        "learning_rate": config.lr_schedule.create()(state.step),
    }
    if config.stage2_smoke_test and stage2 == "partial_ft":
        # Verify the real update does not change ANY frozen leaf. This reduction
        # runs on device before donation, without copying the checkpoint to host.
        before = state.params.filter(nnx.All(nnx.Param, config.freeze_filter))
        after = new_params.filter(nnx.All(nnx.Param, config.freeze_filter))
        info["frozen_unchanged"] = jnp.all(jnp.stack(jax.tree.leaves(
            jax.tree.map(jnp.array_equal, before, after)
        )))
    return new_state, info


def eval_step(model_def, params, rng, batch):
    """Same evaluation mode for train-reference and holdout; no autodiff/updates."""
    model = nnx.merge(model_def, params)
    model.eval()
    observation, actions = batch
    return jnp.mean(model.compute_loss(rng, observation, actions, train=False))


def _stage2_protocol(config, initial_source):
    """Validate before checkpoint directory setup; record immutable resume inputs."""
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

    mode = _bpx_cf.CONFIGS[config.name]
    if not config.counterfactual_manifest or not config.validation_manifest:
        raise ValueError("Stage-2 requires both the quadruple manifest and its grouped validation split")
    if not config.resume and initial_source is None:
        raise ValueError("New Stage-2 runs must initialize from the common Stage-1 18000 checkpoint")
    if not config.resume and not config.stage2_smoke_test and config.checkpoint_dir.exists():
        raise FileExistsError("Stage-2 never overwrites an existing experiment; choose another name or resume")
    if (config.batch_size != 4 or config.fsdp_devices != 4 or jax.process_count() != 1
            or jax.device_count() != 4 or any(device.platform != "gpu" for device in jax.devices())):
        raise ValueError("BPX Stage-2 requires one host, four GPU devices, global batch size 4, and FSDP=4")
    if not getattr(config.model, "pi05", False) or (config.model.action_horizon, config.model.action_dim) != (30, 32):
        raise ValueError("BPX Stage-2 requires the Stage-1 Pi05 architecture (horizon 30, padded action dim 32)")
    if (config.log_interval < 1 or config.validation_interval < 1 or config.validation_batches < 1
            or config.validation_interval % config.log_interval):
        raise ValueError("Stage-2 validation interval must be a positive multiple of log_interval")
    quad_path = pathlib.Path(config.counterfactual_manifest)
    split_path = pathlib.Path(config.validation_manifest)
    dataset = pathlib.Path(HF_LEROBOT_HOME) / config.data.repo_id
    _bpx_cf.bound_quads(quad_path, dataset, split_path, require_attention=mode == "ccfa_ours")
    stats = config.assets_dirs / config.data.repo_id / "norm_stats.json"
    if not stats.is_file():
        raise ValueError(f"Compute shared Stage-2 TRAIN-ONLY normalization first: {stats}")
    protocol = {
        "schema_version": 2, "method": mode, "repo_id": config.data.repo_id, "seed": config.seed,
        "sampling_protocol": _bpx_cf.SAMPLING_PROTOCOL,
        "quad_sha256": hashlib.sha256(quad_path.read_bytes()).hexdigest(),
        "split_digest": _validation_split.split_digest(split_path),
        "norm_sha256": hashlib.sha256(stats.read_bytes()).hexdigest(),
        "model": repr(config.model), "optimizer": repr(config.optimizer), "lr_schedule": repr(config.lr_schedule),
        "ema_decay": config.ema_decay,
        "initial_checkpoint": str(initial_source) if initial_source else None,
    }
    path = config.checkpoint_dir / "monitoring/stage2_protocol.json"
    if config.resume:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("sampling_protocol") != _bpx_cf.SAMPLING_PROTOCOL:
            raise ValueError("Stage-2 sampler changed; start a new experiment from Stage-1 step 18000")
        protocol["initial_checkpoint"] = previous["initial_checkpoint"]
        if previous != protocol:
            raise ValueError("Stage-2 resume settings differ from the saved experiment protocol")
    return path, protocol


def _compile_train_step(config, state_sharding, data_sharding, replicated_sharding):
    stage2_options = {"static_argnames": ("stage2_aligned",)} if config.name in _bpx_cf.CONFIGS else {}
    return jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(1,),
        **stage2_options,
    )


def _run_stage2_smoke(config, init_rng, train_rng, mesh, data_sharding, replicated_sharding):
    print("[BPX PREFLIGHT] Loading real dataset and initialization weights; no checkpoints or W&B writes.", flush=True)
    loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True, num_batches=2)
    state, state_sharding = init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    if _bpx_cf.CONFIGS[config.name] == "partial_ft":
        frozen = state.params.filter(nnx.All(nnx.Param, config.freeze_filter)).flat_state()
        trainable = state.params.filter(config.trainable_filter).flat_state()
        frozen_paths, trainable_paths = tuple(frozen), tuple(trainable)
        for prefix in (("PaliGemma", "img"), ("PaliGemma", "llm")):
            if not any(path[:2] == prefix for path in frozen_paths):
                raise ValueError(f"Expected frozen VLM parameter subtree missing: {prefix}")
        if not any(path[:2] == ("PaliGemma", "llm") for path in trainable_paths):
            raise ValueError("Action Expert parameters are not trainable")
        print(f"[BPX PREFLIGHT] Frozen parameter leaves={len(frozen_paths)}; trainable={len(trainable_paths)}", flush=True)
    step_fn = _compile_train_step(config, state_sharding, data_sharding, replicated_sharding)
    for step, batch in enumerate(loader):
        aligned = _bpx_cf.is_aligned_step(step)
        print(f"[BPX PREFLIGHT] Compiling and executing {'aligned quad' if aligned else 'whole trajectory'} update", flush=True)
        with sharding.set_mesh(mesh):
            state, info = step_fn(train_rng, state, batch, stage2_aligned=aligned)
        info = jax.device_get(info)
        if (not all(np.isfinite(x).all() for x in info.values()) or info["grad_norm"] <= 0
                or info["update_norm"] <= 0 or not info.get("frozen_unchanged", True)):
            raise FloatingPointError(f"Stage-2 preflight step {step} failed: nonfinite/zero update or changed frozen parameters")
        print(f"Step {step + 1}: loss={info['loss']:.6g}, grad_norm={info['grad_norm']:.6g}, "
              f"update_norm={info['update_norm']:.6g}", flush=True)
    if int(state.step) != 2:
        raise RuntimeError("Stage-2 preflight did not execute both update paths")
    if config.selection_suite:
        # Validate the optional suite binding and bundle before starting a long run.
        _selection_probes.ProbeRuntime(
            config, loader.data_config(), state.model_def, state_sharding.params, data_sharding, mesh)
    print("[BPX PREFLIGHT] Checking one real held-out batch", flush=True)
    validation = next(iter(_data_loader.create_data_loader(config, sharding=data_sharding, split="validation")))
    check_eval = jax.jit(functools.partial(eval_step, state.model_def),
                        in_shardings=(state_sharding.params, replicated_sharding, data_sharding),
                        out_shardings=replicated_sharding)
    with sharding.set_mesh(mesh):
        val_loss = float(check_eval(state.params, jax.random.key(config.validation_seed), validation))
    if not np.isfinite(val_loss):
        raise FloatingPointError("Stage-2 preflight held-out CFM is nonfinite")
    print(f"BPX STAGE-2 PREFLIGHT PASS: two updates, heldout_cfm={val_loss:.6g}; no checkpoint saved", flush=True)


def main(config: _config.TrainConfig):
    config = _validation_split.configure(config)
    source = None
    if config.init_checkpoint:
        source = pathlib.Path(config.init_checkpoint).expanduser().resolve()
        if source.suffix == ".json":
            source = pathlib.Path(json.loads(source.read_text(encoding="utf-8"))["checkpoint_path"])
        if not (source / "params").is_dir():
            raise FileNotFoundError(f"Initialization checkpoint params missing: {source}")
        if source.resolve().is_relative_to(pathlib.Path(config.checkpoint_dir).resolve()):
            raise ValueError("Initialization weights must be in a DIFFERENT experiment; do not overwrite their source run")
        config = dataclasses.replace(config, weight_loader=_weight_loaders.CheckpointWeightLoader(str(source / "params")))
    stage2 = _bpx_cf.CONFIGS.get(config.name)
    stage2_protocol = _stage2_protocol(config, source) if stage2 else None
    if config.selection_suite:
        if jax.process_count() != 1:
            raise NotImplementedError("Checkpoint selection currently supports one host with FSDP, not multi-host")
        config = dataclasses.replace(config, monitoring_enabled=True)
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    if config.stage2_smoke_test:
        _run_stage2_smoke(config, init_rng, train_rng, mesh, data_sharding, replicated_sharding)
        return

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite and stage2 is None,
        resume=config.resume,
        checkpoint_limit=config.checkpoint_limit,
        selection_keep=config.selection_keep if config.selection_suite else None,
    )
    if stage2_protocol:
        path, protocol = stage2_protocol
        if config.resume and not resuming:
            raise ValueError("Stage-2 resume requires a complete resumable checkpoint")
        if not resuming:
            path.parent.mkdir(parents=True, exist_ok=True)
            _selection.atomic_json(path, protocol)
            shutil.copyfile(config.counterfactual_manifest, path.with_name("quadruples.json"))
        logging.info("BPX Stage-2: %s; global batch=4, FSDP=4, init=%s", stage2, protocol["initial_checkpoint"])
    ordinary_keep, _, archive_keep = _checkpoints.retention_budget(
        config.checkpoint_limit, config.keep_period, config.selection_keep if config.selection_suite else None,
    )
    if config.checkpoint_limit is not None:
        logging.info("Checkpoint retention: total <= %d after save/selection; latest resumable=%d, "
                     "selected archives <= %d; keep_period disabled (save_interval unchanged)",
                     config.checkpoint_limit, ordinary_keep, archive_keep or 0)
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    if resuming and not config.validation_manifest and (config.checkpoint_dir / "monitoring/validation_protocol.json").exists():
        raise ValueError("This experiment uses held-out validation; resume with the same validation manifest")

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    monitor = None
    if config.monitoring_enabled:
        monitor = _monitoring.TrainingMonitor(
            config.checkpoint_dir / "monitoring",
            batch_size=config.batch_size,
            plot_interval=config.monitoring_plot_interval,
            visualization_interval=config.monitoring_visualization_interval,
            action_names=config.monitoring_action_names,
            composite_loss_title=(f"{_bpx_cf.QUAD_DIFF_NAME} total loss"
                                  if config.name == _bpx_cf.QUAD_DIFF_CONFIG else None),
        )
        logging.info(f"Durable training monitoring enabled at {monitor.output_dir}")
    else:
        # Preserve the upstream first-batch W&B sanity check for configs that have not opted into durable monitoring.
        images_to_log = [
            wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
        wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
        if stage2:
            data_loader.sampler.start_step = int(train_state.step)
            data_iter = iter(data_loader)
            batch = next(data_iter)

    ptrain_step = _compile_train_step(config, train_state_sharding, data_sharding, replicated_sharding)

    start_step = int(train_state.step)
    if monitor is not None:
        monitor.prepare_resume(start_step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []

    evaluation_history = None
    evaluation_batches = {}
    selector = None
    probe_runtime = None
    if config.validation_manifest:
        if config.validation_interval < 1 or config.validation_batches < 1:
            raise ValueError("Validation interval and batches must be positive")
        if config.validation_interval % config.log_interval:
            raise ValueError("Validation interval must be divisible by log_interval")
        identity = {"split_digest": _validation_split.split_digest(config.validation_manifest),
                    "seed": config.validation_seed, "batches": config.validation_batches,
                    "batch_size": config.batch_size, "mode": "current_params_train_false",
                    "config": config.name, "action_horizon": config.model.action_horizon,
                    "sampling": "task_stratified_without_replacement"}
        if config.selection_suite:
            identity["mode"] = "servable_ema_or_current_train_false"
        if stage2:
            identity["training_method"] = stage2
        stats_asset = data_loader.data_config().asset_id or config.data.repo_id
        stats_path = pathlib.Path(config.assets_dirs) / stats_asset / "norm_stats.json"
        identity["norm_stats_sha256"] = hashlib.sha256(stats_path.read_bytes()).hexdigest()
        evaluation_history = _validation.EvaluationHistory(
            pathlib.Path(config.checkpoint_dir) / "monitoring", identity, start_step=start_step, resume=resuming)
        split_copy = pathlib.Path(config.checkpoint_dir) / "monitoring" / "validation_split.json"
        split_copy.write_text(json.dumps(_validation_split.load_split(config.validation_manifest),
                                        ensure_ascii=False, indent=2), encoding="utf-8")
        for split in ("train_reference", "validation"):
            loader = _data_loader.create_data_loader(config, sharding=data_sharding, split=split)
            # Bound host RAM and avoid holding validation images on the GPUs between evaluations.
            evaluation_batches[split] = [jax.tree.map(np.asarray, jax.device_get(item)) for item in loader]
        peval = jax.jit(functools.partial(eval_step, train_state.model_def),
                       in_shardings=(train_state_sharding.params, replicated_sharding, data_sharding),
                       out_shardings=replicated_sharding)

    def run_evaluation(step):
        started = time.monotonic()
        results = {}
        for split, batches in evaluation_batches.items():
            def evaluate(index, host_batch):
                key = jax.random.fold_in(jax.random.key(config.validation_seed), index)
                device_batch = jax.tree.map(
                    lambda x: jax.make_array_from_process_local_data(data_sharding, x), host_batch)
                with sharding.set_mesh(mesh):
                    params = (train_state.ema_params if train_state.ema_params is not None else train_state.params) if selector is not None else train_state.params
                    return jax.device_get(peval(params, key, device_batch))
            results[split] = _validation.evaluate_batches(batches, evaluate)
        train_loss, train_se = results["train_reference"]
        val_loss, val_se = results["validation"]
        record = evaluation_history.append(
            step, train_loss, val_loss, train_stderr=train_se, val_stderr=val_se,
            seconds=time.monotonic() - started,
            checkpoint_saved=False if selector is not None else ((step % config.save_interval == 0 and step > start_step)
                              or step == config.num_train_steps - 1))
        return {key: value for key, value in record.items()
                if key in ("train_eval_loss", "val_loss", "generalization_gap", "val_stderr",
                           "best_validation_step", "best_validation_loss", "overfit_warning", "evaluation_seconds")}

    def flush_metrics(step: int, visualized_batch, *, final: bool = False, evaluate: bool = True) -> None:
        nonlocal infos
        if not infos:
            return
        stacked_infos = common_utils.stack_forest(infos)
        reduced_info = {
            key: jnp.mean(value, axis=0) if key == "loss_by_horizon" else jnp.mean(value)
            for key, value in stacked_infos.items()
        }
        reduced_info = jax.device_get(reduced_info)
        if evaluation_history is not None and evaluate and (step % config.validation_interval == 0 or final):
            reduced_info.update(run_evaluation(step))
        scalar_info = {key: value for key, value in reduced_info.items() if np.asarray(value).ndim == 0}
        info_str = ", ".join(f"{key}={float(value):.4g}" for key, value in scalar_info.items())
        pbar.write(f"Step {step}: {info_str}")

        wandb_payload = dict(scalar_info)
        loss_by_horizon = np.asarray(reduced_info["loss_by_horizon"])
        wandb_payload.update(
            {f"loss_by_horizon/{index:02d}": float(value) for index, value in enumerate(loss_by_horizon)}
        )
        nonfinite_metrics = [key for key, value in reduced_info.items() if not np.isfinite(np.asarray(value)).all()]
        if monitor is not None:
            monitor.log_step_losses(step, jax.device_get(stacked_infos["loss"]))
            plot_paths = monitor.log_metrics(step, reduced_info, force_plot=final)
            if plot_paths:
                wandb_payload["monitor/training_overview"] = wandb.Image(str(plot_paths[0]))
            if final or monitor.should_visualize(step):
                sample_count = config.monitoring_max_samples
                host_images = {
                    name: np.asarray(jax.device_get(image[:sample_count]))
                    for name, image in visualized_batch[0].images.items()
                }
                host_actions = np.asarray(jax.device_get(visualized_batch[1][:sample_count]))
                visual_paths = monitor.save_batch_visualizations(
                    step,
                    host_images,
                    host_actions,
                    max_samples=sample_count,
                )
                wandb_payload["monitor/camera_views"] = wandb.Image(str(visual_paths[0]))
                if len(visual_paths) > 1:
                    wandb_payload["monitor/action_targets"] = wandb.Image(str(visual_paths[1]))
        wandb.log(wandb_payload, step=step)
        infos = []
        if nonfinite_metrics:
            raise FloatingPointError(f"Non-finite training metrics at step {step}: {nonfinite_metrics}")

    if config.selection_suite:
        probe_runtime = _selection_probes.ProbeRuntime(
            config, data_loader.data_config(), train_state.model_def, train_state_sharding.params, data_sharding, mesh)
        selection_identity = {"schema_version": 1, "suite_sha256": probe_runtime.digest,
                              "validation": identity, "keep": archive_keep,
                              "denoise_steps": config.selection_denoise_steps, "seed": config.seed,
                              "action_names": list(config.monitoring_action_names),
                              "inference_params": "ema" if train_state.ema_params is not None else "current",
                              "initial_checkpoint": (stage2_protocol[1]["initial_checkpoint"] if stage2_protocol
                                                     else config.init_checkpoint),
                              "selection_policy": probe_runtime.manifest.get("selection_policy", {}),
                              "loss_function": (f"Stage-2 {stage2}; diagnostics never enter gradients" if stage2
                                                else "UNCHANGED compute_loss; diagnostics never enter gradients")}
        if config.checkpoint_limit is not None:
            selection_identity["checkpoint_limit"] = config.checkpoint_limit
        selection_identity.update(
            probe_code_sha256=hashlib.sha256(pathlib.Path(_selection_probes.__file__).read_bytes()).hexdigest(),
            selector_code_sha256=hashlib.sha256(pathlib.Path(_selection.__file__).read_bytes()).hexdigest(),
            velocity_model_code_sha256=hashlib.sha256(pathlib.Path(_model.__file__).with_name("pi0.py").read_bytes()).hexdigest())
        selector = _selection.SelectionHistory(
            pathlib.Path(config.checkpoint_dir), selection_identity, keep=archive_keep,
            start_step=start_step, resume=resuming, gates=probe_runtime.manifest.get("gates"),
            external_assessments=probe_runtime.manifest.get("external_assessments"))
        _selection.atomic_json(selector.root / "training_config.json", {"config_repr": repr(config),
            "config_source_sha256": hashlib.sha256(pathlib.Path(_config.__file__).read_bytes()).hexdigest(),
            "train_source_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
            "jax_version": jax.__version__, "numpy_version": np.__version__, "python": platform.python_version()})
        original_suite = pathlib.Path(config.selection_suite).resolve()
        suite_copy = selector.root / "suite.json"
        if original_suite != suite_copy.resolve():
            # Preserve the exact bytes used for suite_sha256, not a re-serialized JSON.
            temporary = suite_copy.with_suffix(".json.tmp")
            shutil.copy2(original_suite, temporary)
            temporary.replace(suite_copy)
        source_bundle = original_suite.parent / probe_runtime.manifest["bundle"]
        if not (selector.root / "probes.npz").is_file():
            shutil.copy2(source_bundle, selector.root / "probes.npz")

    def select_current(step, *, complete):
        if not selector:
            return
        if not evaluation_history.records or evaluation_history.records[-1]["step"] != step:
            run_evaluation(step)
        if complete:
            evaluation_history.mark_checkpoint_complete(step)
        params = train_state.ema_params if train_state.ema_params is not None else train_state.params
        started = time.monotonic()
        metrics, rows, artifacts = probe_runtime.evaluate(params)
        loss_record = evaluation_history.records[-1]
        metrics.update({key: loss_record[key] for key in ("train_eval_loss", "val_loss", "val_stderr", "generalization_gap")})
        metrics["probe_seconds"] = time.monotonic()-started
        decision = selector.append(step, metrics, rows, artifacts,
                                   optimizer_step=int(train_state.step), checkpoint_complete=complete)
        if complete:
            monitor.attach_evaluation(step, metrics)
            monitor.render_plots(step)
        logging.info("Checkpoint selection: %s, recommended=%s", decision["status"], decision["checkpoint_path"])
        for warning in decision["warnings"]:
            logging.warning("CHECKPOINT DIAGNOSTIC: %s", warning)

    # Baseline BEFORE the first optimizer update: required for retention trends.
    if selector and not selector.records:
        select_current(start_step-1, complete=False)

    last_trained_batch = batch
    last_completed_step = start_step-1
    try:
        for step in pbar:
            last_trained_batch = batch
            with sharding.set_mesh(mesh):
                if stage2:
                    train_state, info = ptrain_step(
                        train_rng, train_state, batch, stage2_aligned=_bpx_cf.is_aligned_step(step))
                else:
                    train_state, info = ptrain_step(train_rng, train_state, batch)
            last_completed_step = step
            infos.append(info)
            if step % config.log_interval == 0:
                flush_metrics(step, last_trained_batch, final=step == config.num_train_steps - 1)
            batch = next(data_iter)

            if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
                if selector and infos:
                    flush_metrics(step, last_trained_batch, final=step == config.num_train_steps-1)
                _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
                if selector:
                    checkpoint_manager.wait_until_finished()
                    try:
                        select_current(step, complete=True)
                    except Exception as error:
                        # Keep optimizer/training untouched; never promote an unscored checkpoint.
                        logging.exception("Checkpoint probes failed; previous recommendation retained")
                        _selection.atomic_json(selector.root / f"evaluation_error_{step}.json",
                                               {"step": step, "error": str(error), "selected": False})

        flush_metrics(config.num_train_steps - 1, last_trained_batch, final=True)
        logging.info("Waiting for checkpoint manager to finish")
        checkpoint_manager.wait_until_finished()
    finally:
        # Persist pending scalar losses and refresh charts even after an interrupt.
        # Do not launch fresh expensive evaluation while handling an exception.
        if monitor is not None:
            try:
                if infos and last_completed_step >= start_step:
                    flush_metrics(last_completed_step, last_trained_batch, final=True, evaluate=False)
                elif monitor._history:
                    monitor.render_plots(monitor._history[-1]["step"])
            except Exception:
                logging.exception("Could not finalize diagnostic plots; raw metric files remain available")


if __name__ == "__main__":
    main(_config.cli())
