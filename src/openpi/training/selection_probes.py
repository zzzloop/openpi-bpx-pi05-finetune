"""Frozen held-out probes and forward-only checkpoint diagnostics.

These metrics never enter compute_loss, autodiff, optimizer state or training RNG.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import pathlib

import numpy as np


def norm_digest(data_config) -> str:
    values = {key: {name: np.asarray(getattr(stats, name)).tolist()
                    for name in ("mean", "std", "q01", "q99") if getattr(stats, name) is not None}
              for key, stats in data_config.norm_stats.items()}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_flatten(item, name))
        elif item is not None:
            result[name] = np.asarray(item)
    return result


def counterfactual_squared_l2(left, right, active_dims=None) -> float:
    """Formula 4.4 for ONE shared epsilon; expectation is across fixed seeds."""
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("CF velocities must have equal finite (horizon, action_dim) shape")
    difference = a-b
    if active_dims is not None:
        difference = difference[:, active_dims]
    return float(np.sum(difference**2))


def _probe_observation(data: dict):
    """Materialize array leaves before typed, unbatched Observation creation.

    Training collation and policy inference already convert scalar leaves to
    arrays. Frozen single-frame probes bypass those paths: np.bool_ image masks
    must become rank-zero bool ndarrays, without adding a batch axis or changing
    any dtype/value. Keep the model's shape and dtype checks enabled.
    """
    import jax

    from openpi.models.model import Observation

    return Observation.from_dict(jax.tree.map(np.asarray, data))


def prepare_suite(config, output: pathlib.Path, *, case_spec: pathlib.Path | None = None, max_cases: int = 16):
    """Server-side preparation, using the SAME repack/norm/tokenizer as training."""
    from openpi import transforms
    from openpi.training import data_loader, validation_split
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    config = validation_split.configure(config)
    if not config.validation_manifest:
        raise ValueError("Prepare a grouped holdout before checkpoint probes")
    if output.exists() or max_cases < 1:
        raise ValueError("Suite output must be new, and max_cases must be positive")
    split = validation_split.load_split(config.validation_manifest)
    data = config.data.create(config.assets_dirs, config.model)
    if not data.norm_stats:
        raise ValueError("Compute TRAIN-ONLY norm statistics before preparing probes")
    metadata = LeRobotDatasetMetadata(data.repo_id)
    if not math.isfinite(float(metadata.fps)) or metadata.fps <= 0:
        raise ValueError("Probe dataset FPS must be finite and positive")
    if str(pathlib.Path(split["dataset_root"]).resolve()) != str(metadata.root.resolve()):
        raise ValueError("Suite and validation split must reference the same local dataset")
    records, fingerprint = validation_split.inventory(metadata.root)
    if fingerprint != split["metadata_sha256"]:
        raise ValueError("Dataset metadata changed since the validation split")
    spec = json.loads(case_spec.read_text(encoding="utf-8")) if case_spec else {}
    gates = spec.get("gates", {})
    allowed = {"max_action_mse", "max_jerk_excess", "min_s_cf", "min_s_cf_active", "min_task_success", "vlm_noninferiority_margin"}
    if set(gates)-allowed or any(value is not None and
        (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0) for value in gates.values()):
        raise ValueError("Unknown/non-finite gate; thresholds must be explicit, finite and nonnegative")
    for key in ("min_task_success", "vlm_noninferiority_margin"):
        if gates.get(key) is not None and gates[key] > 1:
            raise ValueError(f"{key} must lie in [0,1]")
    policy = spec.get("selection_policy", {"uncertainty_multiplier": 2., "patience": 3})
    if (not isinstance(policy, dict)
            or type(policy.get("uncertainty_multiplier", 2.)) not in (int, float)
            or not math.isfinite(policy.get("uncertainty_multiplier", 2.))
            or not 0 < policy.get("uncertainty_multiplier", 2.) <= 10
            or type(policy.get("patience", 3)) is not int or not 2 <= policy.get("patience", 3) <= 20):
        raise ValueError("Invalid selection uncertainty/patience policy")
    if spec.get("external_assessments"):
        evidence = pathlib.Path(spec["external_assessments"]).expanduser()
        if not evidence.is_absolute():
            evidence = case_spec.resolve().parent / evidence
        spec["external_assessments"] = str(evidence.resolve())
    cases = spec.get("cases")
    if cases is None:
        # A usable DIAGNOSTIC template, not fabricated semantic annotations.
        tasks = sorted({records[i]["tasks"][0] for i in split["validation_episodes"]})
        by_task = {task: [] for task in tasks}
        used_groups = set()
        for index in split["validation_episodes"]:
            group = split["groups"][index]
            if group not in used_groups:
                by_task[records[index]["tasks"][0]].append(index)
                used_groups.add(group)
        order = []
        while any(by_task.values()):
            for task in tasks:
                if by_task[task]:
                    order.append(by_task[task].pop(0))
        if len(tasks) < 2:
            raise ValueError("Single-task data needs explicit, human-reviewed counterfactual prompts")
        cases = [{"id": f"episode_{index}", "episode_index": index, "frame_index": 0,
                  "counterfactual_prompt": next(task for task in tasks if task != records[index]["tasks"][0]),
                  "paraphrase_prompt": None, "semantic_reviewed": False}
                 for index in order[:max_cases]]
    if not cases:
        raise ValueError("Probe cases cannot be empty")
    active = spec.get("active_dims", list(range(len(data.norm_stats["actions"].mean))))
    names = list(config.monitoring_action_names or metadata.features.get("action", {}).get("names") or [])
    continuous = spec.get("continuous_dims", [i for i in active if i >= len(names)
                          or not any(word in str(names[i]).lower() for word in ("gripper", "jaw", "claw"))])
    for values in (active, continuous):
        if not values or len(values) != len(set(values)) or any(type(i) is not int or i < 0 or i >= len(data.norm_stats["actions"].mean) for i in values):
            raise ValueError("Action dimensions must be unique, valid, nonempty integer lists")
    if not set(continuous) <= set(active):
        raise ValueError("continuous_dims must be a subset of active_dims")
    seeds = spec.get("noise_seeds", [101, 202, 303])
    if len(seeds) < 2 or len(seeds) != len(set(seeds)) or any(type(i) is not int or i < 0 for i in seeds):
        raise ValueError("Use at least two distinct, fixed, nonnegative noise seeds")
    raw_dataset = data_loader.create_torch_dataset(
        dataclasses.replace(data, episode_indices=None, balance_by_task=False), config.model.action_horizon, config.model)
    pipeline = transforms.compose([
        *data.repack_transforms.inputs, *data.data_transforms.inputs,
        transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm), *data.model_transforms.inputs,
    ])
    payload = {f"noise/{seed}": np.random.default_rng(seed).standard_normal(
        (config.model.action_horizon, config.model.action_dim)).astype(np.float32) for seed in seeds}
    frozen, ids = [], set()
    for number, case in enumerate(cases):
        index, frame = int(case["episode_index"]), int(case.get("frame_index", 0))
        if index not in split["validation_episodes"] or not 0 <= frame < int(records[index]["length"]):
            raise ValueError("Every probe must be a valid frame in a held-out episode")
        case_id = str(case.get("id", f"case_{number}"))
        if case_id in ids:
            raise ValueError("Probe case IDs must be unique")
        ids.add(case_id)
        task = records[index]["tasks"][0]
        contra = str(case["counterfactual_prompt"]).strip()
        para = case.get("paraphrase_prompt")
        if para is not None and (not str(para).strip() or str(para).strip() == task.strip()):
            raise ValueError("A paraphrase must be a genuinely different wording, not the identical prompt")
        if not contra or contra == task:
            raise ValueError("Counterfactual prompt must differ from the recorded task")
        # Prefix sums avoid depending on any private LeRobot wrapper structure.
        global_index = sum(int(r["length"]) for r in records[:index]) + frame
        raw = raw_dataset[global_index]
        common = None
        for variant, prompt in (("matched", task), ("counterfactual", contra), ("paraphrase", para)):
            if not prompt:
                continue
            transformed = pipeline({**copy.deepcopy(raw), "prompt": str(prompt)})
            obs = _probe_observation(transformed)
            flat = _flatten(obs.to_dict())
            shared = {key: value for key, value in flat.items() if key not in ("tokenized_prompt", "tokenized_prompt_mask")}
            if common is None:
                common = shared
                for key, value in shared.items():
                    payload[f"{number}/common/{key}"] = value
                payload[f"{number}/target"] = np.asarray(transformed["actions"])
            else:
                if shared.keys() != common.keys() or any(not np.array_equal(shared[key], common[key]) for key in common):
                    raise ValueError("Prompt variants changed observation/state; invalid counterfactual control")
            for key in ("tokenized_prompt", "tokenized_prompt_mask"):
                payload[f"{number}/{variant}/{key}"] = flat[key]
        if np.array_equal(payload[f"{number}/matched/tokenized_prompt"], payload[f"{number}/counterfactual/tokenized_prompt"]):
            raise ValueError("Different instructions tokenize identically: check truncation/max_token_len")
        if para and np.array_equal(payload[f"{number}/matched/tokenized_prompt"], payload[f"{number}/paraphrase/tokenized_prompt"]):
            raise ValueError("Paraphrase tokenizes identically; choose a genuinely different phrasing")
        frozen.append({**case, "id": case_id, "episode_index": index, "frame_index": frame,
                       "matched_prompt": task, "group_id": split["groups"][index], "task": task,
                       "valid_horizon": min(config.model.action_horizon, int(records[index]["length"])-frame),
                       "semantic_reviewed": case.get("semantic_reviewed") is True,
                       "paraphrase_prompt": para})
        print(f"PREPARED {number+1}/{len(cases)}: episode={index}, frame={frame}, reviewed={frozen[-1]['semantic_reviewed']}", flush=True)
    if any(np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all() for value in payload.values()):
        raise ValueError("Prepared probe arrays contain non-finite values")
    output.mkdir(parents=True)
    bundle = output / "probes.npz"
    np.savez_compressed(bundle, **payload)
    manifest = {"schema_version": 1, "config_name": config.name, "repo_id": data.repo_id,
                "dataset_fingerprint": fingerprint, "split_digest": validation_split.split_digest(config.validation_manifest),
                "norm_digest": norm_digest(data), "bundle": bundle.name,
                "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                "action_dim": config.model.action_dim, "action_horizon": config.model.action_horizon,
                "active_dims": active, "continuous_dims": continuous, "fps": float(metadata.fps),
                "noise_seeds": seeds, "cases": frozen, "gates": spec.get("gates", {}),
                "selection_policy": policy,
                "external_assessments": spec.get("external_assessments"),
                "cf_definition": "E_noise sum_(h,d) (v_L1-v_L2)^2; paper noise t=0 -> OpenPI t=1",
                "limits": "offline probes, not task success or a general VLM benchmark"}
    (output / "suite.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "editable_case_spec.json").write_text(json.dumps({"cases": frozen, "noise_seeds": seeds,
        "active_dims": active, "continuous_dims": continuous, "gates": spec.get("gates", {}),
        "selection_policy": policy,
        "external_assessments": spec.get("external_assessments")}, ensure_ascii=False, indent=2), encoding="utf-8")
    return output / "suite.json"


def _aggregate(rows: list[dict]) -> dict:
    """Average seeds/cases within source groups before estimating across-group SE."""
    output = {}
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float)) and key != "noise_seed"})
    for key in keys:
        groups = {}
        for row in rows:
            if key in row and row[key] is not None:
                groups.setdefault(row["group_id"], []).append(float(row[key]))
        means = np.asarray([np.mean(values) for values in groups.values()])
        if means.size:
            output[key] = float(means.mean())
            output[f"{key}_se"] = float(means.std(ddof=1)/np.sqrt(len(means))) if len(means) > 1 else 0.
    output["independent_groups"] = len({row["group_id"] for row in rows})
    return output


class ProbeRuntime:
    def __init__(self, config, data_config, model_def, params_sharding, data_sharding, mesh):
        import flax.nnx as nnx
        import jax
        from openpi.training import validation_split

        self.config, self.data, self.mesh, self.sharding = config, data_config, mesh, data_sharding
        path = pathlib.Path(config.selection_suite).resolve()
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest = self.manifest
        if manifest.get("schema_version") != 1 or manifest["split_digest"] != validation_split.split_digest(config.validation_manifest):
            raise ValueError("Probe suite validation split differs from training")
        if (manifest["config_name"] != config.name or manifest["norm_digest"] != norm_digest(data_config)
                or manifest["action_dim"] != config.model.action_dim or manifest["action_horizon"] != config.model.action_horizon):
            raise ValueError("Probe suite config/norm/model contract changed: rebuild the suite")
        bundle = path.parent / manifest["bundle"]
        if hashlib.sha256(bundle.read_bytes()).hexdigest() != manifest["bundle_sha256"]:
            raise ValueError("Frozen probe bundle checksum mismatch")
        with np.load(bundle, allow_pickle=False) as arrays:
            self.arrays = {key: arrays[key].copy() for key in arrays.files}
        self.digest = hashlib.sha256(path.read_bytes()).hexdigest()

        def forward(params, observation, noise):
            model = nnx.merge(model_def, params)
            model.eval()
            if not hasattr(model, "evaluation_velocity"):
                raise ValueError("CF velocity probes support Pi0/Pi05, not FAST/PyTorch")
            velocity = model.evaluation_velocity(observation, noise, timestep=1.0)
            actions = model.sample_actions(jax.random.key(0), observation, noise=noise,
                                           num_steps=config.selection_denoise_steps)
            return actions, velocity

        self.forward = jax.jit(forward, in_shardings=(params_sharding, data_sharding, data_sharding))

    def evaluate(self, params) -> tuple[dict, list[dict], dict]:
        import jax
        from openpi.models.model import Observation
        from openpi.training import sharding

        manifest, arrays, rows, artifacts = self.manifest, self.arrays, [], {}
        dims, continuous = manifest["active_dims"], manifest["continuous_dims"]
        for number, case in enumerate(manifest["cases"]):
            common = {key.split("/common/", 1)[1]: value for key, value in arrays.items() if key.startswith(f"{number}/common/")}
            observations = {}
            for variant in ("matched", "counterfactual", "paraphrase"):
                if f"{number}/{variant}/tokenized_prompt" not in arrays:
                    continue
                data = {"image": {}, "image_mask": {}}
                for key, value in common.items():
                    if key.startswith("image/"):
                        data["image"][key.split("/", 1)[1]] = value
                    elif key.startswith("image_mask/"):
                        data["image_mask"][key.split("/", 1)[1]] = value
                    else:
                        data[key] = value
                for key in ("tokenized_prompt", "tokenized_prompt_mask"):
                    data[key] = arrays[f"{number}/{variant}/{key}"]
                obs = Observation.from_dict(data)
                host = jax.tree.map(lambda value: np.repeat(np.asarray(value)[None], self.config.batch_size, axis=0), obs)
                observations[variant] = jax.tree.map(
                    lambda value: jax.make_array_from_process_local_data(self.sharding, value), host)
            horizon = int(case["valid_horizon"])
            target = arrays[f"{number}/target"][:horizon]
            for seed in manifest["noise_seeds"]:
                # Same observation, state and epsilon for ALL prompt variants.
                noise = arrays[f"noise/{seed}"][None]
                noise = np.repeat(noise, self.config.batch_size, axis=0)
                noise = jax.make_array_from_process_local_data(self.sharding, noise)
                outputs = {}
                for variant, obs in observations.items():
                    with sharding.set_mesh(self.mesh):
                        action, velocity = jax.device_get(self.forward(params, obs, noise))
                    outputs[variant] = (np.asarray(action[0], dtype=np.float32), np.asarray(velocity[0], dtype=np.float32))
                predicted, velocity = outputs["matched"]
                other, other_velocity = outputs["counterfactual"]
                difference = velocity.astype(np.float64)-other_velocity.astype(np.float64)
                row = {"case_id": case["id"], "group_id": case["group_id"], "task": case["task"],
                       "noise_seed": seed, "action_mse": float(np.mean((predicted[:horizon, dims]-target[:, dims])**2)),
                       "s_cf": counterfactual_squared_l2(velocity, other_velocity),
                       "s_cf_active": counterfactual_squared_l2(velocity, other_velocity, dims),
                       "grounding_margin": float(np.mean((other[:horizon, dims]-target[:, dims])**2)
                                                 - np.mean((predicted[:horizon, dims]-target[:, dims])**2))}
                stats = self.data.norm_stats["actions"]
                scales = ((np.asarray(stats.q99)-np.asarray(stats.q01)+1e-6)/2 if self.data.use_quantile_norm
                          else np.asarray(stats.std)+1e-6)
                row["native_rmse_by_active_dimension"] = np.sqrt(np.mean(
                    ((predicted[:horizon, dims]-target[:, dims])*scales[dims])**2, axis=0)).tolist()
                prefix = f"case_{number}/seed_{seed}"
                artifacts[f"{prefix}/predicted_normalized"] = predicted
                artifacts[f"{prefix}/target_normalized"] = target
                artifacts[f"{prefix}/counterfactual_normalized"] = other
                artifacts[f"{prefix}/initial_velocity_difference"] = difference
                # Smoothness is measured on trajectories, NOT diffusion velocity.
                for order, label in ((1, "speed"), (2, "acceleration"), (3, "jerk")):
                    if horizon <= order:
                        continue
                    scale = manifest["fps"] ** order
                    generated = np.diff(predicted[:horizon, continuous], n=order, axis=0) * scale
                    reference = np.diff(target[:, continuous], n=order, axis=0) * scale
                    row[f"normalized_{label}_rms"] = float(np.sqrt(np.mean(generated**2)))
                    row[f"reference_{label}_rms"] = float(np.sqrt(np.mean(reference**2)))
                if "normalized_jerk_rms" in row:
                    row["jerk_excess"] = max(0., row["normalized_jerk_rms"]-row["reference_jerk_rms"])
                if "paraphrase" in outputs:
                    paraphrase_velocity = outputs["paraphrase"][1]
                    row["s_paraphrase_active"] = float(np.sum((velocity[:, dims]-paraphrase_velocity[:, dims]).astype(np.float64)**2))
                    row["instruction_specificity"] = row["s_cf_active"] / max(1e-12, row["s_cf_active"]+row["s_paraphrase_active"])
                if not all(math.isfinite(value) for value in row.values() if isinstance(value, (int, float))):
                    raise FloatingPointError("Non-finite checkpoint probe metric")
                rows.append(row)
        result = _aggregate(rows)
        result["low_cf_scene_ids"] = []
        delta = manifest.get("gates", {}).get("min_s_cf")
        if delta is not None:
            for case in manifest["cases"]:
                values = np.asarray([row["s_cf"] for row in rows if row["case_id"] == case["id"]])
                se = values.std(ddof=1)/np.sqrt(len(values))
                if values.mean()+manifest.get("selection_policy", {}).get("uncertainty_multiplier", 2)*se < delta:
                    result["low_cf_scene_ids"].append(case["id"])
        result["semantic_reviewed"] = all(case.get("semantic_reviewed") is True and case.get("paraphrase_prompt") for case in manifest["cases"])
        return result, rows, artifacts
