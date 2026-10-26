"""Auditable multi-objective candidate retention. No model updates or loss terms."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pathlib
import shutil
import time

import numpy as np


def atomic_json(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _finite(record, key):
    value = record.get(key)
    return isinstance(value, (int, float)) and math.isfinite(value)


def _upper(record, key, multiplier):
    return float(record[key])+multiplier*float(record.get(f"{key}_se", 0))


def _wilson(successes, trials, z=1.96):
    if type(successes) is not int or type(trials) is not int or not 0 <= successes <= trials or trials < 1:
        raise ValueError("External assessments require valid integer success/total counts")
    p = successes/trials
    center = (p+z*z/(2*trials))/(1+z*z/trials)
    radius = z*math.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/(1+z*z/trials)
    return max(0., center-radius), min(1., center+radius)


def external_metrics(evidence, *, suite_sha256, run_dir):
    """Validate evidence as a whole before adding benchmark claims to records."""
    if (evidence.get("schema_version") != 1 or evidence.get("suite_sha256") != suite_sha256
            or pathlib.Path(evidence.get("run_dir", "")).resolve() != run_dir.resolve()):
        raise ValueError("External assessment is bound to a different schema/suite/run")
    updates, protocols = {}, {}
    baseline = None
    for item in evidence["records"]:
        step = item["checkpoint_step"]
        if type(step) is not int or step < 0 or step in updates:
            raise ValueError("External checkpoint steps must be distinct nonnegative integers")
        updates[step] = {}
        if item.get("operator_reviewed") is not True:
            continue
        for kind in ("task_success", "vlm_retention"):
            if kind not in item:
                continue
            counts = item[kind]
            protocol = str(counts.get("protocol_id", "")).strip()
            if not protocol or not str(counts.get("evidence_uri", "")).strip():
                raise ValueError("Reviewed external metrics require protocol_id and evidence_uri")
            if kind in protocols and protocol != protocols[kind]:
                raise ValueError("Do not compare external metrics from different protocols in one run")
            protocols[kind] = protocol
            if kind == "task_success":
                low, high = _wilson(counts["successes"], counts["trials"])
                updates[step].update(task_success_lower=low, task_success_upper=high)
            else:
                low, high = _wilson(counts["correct"], counts["total"])
                base_low, base_high = _wilson(counts["baseline_correct"], counts["baseline_total"])
                baseline_id = str(counts.get("baseline_id", "")).strip()
                if not baseline_id or counts["total"] != counts["baseline_total"]:
                    raise ValueError("VLM comparison requires a named baseline and the same benchmark sample count")
                identity = (baseline_id, counts["baseline_correct"], counts["baseline_total"])
                if baseline is not None and identity != baseline:
                    raise ValueError("VLM baseline changed across checkpoints")
                baseline = identity
                updates[step].update(vlm_difference_lower=low-base_high, vlm_difference_upper=high-base_low)
        updates[step]["external_protocols"] = {kind: item[kind]["protocol_id"] for kind in protocols if kind in item}
    return updates


def choose_candidates(records: list[dict], *, keep: int, gates: dict | None = None,
                      multiplier: float = 2., patience: int = 3) -> dict:
    """One-SE-style admissible region + Pareto frontier + earliest stable tie.

    No fixed step window, arbitrary weighted score, or 50% retention cutoff.
    Absolute task adequacy cannot be inferred without calibrated user gates.
    """
    gates = gates or {}
    required = ("val_loss", "action_mse", "jerk_excess", "s_cf_active")
    available = [row for row in records if row.get("available") and all(_finite(row, key) for key in required)]
    warnings = []
    scored = [row for row in records if all(_finite(row, key) for key in required)]
    recent = scored[-patience:]
    if len(scored) > 1 and scored[-1]["action_mse"]-multiplier*scored[-1].get("action_mse_se", 0) > _upper(scored[0], "action_mse", multiplier):
        warnings.append("adaptation_worse_than_initial_reference: do not infer task competence from relative ranking")
    if scored and scored[-1].get("low_cf_scene_ids"):
        warnings.append("low_cf_scene_response: review/collect counterfactual demonstrations for "+", ".join(scored[-1]["low_cf_scene_ids"]))
    if scored and scored[-1].get("vlm_difference_upper", 0) < 0:
        warnings.append("external_vlm_benchmark_decline: current accuracy band below the matched baseline; benchmark-specific evidence")
    if len(recent) == patience:
        best_val = min(scored, key=lambda row: row["val_loss"])
        if (all(row["val_loss"]-multiplier*row.get("val_stderr", 0) >
                best_val["val_loss"]+multiplier*best_val.get("val_stderr", 0) for row in recent)
                and recent[-1].get("train_eval_loss", math.inf) < recent[0].get("train_eval_loss", -math.inf)):
            warnings.append("overfit_suspected: held-out loss worsens while train-reference loss decreases")
        references = scored[:-patience] or scored[:1]
        reference = max(references, key=lambda row: row["s_cf_active"]-multiplier*row.get("s_cf_active_se", 0))
        if all(_upper(row, "s_cf_active", multiplier) < reference["s_cf_active"]-multiplier*reference.get("s_cf_active_se", 0) for row in recent):
            warnings.append(f"instruction_response_decline: CF response below earlier step {reference['step']} uncertainty band; review/collect counterfactual scenes")
        if len(scored) > patience:
            previous = scored[-patience-1]
            if abs(recent[-1]["val_loss"]-previous["val_loss"]) <= multiplier*(recent[-1].get("val_stderr", 0)+previous.get("val_stderr", 0)):
                warnings.append("validation_plateau_possible: improvement smaller than measured batch uncertainty")
    if not available:
        return {"status": "insufficient_evidence", "recommended_step": None, "retained_steps": [],
                "warnings": warnings+["No complete, available checkpoint with finite required metrics"], "frontier": []}
    # User gates are optional and must be calibrated on this embodiment/suite.
    eligible = [row for row in available if
        (gates.get("max_action_mse") is None or row["action_mse"] <= gates["max_action_mse"])
        and (gates.get("max_jerk_excess") is None or row["jerk_excess"] <= gates["max_jerk_excess"])
        and (gates.get("min_s_cf_active") is None or row["s_cf_active"] >= gates["min_s_cf_active"])
        and (gates.get("min_s_cf") is None or row.get("s_cf", -math.inf) >= gates["min_s_cf"])
        and (gates.get("min_task_success") is None or row.get("task_success_lower", -1) >= gates["min_task_success"])
        and (gates.get("vlm_noninferiority_margin") is None or row.get("vlm_difference_lower", -math.inf) >= -gates["vlm_noninferiority_margin"])]
    if not eligible:
        return {"status": "no_candidate_meets_user_gates", "recommended_step": None,
                "retained_steps": [], "warnings": warnings, "frontier": []}
    best_action = min(eligible, key=lambda row: row["action_mse"])
    best_val = min(eligible, key=lambda row: row["val_loss"])
    region = [row for row in eligible if
              row["action_mse"] <= _upper(best_action, "action_mse", multiplier)+multiplier*row.get("action_mse_se", 0)
              and row["val_loss"] <= best_val["val_loss"]+multiplier*(best_val.get("val_stderr", 0)+row.get("val_stderr", 0))]
    # If the two quality optima conflict, preserve Pareto alternatives rather
    # than inventing a sum of differently scaled losses/semantic scores.
    pool = region or eligible
    include_specificity = all(_finite(row, "instruction_specificity") for row in pool)

    def vector(row):
        return (row["action_mse"], row["val_loss"], row["jerk_excess"],
                *([-row["instruction_specificity"]] if include_specificity else []))

    frontier = [row for row in pool if not any(
        all(a <= b for a, b in zip(vector(other), vector(row), strict=True))
        and any(a < b for a, b in zip(vector(other), vector(row), strict=True)) for other in pool if other is not row)]
    best_jerk = min(frontier, key=lambda row: row["jerk_excess"])
    balanced = [row for row in frontier if row["jerk_excess"] <= _upper(best_jerk, "jerk_excess", multiplier)+multiplier*row.get("jerk_excess_se", 0)]
    language_supported = [row for row in balanced if row.get("semantic_reviewed")
        and row.get("independent_groups", 0) >= 2 and _finite(row, "s_paraphrase_active")
        and row["s_cf_active"]-multiplier*row.get("s_cf_active_se", 0) > _upper(row, "s_paraphrase_active", multiplier)
        and row.get("grounding_margin", -math.inf)-multiplier*row.get("grounding_margin_se", 0) > 0]
    recommended = min(language_supported or balanced, key=lambda row: row["step"])
    representatives = [recommended, min(frontier, key=lambda row: row["action_mse"])]
    if include_specificity:
        representatives.append(max(frontier, key=lambda row: row["instruction_specificity"]))
    representatives += [best_jerk, min(frontier, key=lambda row: row["val_loss"])]
    ordered = []
    for row in representatives+sorted(frontier, key=lambda row: row["step"]):
        if row not in ordered:
            ordered.append(row)
    # Every outcome remains an OFFLINE recommendation, never proof of task
    # success or of preserving general VLM/VQA capabilities.
    calibrated = gates.get("max_action_mse") is not None and gates.get("max_jerk_excess") is not None
    status = "offline_balanced_candidate" if language_supported and calibrated and region else "provisional_offline_candidate"
    if (language_supported and region and gates.get("min_task_success") is not None
            and gates.get("vlm_noninferiority_margin") is not None):
        status = "candidate_with_external_benchmark_support"
    if not calibrated:
        warnings.append("absolute_adaptation_not_calibrated: relative quality does not prove a usable grasp")
    if not language_supported:
        warnings.append("semantic_retention_unconfirmed: needs reviewed instruction pairs/paraphrases and positive grounding evidence")
    if not region:
        warnings.append("action_validation_tradeoff: no candidate lies near both observed optima")
    return {"status": status, "recommended_step": recommended["step"],
            "retained_steps": [row["step"] for row in ordered[:keep]],
            "frontier": [row["step"] for row in frontier], "warnings": warnings,
            "rationale": "near-best action/held-out loss, Pareto smoothness/specificity, earliest candidate within configured uncertainty heuristic",
            "scope": "offline action/language proxies; not robot success, balance safety or general VLM validation"}


class SelectionHistory:
    def __init__(self, run_dir: pathlib.Path, identity: dict, *, keep: int, start_step: int,
                 resume: bool, gates: dict | None = None, external_assessments: str | None = None):
        self.run_dir, self.keep, self.gates = run_dir.resolve(), keep, gates or {}
        self.external_assessments = pathlib.Path(external_assessments).expanduser().resolve() if external_assessments else None
        self.policy = identity.get("selection_policy", {})
        self.root = self.run_dir / "monitoring/selection"
        self.archives = self.run_dir / "selected_checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        self.archives.mkdir(exist_ok=True)
        self.path = self.root / "evaluations.jsonl"
        protocol = self.root / "protocol.json"
        if resume and protocol.exists() and json.loads(protocol.read_text(encoding="utf-8")) != identity:
            raise ValueError("Selection protocol changed: start a new experiment")
        if resume and not protocol.exists() and start_step > 0:
            raise ValueError("Do not retrofit initialization-retention claims onto an old run; use offline evaluation or a new experiment")
        atomic_json(protocol, identity)
        self.identity = identity
        self.records = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()] if self.path.exists() else []
        self.records = [row for row in self.records if row["step"] < start_step]
        self._persist()
        previous_recommendation = self.root / "recommended.json"
        if previous_recommendation.exists():
            prior = json.loads(previous_recommendation.read_text(encoding="utf-8"))
            if prior.get("recommended_step") is not None and prior["recommended_step"] >= start_step:
                atomic_json(previous_recommendation, {"status": "awaiting_resumed_evaluation", "checkpoint_path": None,
                    "recommended_step": None, "retained_steps": [], "warnings": ["Restored an earlier training state; future-run metrics are not valid for this branch"]})

    def _persist(self):
        with self.path.open("w", encoding="utf-8") as stream:
            for row in self.records:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        keys = sorted({key for row in self.records for key, value in row.items() if not isinstance(value, (dict, list))})
        with (self.root / "evaluations.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.records)

    def _source(self, step):
        archived = self.archives / str(step)
        return archived if (archived / "params").is_dir() and (archived / "assets").is_dir() else self.run_dir / str(step)

    def _archive(self, step):
        destination = self.archives / str(step)
        if destination.exists():
            marker = destination / "selection.json"
            if (destination.is_symlink() or destination.resolve().parent != self.archives.resolve()
                    or not marker.is_file()
                    or json.loads(marker.read_text(encoding="utf-8")).get("owner_run") != str(self.run_dir)):
                raise ValueError(f"Candidate directory is not owned by this selector: {destination}")
            return
        source = self._source(step)
        if not (source / "params").is_dir() or not (source / "assets").is_dir():
            raise FileNotFoundError(f"Checkpoint has not completed saving: {source}")
        staging = self.archives / f".{step}.partial"
        if staging.exists():
            marker = staging / "selection.json"
            if (staging.is_symlink() or staging.resolve().parent != self.archives.resolve() or not marker.is_file()
                    or json.loads(marker.read_text(encoding="utf-8")).get("owner_run") != str(self.run_dir)):
                raise FileExistsError(f"Unowned incomplete archive exists; inspect before retry: {staging}")
            shutil.rmtree(staging)
        staging.mkdir()
        try:
            atomic_json(staging / "selection.json", {"owner_run": str(self.run_dir), "step": step,
                "protocol": self.identity, "usage": "inference or weight-only NEW finetuning; not optimizer resume"})
            for item in ("params", "assets"):
                shutil.copytree(source / item, staging / item, copy_function=os.link)
            evaluation = staging / "evaluation"
            evaluation.mkdir()
            for name in ("protocol.json", "suite.json", "probes.npz", "training_config.json"):
                if (self.root / name).is_file():
                    os.link(self.root / name, evaluation / name)
            os.replace(staging, destination)
        except BaseException:
            # Never silently copy enormous weights or touch original checkpoints.
            raise RuntimeError(f"Candidate archival failed; originals unchanged. Check same-filesystem hard-link support and {staging}")

    def append(self, step, metrics, rows, artifacts, *, optimizer_step, checkpoint_complete):
        record = {"step": int(step), "optimizer_step": int(optimizer_step), "timestamp": time.time(),
                  **metrics, "checkpoint_complete": bool(checkpoint_complete)}
        if any(not math.isfinite(value) for value in record.values() if isinstance(value, (int, float))):
            raise FloatingPointError("Selection metrics are non-finite; do not smooth or select them")
        if any(row["step"] == step for row in self.records):
            raise ValueError("Duplicate selection step")
        case_dir = self.root / "probes"
        case_dir.mkdir(exist_ok=True)
        atomic_json(case_dir / f"step_{step}.json", rows)
        np.savez_compressed(case_dir / f"step_{step}.npz", **artifacts)
        self.records.append(record)
        if self.external_assessments and self.external_assessments.is_file():
            content = self.external_assessments.read_bytes()
            evidence = json.loads(content)
            updates = external_metrics(evidence, suite_sha256=self.identity["suite_sha256"], run_dir=self.run_dir)
            for previous in self.records:
                if previous["step"] not in updates:
                    continue
                for key in ("task_success_lower", "task_success_upper", "vlm_difference_lower", "vlm_difference_upper", "external_protocols"):
                    previous.pop(key, None)
                previous.update(updates[previous["step"]])
                previous["external_evidence_sha256"] = hashlib.sha256(content).hexdigest()
            atomic_json(self.root / f"external_evidence_{step}.json", evidence)
        for previous in self.records:
            source = self._source(previous["step"])
            previous["available"] = previous["checkpoint_complete"] and (source / "params").is_dir() and (source / "assets").is_dir()
        self._persist()
        decision = choose_candidates(self.records, keep=self.keep, gates=self.gates,
            multiplier=self.policy.get("uncertainty_multiplier", 2.), patience=self.policy.get("patience", 3))
        for candidate in decision["retained_steps"]:
            self._archive(candidate)
        chosen = decision["recommended_step"]
        decision["checkpoint_path"] = str(self.archives / str(chosen)) if chosen is not None else None
        decision["run_dir"] = str(self.run_dir)
        decision["updated_at"] = time.time()
        atomic_json(self.root / "recommended.json", decision)
        atomic_json(self.root / f"decision_{step}.json", decision)
        # Delete only previously generated candidate links, never original
        # Orbax checkpoints or any unowned folder. Top-K bounds disk retention.
        pruned = []
        for directory in self.archives.iterdir():
            owner = directory / "selection.json"
            if directory.name.isdigit() and int(directory.name) not in decision["retained_steps"] and owner.is_file():
                if (directory.is_symlink() or directory.resolve().parent != self.archives.resolve()
                        or json.loads(owner.read_text(encoding="utf-8")).get("owner_run") != str(self.run_dir)):
                    raise ValueError("Refusing to prune an unowned candidate")
                shutil.rmtree(directory)
                pruned.append(int(directory.name))
        if pruned:
            atomic_json(self.root / f"pruned_links_{step}.json", {"steps": pruned, "original_checkpoints_deleted": False})
        self._persist()
        self.render()
        return decision

    def render(self):
        if not self.records:
            return
        import matplotlib.pyplot as plt
        plt.switch_backend("Agg")
        figure, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
        series = [("action_mse", "Held-out sampled-action MSE (normalized)"),
                  ("jerk_excess", "Jerk excess vs demonstration (not closed-loop stability)"),
                  ("s_cf_active", "CF squared L2 / active dimensions"),
                  ("s_cf", "CF squared L2 / full model dimensions"),
                  ("grounding_margin", "Wrong-prompt MSE minus matched-prompt MSE"),
                  ("instruction_specificity", "CF / (CF + paraphrase response)")]
        for axis, (key, title) in zip(axes.flat, series, strict=True):
            rows = [row for row in self.records if _finite(row, key)]
            x, y = [row["step"] for row in rows], np.asarray([row[key] for row in rows])
            se = np.asarray([row.get(key+"_se", 0) for row in rows])
            axis.plot(x, y, marker=".")
            axis.fill_between(x, y-2*se, y+2*se, alpha=.12)
            axis.set_title(title, fontsize=9)
            axis.set_xlabel("checkpoint directory step (-1 = initialization baseline)")
            axis.grid(alpha=.2)
        figure.suptitle("Offline checkpoint probes: raw metrics, fixed scenes/noise; not a general VLM benchmark")
        path = self.root / "selection_latest.png"
        temporary = path.with_suffix(".png.tmp")
        figure.savefig(temporary, format="png", dpi=140)
        plt.close(figure)
        os.replace(temporary, path)
