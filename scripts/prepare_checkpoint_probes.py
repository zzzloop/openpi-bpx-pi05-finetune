"""Prepare immutable checkpoint probes on the SERVER, never during training."""
import argparse
import dataclasses
import pathlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--case-spec", type=pathlib.Path)
    parser.add_argument("--max-cases", type=int, default=16)
    args = parser.parse_args()
    from openpi.training import config, selection_probes
    selected = dataclasses.replace(config.get_config(args.config_name), validation_manifest=args.validation_manifest)
    result = selection_probes.prepare_suite(selected, args.output_dir, case_spec=args.case_spec, max_cases=args.max_cases)
    print(f"PROBE SUITE PASS: {result}", flush=True)
    print("Auto-generated cases are UNREVIEWED; edit editable_case_spec.json and rebuild to assert semantic validity.", flush=True)


if __name__ == "__main__":
    main()
