"""Read-only checks against a running Studio; no training jobs are started."""
import argparse
import json
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8866")
    args = parser.parse_args()

    def get(path):
        with urllib.request.urlopen(args.url.rstrip("/") + path, timeout=30) as response:
            return json.load(response)

    assert get("/healthz")["ok"], "Health check failed"
    config = get("/api/config")
    print("PASS: API accessible without a token")
    print("Data root:", config["data_root"])
    profile = get("/api/config-profile")
    assert profile["dataset_path"] == config["clean_dataset"], "Default dataset paths differ"
    assert profile["stats_paths_match"], "Normalization output and training input differ"
    print("PASS: CLEAN -> CONFIG -> STATS paths agree")
    print("Dataset exists:", profile["dataset_exists"], "Stats exist:", profile["loaded_stats_exists"])
    system = get("/api/system")
    assert "error" not in system["disk"], system["disk"]
    print("PASS: disk status", system["disk"]["path"])
    print("GPUs:", json.dumps(system["gpus"], ensure_ascii=False))
    if system.get("gpu_error"):
        print("GPU diagnostic:", system["gpu_error"])
    print("Complete checkpoints:", len(get("/api/checkpoints")))
    print("Jobs:", len(get("/api/jobs")))
    print("Read-only verification complete; real training/inference was not run.")


if __name__ == "__main__":
    main()
