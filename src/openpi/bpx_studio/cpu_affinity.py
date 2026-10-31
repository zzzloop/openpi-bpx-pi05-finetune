"""Opt-in Linux CPU-socket binding before importing any ML runtime.

Socket IDs are physical_package_id values, not logical CPU or NUMA node IDs.
Only the new process and its children are affected; no host settings change.
Disable hwloc's x86 probe, which temporarily rebinds threads outside this mask.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys


def socket_cpus(socket_id: int, allowed: set[int], topology_root: pathlib.Path) -> set[int]:
    """Intersect the requested socket with the current process's allowed CPUs."""
    if socket_id < 0:
        raise ValueError("CPU socket ID must be non-negative")
    selected = set()
    for cpu in sorted(allowed):
        path = topology_root / f"cpu{cpu}" / "topology" / "physical_package_id"
        try:
            package = int(path.read_text(encoding="ascii").strip())
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot read CPU topology: {path}: {error}") from error
        if package == socket_id:
            selected.add(cpu)
    if not selected:
        raise ValueError(
            f"No allowed online CPUs belong to physical socket {socket_id}; "
            "check lscpu -e=CPU,SOCKET,NODE,ONLINE. Refusing to fall back to another socket."
        )
    return selected


def bind_socket(socket_id: int) -> set[int]:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise ValueError("CPU socket binding requires Linux sched affinity and sysfs topology")
    selected = socket_cpus(socket_id, set(os.sched_getaffinity(0)), pathlib.Path("/sys/devices/system/cpu"))
    os.sched_setaffinity(0, selected)
    actual = set(os.sched_getaffinity(0))
    if actual != selected:
        raise ValueError(f"CPU affinity verification failed: requested={sorted(selected)}, actual={sorted(actual)}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=int, required=True, help="Linux physical CPU socket ID")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("provide a command after --")
    try:
        cpus = bind_socket(args.socket)
        cpu_list = ",".join(str(cpu) for cpu in sorted(cpus))
        print(f"[BPX CPU] physical socket={args.socket}; logical CPUs={cpu_list}; count={len(cpus)}", flush=True)
        print("[BPX CPU] New process and workers inherit this affinity; existing jobs are unchanged.", flush=True)
        environment = os.environ.copy()
        # Consume the launch selector to avoid re-exec recursion. Kernel affinity
        # and an audit label persist in the new program and all its children.
        environment.pop("BPX_CPU_SOCKET", None)
        environment["BPX_BOUND_CPU_SOCKET"] = str(args.socket)
        # sched affinity is not a cgroup fence: hwloc's CPUID discovery can
        # temporarily bind a native worker to CPU0 anyway. Disable just that
        # component BEFORE importing JAX/CUDA; Linux/sysfs discovery remains.
        # Keep other user-selected components, and never change the parent env.
        components = [part.strip() for part in environment.get("HWLOC_COMPONENTS", "").split(",") if part.strip()]
        if "-x86" not in components:
            components.insert(0, "-x86")
        environment["HWLOC_COMPONENTS"] = ",".join(components)
        print(f"[BPX CPU] HWLOC_COMPONENTS={environment['HWLOC_COMPONENTS']}; x86 CPUID rebinding probe disabled.",
              flush=True)
        os.execvpe(command[0], command, environment)
    except (OSError, ValueError) as error:
        print(f"[ERROR] CPU socket launch failed: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
