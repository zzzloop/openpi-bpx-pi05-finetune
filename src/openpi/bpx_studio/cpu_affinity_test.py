"""Topology-only tests; no torch, JAX, GPU or robot imports."""

from __future__ import annotations

import pathlib

import pytest

from openpi.bpx_studio.cpu_affinity import socket_cpus


def _topology(root: pathlib.Path, mapping: dict[int, int]) -> pathlib.Path:
    for cpu, socket_id in mapping.items():
        target = root / f"cpu{cpu}" / "topology" / "physical_package_id"
        target.parent.mkdir(parents=True)
        target.write_text(str(socket_id), encoding="ascii")
    return root


def test_socket_selection_uses_topology_not_a_contiguous_half(tmp_path: pathlib.Path) -> None:
    root = _topology(tmp_path, {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1})
    assert socket_cpus(1, {0, 1, 2, 3, 4, 5}, root) == {1, 3, 5}


def test_socket_selection_does_not_expand_existing_cpuset(tmp_path: pathlib.Path) -> None:
    root = _topology(tmp_path, {0: 0, 1: 1, 3: 1, 5: 1})
    assert socket_cpus(1, {0, 3}, root) == {3}


def test_missing_socket_fails_instead_of_using_faulty_socket(tmp_path: pathlib.Path) -> None:
    root = _topology(tmp_path, {0: 0, 2: 0})
    with pytest.raises(ValueError, match="Refusing to fall back"):
        socket_cpus(1, {0, 2}, root)


def test_unreadable_topology_fails_closed(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="Cannot read CPU topology"):
        socket_cpus(1, {1}, tmp_path)


def test_negative_socket_is_rejected(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        socket_cpus(-1, {0}, tmp_path)


def test_binding_is_verified_before_return(monkeypatch: pytest.MonkeyPatch) -> None:
    from openpi.bpx_studio import cpu_affinity

    calls = []
    affinity = iter([{0, 1, 2, 3}, {1, 3}])
    monkeypatch.setattr(cpu_affinity.os, "sched_getaffinity", lambda pid: next(affinity), raising=False)
    monkeypatch.setattr(cpu_affinity.os, "sched_setaffinity", lambda pid, cpus: calls.append((pid, cpus)), raising=False)
    monkeypatch.setattr(cpu_affinity, "socket_cpus", lambda socket, allowed, root: {1, 3})
    assert cpu_affinity.bind_socket(1) == {1, 3}
    assert calls == [(0, {1, 3})]


def test_cli_execs_after_binding_and_consumes_launcher_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    from openpi.bpx_studio import cpu_affinity

    monkeypatch.setattr(cpu_affinity.sys, "argv", ["cpu_affinity", "--socket", "1", "--", "python", "-V"])
    monkeypatch.setenv("BPX_CPU_SOCKET", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.delenv("HWLOC_COMPONENTS", raising=False)
    monkeypatch.setattr(cpu_affinity, "bind_socket", lambda socket: {1, 3})
    captured = []
    monkeypatch.setattr(cpu_affinity.os, "execvpe", lambda file, argv, env: captured.append((file, argv, env)))
    cpu_affinity.main()
    file, argv, env = captured[0]
    assert file == "python" and argv == ["python", "-V"]
    assert "BPX_CPU_SOCKET" not in env
    assert env["BPX_BOUND_CPU_SOCKET"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert env["HWLOC_COMPONENTS"] == "-x86"
    assert "HWLOC_COMPONENTS" not in cpu_affinity.os.environ


@pytest.mark.parametrize(
    ("components", "expected"),
    [("", "-x86"), ("linux", "-x86,linux"), ("-pci", "-x86,-pci"),
     ("linux,stop", "-x86,linux,stop"), ("-x86,-pci", "-x86,-pci")],
)
def test_cli_disables_only_x86_and_preserves_other_environment(monkeypatch, components, expected):
    from openpi.bpx_studio import cpu_affinity

    monkeypatch.setattr(cpu_affinity.sys, "argv", ["cpu_affinity", "--socket", "1", "--", "python", "-V"])
    monkeypatch.setenv("HWLOC_COMPONENTS", components)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.96")
    monkeypatch.setattr(cpu_affinity, "bind_socket", lambda socket: {28, 29, 84, 85})
    captured = []
    monkeypatch.setattr(cpu_affinity.os, "execvpe", lambda file, argv, env: captured.append(env))
    cpu_affinity.main()
    assert captured[0]["HWLOC_COMPONENTS"] == expected
    assert captured[0]["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert captured[0]["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.96"
    assert cpu_affinity.os.environ["HWLOC_COMPONENTS"] == components


def test_cli_does_not_exec_after_binding_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from openpi.bpx_studio import cpu_affinity

    monkeypatch.setattr(cpu_affinity.sys, "argv", ["cpu_affinity", "--socket", "1", "--", "python", "-V"])

    def reject(socket):
        raise ValueError("No allowed online CPUs")

    monkeypatch.setattr(cpu_affinity, "bind_socket", reject)
    captured = []
    monkeypatch.setattr(cpu_affinity.os, "execvpe", lambda *args: captured.append(args))
    with pytest.raises(SystemExit) as error:
        cpu_affinity.main()
    assert error.value.code == 2 and not captured
