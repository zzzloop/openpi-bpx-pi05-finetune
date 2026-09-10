# BPX Studio offline wheels

This directory intentionally contains the small, platform-independent Python wheels required by the BPX Studio web server. `scripts/start_bpx_studio.sh` adds the wheel files directly to `PYTHONPATH`; it never installs them and never contacts a package index.

The versions are identical to `uv.lock`. Pydantic and `typing-extensions` are not duplicated here because they are already required, mutually compatible OpenPI dependencies. The wheel archives include their upstream package metadata and license files.
