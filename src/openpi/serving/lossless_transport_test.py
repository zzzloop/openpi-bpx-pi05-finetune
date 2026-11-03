"""Pure stdlib wire tests; can run directly without importing OpenPI/JAX/ROS."""
import ast
from pathlib import Path
import random
import unittest
import zlib


def load_decoder():
    source = Path(__file__).with_name("websocket_policy_server.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_decode_request")
    namespace = {"zlib": zlib}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_decode_request"]


class LosslessTransportTest(unittest.TestCase):
    def test_legacy_binary_is_unchanged(self):
        payload = b"\x83qpos-and-images-legacy-msgpack"
        self.assertIs(load_decoder()(payload), payload)

    def test_compressed_bytes_round_trip_exactly(self):
        rng = random.Random(9)
        payload = bytes(rng.getrandbits(8) for _ in range(8192)) + b"rgb-camera" * 20000
        encoded = b"BPXZ1" + zlib.compress(payload, level=1)
        self.assertLess(len(encoded), len(payload))
        self.assertEqual(load_decoder()(encoded), payload)

    def test_truncated_corrupted_and_trailing_data_rejected(self):
        valid = b"BPXZ1" + zlib.compress(b"valid payload" * 100)
        for invalid in (valid[:-2], valid + b"unexpected", b"BPXZ1not-zlib"):
            with self.subTest(payload=invalid[:12]):
                with self.assertRaises((ValueError, zlib.error)):
                    load_decoder()(invalid)

    def test_text_frames_rejected(self):
        with self.assertRaises(ValueError):
            load_decoder()("not a binary frame")

    def test_size_limit_for_raw_and_compressed_payloads(self):
        decode = load_decoder()
        self.assertEqual(decode(b"a" * 32, limit=32), b"a" * 32)
        for payload in (b"a" * 33, b"BPXZ1" + zlib.compress(b"a" * 33)):
            with self.assertRaises(ValueError):
                decode(payload, limit=32)


if __name__ == "__main__":
    unittest.main()
