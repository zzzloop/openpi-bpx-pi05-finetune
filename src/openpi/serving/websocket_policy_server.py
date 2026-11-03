import asyncio
import http
import logging
import time
import traceback
import zlib

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


def _decode_request(payload, limit=64 * 1024 * 1024):
    """Decode the negotiated lossless envelope, with a bounded expansion size."""
    if not isinstance(payload, bytes):
        raise ValueError("Inference request must be a binary WebSocket frame")
    if len(payload) > limit:
        raise ValueError("Inference request exceeds 64 MiB")
    if not payload.startswith(b"BPXZ1"):
        return payload
    decoder = zlib.decompressobj()
    decoded = decoder.decompress(payload[5:], limit + 1)
    if (len(decoded) > limit or decoder.unconsumed_tail or decoder.unused_data or not decoder.eof):
        raise ValueError("Invalid or oversized compressed inference request")
    return decoded


def _warmup_bpx_policy(policy, metadata: dict) -> None:
    """Compile BPX inference before listening; never send warmup actions anywhere.

    Other embodiments are deliberately untouched. Warmup uses the normal input
    transforms and sampler, so it also initializes tokenizer/device work. Like
    two ordinary inference calls it consumes sampler RNG, not training updates.
    """
    if metadata.get("raw_action_dim") != 23 or metadata.get("effective_action_dim") != 18:
        return
    import numpy as np

    from openpi.policies import bpx_constants

    logger.info("BPX WARMUP: compiling before listening; no robot connection or motor output")
    for index in range(2):
        observation = {
            "qpos": np.zeros(bpx_constants.BPX_RAW_DIM, dtype=np.float32),
            "images": {
                key: np.zeros(bpx_constants.BPX_SOURCE_IMAGE_SHAPE, dtype=np.uint8)
                for key in bpx_constants.BPX_IMAGE_KEYS
            },
            "prompt": bpx_constants.BPX_DEFAULT_PROMPT,
        }
        started = time.monotonic()
        result = policy.infer(observation)
        # Materialize GPU output: timing must include execution, not just dispatch.
        actions = np.asarray(result["actions"])
        expected = (metadata["action_horizon"], bpx_constants.BPX_RAW_DIM)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"BPX warmup returned invalid actions: {actions.shape}; expected {expected}")
        logger.info("BPX WARMUP %d/2 finished in %.1f ms", index + 1, (time.monotonic() - started) * 1000)
    logger.info("BPX WARMUP READY: accepting real observations next; synthetic actions discarded")


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = dict(metadata or {})
        self._metadata["request_compression"] = "zlib-v1"
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        _warmup_bpx_policy(self._policy, self._metadata)
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        request_number = 0
        while True:
            try:
                start_time = time.monotonic()
                raw_obs = await websocket.recv()
                request_received_time = time.monotonic()
                request_received_unix_ns = time.time_ns()

                unpack_time = time.monotonic()
                decoded_obs = _decode_request(raw_obs)
                obs = msgpack_numpy.unpackb(decoded_obs)
                unpack_time = time.monotonic() - unpack_time

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                response_ready_time = time.monotonic()
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                    "request_unpack_ms": unpack_time * 1000,
                    "request_wire_bytes": len(raw_obs),
                    "request_uncompressed_bytes": len(decoded_obs),
                    # Time from receiving the complete WebSocket frame through
                    # observation decoding and policy inference. Response
                    # MessagePack serialization and socket delivery happen
                    # after this timestamp and are measured by the client-side
                    # residual rather than being misreported as model time.
                    "pre_response_ms": (response_ready_time - request_received_time) * 1000,
                    "request_received_unix_ns": request_received_unix_ns,
                    "response_ready_unix_ns": time.time_ns(),
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                request_number += 1
                if request_number <= 3 or request_number % 30 == 0 or infer_time > 0.5:
                    logger.info(
                        "Inference request=%d peer=%s wire=%dB raw=%dB decode=%.1fms infer=%.1fms",
                        request_number, websocket.remote_address, len(raw_obs), len(decoded_obs),
                        unpack_time * 1000, infer_time * 1000,
                    )

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
