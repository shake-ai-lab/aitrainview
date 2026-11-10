"""Unit tests for the parts that need no GPU and no model download.

Run:  npm run test:backend   (or: python -m unittest discover -s backend/tests)
"""
from __future__ import annotations

import asyncio
import json
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aitrainview import events, server  # noqa: E402
from aitrainview.st_reader import SafetensorsCheckpoint, to_float32  # noqa: E402
from aitrainview.thumbs import quantize, thumbnail  # noqa: E402

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def write_safetensors(path: str, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    """Minimal writer: {name: (dtype tag, array of the storage dtype)}."""
    header, blobs, off = {}, [], 0
    for name, (dtype, arr) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    h += b" " * ((-len(h)) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def to_bf16_bits(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


class ReaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        rng = np.random.default_rng(0)
        self.w32 = rng.standard_normal((300, 517)).astype(np.float32)
        self.w16 = rng.standard_normal((64, 40)).astype(np.float16)
        self.wbf = to_bf16_bits(rng.standard_normal((130, 70)))
        write_safetensors(os.path.join(self.dir, "model.safetensors"), {
            "model.layers.0.mlp.up_proj.weight": ("F32", self.w32),
            "model.layers.0.self_attn.q_proj.weight": ("F16", self.w16),
            "model.layers.0.self_attn.k_proj.weight": ("BF16", self.wbf),
            "model.layers.0.input_layernorm.weight": ("F32", np.ones(40, np.float32)),
        })
        self.ck = SafetensorsCheckpoint(self.dir)

    def tearDown(self):
        self.ck.close()
        self.tmp.cleanup()

    def test_decode(self):
        np.testing.assert_array_equal(to_float32(self.ck.raw("model.layers.0.mlp.up_proj.weight"), "F32"), self.w32)
        np.testing.assert_array_equal(to_float32(self.ck.raw("model.layers.0.self_attn.q_proj.weight"), "F16"),
                                      self.w16.astype(np.float32))
        bf = to_float32(self.ck.raw("model.layers.0.self_attn.k_proj.weight"), "BF16")
        np.testing.assert_array_equal(bf, (self.wbf.astype(np.uint32) << 16).view(np.float32))

    def test_sample_fast_path_matches_full_read(self):
        for name in ("model.layers.0.mlp.up_proj.weight", "model.layers.0.self_attn.k_proj.weight"):
            fast, _ = thumbnail(self.ck, name, 64, "sample")
            full, stats = thumbnail(self.ck, name, 64, "sample", want_stats=True)
            np.testing.assert_array_equal(fast, full)
            self.assertIsNotNone(stats)

    def test_stats(self):
        _, s = thumbnail(self.ck, "model.layers.0.mlp.up_proj.weight", 32, "rms", want_stats=True)
        w = self.w32.astype(np.float64)
        self.assertAlmostEqual(s["mean"], w.mean(), places=6)
        self.assertAlmostEqual(s["rms"], np.sqrt((w * w).mean()), places=6)
        self.assertAlmostEqual(s["std"], w.std(), places=5)

    def test_rms_thumbnail(self):
        th, _ = thumbnail(self.ck, "model.layers.0.mlp.up_proj.weight", 300, "rms")
        self.assertEqual(th.shape, (300, 300))
        q, lo, hi = quantize(th, signed=False)
        self.assertEqual(q.dtype, np.uint8)
        self.assertLess(lo, hi)

    def test_missing_shard(self):
        d = os.path.join(self.dir, "sharded")
        os.mkdir(d)
        write_safetensors(os.path.join(d, "model-00001-of-00002.safetensors"), {"a": ("F32", self.w32)})
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as fh:
            json.dump({"weight_map": {"a": "model-00001-of-00002.safetensors",
                                      "b": "model-00002-of-00002.safetensors"}}, fh)
        with self.assertRaises(FileNotFoundError) as cm:
            SafetensorsCheckpoint(d)
        self.assertIn("model-00002-of-00002.safetensors", str(cm.exception))
        self.assertFalse(server._snapshot_complete(d))


class EventsTest(unittest.TestCase):
    def test_frame_roundtrip(self):
        buf = events.frame({"type": "x", "v": [1, 2], "nan": float("nan")}, b"\x01\x02\x03")
        (n,) = struct.unpack("<I", buf[:4])
        self.assertEqual((4 + n) % 4, 0)
        self.assertEqual(json.loads(buf[4:4 + n]), {"type": "x", "v": [1, 2], "nan": None})
        self.assertEqual(buf[4 + n:], b"\x01\x02\x03")

    def test_backlog_keeps_critical_events(self):
        loop = asyncio.new_event_loop()
        try:
            bus = events.Bus()
            bus.loop = loop
            q = bus.subscribe()
            for i in range(events.MAX_BACKLOG + 50):
                bus.publish({"type": "infer_step", "step": i})
            bus.publish({"type": "infer_done"})
            loop.run_until_complete(asyncio.sleep(0))
            self.assertEqual(q.qsize(), events.MAX_BACKLOG + 1)
            items = [q.get_nowait() for _ in range(q.qsize())]
            self.assertIn('"infer_done"', items[-1])
        finally:
            loop.close()


class AccessTest(unittest.TestCase):
    def tearDown(self):
        server.configure_access(8765)

    def test_origins(self):
        server.configure_access(9001)
        ok = server.origin_allowed
        self.assertTrue(ok(None))
        self.assertTrue(ok("http://tauri.localhost"))
        self.assertTrue(ok("http://127.0.0.1:9001"))
        self.assertTrue(ok("http://localhost:5173"))
        self.assertFalse(ok("http://localhost:3000"))
        self.assertFalse(ok("https://example.com"))
        self.assertFalse(ok("null"))

    def test_hosts(self):
        server.configure_access(9001)
        ok = server.host_allowed
        self.assertTrue(ok("127.0.0.1:9001"))
        self.assertTrue(ok("localhost"))
        self.assertTrue(ok("[::1]:9001"))
        self.assertFalse(ok("evil.example:9001"))
        server.configure_access(9001, host="0.0.0.0")  # explicitly served to the LAN
        self.assertTrue(ok("192.168.1.5:9001"))


class CharTok:
    """Character-level stand-in for a tokenizer without a chat template."""
    chat_template = None

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) for c in text]}


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class DatasetTest(unittest.TestCase):
    def test_parse_formats(self):
        from aitrainview.trainer import parse_dataset
        rows = parse_dataset("\n".join([
            json.dumps({"instruction": "a", "input": "b", "output": "c"}),
            json.dumps({"prompt": "p", "completion": "q"}),
            json.dumps({"conversations": [{"from": "human", "value": "h"}, {"from": "gpt", "value": "g"}]}),
            json.dumps({"text": "t"}),
        ]))
        self.assertEqual(rows[0]["messages"][0]["content"], "a\n\nb")
        self.assertEqual(rows[1]["messages"][1]["content"], "q")
        self.assertEqual(rows[2]["messages"][1]["role"], "assistant")
        self.assertEqual(rows[3], {"text": "t"})
        self.assertEqual(parse_dataset("just some text"), [{"text": "just some text"}])

    def test_labels_cover_only_the_last_reply(self):
        from aitrainview.trainer import encode_examples
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"},
                {"role": "user", "content": "trailing question"}]
        (ids, labels), = encode_examples(CharTok(), [{"messages": msgs}], 512)
        target = "".join(chr(t) for t in labels if t != -100)
        self.assertEqual(target, " yo")
        self.assertNotIn("trailing", "".join(chr(t) for t in ids))

    def test_chat_without_reply_trains_on_everything(self):
        from aitrainview.trainer import encode_examples
        (ids, labels), = encode_examples(CharTok(), [{"messages": [{"role": "user", "content": "hello"}]}], 512)
        self.assertEqual(ids, labels)


if __name__ == "__main__":
    unittest.main()
