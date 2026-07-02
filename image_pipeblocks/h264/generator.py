#!/usr/bin/env python3
"""
Fast raw H.264 Annex-B -> MP4 remux test using:

    Python H264FileChunkGenerator
    -> appsrc
    -> h264parse
    -> timestamp pad-probe on parsed AU buffers
    -> mp4mux
    -> filesink

Why the pad-probe exists:
    mp4mux requires timestamped access-unit buffers. Raw .h264 elementary
    streams do not carry container timestamps, and arbitrary appsrc byte chunks
    have no frame timing. h264parse creates AU-aligned buffers; this script
    stamps those parsed AU buffers at a fixed FPS.

Usage:
    python3 this.py rgb_full_test.h264 rgb_full_test.mp4 --fps 30
"""

import argparse
import os
import threading
import time
import uuid
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from pydantic import BaseModel, PrivateAttr


logger = print


def _quote_gst_path(path: str) -> str:
    # GStreamer parse_launch accepts quoted property strings.
    return '"' + os.path.abspath(path).replace('"', '\\"') + '"'


def _parse_fps(value: str) -> tuple[int, int]:
    """Accept '30', '30000/1001', '29.97'. Return numerator, denominator."""
    frac = Fraction(value).limit_denominator(1001)
    if frac.numerator <= 0 or frac.denominator <= 0:
        raise ValueError(f"invalid fps: {value}")
    return frac.numerator, frac.denominator


class H264Chunk(BaseModel):
    source_idx: int
    source: str
    data: bytes
    chunk_idx: int
    offset: int
    offset_end: int
    pts_ns: Optional[int] = None
    dts_ns: Optional[int] = None
    duration_ns: Optional[int] = None

class H264ChunkGenerator(BaseModel):
    sources: List[str]
    uuid: str = ""
    chunk_size: int = 1024 * 1024
    loop: bool = False
    fps: int = -1
    require_annex_b: bool = True

    _min_chunk_time: float = PrivateAttr(default=0.0)
    _resources: list = PrivateAttr(default_factory=list)
    _chunk_generators: list = PrivateAttr(default_factory=list)

    def model_post_init(self, context):
        self._min_chunk_time = 1.0 / self.fps if self.fps and self.fps > 0 else 0.0
        self.uuid = f"{self.__class__.__name__}:{uuid.uuid4()}"

        if len(self.sources) == 0:
            raise ValueError("empty sources.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        self._chunk_generators = [
            self.create_chunk_generator(i, src)
            for i, src in enumerate(self.sources)
        ]
        return super().model_post_init(context)

    def register_resource(self, resource):
        self._resources.append(resource)
        return resource

    @staticmethod
    def has_func(obj, name):
        return callable(getattr(obj, name, None))

    def release_resources(self):
        cleanup_methods = [
            "exit", "end", "teardown",
            "stop", "shutdown", "terminate",
            "join", "cleanup", "deactivate",
            "release", "close", "disconnect",
            "destroy",
        ]
        for res in self._resources:
            for method in cleanup_methods:
                if self.has_func(res, method):
                    try:
                        getattr(res, method)()
                    except Exception as e:
                        logger(f"Error during {method} on {res}: {e}")
        self._resources.clear()

    def create_chunk_generator(self, idx: int, source: str):
        raise NotImplementedError("Subclasses must implement `create_chunk_generator`")

    def __iter__(self):
        return self

    def __next__(self):
        start_time = time.time()
        try:
            chunks = [next(g) for g in self._chunk_generators]
            if not chunks or any(c is None for c in chunks):
                raise StopIteration

            if self._min_chunk_time > 0:
                elapsed = time.time() - start_time
                sleep_time = self._min_chunk_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            return chunks
        except StopIteration:
            raise StopIteration

    def reset_generators(self):
        self.release_resources()
        self._chunk_generators = [
            self.create_chunk_generator(i, src)
            for i, src in enumerate(self.sources)
        ]

    def release(self):
        self.release_resources()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

class H264FileChunkGenerator(H264ChunkGenerator):
    pass


def _validate_annex_b_file(path: str):
    with open(path, "rb") as f:
        probe = f.read(4096)
    if b"\x00\x00\x01" not in probe and b"\x00\x00\x00\x01" not in probe:
        raise ValueError(
            f"{path} does not look like Annex-B H.264. "
            "Expected 00 00 01 or 00 00 00 01 start codes."
        )


def _create_chunk_generator_impl(self, idx: int, source: str):
    path = os.path.abspath(source)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if self.require_annex_b:
        _validate_annex_b_file(path)

    f = self.register_resource(open(path, "rb"))

    def gen(f=f, path=path, source_idx=idx, chunk_size=self.chunk_size, loop=self.loop):
        chunk_idx = 0
        offset = 0
        while True:
            data = f.read(chunk_size)
            if data:
                offset_end = offset + len(data)
                yield H264Chunk(
                    source_idx=source_idx,
                    source=path,
                    data=data,
                    chunk_idx=chunk_idx,
                    offset=offset,
                    offset_end=offset_end,
                )
                chunk_idx += 1
                offset = offset_end
                continue

            if loop:
                f.seek(0)
                chunk_idx = 0
                offset = 0
                continue

            break

    return gen()


# Attach the method for both pydantic and fallback class paths.
H264FileChunkGenerator.create_chunk_generator = _create_chunk_generator_impl


def remux_h264_to_mp4_appsrc_stamped(
    h264_path: str,
    mp4_path: str,
    *,
    fps: str = "30",
    chunk_size: int = 1024 * 1024,
    require_annex_b: bool = True,
):
    Gst.init(None)

    fps_num, fps_den = _parse_fps(fps)
    frame_duration = Gst.util_uint64_scale_int(Gst.SECOND, fps_den, fps_num)

    h264_path = os.path.abspath(h264_path)
    mp4_path = os.path.abspath(mp4_path)

    os.makedirs(os.path.dirname(mp4_path) or ".", exist_ok=True)
    if os.path.exists(mp4_path):
        os.remove(mp4_path)

    pipeline_desc = f"""
appsrc name=h264src is-live=false block=true format=time do-timestamp=false
    caps=video/x-h264,stream-format=(string)byte-stream
!
queue max-size-buffers=0 max-size-time=0 max-size-bytes=0
!
h264parse name=parse config-interval=-1
!
video/x-h264,stream-format=(string)avc,alignment=(string)au
!
mp4mux faststart=true
!
filesink location="{_quote_gst_path(mp4_path)}" sync=false
"""

    print("pipeline:")
    print(pipeline_desc)

    pipeline = Gst.parse_launch(pipeline_desc)
    appsrc = pipeline.get_by_name("h264src")
    parser = pipeline.get_by_name("parse")
    bus = pipeline.get_bus()

    if appsrc is None:
        raise RuntimeError("could not find appsrc named h264src")
    if parser is None:
        raise RuntimeError("could not find h264parse named parse")

    frame_state = {"i": 0}

    def stamp_parsed_au_buffers(_pad, info):
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK

        i = frame_state["i"]
        ts = i * frame_duration

        # h264parse emits AU-aligned buffers here. mp4mux requires PTS.
        # For a fast sanity test, set PTS == DTS. If your stream has B-frames,
        # add h264timestamper or provide true DTS/PTS later.
        buf.pts = ts
        buf.dts = ts
        buf.duration = frame_duration
        buf.offset = i
        buf.offset_end = i + 1

        frame_state["i"] = i + 1
        return Gst.PadProbeReturn.OK

    parser.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER,
        stamp_parsed_au_buffers,
    )

    gen = H264FileChunkGenerator(
        sources=[h264_path],
        chunk_size=chunk_size,
        loop=False,
        require_annex_b=require_annex_b,
    )

    feeder_error = []

    def feeder():
        try:
            for chunks in gen:
                chunk = chunks[0]
                buf = Gst.Buffer.new_allocate(None, len(chunk.data), None)
                buf.fill(0, chunk.data)

                # Byte offsets are useful for debugging, but not timestamps.
                buf.offset = chunk.offset
                buf.offset_end = chunk.offset_end

                ret = appsrc.emit("push-buffer", buf)
                if ret != Gst.FlowReturn.OK:
                    raise RuntimeError(f"appsrc push-buffer failed: {ret}")

            ret = appsrc.emit("end-of-stream")
            if ret != Gst.FlowReturn.OK:
                raise RuntimeError(f"appsrc end-of-stream failed: {ret}")

        except Exception as e:
            feeder_error.append(e)
            try:
                appsrc.emit("end-of-stream")
            except Exception:
                pass
        finally:
            gen.release()

    pipeline.set_state(Gst.State.PLAYING)

    t = threading.Thread(target=feeder, daemon=True)
    t.start()

    ok = False
    while True:
        msg = bus.timed_pop_filtered(
            30 * Gst.SECOND,
            Gst.MessageType.ERROR | Gst.MessageType.EOS,
        )

        if msg is None:
            print("timeout waiting for EOS/error")
            break

        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print("GStreamer ERROR:", err)
            print("debug:", debug)
            break

        if msg.type == Gst.MessageType.EOS:
            ok = True
            print("EOS")
            break

    pipeline.set_state(Gst.State.NULL)
    t.join(timeout=2.0)

    if feeder_error:
        raise feeder_error[0]
    if not ok:
        raise RuntimeError("pipeline did not finish cleanly")
    if not os.path.exists(mp4_path) or os.path.getsize(mp4_path) == 0:
        raise RuntimeError(f"output MP4 was not created or is empty: {mp4_path}")

    print("parsed/stamped frames:", frame_state["i"])
    print("wrote:", mp4_path)
    print("size:", os.path.getsize(mp4_path))


def test1(h264_path:str):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("input", nargs="?", default="rgb_full_test.h264")
    # parser.add_argument("output", nargs="?", default=None)
    # parser.add_argument("--fps", default="30", help="e.g. 30, 30000/1001, 29.97")
    # parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    # parser.add_argument("--no-require-annex-b", action="store_true")
    # args = parser.parse_args()

    # output = args.output or os.path.splitext(args.input)[0] + ".mp4"

    remux_h264_to_mp4_appsrc_stamped(
        h264_path=h264_path,#args.input,
        mp4_path=h264_path.replace(".h264",".mp4"),#args.output,
        fps=30,#args.fps,
        chunk_size=1024 * 1024,#args.chunk_size,
        require_annex_b=True,#not args.no_require_annex_b,
    )


if __name__ == "__main__":
    # test1()
    pass
