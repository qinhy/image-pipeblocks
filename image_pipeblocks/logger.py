from datetime import datetime

def logger(msg, level="info"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} [{level.upper()}] {msg}")

try:

    import asyncio
    import os
    import re
    import socket
    import time
    import traceback
    from typing import Any, Optional

    import redis.asyncio as redis


    class AsyncRedisLogger:
        """
        Usage:
            logger = AsyncRedisLogger()
            await logger.start()

            logger("[Amodule:member1] say something")
            logger("[Amodule:member1] failed", level="error")

            await logger.close()
        """

        def __init__(
            self,
            redis_url: str = "redis://localhost:6379/0",
            key_prefix: str = "log:",
            max_queue_size: int = 10_000,
            max_stream_len: int = 100_000,
            print_also: bool = True,
        ):
            self.redis_url = redis_url
            self.key_prefix = key_prefix
            self.max_stream_len = max_stream_len
            self.print_also = print_also

            self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=max_queue_size)

            self.redis: Optional[redis.Redis] = redis.from_url(self.redis_url, decode_responses=True)
            if not self.redis.ping():
                raise RuntimeError("Failed to connect to Redis")

            self.worker_task: Optional[asyncio.Task] = None
            self.closed = False

            self.hostname = socket.gethostname()
            self.pid = os.getpid()

        async def start(self):
            """
            Start Redis connection and background worker.
            """
            self.redis = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis.ping()

            self.closed = False
            self.worker_task = asyncio.create_task(self._worker())

        def __call__(self, raw_msg: str, level: str = "info", **extra):
            """
            Print-like API.

            Example:
                logger("[Amodule:member1] say something")
                logger("[Auth:login] failed password", level="warning", user_id=123)
            """
            if self.closed:
                return

            parsed_key, message = self.parse(raw_msg)
            redis_key = self.make_redis_key(parsed_key)

            record = {
                "redis_key": redis_key,
                "key": parsed_key,
                "level": level.upper(),
                "message": message,
                "raw": raw_msg,
                "ts": str(time.time()),
                "hostname": self.hostname,
                "pid": str(self.pid),
            }

            for k, v in extra.items():
                record[f"extra:{k}"] = str(v)

            if self.print_also:
                print(f"[{record['level']}] [{parsed_key}] {message}")

            try:
                self.queue.put_nowait(record)
            except asyncio.QueueFull:
                print("[LOGGER WARNING] queue full; dropped log:", raw_msg)

        def exception(self, raw_msg: str, **extra):
            """
            Use inside except block.

            Example:
                try:
                    1 / 0
                except Exception:
                    logger.exception("[Calc:divide] failed")
            """
            extra["traceback"] = traceback.format_exc()
            self(raw_msg, level="error", **extra)

        async def _worker(self):
            """
            Background task that writes queued logs to Redis.
            """
            while not self.closed or not self.queue.empty():
                try:
                    record = await asyncio.wait_for(self.queue.get(), timeout=0.5)

                    redis_key = record.pop("redis_key")

                    if self.redis is None:
                        raise RuntimeError("Redis logger is not started")

                    await self.redis.xadd(
                        redis_key,
                        record,
                        maxlen=self.max_stream_len,
                        approximate=True,
                    )

                    self.queue.task_done()

                except asyncio.TimeoutError:
                    continue

                except asyncio.CancelledError:
                    break

                except Exception as exc:
                    print("[LOGGER ERROR] failed to write to Redis:", repr(exc))

                    try:
                        self.queue.task_done()
                    except ValueError:
                        pass

                    await asyncio.sleep(0.5)

        async def close(self):
            """
            Flush logs and close Redis connection.
            redis-py async clients should be closed when finished. 
            """
            self.closed = True

            await self.queue.join()

            if self.worker_task:
                self.worker_task.cancel()
                try:
                    await self.worker_task
                except asyncio.CancelledError:
                    pass

            if self.redis:
                await self.redis.aclose()

        def parse(self, raw_msg: str) -> tuple[str, str]:
            """
            Parse:
                "[Amodule:member1] say something"

            Into:
                key = "Amodule:member1"
                message = "say something"
            """
            raw_msg = str(raw_msg).strip()

            match = re.match(r"^\[([^\]]+)\]\s*(.*)$", raw_msg)

            if not match:
                return "unknown", raw_msg

            key = match.group(1).strip()
            message = match.group(2).strip()

            if not key:
                key = "unknown"

            return key, message

        def make_redis_key(self, key: str) -> str:
            """
            Convert:
                Amodule:member1

            Into:
                log:Amodule:member1
            """
            return f"{self.key_prefix}{key}"


    logger = AsyncRedisLogger()
except:
    pass



