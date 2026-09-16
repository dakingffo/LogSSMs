from __future__ import annotations

from typing import Callable, Generic, TypeVar

import msgpack
import zmq
import zmq.asyncio


T = TypeVar("T")


class ZmqPushQueue(Generic[T]):
    def __init__(
        self,
        addr   : str,
        create : bool,
        encoder: Callable[[T], dict],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put_raw(self, raw: bytes):
        self.socket.send(raw, copy=False)

    def put(self, obj: T):
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqAsyncPushQueue(ZmqPushQueue[T]):
    def __init__(
        self,
        addr   : str,
        create : bool,
        encoder: Callable[[T], dict],
    ):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    async def put_raw(self, raw: bytes):
        await self.socket.send(raw, copy=False)

    async def put(self, obj: T):
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        await self.socket.send(event, copy=False)


class ZmqPullQueue(Generic[T]):
    def __init__(
        self,
        addr   : str,
        create : bool,
        decoder: Callable[[dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    def get_raw(self) -> bytes:
        return self.socket.recv()

    def decode(self, raw: bytes) -> T:
        return self.decoder(msgpack.unpackb(raw, raw=False))

    def get(self) -> T:
        return self.decode(self.get_raw())

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def wait(self, timeout_ms: int) -> bool:
        return self.socket.poll(timeout=timeout_ms, flags=zmq.POLLIN) != 0

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqAsyncPullQueue(ZmqPullQueue[T]):
    def __init__(
        self,
        addr   : str,
        create : bool,
        decoder: Callable[[dict], T],
    ):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    async def get_raw(self) -> bytes:
        return await self.socket.recv()

    async def get(self) -> T:
        event = await self.get_raw()
        return self.decoder(msgpack.unpackb(event, raw=False))

    async def wait(self, timeout_ms: int) -> bool:
        return await self.socket.poll(timeout=timeout_ms, flags=zmq.POLLIN) != 0


class ZmqPubQueue(Generic[T]):
    def __init__(
        self,
        addr   : str,
        create : bool,
        encoder: Callable[[T], dict],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put_raw(self, raw: bytes):
        self.socket.send(raw, copy=False)

    def put(self, obj: T):
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqSubQueue(Generic[T]):
    def __init__(
        self,
        addr   : str,
        create : bool,
        decoder: Callable[[dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.decoder = decoder

    def get_raw(self) -> bytes:
        return self.socket.recv()

    def get(self) -> T:
        event = self.get_raw()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()
