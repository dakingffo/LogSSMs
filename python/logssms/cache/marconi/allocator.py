from collections import deque


class BlockAllocator:
    def __init__(self, size: int) -> None:
        self.free = deque(range(size))
        self.refs = [0] * size

    def allocate(self, count: int) -> tuple[int, ...]:
        if count > len(self.free):
            raise RuntimeError(f"Need {count} blocks, only {len(self.free)} free")
        blocks = tuple(self.free.popleft() for _ in range(count))
        for block in blocks:
            assert self.refs[block] == 0
            self.refs[block] = 1
        return blocks

    def retain(self, blocks) -> None:
        for block in blocks:
            assert self.refs[block] > 0
            self.refs[block] += 1

    def release(self, blocks) -> None:
        for block in blocks:
            assert self.refs[block] > 0
            self.refs[block] -= 1
            if self.refs[block] == 0:
                self.free.append(block)
