class BlockAllocator:
    def __init__(self, size: int) -> None:
        self.free = set(range(size))
        self.refs = [0] * size
        self.intervals = [(0, size)] if size else []

    def _claim(self, blocks):
        for block in blocks:
            assert self.refs[block] == 0
            self.refs[block] = 1
            self.free.remove(block)
        return tuple(blocks)

    def allocate(self, count: int) -> tuple[int, ...]:
        if count < 0 or count > len(self.free):
            raise RuntimeError(f"Need {count} blocks, only {len(self.free)} free")
        blocks = []
        while count:
            begin, end = self.intervals[0]
            take = min(count, end - begin)
            blocks.extend(range(begin, begin + take))
            if begin + take == end:
                self.intervals.pop(0)
            else:
                self.intervals[0] = (begin + take, end)
            count -= take
        return self._claim(blocks)

    def can_allocate_groups(self, count: int, width: int) -> bool:
        if count == 0:
            return True
        return sum((end - begin) // width for begin, end in self.intervals) >= count

    def allocate_groups(self, count: int, width: int) -> tuple[int, ...]:
        if not self.can_allocate_groups(count, width):
            raise RuntimeError(f"Need {count} contiguous groups of {width} blocks")
        blocks = []
        for i in range(len(self.intervals) - 1, -1, -1):
            begin, end = self.intervals[i]
            take = min(count, (end - begin) // width)
            if not take:
                continue
            blocks.extend(range(end - take * width, end))
            end -= take * width
            if begin == end:
                self.intervals.pop(i)
            else:
                self.intervals[i] = (begin, end)
            count -= take
            if not count:
                break
        return self._claim(blocks)

    def retain(self, blocks) -> None:
        for block in blocks:
            assert self.refs[block] > 0
            self.refs[block] += 1

    def release(self, blocks) -> None:
        released = []
        for block in blocks:
            assert self.refs[block] > 0
            self.refs[block] -= 1
            if self.refs[block] == 0:
                self.free.add(block)
                released.append((block, block + 1))
        if not released:
            return
        intervals = []
        for begin, end in sorted(self.intervals + released):
            if intervals and intervals[-1][1] == begin:
                intervals[-1] = (intervals[-1][0], end)
            else:
                intervals.append((begin, end))
        self.intervals = intervals
