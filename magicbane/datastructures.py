class CircularBuffer:

    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.pointer = 0
        self.buffer = [None for i in range(maxlen)]
        self.current_len = 0

    def insert(self, item):
        self.buffer[self.pointer] = item
        self.pointer += 1
        if self.current_len < self.maxlen:
            self.current_len += 1
        if self.pointer >= self.maxlen:
            self.pointer = 0

    def __getitem__(self, index):
        return self.buffer[index]

    def __len__(self):
        return self.current_len

if __name__ == "__main__":
    import time
    from collections import deque
    import random

    buffer_size = 100000

    mybuf = CircularBuffer(buffer_size)
    dqbuf = deque([], buffer_size)

    start = time.time()
    for i in range(200000):
        mybuf.insert(random.randint(0, 300))
    print("mybuf insertion", time.time() - start)

    start = time.time()
    for i in range(200000):
        dqbuf.append(random.randint(0, 300))
    print("dqbuf insertion", time.time() - start)

    start = time.time()
    for i in range(200000):
        mybuf[random.randint(0, buffer_size-1)]
    print("mybuf read", time.time() - start)

    start = time.time()
    for i in range(200000):
        dqbuf[random.randint(0, buffer_size-1)]
    print("dqbuf read", time.time() - start)
