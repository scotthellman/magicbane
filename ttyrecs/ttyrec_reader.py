import bz2

class TTYReader:

    def __init__(self, fn):
        frames = []
        with bz2.open(fn) as f:
            next_frame = self.read_frame(f)
            while next_frame:
                frames.append(next_frame)
                next_frame = self.read_frame(f)
        self.frames = frames

    @staticmethod
    def read_frame(f):
        chunk_start = f.read(4)
        if chunk_start == "":
            return None
        sec = int.from_bytes(chunk_start, byteorder="little")
        usec = int.from_bytes(f.read(4), byteorder="little")
        length = int.from_bytes(f.read(4), byteorder="little")
        frame = f.read(length)
        return frame





if __name__ == "__main__":
    foo = TTYReader("2010-01-22.07 49 29.ttyrec.bz2")
    print(len(foo.frames))
    print(foo.frames[0])
    print(foo.frames[10])
    print(foo.frames[-1])
