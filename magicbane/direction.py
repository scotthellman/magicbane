from enum import Enum


class Direction(Enum):
    N = (-1, 0)
    S = (1, 0)
    W = (0, -1)
    E = (0, 1)

    NW = (-1, -1)
    NE = (-1, 1)
    SW = (1, -1)
    SE = (1, 1)

    @staticmethod
    def cardinal_directions():
        for d in Direction:
            if d.is_cardinal():
                yield d

    def __getitem__(self, i):
        return self.value[i]

    def apply(self, point):
        return (point[0] + self[0], point[1] + self[1])

    def is_cardinal(self):
        cardinal = self is Direction.N
        cardinal |= self is Direction.S
        cardinal |= self is Direction.W
        cardinal |= self is Direction.E
        return cardinal

    def is_orthogonal(self, other):
        combined_y = self.value[0] + other.value[0]
        combined_x = self.value[1] + other.value[1]

        return abs(combined_y) + abs(combined_x) == 2

    def combine(self, other):
        if not self.is_orthogonal(other):
            return None
        combined = (self.value[0] + other.value[0], self.value[1] + other.value[1])
        reduced = tuple(c/abs(c) if c != 0 else 0 for c in combined)
        for d in Direction:
            if d.value == reduced:
                return d
        assert False

    def action(self):
        if self is Direction.N:
            return 1
        elif self is Direction.S:
            return 3
        elif self is Direction.W:
            return 4
        elif self is Direction.E:
            return 2
        elif self is Direction.NW:
            return 8
        elif self is Direction.NE:
            return 5
        elif self is Direction.SW:
            return 7
        elif self is Direction.SE:
            return 6
