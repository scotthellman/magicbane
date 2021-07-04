import gym
import nle
import time
from collections import Counter
from enum import Enum
import numpy as np

STEPS = 100
EMPTY = 2359
FLOOR = 2379
# I'm guessing about 63-65
WALLS = set([2360, 2361, 2362, 2363, 2364, 2365])

MAP_SIZE = (21, 79)

"""
1 CompassDirection.N
2 CompassDirection.E
3 CompassDirection.S
4 CompassDirection.W
5 CompassDirection.NE
6 CompassDirection.SE
7 CompassDirection.SW
8 CompassDirection.NW
"""

# FIXME obvious this all belongs in a class
# (or at least not in a toplevel py file)
# tbh this isn't even vision really, just a "is mr. empty here a wall or unseen" flag
VISION_RADIUS = 1  # So this is dependent on lighting and should be grabbed from the state somehow
SEEN = np.zeros(MAP_SIZE)


class Direction(Enum):
    N = (-1, 0)
    S = (1, 0)
    W = (0, -1)
    E = (0, 1)

    NW = (-1, -1)
    NE = (-1, 1)
    SW = (1, -1)
    SE = (1, 1)

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


def get_neighbors(point, cardinal=False):
    for d in Direction:
        if cardinal and not d.is_cardinal():
            continue
        next_point = d.apply(point)
        if next_point[0] < 0 or next_point[0] >= MAP_SIZE[0]:
            continue
        if next_point[1] < 0 or next_point[1] >= MAP_SIZE[1]:
            continue
        yield next_point


def get_player_loc(obs):
    # blstats shows it in x,y so reverse
    return obs["blstats"][:2][::-1]


def update_vision(p):
    y_bound = (p[0]-VISION_RADIUS, p[0]+VISION_RADIUS+1)
    x_bound = (p[1]-VISION_RADIUS, p[1]+VISION_RADIUS+1)
    if y_bound[0] < 0:
        y_bound[0] = 0
    if y_bound[1] >= MAP_SIZE[0]:
        y_bound[1] = MAP_SIZE[0]
    if x_bound[0] < 0:
        x_bound[0] = 0
    if x_bound[1] >= MAP_SIZE[1]:
        x_bound[1] = MAP_SIZE[1]
    SEEN[y_bound[0]:y_bound[1], x_bound[0]:x_bound[1]] = 1


def direction_from_to(point, target):
    delta_y = target[0] - point[0]
    delta_x = target[1] - point[1]

    y_dir = None
    if delta_y < 0:
        y_dir = Direction.N
    elif delta_y > 0:
        y_dir = Direction.S

    x_dir = None
    if delta_x < 0:
        x_dir = Direction.W
    elif delta_x > 0:
        x_dir = Direction.E

    possible_dirs = []
    if x_dir is not None and y_dir is not None:
        possible_dirs.append(x_dir.combine(y_dir))

    possible_dirs.extend([y_dir, x_dir])
    return y_dir, x_dir


def find_nearest(point, glyphs, target_predicate, blocking_predicate, cardinal=False):
    point = tuple(point)
    seen = set([point])
    queue = [point]
    while len(queue) > 0:
        current = queue.pop(0)
        for neighbor in get_neighbors(current, cardinal=cardinal):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            if blocking_predicate(neighbor, glyphs):
                continue
            queue.append(neighbor)
            if target_predicate(neighbor, glyphs):
                return neighbor
    return None


def not_traversable(point, glyphs):
    glyph = glyphs[point]
    if glyph in WALLS:
        return True
    if glyph == EMPTY:
        return True
    return False


def unexplored_adjacent(point, glyphs):
    return any(glyphs[n] == EMPTY and not SEEN[n] for n in get_neighbors(point))


def get_direction_to_nearest_unexplored(point, glyphs):
    target = find_nearest(point, glyphs, unexplored_adjacent, not_traversable, cardinal=True)
    if target is None:
        return None
    directions = [d for d in direction_from_to(point, target) if d is not None]
    directions = [d for d in directions if not not_traversable(d.apply(point), glyphs)]
    if len(directions) == 0:
        return None
    print(f"with @ at {point}, we are going {directions[0]}({directions[0].action()}) towards {target}")
    return directions[0].action()


env = gym.make("NetHackScore-v0")
obs = env.reset()  # each reset generates a new dungeon
for _ in range(STEPS):
    env.render()
    player_loc = get_player_loc(obs)
    update_vision(player_loc)
    action = get_direction_to_nearest_unexplored(player_loc, obs["glyphs"])
    if action is None:
        print('going random')
        action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
env.close()
