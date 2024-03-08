from dataclasses import dataclass
from typing import Callable
from math import sqrt
from collections import deque
import platform
USE_NP_ARRAYS = False


def dominant_axis(points):
    dimensions = len(points[0])
    min_point = [float('inf')]*dimensions
    max_point = [float('-inf')]*dimensions
    for i in points:
        for c in range(dimensions):
            min_point[c] = min(min_point[c], i[c])
            max_point[c] = max(max_point[c], i[c])
    biggest_axis = 0
    biggest_diff = 0
    for axis, (i, j) in enumerate(zip(min_point, max_point)):
        diff = j - i
        if diff > biggest_diff:
            biggest_axis = axis
            biggest_diff = diff
    return biggest_axis


# The maximum number of points to place in a leaf
BALLNODE_MAX_POINTS = 10
DEBUG_SEARCH = False


def dprint(*args):
    if DEBUG_SEARCH:
        print(*args)


@dataclass
class BallNode:
    left: "BallNode"
    right: "BallNode"
    pivot: tuple[float, ...] | list[tuple[float, ...]]
    points: None | list[tuple[float]]
    radius: float

    axis: int

    def __init__(self, points, distance_function):
        if len(points) == 1:
            self.pivot = points[0]
            self.left, self.right = None, None
            self.radius = 0.0
            self.axis = 0
            self.points = None
            return
        c = dominant_axis(points)
        sp = sorted(points, key=lambda x: x[c])
        median = len(sp)//2
        self.pivot = sp[median]
        if len(points) >= BALLNODE_MAX_POINTS:
            left, right = sp[:median], sp[median+1:]
            self.left = BallNode(left, distance_function) if len(
                left) > 0 else None
            self.right = BallNode(right, distance_function) if len(
                right) > 0 else None
            self.points = None
        else:
            self.left, self.right = None, None
            self.points = list(filter(lambda x: x != self.pivot, points))
        self.radius = max(
            map(lambda x: distance_function(x, self.pivot), points))
        self.axis = c

    def __iter__(self):
        if self.left:
            for i in self.left:
                yield i
        yield self.pivot
        if self.right:
            for i in self.right:
                yield i
        if self.points is not None:
            for i in self.points:
                yield i

    def find_nearest(self, point):
        if not self.left and not self.right:
            return self.pivot
        if self.pivot[self.axis] > point[self.axis]:
            if not self.left:
                return self.pivot


    def draw(self, image, depth=0):
        pvx, pvy = int(self.pivot[0]), int(self.pivot[1])
        draw = PIL.ImageDraw.Draw(image)
        MAX_WIDTH = 5
        if self.left:
            lx, ly = int(self.left.pivot[0]), int(self.left.pivot[1])
            self.left.draw(image, depth=depth+1)
            draw.line(((pvx, pvy), (lx, ly)), fill='red',
                      width=max(1, MAX_WIDTH-depth))
        if self.right:
            rx, ry = int(self.right.pivot[0]), int(self.right.pivot[1])
            self.right.draw(image, depth=depth+1)
            draw.line(((pvx, pvy), (rx, ry)), fill='green',
                      width=max(1, MAX_WIDTH-depth))
        for i in self.points:
            draw.line([(pvx, pvy), i], fill='purple')
        draw.text((pvx, pvy), str(depth), fill="yellow")


class BallTree():
    def __init__(self, points, distance_function):
        self.distance_function = distance_function
        self.root = BallNode(points, distance_function)


    def __iter__(self):
        for i in self.root:
            yield i

    def range_query(self, p: tuple[float, ...],
                     epsilon: float):
        nodes_to_explore = deque()
        nodes_to_explore.append(self.root)
        result = []
        distance_function = self.distance_function
        while len(nodes_to_explore) > 0:
            current = nodes_to_explore.popleft()
            distance = distance_function(p, current.pivot)
            if distance <= epsilon:
                result.append(current.pivot)
            if distance + current.radius <= epsilon:
                if current.points is not None:
                    result.extend(current.points)
                else:
                    if current.left:
                        result.extend(current.left)
                    if current.right:
                        result.extend(current.right)
            elif distance - current.radius <= epsilon:
                if current.points is not None:
                    result.extend(
                        filter(
                            lambda x: distance_function(p, x) <= epsilon,
                            current.points))
                else:
                    if current.left:
                        nodes_to_explore.append(current.left)
                    if current.right:
                        nodes_to_explore.append(current.right)
            else:
                # Excluded
                pass
        return result


    # TODO The range query here might be better turned into a looping
    # structure, the recursion cannot be optimized out and thus will contribute
    # to substantial data shuffling

if __name__ == '__main__':
    import random
    import PIL
    import PIL.ImageDraw
    comparisons = 0

    def dist(a, b):
        global comparisons
        comparisons += 1
        return sqrt(sum(map(lambda x, y: (x-y)**2, a, b)))
    points = [(random.uniform(0, 1000.0), random.uniform(0, 1000.0))
              for _ in range(1000)]
    bn = BallNode(points, dist)
    comparisons = 0
    c = 0
    for i in bn:
        c += 1
    assert (c == 1000)
    image = PIL.Image.new('RGB', (1000, 1000))
    bn.draw(image)
    nearest, distance = nearest_neighbor(bn, (500, 500))
    print(comparisons)
    draw = PIL.ImageDraw.Draw(image)
    draw.chord([(495, 495), (505, 505)], 0, 360, fill='purple')
    draw.chord([(nearest[0]-5, nearest[1]-5),
               (nearest[0]+5, nearest[1]+5)], 0, 360, fill="pink")
    search_radius = 300
    comparisons = 0
    rq = bn.range_query((400, 400), search_radius)
    comparisons_bn, comparisons = comparisons, 0
    linear = list(filter(lambda x: dist(x, (400, 400)) < search_radius, bn))
    comparisons_linear = comparisons
    print(
        f"Found {len(rq)} points with the balltree versus {len(linear)} with a complete search")
    print(f"Took {comparisons_bn} comparisons to find with the ball tree versus {comparisons_linear} for the linear scan")
    for z in linear:
        col = 'pink' if z in rq else 'gray'
        draw.chord([(z[0]-5, z[1]-5),
                    (z[0]+5, z[1]+5)], 0, 360, fill=col)
    image.show()
