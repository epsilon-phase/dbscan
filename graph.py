import queue
from typing import *
import random
import PIL
import PIL.Image
import PIL.ImageColor
import PIL.ImageDraw
import math
import time
from collections import defaultdict, deque
import functools
import kdtree as kdpy
import grid as pygrid
import balltree as pyball

NO_SCIPY = False
try:
    import scipy.spatial
except ImportError:
    NO_SCIPY = True

ACCELERATION = 'python-ball'
INPUT = None
PRINT_PROGRESS = True
RANDOMIZE_POINT_ORDER = False
OUTPUT = "output.png"
RADIUS = 20.

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    accelerators = ['python-ball', 'python-kdtree', 'python-grid']
    if not NO_SCIPY:
        accelerators.append('scipy-kdtree')
    parser.add_argument("--acceleration", dest="acceleration",
                        choices=accelerators, default="python-ball")
    parser.add_argument('--balltree-leaf-size', dest='balltree_width',
                        type=int, default=10)
    parser.add_argument('--balltree-search-debug',
                        dest="balltree_search_debug", action='store_const',
                        const=True, default=False)
    parser.add_argument('--print-progress', dest='print_progress',
                        action='store_const',
                        const=True, default=False)
    parser.add_argument("-o", dest="output")
    parser.add_argument("--radius", dest='radius', type='float', default=20.0)
    args = parser.parse_args()
    ACCELERATION = args.acceleration
    pyball.BALLNODE_MAX_POINTS = args.balltree_width
    pyball.DEBUG_SEARCH = args.balltree_search_debug
    PRINT_PROGRESS = args.print_progress
    INPUT = args.filename
    OUTPUT = args.output
    RADIUS = args.radius

Point = tuple[float, ...]


def distance(a: Point, b: Point):
    s = 0.0
    for i in range(len(a)):
        s += (a[i]-b[i])**2
    return math.sqrt(s)


def in_bounding_box(a: Point, epsilon: float, c: Point):
    return all(map(lambda a, b: a-epsilon > b and b < a+epsilon, a, c))


if ACCELERATION == 'python-kdtree':
    def range_query(idx: kdpy.KDTree,
                    distance_func: Callable[[Point, Point], float],
                    point: Point, epsilon: float) -> Iterable[Point]:
        # r = []
        # for i in points:
        #     if distance_func(i, point) <= epsilon:
        #         r.append(i)
        return idx.points_within_distance(point, epsilon)
elif ACCELERATION == 'scipy-kdtree':
    def range_query(indx: scipy.spatial.KDTree, points,
                    distance_func: Callable[[Point, Point], float],
                    point: Point, epsilon: float) -> Iterable[Point]:
        r = []
        z = indx.query_ball_point(list(point), epsilon)

        return list(map(lambda x: points[x], z))
elif ACCELERATION == 'python-grid':
    def range_query(indx: pygrid.Grid,
                    distance_func: Callable[[Point, Point], float],
                    point: Point, epsilon: float):
        return indx.range_query(point, distance_func, epsilon)
elif ACCELERATION == 'python-ball':
    def range_query(indx: pyball.BallNode,
                    _,
                    point: Point,
                    epsilon: float):
        return indx.range_query(point, epsilon)


def dbscan(points, distance_function, minpoints, epsilon):
    c = 0
    idx = None
    kd_start = time.time()
    if ACCELERATION == 'scipy-kdtree':
        idx = scipy.spatial.KDTree(points)
    elif ACCELERATION == 'python-kdtree':
        idx = kdpy.KDTree.from_points(points, 2, 0)
        # for i in points:
        # idx.insert(i)
    elif ACCELERATION == 'python-ball':
        idx = pyball.BallTree(points, distance_function)
    else:
        idx = pygrid.Grid.from_points(points, 100)
    kd_end = time.time()
    print(f"Finished constructing {ACCELERATION}({kd_end-kd_start}s)")
    labels = {}
    core_points = 0
    noise_points = 0
    border_points = 0
    # unlabelled = set(points)
    start = time.time()
    border_collection = {}
    centrality = {}
    for p in points:
        if p in labels:
            continue
        neighbors = []
        if ACCELERATION != 'scipy-kdtree':
            neighbors = range_query(idx, distance_function, p, epsilon)
        else:
            neighbors = range_query(idx, points, distance_function, p, epsilon)

        centrality[p] = len(neighbors)
        if len(neighbors) < minpoints:
            labels[p] = -1
            noise_points += 1
            continue
        c += 1
        labels[p] = c
        seed_set = deque(neighbors)
        while 0 < len(seed_set):
            s = seed_set.pop()
            if s is None:
                break
            if s in labels:
                if labels[s] == -1:
                    labels[s] = c  # Become border point!
                    border_collection[s] = c
                    border_points += 1
                    noise_points -= 1
                continue

            labels[s] = c
            # unlabelled.remove(s)
            n2 = []
            if ACCELERATION != 'scipy-kdtree':
                n2 = range_query(idx, distance_function, s, epsilon)
            else:
                n2 = range_query(idx, points, distance_function, s, epsilon)
            centrality[s] = len(n2)
            if len(n2) >= minpoints:
                seed_set.extendleft(
                    filter(lambda x: x not in labels or labels[x] == -1, n2))
                # print("Adding core point to queue")
                core_points += 1
        cluster_end = time.time()
        if PRINT_PROGRESS:
            print(
                f"Cluster {c}, {core_points} core points, {border_points} border points, {noise_points} noise, {cluster_end-start:0.4f} seconds elapsed")
            print(
                f"{len(points) - len(labels)} points remaining({100*(len(labels)/len(points)):.3f}%)")
        start = time.time()
    bd = defaultdict(list)
    for point, cluster in border_collection.items():
        bd[cluster].append(point)
    return labels, bd, centrality


def draw_border_lines(image, border: list[Point], color: tuple[int, int, int]):
    # Find the centroid of the border
    centroid = [0.0, 0.0]
    for i in border:
        centroid[0] += i[0]
        centroid[1] += i[1]
    centroid[0] /= len(border)
    centroid[1] /= len(border)

    def compare_points(p1, p2):
        """
        Compare two points by how clockwise they are from the center. Necessary to draw a reasonable polygon
        """
        if p1[0]-centroid[0] >= 0 and p2[0] - centroid[0] < 0:
            return -1
        if p1[0]-centroid[0] < 0 and p2[0] - centroid[0] >= 0:
            return 1
        if p1[0]-centroid[0] == 0 and p2[0] - centroid[0] == 0:
            if p1[1]-centroid[1] >= 0 or p2[1]-centroid[1] >= 0:
                return 1 if p1[1] > p2[1] else -1
            return -1 if p2[1] > p1[1] else 1
        det = (p1[0]-centroid[0])*(p2[1]-centroid[1]) - \
            (p2[0]-centroid[0])*(p1[1]-centroid[1])
        if det < 0:
            return -1
        elif det > 0:
            return 1

        d1 = (p1[0] - centroid[0]) ** 2 + (p1[1]-centroid[1])**2
        d2 = (p2[0] - centroid[0]) ** 2 + (p2[1]-centroid[1])**2
        return 1 if d1 > d2 else 0 if d1 == d2 else -1
    fill_color = color + (100,)
    border = border.copy()
    border.sort(key=functools.cmp_to_key(compare_points))
    draw = PIL.ImageDraw.Draw(image, 'RGB')
    draw.polygon(border, outline=color, fill=fill_color)


def from_image(image: PIL.Image):
    points = []
    for x in range(image.width):
        for y in range(image.height):
            pix = image.getpixel((x, y))
            if pix != (255, 255, 255):
                points.append((x, y, pix[0], pix[1], pix[2]))
    return points, image.width, image.height


def manhattan(a, b):
    s = 0
    for (an, bn) in zip(a, b):
        s += abs(an-bn)
    return s


width, height = 1000, 1000
if __name__ == "__main__":
    with open(INPUT, 'rb') as fp:
        points, width, height = from_image(PIL.Image.open(fp))
        print(f"Points: {len(points)}")
        if RANDOMIZE_POINT_ORDER:
            points.sort(key=lambda _: random.random())

        labels, borders, centrality = dbscan(points, distance, 8, RADIUS)

        image = PIL.Image.new("RGB", (width+1, height+1), color=(255, 255, 255))

        colors = list(
            map(lambda x: PIL.ImageColor.getrgb(x),
                PIL.ImageColor.colormap.keys())
        )
        draw = PIL.ImageDraw.Draw(image, 'RGB')

        for (position, cluster) in labels.items():
            color = colors[cluster % len(colors)]
            if cluster == -1:
                color = (0, 0, 0)
            x = round(position[0])
            y = round(position[1])
            image.putpixel((x, y), color)
            off = 1
            if (x, y) in centrality.keys():
                off = centrality[(x, y)]
            # draw.ellipse([(x-off,y-off),(x+off,y+off)], fill=color)

        for (cluster, points) in borders.items():
            if len(points) < 2:
                continue
            coords = list(map(lambda x: x[:2], points))
            draw_border_lines(image, coords,
                              colors[int(cluster) % len(colors)])
        image.save(OUTPUT)
