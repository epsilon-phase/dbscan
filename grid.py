from collections import defaultdict
from math import ceil, floor


class Grid:
    partition_count: int
    points: defaultdict[tuple[int, int], list[tuple[float, float]]]
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    step_x: float
    step_y: float

    def _to_grid(self, point):
        return (floor((point[0]-self.min_x)/self.step_x),
                floor((point[1]-self.min_y)/self.step_y))

    @staticmethod
    def from_points(points, partitions):
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        for i in points:
            min_x, max_x = min(i[0], min_x), max(i[0], max_x)
            min_y, max_y = min(i[1], min_y), max(i[1], max_y)
        grid = Grid(min_x, max_x, min_y, max_y, partitions)
        for i in points:
            grid.add_point(i)
        return grid

    def __init__(self, min_x, max_x, min_y, max_y, partitions):
        self.partition_count = partitions
        self.min_x, self.min_y, self.max_x, self.max_y = min_x, min_y, max_x, max_y
        self.step_x = (max_x-min_x)/partitions
        self.step_y = (max_y-min_y)/partitions
        self.points = defaultdict(list)

    def add_point(self, point):
        self.points[self._to_grid(point)].append(point)

    def neighborhood_cells(self, grid, epsilon):
        y_radius = ceil(epsilon/self.step_y)
        x_radius = ceil(epsilon/self.step_x)
        for x in range(-x_radius, x_radius+1):
            for y in range(-y_radius, y_radius+1):
                yield (grid[0]+x, grid[1]+y)

    def range_query(self, point, distance_function, epsilon):
        r = []
        grid = self._to_grid(point)
        for i in self.neighborhood_cells(grid, epsilon):
            r.extend(filter(lambda x: distance_function(
                x, point) <= epsilon, self.points[i]))
        return r
