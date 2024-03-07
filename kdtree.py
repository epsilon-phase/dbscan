from dataclasses import dataclass

# TODO Rewrite this to use a supplied distance function instead of bringing its own

@dataclass
class Node:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right

    def __str__(self):
        return f"Node(point={self.point}, left={self.left}, right={self.right})"

    @staticmethod
    def from_points(points: list, dimensions, depth=0):
        axis = depth % dimensions
        if len(points) == 1:
            return Node(points[0], None, None)
        elif len(points) == 0:
            return None
        median_index = len(points) // 2
        p2 = sorted(points, key=lambda x: x[axis])
        return Node(p2[median_index], Node.from_points(p2[:median_index], dimensions, depth+1),
                    Node.from_points(p2[median_index:], dimensions, depth+1))


class KDTree:
    def __init__(self):
        self.root = None

    @staticmethod
    def from_points(points, dimensions, depth=0):
        kd = KDTree()
        kd.root = Node.from_points(points, len(points[0]))
        return kd

    def _insert(self, node, point, depth):
        if node is None:
            return Node(point)

        axis = depth % len(point)  # Alternate between dimensions
        if point[axis] < node.point[axis]:
            node.left = self._insert(node.left, point, depth + 1)
        else:
            node.right = self._insert(node.right, point, depth + 1)
        return node

    def insert(self, point):
        self.root = self._insert(self.root, point, 0)

    def _nearest_neighbor(self, node, point, depth):
        if node is None:
            return None

        axis = depth % len(point)
        nearest = self._nearest_neighbor(
            node.left if point[axis] < node.point[axis] else node.right, point, depth + 1)

        # Check if closer neighbor found in another branch
        d = abs(point[axis] - node.point[axis])
        if nearest is None or d < distance(point, nearest.point):
            nearest = node

        # Check neighbor on same side if distance is within bounds
        if nearest is not None and distance(point, nearest.point) > d**2:
            better_neighbor = self._nearest_neighbor(
                node.right if point[axis] < node.point[axis] else node.left, point, depth + 1)
            if better_neighbor is not None and distance(point, better_neighbor.point) < distance(point, nearest.point):
                nearest = better_neighbor

        return nearest

    def nearest_neighbor(self, point):
        return self._nearest_neighbor(self.root, point, 0)

    def _points_within_distance(self, node, point, depth, radius):
        if node is None:
            return []

        axis = depth % len(point)
        d = abs(point[axis] - node.point[axis])
        near_points = []

        # Check if current node within radius
        if distance(point, node.point) <= radius**2:
            near_points.append(node.point)

        # Explore closer branch first
        if point[axis] < node.point[axis]:
            near_points.extend(self._points_within_distance(
                node.left, point, depth + 1, radius))
            # Explore other branch if necessary
            if radius**2 > d**2:
                near_points.extend(self._points_within_distance(
                    node.right, point, depth + 1, radius))
        else:
            near_points.extend(self._points_within_distance(
                node.right, point, depth + 1, radius))
            if radius**2 > d**2:
                near_points.extend(self._points_within_distance(
                    node.left, point, depth + 1, radius))

        return near_points

    def points_within_distance(self, point, radius):
        return self._points_within_distance(self.root, point, 0, radius)


def distance(p1, p2):
    # Euclidean distance
    s = 0.0
    for a, b in zip(p1, p2):
        s += (a-b)**2
    return s
    return sum((a - b) ** 2 for a, b in zip(p1, p2))
