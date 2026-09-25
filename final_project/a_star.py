import heapq
import numpy as np
from math import hypot


class OccupancyGridAStar:

    def __init__(self, map_msg, start, goal, inflation_cells=0):
        self.map_msg = map_msg
        self.width = map_msg.info.width
        self.height = map_msg.info.height
        self.grid = map_msg.data

        # Occupied cells grown by the robot radius (configuration space), so that paths keep a distance to obstacles
        self.inflated = self.get_inflated_grid(inflation_cells) if inflation_cells > 0 else None

        # Convert to integer cell indices
        self.sx, self.sy = int(round(start[0])), int(round(start[1]))
        self.gx, self.gy = int(round(goal[0])), int(round(goal[1]))

    def search(self):
        if not self.in_bounds(self.sx, self.sy) or not self.in_bounds(self.gx, self.gy):
            return []
        if not self.is_free(self.sx, self.sy) or not self.is_free(self.gx, self.gy):
            return []

        # 8-connected neighbors
        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (1, 1, 2 ** 0.5),
        ]

        open_heap = []
        heapq.heappush(open_heap, (self.heuristic(self.sx, self.sy), 0.0, (self.sx, self.sy)))
        g_cost = {(self.sx, self.sy): 0.0}
        came_from = {}
        closed = set()

        while open_heap:
            f, g, (x, y) = heapq.heappop(open_heap)

            if (x, y) in closed:
                continue
            closed.add((x, y))

            if (x, y) == (self.gx, self.gy):
                # Reconstruct path
                path = [(x, y)]
                while (x, y) in came_from:
                    x, y = came_from[(x, y)]
                    path.append((x, y))
                path.reverse()
                return path

            for dx, dy, step_cost in neighbors:
                nx, ny = x + dx, y + dy
                if not self.in_bounds(nx, ny):
                    continue
                if not self.is_free(nx, ny):
                    continue
                tentative_g = g_cost[(x, y)] + step_cost
                if tentative_g < g_cost.get((nx, ny), float('inf')):
                    g_cost[(nx, ny)] = tentative_g
                    came_from[(nx, ny)] = (x, y)
                    heapq.heappush(open_heap, (tentative_g + self.heuristic(nx, ny), tentative_g, (nx, ny)))

        # No path found
        return []

    def get_inflated_grid(self, radius):
        occupied = np.array(self.grid, dtype=np.int16).reshape(self.height, self.width) >= 50
        inflated = np.zeros_like(occupied)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx ** 2 + dy ** 2 > radius ** 2:
                    continue
                # Shift occupied cells by (dx, dy)
                inflated[max(0, dy):self.height + min(0, dy), max(0, dx):self.width + min(0, dx)] |= \
                    occupied[max(0, -dy):self.height + min(0, -dy), max(0, -dx):self.width + min(0, -dx)]

        return inflated

    def heuristic(self, x, y):
        return hypot(self.gx - x, self.gy - y)

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, x, y):
        if self.inflated is not None and self.inflated[y, x]:
            # Too close to an obstacle for the robot
            return False
        idx = y * self.width + x
        val = self.grid[idx]
        if val < 0:
            # Unknown treated as not traversable
            return False
        return val < 50  # free if occupancy < 50