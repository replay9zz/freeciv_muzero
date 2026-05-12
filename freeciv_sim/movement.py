import numpy as np


class FreecivMovement:
    def __init__(self, map_width=16, map_height=16):
        self.native_w = map_width
        self.native_h = map_height

    def get_native_neighbors(self, native_x, native_y):
        """Return the six hex neighbors in [N, NE, SE, S, SW, NW] order."""
        if native_y % 2 == 0:
            offsets = [
                (0, -2),
                (0, -1),
                (0, +1),
                (0, +2),
                (-1, +1),
                (-1, -1),
            ]
        else:
            offsets = [
                (0, -2),
                (+1, -1),
                (+1, +1),
                (0, +2),
                (0, +1),
                (0, -1),
            ]

        neighbors = []
        for dx, dy in offsets:
            nx = native_x + dx
            ny = native_y + dy
            if 0 <= nx < self.native_w and 0 <= ny < self.native_h:
                neighbors.append((nx, ny))
            else:
                neighbors.append((None, None))

        return neighbors

    def parse_visibility_string(self, visibility_string, width, height):
        """
        Convert a visibility string into a dictionary keyed by native coordinates.

        The source string is interpreted column-by-column in a zig-zag layout.
        """
        flat_visibility = ''.join(visibility_string.split())

        visibility_dict = {}
        index = 0

        for col in range(width):
            for row in range(height):
                native_y = col * 2 + (row % 2)
                native_x = row // 2

                if index < len(flat_visibility):
                    visibility_dict[(native_x, native_y)] = flat_visibility[index]
                    index += 1

        return visibility_dict

    def get_movable_tiles(self, current_x, current_y, visibility_dict):
        """
        Return all adjacent available tiles reachable from the current position.

        Raises if the current tile is unavailable or outside the known map.
        """
        if (current_x, current_y) not in visibility_dict:
            raise ValueError(f"Coordinate ({current_x}, {current_y}) is outside the map bounds.")

        current_tile_status = visibility_dict[(current_x, current_y)]
        if current_tile_status == 'U':
            raise ValueError(f"Coordinate ({current_x}, {current_y}) is an unavailable tile.")

        adjacent_tiles = self.get_native_neighbors(current_x, current_y)

        movable = []
        for nx, ny in adjacent_tiles:
            if (nx, ny) in visibility_dict and visibility_dict[(nx, ny)] == 'A':
                movable.append((nx, ny))

        return movable
