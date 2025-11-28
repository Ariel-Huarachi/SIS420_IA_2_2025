import numpy as np

class Maze:
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.walls = []
        self._create_maze()

    def _create_maze(self):
        """Define las paredes reales del laberinto (líneas)."""
        # Paredes exteriores
        self.walls.append(((0, 0), (self.width, 0)))  
        self.walls.append(((0, self.height), (self.width, self.height)))
        self.walls.append(((0, 0), (0, self.height)))
        self.walls.append(((self.width, 0), (self.width, self.height)))

        # Paredes internas (tu laberinto existente)
        self.walls.append(((3, 0), (3, 8)))
        self.walls.append(((7, 4), (7, 12)))
        self.walls.append(((3, 12), (10, 12)))
        self.walls.append(((10, 8), (10, 16)))
        self.walls.append(((14, 0), (14, 10)))
        self.walls.append(((6, 16), (17, 16)))
        self.walls.append(((17, 8), (17, 16)))
        self.walls.append(((10, 4), (14, 4)))
        self.walls.append(((3, 8), (7, 8)))

    def check_collision(self, x, y, radius):
        """Verifica si un punto (x,y) toca alguna pared."""
        # límites
        if x - radius < 0 or x + radius > self.width:
            return True
        if y - radius < 0 or y + radius > self.height:
            return True

        # colisión con paredes reales (segmentos)
        for wall in self.walls:
            if self._distance_to_segment(x, y, wall) < radius:
                return True

        return False

    def _distance_to_segment(self, px, py, segment):
        """Distancia mínima de un punto a una línea."""
        (x1, y1), (x2, y2) = segment

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)

        t = max(0, min(1, ((px - x1)*dx + (py - y1)*dy) / (dx*dx + dy*dy)))

        cx = x1 + t*dx
        cy = y1 + t*dy

        return np.sqrt((px - cx)**2 + (py - cy)**2)
