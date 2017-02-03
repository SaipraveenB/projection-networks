import numpy as np
import math
class Circle:
    def __init__(self, radius, position, color):
        self.radius = radius
        self.pos = position
        self.color = color

    def intersect(self, ray):
        # Ensure 'v' is normalised.
        o, v = ray

        c = self.pos
        r = self.radius

        # Vectorise these terms soon.
        delta = np.square(np.dot((o-c), v)) - np.dot((o-c), (o-c)) + r * r

        if delta > 0:
            # HIT
            dist1 = np.dot(c - o, v) + np.sqrt(delta)
            dist2 = np.dot(c - o, v) - np.sqrt(delta)
            if dist1 > dist2 and dist2 > 0:
                return True, self.color, dist2, None
            elif dist1 > 0:
                return True, self.color, dist1, None
            else:
                return False, np.zeros((4,)), 1000, None

        return False, np.zeros((4,)), 1000, None
