import numpy as np
import math
class Line:
    def __init__(self, spos, epos, color):
        self.start_pos = spos
        self.end_pos = epos
        self.color = color

    def intersect(self, ray):

        # Ensure 'v' is normalised.
        o, v = ray
        a = self.start_pos
        b = self.end_pos

        st = o - b
        xy = v
        pq = a - b

        lam = -(st[0]*pq[1] - st[1]*pq[0]) / (xy[0]*pq[1] - xy[1]*pq[0])
        gamma = (st[0]*xy[1] - st[1]*xy[0]) / (xy[1]*pq[0] - xy[0]*pq[1])

        if( gamma < 0 or gamma > 1):
            return False, np.zeros((4,)), 1000, None
        elif ( lam < 0 ):
            return False, np.zeros((4,)), 1000, None
        else:
            return True, self.color, lam, None