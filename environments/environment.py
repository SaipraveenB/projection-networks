import project2d
import numpy as np
import math

RIGHT_ROT = 0
LEFT_ROT = 1
RIGHT_MOVE = 2
LEFT_MOVE = 3
FRONT_MOVE = 4
BACK_MOVE = 5
INITIALISE = 6


class Environment2D:
    def __init__(self):
        self.shape_list = []
        self.position = np.asarray([0, 0])
        self.f_direction = np.asarray([[0, 1]])
        self.r_direction = np.asarray([[1, 0]])
        self.old_position = np.array( self.position )
        self.old_f_direction = np.array( self.f_direction )
        self.old_r_direction = np.array( self.r_direction )

        # Initialize transform matrices..
        self.right_rot = np.identity(3)
        self.left_rot = np.identity(3)
        self.move_delta = 0.1

    def reset(self):
        self.f_direction = np.array( self.old_f_direction )
        self.r_direction = np.array( self.old_r_direction )
        self.position = np.array( self.old_position )

    def set_position(self, pos):
        self.old_position = np.array( pos )
        self.position = np.array( pos )

    def set_move_amount(self, amt):
        self.move_delta = amt

    def set_directions(self, dir1, dir2):
        self.old_f_direction = np.array( dir1 )
        self.f_direction = np.array( dir1 )
        self.old_r_direction = np.array( dir2 )
        self.r_direction = np.array( dir2 )

    def set_rot_amount(self, rot):
        rot = (rot * math.pi)/180.0
        self.right_rot = np.asarray([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
        self.left_rot = np.asarray([[math.cos(rot), math.sin(rot)], [-math.sin(rot), math.cos(rot)]])

    def add_shapes(self, shape):
        self.shape_list.append(shape)
        pass

    # performs an action.
    def action(self, num):

        # Actions and their effects.
        if num == RIGHT_ROT:
            self.r_direction = np.dot(self.r_direction, self.right_rot.transpose())
            self.f_direction = np.dot(self.f_direction, self.right_rot.transpose())
        elif num == LEFT_ROT:
            self.r_direction = np.dot(self.r_direction, self.left_rot.transpose())
            self.f_direction = np.dot(self.f_direction, self.left_rot.transpose())

        elif num == RIGHT_MOVE:
            self.position += self.move_delta * self.r_direction;
        elif num == LEFT_MOVE:
            self.position -= self.move_delta * self.r_direction;

        elif num == FRONT_MOVE:
            self.position += self.move_delta * self.f_direction;
        elif num == BACK_MOVE:
            self.position -= self.move_delta * self.f_direction;

        elif num == INITIALISE:
            # Do nothing.
            pass

    # Gets the current observation.
    def get_observation(self):
        # Ray trace the shapes in the scene.
        return project2d.project(self.position, (self.f_direction,self.r_direction), self.shape_list)
