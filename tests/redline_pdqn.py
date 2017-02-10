from environments.d2.line import Line
from environments.environment import Environment2D
import numpy as np
from models.mpbacpnn import ModelPredictivePDQN

# Create a Environment Factory
class EnvBuilder:
    def __init__(self):
        pass

    def build(self):
        testenv = Environment2D()

        # Add a shape to the 2D space.
        testenv.add_shapes(Line(np.asarray([-1, 0]), np.asarray([1, 0]), np.asarray([1, 0, 0, 1])))
        # Set the agent's position to right of the circle.
        testenv.set_position(np.asarray([0.0, 2.0]))
        # Set rotation quanta in degrees.
        testenv.set_rot_amount(30)
        # Set the Forward and Right directions for the agent.
        testenv.set_directions(np.asarray([0.0, -1.0]), np.asarray([1.0, 0.0]))
        # Set the movement amount to 0.1 units per action.
        testenv.set_move_amount(0.4)

        return testenv

pdqn = ModelPredictivePDQN( EnvBuilder() ,
                            num_episodes=500,
                            num_actions=7,
                            pnn_epochs=600,
                            dqn_epochs=2000,
                            reward_multiplier=0.2,
                            buffer_samples=500)

pdqn.run( episode_length=10 )