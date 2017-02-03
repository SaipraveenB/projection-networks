import random
import numpy as np

class Recorder:
    def __init__(self, env):
        self.obs = []
        self.obss = []
        self.env = env
        pass;
    def action(self, ac):
        self.env.action(ac);
        ims = self.env.get_observation()
        self.obs.append((ims, ac, np.array(self.env.position), np.array(self.env.f_direction)))

    def flush(self):
        if len(self.obs) > 0:
            self.obss.append(self.obs)
            self.obs = []

    def reset(self):
        self.env.reset()
        self.flush()

    def get_trace(self):
        return self.obss

# Randomly sample paths in the grid-direction space.
# Returns a list of list of tuples, each consisting of a position and direction.
def sample_paths( env, num_episodes=1000, episode_length=7):

    rec = Recorder(env)
    #obss = [];
    for i in range(0,num_episodes):
        # Reset internal state.
        #obs = []
        #env.reset()
        rec.reset()

        # Initialise.
        rec.action(6)
        for j in range(0,episode_length):
            k = random.randint(0,5)
            rec.action(k)
            #env.action(k)
            #ims = env.get_observation()
            #obs.append((ims,k,np.array(env.position),np.array(env.f_direction)))

        #obss.append(obs)
    rec.flush()
    return rec.get_trace()

def sample_lines( env, num_episodes=1, num_rows=5, num_cols=5, num_turns=12):

    rec = Recorder(env)

    # Initialise.
    rec.action(6)

    for c in range(0, num_cols/2):
        rec.action(3)

    for r in range(0, num_rows/2):
        for c in range(0, num_cols/4):
            for t in range(0, num_turns):
                rec.action(0)
            for t2 in range(0, 4):
                rec.action(2)

        rec.action(5)
        for c in range(0, num_cols/4):
            for t in range(0, num_turns):
                rec.action(1)
            for t2 in range(0, 4):
                rec.action(3)

        rec.action(5)

    rec.flush()
    return rec.get_trace()