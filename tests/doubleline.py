import numpy as np

from environments.d2.line import Line
from environments.visualisers.pg2dvis import PyGame2D
from models.pnn import ACPNN
from environments.environment import Environment2D
from environments.d2.circle import Circle
from agents.pathsampler import sample_paths, sample_lines

def to_one_hot(num, tot):
    return [0] * (num) + [1] + [0] * (tot - (num + 1))

def transform( uvs, obs, acs, states ):
    # Extend N_xTxW to N_xTxWxP
    uvs = uvs.reshape(uvs.shape + (1,))
    # N_xTxWxP to TxNxP
    uvs = uvs.transpose([1, 0, 2, 3]).reshape([uvs.shape[1], uvs.shape[0] * uvs.shape[2], uvs.shape[3]])
    W = obs.shape[2];

    # Convert the rest
    # TxNxC
    obs = obs.transpose([1, 0, 2, 3]).reshape([obs.shape[1], obs.shape[0] * obs.shape[2], obs.shape[3]])

    # TxNxA
    acs = np.tile(acs.transpose([1, 0, 2]).reshape([acs.shape[1], acs.shape[0], 1, acs.shape[2]]), [1, 1, W, 1]) \
        .reshape([acs.shape[1], acs.shape[0] * W, acs.shape[2]])

    # TxNxS
    states = np.tile(states.transpose([1, 0, 2]).reshape([states.shape[1], states.shape[0], 1, states.shape[2]]), [1, 1, W, 1]) \
        .reshape([states.shape[1], states.shape[0] * W, states.shape[2]])
    return uvs, obs, acs, states

testenv = Environment2D()

# Add a shape to the 2D space.
testenv.add_shapes( Line( np.asarray([-1,0]), np.asarray([1,0]), np.asarray([1,0,0,1])) )
# Set the agent's position to right of the circle.
testenv.set_position( np.asarray([0.0,2.0]) )
# Set rotation quanta in degrees.
testenv.set_rot_amount( 30 )
# Set the Forward and Right directions for the agent.
testenv.set_directions( np.asarray([0.0,-1.0]), np.asarray([1.0,0.0]) )
# Set the movement amount to 0.1 units per action.
testenv.set_move_amount( 0.4 )

#outputss = sample_lines( testenv )
outputss = sample_paths( testenv )

# N_xTxWxC
obs = np.asarray( [ [ output[0][0] for output in outputs ] for outputs in outputss ])
# N_xTxA
acs = np.asarray( [ [ to_one_hot(output[1],7) for output in outputs ] for outputs in outputss ])
# N_xTx2
poss = np.asarray([ [ output[2] for output in outputs ] for outputs in outputss ])
# N_xTx2
dirs = np.asarray([ [ output[3] for output in outputs ] for outputs in outputss ])
# N_xTxW
uvs = np.asarray( [ [ np.linspace( -1, 1, obs.shape[2] ) for output in outputs ] for outputs in outputss ])

states = np.concatenate([poss, dirs], axis=2)

N_ = obs.shape[0]
W = obs.shape[2]

# 80% for train and 20% for test.
N_split = N_ - int(N_ / 5)
train_uvs, train_obs, train_acs, train_states = transform( uvs[:N_split], obs[:N_split], acs[:N_split], states[:N_split] )
test_uvs, test_obs, test_acs, test_states = transform( uvs[N_split:], obs[N_split:], acs[N_split:], states[N_split:] )


# Uncomment to see visualisation.
#plotter = PyGame2D()
#plotter.draw_path(testenv, outputss[0])

pnn = ACPNN()

# Start fitting the model.
# pnn.fit(train_acs, train_uvs, train_obs, epochs=2)

# TxNxC
preds = pnn.predict(test_acs, test_uvs)

verify_cost = pnn.verify(test_acs, test_uvs, test_obs)

print("Cross-validation cost: ", verify_cost)
# Unwrap preds.
# N_xTxWxC
preds = preds.reshape([preds.shape[0], N_ - N_split, W, preds.shape[2]]).transpose([1,0,2,3])

plotter = PyGame2D()
plotter.draw_path_cmp(testenv, outputss[-1], preds[-1])

#print(outputss[-1])
#print(preds[-1])
#print(train_states[:,-1,:])