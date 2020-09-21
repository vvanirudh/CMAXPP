'''
Script to check if the generated mprims have a good coverage
'''
from szeth.envs.car_racing_env_revised import construct_params
from szeth.utils.mprim import read_primitives
import os

mprim_path = os.path.join(
    os.environ['HOME'], 'workspaces/szeth_ws/src/szeth/save/car.mprim')
params = construct_params()

mprims, params = read_primitives(mprim_path, params)

for heading in mprims.keys():
    mprims_heading = mprims[heading]
    print('Heading: ', heading)
    for mprim in mprims_heading:
        print(mprim.final_point)
