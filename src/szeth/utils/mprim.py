import os
import numpy as np
from numpy import array2string as a2s
from numpy import fromstring as s2a


class MPrim:
    def __init__(self, initial_heading,
                 T, seq, true_states, discrete_states,
                 final_point, id=None):
        self.initial_heading = initial_heading  # Initial heading
        self.T = T  # Length of the motion primitive
        self.seq = [c for c in seq]  # Sequence of controls
        self.true_states = [state.copy() for state in true_states]
        self.discrete_states = [state.copy() for state in discrete_states]
        # Final lattice center reached
        self.final_point = final_point.copy()
        if id is not None:
            self.id = id

    def __str__(self):
        rep = 'MPrim of length ' + \
            str(self.T)+' and initial heading ' + \
            str(self.initial_heading)+' with controls\n'
        for c in self.seq:
            rep += np.array2string(c)+'\n'
        rep += 'with final point at '+np.array2string(self.final_point)
        return rep

    def compute_cost(self, costs):
        cost = 0.0
        for control in self.seq:
            cost += costs[tuple(control)]

        return cost

    def __eq__(self, other):
        return self.id == other.id


def write_primitives(mprims_all_headings_no_duplicates,
                     params):

    # Delete the file if it exists
    path = os.path.join(os.environ['HOME'],
                        'workspaces/szeth_ws/src/szeth/save/car.mprim')

    if os.path.exists(path):
        os.remove(path)

    mprim_file = open(path, 'a+')

    # First, write metadata
    # mprim_file.write('PLAYFIELD: '+str(params['PLAYFIELD'])+'\n')
    mprim_file.write('X_BOUNDS: '+a2s(params['X_BOUNDS'])+'\n')
    mprim_file.write('Y_BOUNDS: '+a2s(params['Y_BOUNDS'])+'\n')
    mprim_file.write('X_DISCRETIZATION: '+str(params['X_DISCRETIZATION'])+'\n')
    mprim_file.write('Y_DISCRETIZATION: '+str(params['Y_DISCRETIZATION'])+'\n')
    mprim_file.write('THETA_DISCRETIZATION: ' +
                     str(params['THETA_DISCRETIZATION'])+'\n')
    mprim_file.write('X_THRESHOLD: '+str(params['X_THRESHOLD'])+'\n')
    mprim_file.write('Y_THRESHOLD: '+str(params['Y_THRESHOLD'])+'\n')
    mprim_file.write('THETA_THRESHOLD: '+str(params['THETA_THRESHOLD'])+'\n')
    mprim_file.write('NUMBER_OF_PRIMITIVES: ' +
                     str(params['number_of_primitives'])+'\n')

    # Write mprims one by one
    mprim_id = 0
    for heading, mprims in enumerate(mprims_all_headings_no_duplicates):
        for mprim in mprims:
            # Write mprim_id
            mprim_file.write('mprim_id: '+str(mprim_id)+'\n')
            # Write initial heading
            mprim_file.write('initial_heading: '+str(heading)+'\n')
            # Write final point
            mprim_file.write('final_center: '+a2s(mprim.final_point)+'\n')
            # Write length of mprim
            mprim_file.write('length: '+str(mprim.T)+'\n')
            # Write sequence of controls
            for control in mprim.seq:
                mprim_file.write(a2s(control)+'\n')
            # Write sequence of true states
            for true_state in mprim.true_states:
                mprim_file.write(a2s(true_state)+'\n')
            # Write sequence of discrete states
            for discrete_state in mprim.discrete_states:
                mprim_file.write(a2s(discrete_state)+'\n')
            # Increment mprim_id
            mprim_id += 1

    # Close the file
    mprim_file.close()


def parse_line(line):
    components = line.split(': ')
    return components[1]


def parse_np_array(line):
    components = line.split('[')
    components = components[1].split(']')
    return components[0]


def read_primitives(path, params):
    if not os.path.exists(path):
        raise Exception('Mprim file not found')

    mprim_file = open(path, 'r')

    # First read metadata
    # PLAYFIELD = float(parse_line(mprim_file.readline()))
    # assert PLAYFIELD == params['PLAYFIELD'], "PLAYFIELD does not match"
    X_BOUNDS = s2a(parse_np_array(parse_line(mprim_file.readline())),
                   dtype=float,
                   sep=' ')
    assert np.array_equal(
        X_BOUNDS, params['X_BOUNDS']), "X_BOUNDS do not match"
    Y_BOUNDS = s2a(parse_np_array(parse_line(mprim_file.readline())),
                   dtype=float,
                   sep=' ')
    assert np.array_equal(
        Y_BOUNDS, params['Y_BOUNDS']), "Y_BOUNDS do not match"
    X_DISCRETIZATION = int(parse_line(mprim_file.readline()))
    assert X_DISCRETIZATION == params['X_DISCRETIZATION'], "X_DISCRETIZATION does not match"
    Y_DISCRETIZATION = int(parse_line(mprim_file.readline()))
    assert Y_DISCRETIZATION == params['Y_DISCRETIZATION'], "Y_DISCRETIZATION does not match"
    THETA_DISCRETIZATION = int(parse_line(mprim_file.readline()))
    assert THETA_DISCRETIZATION == params['THETA_DISCRETIZATION'], "THETA_DISCRETIZATION does not match"
    X_THRESHOLD = float(parse_line(mprim_file.readline()))
    Y_THRESHOLD = float(parse_line(mprim_file.readline()))
    THETA_THRESHOLD = float(parse_line(mprim_file.readline()))
    params['X_THRESHOLD'] = X_THRESHOLD
    params['Y_THRESHOLD'] = Y_THRESHOLD
    params['THETA_THRESHOLD'] = THETA_THRESHOLD
    number_of_primitives = int(parse_line(mprim_file.readline()))
    params['number_of_primitives'] = number_of_primitives

    mprims = {}

    # Read mrpims one by one
    for mprim_id in range(number_of_primitives):
        # Skip the id line
        id = int(parse_line(mprim_file.readline()))
        # heading
        initial_heading = int(parse_line(mprim_file.readline()))
        # final point
        final_point = s2a(
            parse_np_array(parse_line(mprim_file.readline())),
            dtype=int,
            sep=' ')
        # Length of mprim
        T = int(parse_line(mprim_file.readline()))
        # Read in controls
        seq = []
        for t in range(T):
            seq.append(s2a(
                parse_np_array(mprim_file.readline()),
                dtype=int,
                sep=' '))
        # Read in true states
        true_states = []
        for t in range(T+1):
            true_states.append(
                s2a(
                    parse_np_array(mprim_file.readline()),
                    dtype=float,
                    sep=' '))

        # Read in discrete states
        discrete_states = []
        for t in range(T+1):
            discrete_states.append(
                s2a(
                    parse_np_array(mprim_file.readline()),
                    dtype=int,
                    sep=' '))

        # Construct mprim
        mprim = MPrim(initial_heading, T, seq, true_states, discrete_states,
                      final_point, id=id)

        # Add to dict
        if initial_heading not in mprims:
            mprims[initial_heading] = []
        mprims[initial_heading].append(mprim)

    return mprims, params
