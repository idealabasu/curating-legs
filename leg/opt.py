import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.optimize import differential_evolution
from leg import four_bar
from leg import model
from leg import helper

MAX_LEG_LENGTH = 0.14
MAX_LINK_LENGTH = 0.1
BOUNDS = [
    (0.02, MAX_LINK_LENGTH),  # crank
    (0.02, MAX_LINK_LENGTH),  # coupler
    (0.02, MAX_LINK_LENGTH),  # rocker
    (0.02, MAX_LINK_LENGTH),  # ground
    (0.02, MAX_LINK_LENGTH),  # coupler extension
    (-np.pi, np.pi),  # initial crank angle
    (0.005, MAX_LINK_LENGTH),  # parallel spring length
    (0.005, MAX_LINK_LENGTH),  # series spring length
]
CONSTRAINTS_MAX = np.array([
    0.02,  # trajectory error
    -0.02,  # gap
    0.01,  # center x
    60 / 180 * np.pi,  # transmission angle
    0.02,  # parallel stiffness error
    0.02,  # series stiffness error
    0.1,  # parallel stiffness local torque
    -0.028,  # femur min length
    MAX_LINK_LENGTH,  # femur max length
    MAX_LINK_LENGTH,  # ps coupler max length
    0.05  # max leg width
])
CONSTRAINTS_WEIGHTS = 10 * np.array([1, 1, 1, 0.04, 1, 1, 1, 1, 1, 1, 1])


def obj_with_constraints(x, p, plot=False):
    result = model.sim(x, p)
    l_ab, l_bc, l_cd, l_ad, l_be, input_offset, l_ps, l_ss = x
    travel_offset, travel_length, input_range, parallel_stiffness, series_stiffness = p

    has_ps = parallel_stiffness != 0
    if not has_ps:
        result['legs'] = np.array([
            leg[np.r_[0, 4:len(leg)]]
            for leg in result['legs']
        ])

    # Cost
    l_femur = helper.link_length(result['femur'])
    l_af = helper.link_length(result['legs'][0][0])
    l_aj = helper.link_length(result['legs'][0][4])
    cost_length = np.sum([
        l_ab, l_bc, l_cd, l_ad, l_be, l_femur,  # main
        l_af * 2, l_aj  # series spring
    ])
    if has_ps:
        l_fg = helper.link_length(result['legs'][0][1])
        l_gh = helper.link_length(result['legs'][0][2])
        l_ah = helper.link_length(result['legs'][0][3])
        cost_length += np.sum([l_fg, l_gh, l_ah])

    cost = cost_length

    # Constraints
    constraint_trajectory = np.amax(
        np.abs((result['feet'] - result['feet_ref']))
    ) / travel_length

    gap_y = np.array([
        np.amin(leg.reshape((-1, 2))[:-1, 1] - foot[1])
        for leg, foot in zip(result['legs'], result['feet'])
    ])
    constraint_gap_y = -np.amin(gap_y)

    pts_x = result['legs'].reshape(-1, 2)[:, 0] - result['feet'][0, 0]
    constraint_centroid_x = np.abs(np.mean(pts_x))

    transmission_angles = np.array([
        four_bar.transmission_angle(leg[5:9]) if has_ps else
        four_bar.transmission_angle(leg[2:6])
        for leg in result['legs']
    ])
    constraint_transmission_angle = np.amax(np.pi / 2 - transmission_angles)

    if has_ps:
        constraint_parallel_stiffness = np.abs(
            result["ps_torques"][-1] / result["inputs"][-1] -
            parallel_stiffness
        ) / parallel_stiffness
    else:
        constraint_parallel_stiffness = 0

    constraint_series_stiffness = np.abs(
        result["ss"] - series_stiffness
    ) / series_stiffness

    if has_ps:
        constraint_ps_local_torque = np.amax(
            np.abs(result['ps_torques_local'])
        )
    else:
        constraint_ps_local_torque = 0

    constraint_min_femur_length = -l_femur
    constraint_max_femur_length = l_femur
    if has_ps:
        constraint_max_ps_coupler_length = l_fg
    else:
        constraint_max_ps_coupler_length = 0

    constraint_max_leg_width = np.amax(np.abs([
        np.amin(pts_x), np.amax(pts_x)
    ]))

    constraints = np.array([
        constraint_trajectory,
        constraint_gap_y,
        constraint_centroid_x,
        constraint_transmission_angle,
        constraint_parallel_stiffness,
        constraint_series_stiffness,
        constraint_ps_local_torque,
        constraint_min_femur_length,
        constraint_max_femur_length,
        constraint_max_ps_coupler_length,
        constraint_max_leg_width
    ])
    constraints = np.maximum(constraints, CONSTRAINTS_MAX) - CONSTRAINTS_MAX

    if plot:
        print('x: ', ', '.join([f'{c:.5f}' for c in x]))
        print('p: ', ', '.join([f'{c:.2f}' for c in p]))
        print(f'cost: {cost:.3f}')
        print(
            'constraints: ',
            ', '.join([f'{c:.2f}' for c in constraints])
        )
        print(
            'weighted constraints: ',
            ', '.join([f'{c:.2f}' for c in CONSTRAINTS_WEIGHTS * constraints])
        )
        model.plot(result, leg_only=plot == 'leg_only')
        if plot == 'leg_only':
            plt.text(
                0.5, 0.02,
                ', '.join([f'{c:.2f}' for c in p]),
                ha='center', va='bottom', transform=plt.gca().transAxes
            )

    return cost, constraints


def obj(x, p, plot=False):
    try:
        cost, constraints = obj_with_constraints(x, p, plot=plot)
    except AssertionError:
        return 10
    return cost + np.sum(CONSTRAINTS_WEIGHTS * constraints)


def optimize(p, seed=None):
    assert p[0] + p[1] < MAX_LEG_LENGTH + 1e-4
    return differential_evolution(
        obj,
        args=(p,),
        bounds=BOUNDS,
        popsize=10,
        maxiter=500,
        tol=0.01,
        workers=-1,
        updating='deferred',
        polish=False,
        seed=seed
    )


if __name__ == '__main__':
    p = [float(v) for v in sys.argv[1:]]
    for i in range(5):
        print(f'trial: {i}')
        try:
            r = optimize(p)
            cost, constraints = obj_with_constraints(r.x, p)
            valid = np.sum(constraints) == 0
        except AssertionError:
            valid = False

        if valid:
            break

    if valid:
        print(str(list(r.x)))
        obj(r.x, p, plot=True)
        plt.show()
