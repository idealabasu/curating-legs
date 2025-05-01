import os
import matplotlib.pyplot as plt
import numpy as np
from leg import four_bar
from leg import helper

NUM_KEYFRAMES = 11
E_MAX = 1
H_JOINT = 0.1 / 1000
H_ADHESIVE = 0.015 / 1000
H_LINK = 0.72 / 1000
H_LINK_THIN = 0.45 / 1000
H_LINK_THICK = 1.7 / 1000
L_AF = 20 / 1000
L_AI = 15 / 1000
L_J = 10 / 1000
GAMMA = 0.85
E = 10.3e9
K_THETA = 2.65
W_PS = 15 / 1000
W_SS = 15 / 1000

Y_FEMUR = 0.01565
H_FEMUR = 0.016

Y_SERVO = 0.028  # offset from midplane of main four-bar
L_SERVO = 0.034
W_SERVO = 0.023
H_SERVO = 0.02
L_SERVO_OFFSET = 0.0075  # distance between center of servo and knee servo joint

H_LINK_TOTAL = np.sum([
    H_JOINT, H_ADHESIVE * 2, H_LINK, H_LINK_THIN, H_LINK_THICK
])
H_LINK_OFFSET = H_LINK_TOTAL / 2 - (H_LINK_THIN + H_ADHESIVE + H_JOINT / 2)
W_LINK = 0.015

W_INPUT = 0.021
Y_INPUT = 0.001 + W_LINK / 2 + W_INPUT / 2

Y_PS = 0.001 + W_LINK

L_FOOT = 0.01


def sim(x, p):
    l_ab, l_bc, l_cd, l_ad, l_be, input_offset, l_ps, l_ss = x
    travel_offset, travel_length, input_range, parallel_stiffness, series_stiffness = p

    # Constants for main and spring linkage
    h = H_LINK_THIN + H_ADHESIVE + H_JOINT + H_ADHESIVE + H_LINK
    t_cbe = -np.arccos(h / l_bc) - np.arccos(h / l_be)
    pt_e_local = np.array([[l_be * np.cos(t_cbe), l_be * np.sin(t_cbe)]])

    h_offset = H_LINK - H_LINK_THIN
    l_gh = l_ps * GAMMA
    l_ah = ((L_AI + l_ps * (1 - GAMMA))**2 + h_offset**2)**0.5
    t_had = np.arcsin(h_offset / l_ah)
    t_fah = np.pi + input_offset - t_had
    l_fh = (l_ah**2 + L_AF**2 - 2 * l_ah * L_AF * np.cos(t_fah))**0.5
    t_ahf = np.arcsin(L_AF / l_fh * np.sin(t_fah))
    t_fhg = np.pi * 2 - (np.pi - t_had) - t_ahf
    l_fg = (l_fh**2 + l_gh**2 - 2 * l_fh * l_gh * np.cos(t_fhg))**0.5
    pt_j_local = np.array([[-(l_ss * GAMMA + L_J), 0]])

    k_ps = GAMMA * K_THETA * E * (W_PS * H_LINK_THIN**3) / 12 / l_ps
    k_ss = GAMMA * K_THETA * E * (W_SS * H_LINK**3) / 12 / l_ss

    # Key points
    legs = []
    ps_torques = []
    ps_torques_local = []
    inputs = np.linspace(0, input_range, NUM_KEYFRAMES)
    for input in inputs:
        lkg_main = four_bar.solve(
            l_ab, l_bc, l_cd, l_ad, input + input_offset, True
        )
        pt_e = helper.transform_points(
            helper.link_angle(lkg_main[1]),
            lkg_main[1][0, 0], lkg_main[1][0, 1],
            pt_e_local
        )[0]
        lkg_main = np.concatenate([
            lkg_main,
            np.array([[lkg_main[1][0, :], pt_e]])
        ])

        lkg_spring = four_bar.solve(
            L_AF, l_fg, l_gh, l_ah, t_fah + input, False
        )
        tau = (
            -helper.angle_between_links(-lkg_spring[2], lkg_spring[3]) -
            -t_had
        ) * k_ps
        ps_torques.append(four_bar.input_torque(tau, lkg_spring))
        ps_torques_local.append(tau)

        pt_j = helper.transform_points(
            helper.link_angle(lkg_spring[0]),
            0, 0,
            pt_j_local
        )[0]
        lkg_spring = np.concatenate([
            lkg_spring,
            np.array([[[0, 0], pt_j]])
        ])
        lkg_spring = helper.transform_points(
            t_had, 0, 0, lkg_spring.reshape((-1, 2))
        ).reshape(lkg_spring.shape)

        lkg = np.concatenate([lkg_spring, lkg_main])

        legs.append(lkg)
    legs = np.array(legs)

    # Transform based on foot initial and end position
    legs = helper.transform_points(
        helper.link_angle(legs[[0, -1], -1, 1, :]) - np.pi / 2,
        0, 0,
        legs.reshape((-1, 2)),
        inverse=True
    ).reshape(legs.shape)

    feet_ref = np.array([
        # x offset
        np.ones(NUM_KEYFRAMES) * legs[0, -1, 1, 0],
        (
            # Even spacing
            travel_length / input_range * inputs +
            # y offset
            legs[0, -1, 1, 1]
        ),
    ]).T
    feet = legs[:, -1, 1, :]

    femur = np.array([
        [0, 0,],
        [feet_ref[-1, 0], travel_offset + feet_ref[-1, 1]],
    ])

    result = {
        'inputs': inputs,
        'legs': legs,
        'feet': feet,
        'feet_ref': feet_ref,
        'femur': femur,
        'ps_torques': ps_torques,
        'ps_torques_local': ps_torques_local,
        'ps': k_ps,
        'ss': k_ss
    }

    return result


def mj_params(x, p):
    result = sim(x, p)
    leg = result['legs'][0]

    params = {}

    lk_femur = result['femur']
    l_femur = helper.link_length(lk_femur)
    pitch_femur = helper.link_angle(lk_femur)
    center_femur = helper.transform_points(
        pitch_femur, lk_femur[0, 0], lk_femur[0, 1],
        np.array([[l_femur / 2, 0]]),
    )[0]
    params['femur_x'] = center_femur[0] - lk_femur[1, 0]
    params['femur_y'] = Y_FEMUR
    params['femur_z'] = center_femur[1] - lk_femur[1, 1]
    params['femur_pitch'] = -pitch_femur / np.pi * 180
    params['femur_l'] = l_femur / 2
    params['femur_w'] = H_LINK_THICK / 2
    params['femur_h'] = H_FEMUR / 2

    params['knee_x'] = -lk_femur[1, 0]
    params['knee_z'] = -lk_femur[1, 1]

    pitch_servo = (
        pitch_femur +
        np.arccos(L_SERVO_OFFSET / l_femur) +
        np.pi
    )
    center_servo = helper.transform_points(
        pitch_servo,
        lk_femur[0, 0], lk_femur[0, 1],
        np.array([[-L_SERVO_OFFSET, 0]]),
    )[0]
    params['knee_servo_x'] = center_servo[0] - lk_femur[1, 0]
    params['knee_servo_y'] = Y_SERVO
    params['knee_servo_z'] = center_servo[1] - lk_femur[1, 1]
    params['knee_servo_pitch'] = -pitch_servo * 180 / np.pi
    params['knee_servo_l'] = L_SERVO / 2
    params['knee_servo_w'] = W_SERVO / 2
    params['knee_servo_h'] = H_SERVO / 2

    lk_ground = leg[8]
    l_ground = helper.link_length(lk_ground)
    pitch_ground = helper.link_angle(lk_ground)
    center_ground = helper.transform_points(
        pitch_ground, lk_ground[0, 0], lk_ground[0, 1],
        np.array([[l_ground / 2, H_LINK_OFFSET]]),
    )[0]
    params['ground_x'] = center_ground[0] - lk_femur[1, 0]
    params['ground_z'] = center_ground[1] - lk_femur[1, 1]
    params['ground_pitch'] = -pitch_ground * 180 / np.pi
    params['ground_l'] = l_ground / 2
    params['ground_w'] = W_LINK / 2
    params['ground_h'] = H_LINK_TOTAL / 2

    lk_input = leg[0]
    l_input = helper.link_length(lk_input)
    pitch_input = helper.link_angle(lk_input)
    center_input = helper.transform_points(
        pitch_input, lk_input[0, 0], lk_input[0, 1],
        np.array([[l_input / 2, H_LINK_OFFSET]]),
    )[0]
    params['input_x'] = center_input[0] - lk_femur[1, 0]
    params['input_y'] = Y_INPUT
    params['input_z'] = center_input[1] - lk_femur[1, 1]
    params['input_pitch'] = -pitch_input * 180 / np.pi
    params['input_l'] = l_input / 2
    params['input_w'] = W_INPUT / 2
    params['input_h'] = H_LINK_TOTAL / 2

    lk_crank = leg[5]
    l_crank = helper.link_length(lk_crank)
    pitch_crank = helper.link_angle(lk_crank)
    center_crank = helper.transform_points(
        pitch_crank, lk_crank[0, 0], lk_crank[0, 1],
        np.array([[l_crank / 2, -H_LINK_OFFSET]]),
    )[0]
    params['crank_x'] = center_crank[0] - lk_femur[1, 0]
    params['crank_z'] = center_crank[1] - lk_femur[1, 1]
    params['crank_pitch'] = -pitch_crank * 180 / np.pi
    params['crank_l'] = l_crank / 2
    params['crank_w'] = W_LINK / 2
    params['crank_h'] = H_LINK_TOTAL / 2

    lk_ss = leg[4]
    l_ss = helper.link_length(lk_ss)
    pitch_ss = helper.link_angle(lk_ss)
    center_ss = helper.transform_points(
        pitch_ss, lk_ss[0, 0], lk_ss[0, 1],
        np.array([[l_ss / 2, -H_LINK_OFFSET]]),
    )[0]
    params['ss_x'] = center_ss[0] - lk_femur[1, 0]
    params['ss_y'] = Y_INPUT
    params['ss_z'] = center_ss[1] - lk_femur[1, 1]
    params['ss_pitch'] = -pitch_ss * 180 / np.pi
    params['ss_l'] = l_ss / 2
    params['ss_w'] = W_INPUT / 2
    params['ss_h'] = H_LINK_TOTAL / 2

    lk_rocker = leg[7]
    l_rocker = helper.link_length(lk_rocker)
    pitch_rocker = helper.link_angle(lk_rocker)
    center_rocker = helper.transform_points(
        pitch_rocker, lk_rocker[0, 0], lk_rocker[0, 1],
        np.array([[l_rocker / 2, -H_LINK_OFFSET]]),
    )[0]
    params['rocker_x'] = center_rocker[0] - lk_femur[1, 0]
    params['rocker_z'] = center_rocker[1] - lk_femur[1, 1]
    params['rocker_pitch'] = -pitch_rocker * 180 / np.pi
    params['rocker_l'] = l_rocker / 2
    params['rocker_w'] = W_LINK / 2
    params['rocker_h'] = H_LINK_TOTAL / 2

    params['rocker_joint_x'] = lk_rocker[1, 0] - lk_femur[1, 0]
    params['rocker_joint_z'] = lk_rocker[1, 1] - lk_femur[1, 1]

    lk_coupler = np.array([leg[6, 1, :], leg[9, 1, :]])
    l_coupler = helper.link_length(lk_coupler) - L_FOOT
    pitch_coupler = helper.link_angle(lk_coupler)
    center_coupler = helper.transform_points(
        pitch_coupler, lk_coupler[0, 0], lk_coupler[0, 1],
        np.array([[l_coupler / 2, H_LINK_OFFSET]]),
    )[0]
    params['coupler_x'] = center_coupler[0] - lk_femur[1, 0]
    params['coupler_z'] = center_coupler[1] - lk_femur[1, 1]
    params['coupler_pitch'] = -pitch_coupler * 180 / np.pi
    params['coupler_l'] = l_coupler / 2
    params['coupler_w'] = W_LINK / 2
    params['coupler_h'] = H_LINK_TOTAL / 2

    center_foot = helper.transform_points(
        pitch_coupler, lk_coupler[0, 0], lk_coupler[0, 1],
        np.array([[l_coupler + L_FOOT / 2, H_LINK_OFFSET]]),
    )[0]
    params['foot_x'] = center_foot[0] - lk_femur[1, 0]
    params['foot_z'] = center_foot[1] - lk_femur[1, 1]
    params['foot_pitch'] = -pitch_coupler * 180 / np.pi
    params['foot_l'] = L_FOOT / 2
    params['foot_w'] = W_LINK / 2
    params['foot_h'] = H_LINK_TOTAL / 2

    params['coupler_joint_x'] = lk_coupler[0, 0] - lk_femur[1, 0]
    params['coupler_joint_z'] = lk_coupler[0, 1] - lk_femur[1, 1]

    params['crank_anchor_x'] = leg[6, 0, 0] - lk_femur[1, 0]
    params['crank_anchor_z'] = leg[6, 0, 1] - lk_femur[1, 1]

    lk_ps = leg[2]
    l_ps = helper.link_length(lk_ps)
    pitch_ps = helper.link_angle(lk_ps)
    center_ps = helper.transform_points(
        pitch_ps, lk_ps[0, 0], lk_ps[0, 1],
        np.array([[l_ps / 2, H_LINK_OFFSET]]),
    )[0]
    params['ps_x'] = center_ps[0] - lk_femur[1, 0]
    params['ps_y'] = Y_PS
    params['ps_z'] = center_ps[1] - lk_femur[1, 1]
    params['ps_pitch'] = -pitch_ps * 180 / np.pi
    params['ps_l'] = l_ps / 2
    params['ps_w'] = W_LINK / 2
    params['ps_h'] = H_LINK_TOTAL / 2

    params['ps_joint_x'] = lk_ps[1, 0] - lk_femur[1, 0]
    params['ps_joint_z'] = lk_ps[1, 1] - lk_femur[1, 1]

    lk_ps_coupler = leg[1]
    l_ps_coupler = helper.link_length(lk_ps_coupler)
    pitch_ps_coupler = helper.link_angle(lk_ps_coupler)
    center_ps_coupler = helper.transform_points(
        pitch_ps_coupler, lk_ps_coupler[0, 0], lk_ps_coupler[0, 1],
        np.array([[l_ps_coupler / 2, H_LINK_OFFSET]]),
    )[0]
    params['ps_coupler_x'] = center_ps_coupler[0] - lk_femur[1, 0]
    params['ps_coupler_y'] = Y_PS
    params['ps_coupler_z'] = center_ps_coupler[1] - lk_femur[1, 1]
    params['ps_coupler_pitch'] = -pitch_ps_coupler * 180 / np.pi
    params['ps_coupler_l'] = l_ps_coupler / 2
    params['ps_coupler_w'] = W_LINK / 2
    params['ps_coupler_h'] = H_LINK_TOTAL / 2

    params['ps_coupler_joint_x'] = lk_ps_coupler[1, 0] - lk_femur[1, 0]
    params['ps_coupler_joint_z'] = lk_ps_coupler[1, 1] - lk_femur[1, 1]

    params['input_anchor_x'] = leg[1, 0, 0] - lk_femur[1, 0]
    params['input_anchor_z'] = leg[1, 0, 1] - lk_femur[1, 1]

    params['ss_k'] = result['ss']
    params['ps_k'] = result['ps']

    return params


def plot(result, leg_only=False):
    legs = result['legs']
    feet = result['feet']
    feet_ref = result['feet_ref']
    femur = result['femur']
    xc, yc = helper.points_center(np.concatenate([
        legs.reshape((-1, 2)),
        feet,
        feet_ref,
        femur
    ]))
    r = 0.1

    if not leg_only:
        plt.figure(figsize=(6.4, 6.4), dpi=100)
        plt.gcf().subplots_adjust(
            left=0, right=1, top=1, bottom=0,
            wspace=0, hspace=0
        )
    plt.plot(feet_ref[:, 0], feet_ref[:, 1], '.-b')
    plt.plot(feet[:, 0], feet[:, 1], '.-g')
    plt.plot(femur[:, 0], femur[:, 1], '.-k')
    helper.plot_linkage(legs[-1], ls='.--')
    helper.plot_linkage(legs[0])
    plt.xlim(xc - r, xc + r)
    plt.ylim(yc - r, yc + r)
    plt.xticks([])
    plt.yticks([])

    if not leg_only:
        plt.figure()
        plt.plot(result['inputs'], result['ps_torques'])
        plt.plot(result['inputs'], result['ps_torques_local'])
        plt.xlabel('Input (rad)')
        plt.ylabel('Torque (Nm)')
        plt.legend(['Input', 'PRBM Joint'])


if __name__ == '__main__':
    import re
    import mujoco
    import mujoco.viewer

    x = [
        0.06039, 0.02004, 0.03461, 0.04566, 0.03782,
        -0.88251,
        0.02719, 0.01083
    ]
    p = [0.04, 0.04, 0.50, 0.10, 1.00]
    result = sim(x, p)
    plot(result)

    model_params = {
        'gravity_z': 0,
        'hfield_texrepeat': 10,
        'hfield_nr': 500,
        'hfield_r': 5,
        'floor_x': 0,
        'floor_y': 0,
        'floor_z': 0,
        'floor_roll': 0,
        'floor_pitch': 0,
        'floor_yaw': 0,
        'body_x': 0,
        'body_y': 0,
        'body_z': 0.2,
        'body_roll': 0,
        'body_pitch': 0,
        'body_yaw': 0,
        'friction_tan': 0.7,
        'servo_k': 1.8221793693748647,
        'servo_b': 0.05077375928980776,
        'servo_I': 0.0005170412918172306,
        'servo_f': 0.02229709671396319 * 0,
        'servo_tmax': 0.5771893852055175,
        'ss_b': 0,
        'ps_b': 0,
    }
    model_params = {**model_params, **mj_params(x, p)}

    with open(os.path.join('rl', 'model.xml'), 'r') as file:
        model_string = file.read()
    for k, v in model_params.items():
        model_string = re.sub(r'\b%s\b' % k, str(v), model_string)
    m = mujoco.MjModel.from_xml_string(model_string, {})
    d = mujoco.MjData(m)

    max_steps = round(100 / m.opt.timestep)
    inputs_mj = []
    ps_torques_mj = []
    ps_torques_local_mj = []
    for i in range(max_steps):
        input = p[2] * i / max_steps
        d.ctrl = np.array([0, input, 0, 0, 0, 0, 0, 0])
        mujoco.mj_step(m, d)
        inputs_mj.append(d.joint('knee_fl').qpos[0])
        ps_torques_mj.append(d.actuator('knee_fl').force[0])
        ps_torques_local_mj.append(-d.joint('ps_fl').qpos[0] * result['ps'])

    plt.figure(2)
    plt.plot(inputs_mj, ps_torques_mj, '--', color='C0')
    plt.plot(inputs_mj, ps_torques_local_mj, '--', color='C1')
    l1 = plt.legend(plt.gca().get_lines()[0:4:2], ['Model', 'MuJoCo'])
    plt.legend(['Input', 'PRBM Joint'], loc='lower right')
    plt.gca().add_artist(l1)
    plt.show()

    mujoco.viewer.launch(m, d)
