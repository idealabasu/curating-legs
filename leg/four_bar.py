import matplotlib.pyplot as plt
import numpy as np
from leg import helper


def solve(l_ab, l_bc, l_cd, l_ad, theta_bad, crossed):
    l_bd = np.sqrt(l_ad**2 + l_ab**2 - 2 * l_ad * l_ab * np.cos(theta_bad))

    cos_theta_bcd = (l_bc**2 + l_cd**2 - l_bd**2) / (2 * l_bc * l_cd)

    # if np.abs(np.abs(cos_theta_bcd) - 1) < 1e-3:
    #     cos_theta_bcd = 1

    assert np.abs(cos_theta_bcd) <= 1 and not np.isnan(cos_theta_bcd)

    sin_theta_bcd = np.sqrt(1 - cos_theta_bcd**2)
    if crossed:
        sin_theta_bcd = -sin_theta_bcd

    theta_bdc = np.arctan2(
        l_bc / l_bd * sin_theta_bcd,
        (l_bd**2 + l_cd**2 - l_bc**2) / (2 * l_bd * l_cd)
    )

    theta_adb = np.arctan2(
        l_ab / l_bd * np.sin(theta_bad),
        (l_bd**2 + l_ad**2 - l_ab**2) / (2 * l_bd * l_ad)
    )

    theta_adc = theta_adb + theta_bdc

    pt_abcd = np.array([
        [0, 0],
        [l_ab * np.cos(theta_bad), l_ab * np.sin(theta_bad)],
        [
            l_cd * np.cos(np.pi - theta_adc) + l_ad,
            l_cd * np.sin(np.pi - theta_adc)
        ],
        [l_ad, 0]
    ])

    lkg_abcd = np.array([
        pt_abcd[:2, :],
        pt_abcd[1:3, :],
        pt_abcd[2:4, :],
        pt_abcd[[0, 3], :],
    ])

    return lkg_abcd


def transmission_angle(lkg):
    angle = np.abs(helper.angle_between_links(lkg[1], lkg[2]))
    if angle > np.pi / 2:
        # map to 0 to np.pi/2
        angle = np.pi - angle
    return angle


def input_torque(tau, lkg):
    t_dcb = np.pi + helper.angle_between_links(lkg[1], lkg[2])
    assert np.abs(np.sin(t_dcb)) > 1e-3, 'Near singularity'

    l_cd = helper.link_length(lkg[2])
    f = tau / l_cd / np.sin(t_dcb)

    t_cba = np.pi + helper.angle_between_links(lkg[0], lkg[1])
    l_ab = helper.link_length(lkg[0])
    return f * np.sin(t_cba) * l_ab
