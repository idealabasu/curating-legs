import numpy as np
import matplotlib.pyplot as plt


def angle_between_links(lk1, lk2):
    v1 = lk1[1, :] - lk1[0, :]
    v2 = lk2[1, :] - lk2[0, :]

    return np.arctan2(
        v1[0] * v2[1] - v1[1] * v2[0],
        np.dot(v1, v2)
    )


def link_angle(lk):
    p1 = lk[0, :]
    p2 = lk[1, :]
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])


def link_length(lk):
    return np.linalg.norm(lk[1, :] - lk[0, :])


def transform_points(theta, x, y, pts, inverse=False):
    t = np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0, 0, 1]
    ])
    if inverse:
        t = np.linalg.inv(t)

    pts_tf = (
        t @ np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1).T
    ).T[:, :2]

    return pts_tf


def points_center(pts):
    return (
        (np.amax(pts[:, 0]) + np.amin(pts[:, 0])) / 2,
        (np.amax(pts[:, 1]) + np.amin(pts[:, 1])) / 2
    )


def plot_linkage(lkg, ls='.-'):
    for i, lk in enumerate(lkg):
        plt.plot(
            lk[:, 0], lk[:, 1], ls,
            color='r' if i == 0 else 'k'
        )
    plt.axis('scaled')
