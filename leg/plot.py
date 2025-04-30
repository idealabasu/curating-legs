import os
import sys
import numpy as np
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
from leg import opt, model, helper

file_name = os.path.join('data', 'leg_checkpoint.npy')


def plot_leg(leg_index):
    checkpoint = np.load(file_name, allow_pickle=True).item()

    p = np.array(checkpoint['legs'])[leg_index, 8:]
    print(f'travel offset: {p[0]:.2f}')
    print(f'travel length: {p[1]:.2f}')
    print(f'input range: {p[2]:.2f}')
    print(f'parallel sitffness: {p[3]:.2f}')
    print(f'series sitffness: {p[4]:.2f}')

    x = np.array(checkpoint['legs'])[leg_index, :8]
    print(f'l_ab (crank): {x[0]:.5f}')
    print(f'l_bc (coupler): {x[1]:.5f}')
    print(f'l_cd (rocker): {x[2]:.5f}')
    print(f'l_ad (ground): {x[3]:.5f}')
    print(f'l_be (copupler extension): {x[4]:.5f}')
    print(f'input_offset: {x[5]:.5f}')
    print(f'l_ps: {x[6]:.5f}')
    print(f'l_ss: {x[7]:.5f}')

    result = model.sim(x, p)
    l_ao = helper.link_length(result['femur'])
    t_ao = helper.link_angle(result['femur'])
    t_oad = helper.angle_between_links(result['femur'], result['legs'][0][8])
    l_fg = helper.link_length(result['legs'][0][1])
    print(f'l_ao (femur_length): {l_ao:.5f}')
    print(f't_ao (femur_angle): {t_ao:.5f}')
    print(f't_oad (femur_ground_angle): {t_oad:.5f}')
    print(f'l_fg (ps_coupler_length): {l_fg:.5f}')

    if p[3] == 0:
        result['legs'] = np.array([
            leg[np.r_[0, 4:len(leg)]]
            for leg in result['legs']
        ])

    plt.figure(figsize=(4.8, 4.8), dpi=100)
    plt.gcf().subplots_adjust(
        left=0.002, right=0.998, top=0.998, bottom=0.002,
        wspace=0, hspace=0
    )
    model.plot(result, leg_only=True)
    plt.text(
        0.5, 0.02,
        ', '.join([f'{c:.2f}' for c in p]),
        ha='center', va='bottom', transform=plt.gca().transAxes
    )


def plot_centroids():
    checkpoint = np.load(file_name, allow_pickle=True).item()
    params = np.array(checkpoint['legs'])[:, 8:]
    params_min = np.amin(params, axis=0)
    params_max = np.amax(params, axis=0)
    params_n = (
        (params - params_min) / (params_max - params_min)
    )
    centroids_n, label = kmeans2(params_n, 10, iter=10, minit='points', seed=0)
    centroids = (
        centroids_n * (params_max - params_min) +
        params_min
    )
    print('counts for each centroid', np.bincount(label))

    plt.figure(figsize=(16, 6.4), dpi=100)
    plt.gcf().subplots_adjust(
        left=0.001, right=0.999, top=0.998, bottom=0.002,
        wspace=0, hspace=0
    )
    for j, p in enumerate(centroids):
        for i in range(10):
            print(f'trial: {i}')
            try:
                r = opt.optimize(p)
                cost, constraints = opt.obj_with_constraints(r.x, p)
                valid = np.sum(constraints) == 0
            except AssertionError:
                valid = False

            if valid:
                break

        assert valid
        plt.subplot(2, 5, j + 1)
        opt.obj_with_constraints(r.x, p, plot='leg_only')


def plot_space():
    checkpoint = np.load(file_name, allow_pickle=True).item()
    params = np.array(checkpoint['legs'])[:, 8:]
    params_min = np.amin(params, axis=0)
    params_max = np.amax(params, axis=0)
    params_n = (
        (params - params_min) / (params_max - params_min)
    )
    centroids_n, label = kmeans2(params_n, 10, iter=10, minit='points', seed=0)
    centroids = (
        centroids_n * (params_max - params_min) +
        params_min
    )
    print('counts for each centroid', np.bincount(label))

    fig, axes = plt.subplots(
        1, 5,
        sharey=True, sharex=False,
        figsize=(15, 3)
    )
    plt.subplots_adjust(
        left=0.045, right=0.998, top=0.998, bottom=0.15,
        wspace=0.1, hspace=0
    )
    for i, (param, label) in enumerate(zip(
        params.T,
        [
            'Travel offset (m)',
            'Travel length (m)',
            'Input range (rad)',
            'Parallel stiffness (Nm/rad)',
            'Series stiffness (Nm/rad)'
        ],
    )):
        u, uc = np.unique(param.round(decimals=4), return_counts=True)
        axes[i].plot(u, uc, marker='.')
        axes[i].set_xlabel(label)
    axes[0].set_ylabel('Count')

    (
        travel_offset,
        travel_length,
        input_range,
        _parallel_stiffness,
        _series_stiffness
    ) = params.T

    leg_length = travel_offset + travel_length
    moment_arm = travel_length / input_range
    parallel_stiffness = _parallel_stiffness / moment_arm**2
    series_stiffness = _series_stiffness / moment_arm**2

    fig, axes = plt.subplots(
        1, 4,
        sharey=True, sharex=False,
        figsize=(12, 3)
    )
    plt.subplots_adjust(
        left=0.055, right=0.998, top=0.998, bottom=0.15,
        wspace=0.1, hspace=0
    )
    for i, (param, label) in enumerate(zip(
        [leg_length, moment_arm, series_stiffness, parallel_stiffness],
        [
            'Leg length (m)',
            'Moment arm (m)',
            'Series stiffness (N/m)',
            'Parallel stiffness (N/m)'
        ]
    )):
        u, uc = np.unique(param.round(decimals=4), return_counts=True)
        axes[i].plot(u, uc, marker='.')
        axes[i].set_xlabel(label)
    axes[0].set_ylabel('Count')

    for axes in [[0, 1, 2], [0, 3, 4], [1, 3, 4], [2, 3, 4]]:
        pts_dict = {}
        for param in params[:, axes]:
            param = tuple(param)
            if param not in pts_dict:
                pts_dict[param] = 0
            else:
                pts_dict[param] += 1
        pts_counts = np.array(list(pts_dict.values()))
        pts = np.array(list(pts_dict.keys()))

        fig = plt.figure(figsize=(6.4, 6.4))
        ax = fig.add_subplot(projection='3d')
        c = ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=pts_counts, depthshade=False
        )
        ax.scatter(
            centroids[:, axes[0]],
            centroids[:, axes[1]],
            centroids[:, axes[2]],
            s=50,
            marker='^',
            c='r',
            depthshade=False
        )
        ax.set_xticks(np.unique(pts[:, 0],))
        ax.set_yticks(np.unique(pts[:, 1],))
        ax.set_zticks(np.unique(pts[:, 2],))
        labels = [
            'Travel Offset (m)',
            'Travel Length (m)',
            'Input Range (rad)',
            'Parallel Stiffness (Nm/rad)',
            'Series Stiffness (Nm/rad)'
        ]
        ax.set_xlabel(labels[axes[0]])
        ax.set_ylabel(labels[axes[1]])
        ax.set_zlabel(labels[axes[2]])
        ax.set_box_aspect([1, 1, 1])
        fig.colorbar(c, location='bottom', pad=0.1, fraction=0.1)
        plt.gcf().subplots_adjust(
            left=0.1, right=0.9, top=0.9, bottom=0.1,
            wspace=0, hspace=0
        )


if __name__ == '__main__':
    if sys.argv[1] == 'l':
        plot_leg(int(sys.argv[2]))
        plt.show()

    if sys.argv[1] == 's':
        plot_space()
        plt.show()

    if sys.argv[1] == 'c':
        plot_centroids()
        plt.show()
