import os
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from rl.env import Env
from leg import model


def sym():
    r = np.load(os.path.join('data', 'rl_eval.npy'), allow_pickle=True)
    asym_cell_counts = [0, 0, 0, 0]
    total_cell_count = 0

    for _r in r[:]:
        cells = _r['cells']
        metrics = _r['metrics']
        for cell in reversed(cells):
            has_sym = True
            for sign_x, sign_y in [[-1, 1], [1, -1], [-1, -1]]:
                sym_cell = [sign_x * cell[0], sign_y * cell[1]]
                if sym_cell not in cells:
                    has_sym = False
                    break

            if not has_sym:
                index = cells.index(cell)
                cells.pop(index)
                metrics.pop(index)

                if cell[0] > 0 and cell[1] > 0:
                    asym_cell_counts[0] += 1
                if cell[0] < 0 and cell[1] > 0:
                    asym_cell_counts[1] += 1
                if cell[0] < 0 and cell[1] < 0:
                    asym_cell_counts[2] += 1
                if cell[0] > 0 and cell[1] < 0:
                    asym_cell_counts[3] += 1

            total_cell_count += 1

    forward_count = (
        (asym_cell_counts[0] + asym_cell_counts[1]) / total_cell_count
    )
    backward_count = (
        (asym_cell_counts[2] + asym_cell_counts[3]) / total_cell_count
    )
    left_count = (
        (asym_cell_counts[1] + asym_cell_counts[2]) / total_cell_count
    )
    right_count = (
        (asym_cell_counts[0] + asym_cell_counts[3]) / total_cell_count
    )

    print(
        'Moving forward vs backward: '
        f'{forward_count:.3f}, {backward_count:.3f}'
    )
    print(f'Turning right vs left: {left_count:.3f}, {right_count:.3f}')

    return r


env = Env()
th = env.curriculum_score_th


def key_metrics(r):
    cells = np.array(r['cells'])
    metrics = np.array(r['metrics'])
    score = metrics[:, 0]

    vx = (
        np.sum(np.linalg.norm(env.curriculum_vx_step * cells * np.array([1, 0]), axis=1)) +
        0 * (np.mean(score - th))
    )
    wz = (
        np.sum(np.linalg.norm(env.curriculum_wz_step * cells * np.array([0, 1]), axis=1)) +
        0 * (np.mean(score) - th)
    )

    cot = metrics[:, 1]
    indices = np.abs(cells[:, 0]) > 0
    if np.sum(indices) == 0:
        cot = np.nan
    else:
        cot = np.mean(cot[indices])

    return vx, wz, cot


if __name__ == "__main__":
    r = sym()
    # r = np.load(os.path.join('logs', 'rl_eval.npy'), allow_pickle=True)
    d = []
    for _r in r:
        d.append([_r['index'], *_r['params'], *key_metrics(_r)])
    d = np.array(d)

    _params = d[:, 1:6]
    vx = d[:, 6]
    wz = d[:, 7]
    cot = d[:, 8]
    # cot = 1 / d[:, 8]
    # combined = vx + 1 * wz + 10 / cot

    p = 0.1
    indices = [
        np.nonzero(vx > np.amax(vx) * (1 - p))[0],
        np.nonzero(wz > np.amax(wz) * (1 - p))[0],
        np.nonzero(cot < np.amin(cot) * (1 + p))[0],
        # np.nonzero(cot > np.amax(cot) * (1 - p))[0],
        # np.nonzero(combined > np.amax(combined) * (1 - p))[0],
    ]
    # print(indices[0])
    # print(indices[1])
    # print(indices[2])
    # print(vx[indices[0]])
    # print(wz[indices[1]])
    # print(cot[indices[2]])

    _params_min = np.amin(_params, axis=0)
    _params_max = np.amax(_params, axis=0)
    indices_best = []
    for _indices in indices:
        # indices_best.append(np.random.choice(_indices))
        # continue

        _params_best = _params[_indices]
        # _params_best_min = np.amin(_params_best, axis=0)
        # _params_best_max = np.amax(_params_best, axis=0)
        _params_best_n = (
            (_params_best - _params_min) /
            (_params_max - _params_min)
        )
        dist = []
        for _param_n in _params_best_n:
            dist.append(np.mean(np.linalg.norm(
                _param_n - _params_best_n, axis=1
            )))
        indices_best.append(_indices[np.argsort(dist)[0]])

    travel_offset = d[:, 1]
    travel_length = d[:, 2]
    input_range = d[:, 3]
    _parallel_stiffness = d[:, 4]
    _series_stiffness = d[:, 5]
    leg_length = travel_offset + travel_length
    moment_arm = travel_length / input_range
    parallel_stiffness = _parallel_stiffness / moment_arm**2
    series_stiffness = _series_stiffness / moment_arm**2

    params = [
        leg_length,
        moment_arm,
        series_stiffness,
        parallel_stiffness
    ]
    metrics = [vx, wz, cot]
    ylabels = ['Vx', 'Wz', 'CoT']
    xlabels = [
        'Leg length (m)',
        'Moment arm (m)',
        'Series stiffness (N/m)',
        'Parallel stiffness (N/m)'
    ]

    legs = np.array(np.load(
        os.path.join('data', 'leg_checkpoint.npy'),
        allow_pickle=True
    ).item()['legs'])
    assert len(legs) == len(_params)
    w = 3
    fig, axes = plt.subplots(
        1, 3,
        figsize=(w * 3, w), dpi=100
    )
    plt.subplots_adjust(
        left=0.002, right=0.998, top=0.997, bottom=0.002,
        wspace=0, hspace=0
    )
    for i, index in enumerate(indices_best):
        m = [metric[index] for metric in metrics]
        p = legs[index, 8:]
        x = legs[index, :8]
        result = model.sim(x, p)
        if p[3] == 0:
            result['legs'] = np.array([
                leg[np.r_[0, 4:len(leg)]]
                for leg in result['legs']
            ])

        plt.sca(axes[i])
        model.plot(result, leg_only=True)
        plt.text(
            0.98, 0.02,
            ', '.join([f'{_p:.2f}' for _p in p]),
            ha='right', va='bottom', transform=plt.gca().transAxes
        )
        plt.text(
            0.98, 0.98,
            ', '.join([f'{_m:3.3g}' for _m in m]),
            ha='right', va='top', transform=plt.gca().transAxes
        )
        plt.text(
            0.02, 0.98,
            f'{index}',
            ha='left', va='top', transform=plt.gca().transAxes
        )

        _r = r[index]
        cells = np.array(_r['cells'])
        score = np.array(_r['metrics'])[:, 0]

        h_nrow = 5
        h_ncol = 5
        nrow = h_nrow * 2 + 1
        ncol = h_ncol * 2 + 1
        map = np.zeros((nrow, ncol))
        for cell, _score in zip(cells, score):
            map[h_nrow - cell[0], cell[1] + h_ncol] = _score

        inset_ax = inset_axes(
            axes[i], width='30%', height='30%', loc='lower left'
        )
        # inset_ax.axis('off')
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.imshow(map, cmap='Greys', vmin=0, vmax=2)

    fig, axes = plt.subplots(
        3, 4,
        sharey='row', sharex='col',
        figsize=(10, 4), dpi=100
    )
    plt.subplots_adjust(
        left=0.055, right=0.998, top=0.998, bottom=0.11,
        wspace=0, hspace=0
    )
    for i, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        for j, (param, xlabel) in enumerate(zip(params, xlabels)):
            _param = np.round(param, decimals=4)
            uparam, uparam_label = np.unique(_param, return_inverse=True)
            umetric = [metric[uparam_label == k] for k in range(len(uparam))]

            umetric_mean = np.array([
                np.nanmean(_metric) for _metric in umetric
            ])
            axes[i][j].plot(uparam, umetric_mean, color=f'C{i}')

            umetric_max = np.array([np.nanmax(_metric) for _metric in umetric])
            umetric_min = np.array([np.nanmin(_metric) for _metric in umetric])
            if j == 2:
                _, res = find_peaks(umetric_max, plateau_size=(None, None))
                max_indices = np.concatenate([
                    np.arange(l, r + 1)
                    for l, r in zip(res['left_edges'], res['right_edges'])
                ])
                max_indices = np.concatenate(
                    [[0], max_indices, [len(umetric_max) - 1]]
                )
                _, res = find_peaks(-umetric_min, plateau_size=(None, None))
                min_indices = np.concatenate([
                    np.arange(l, r + 1)
                    for l, r in zip(res['left_edges'], res['right_edges'])
                ])
                min_indices = np.concatenate(
                    [[0], min_indices, [len(umetric_min) - 1]]
                )
                _uparam = np.linspace(uparam[0], uparam[-1], 1000)
                umetric_max = np.interp(
                    _uparam, uparam[max_indices], umetric_max[max_indices]
                )
                umetric_min = np.interp(
                    _uparam, uparam[min_indices], umetric_min[min_indices]
                )
                uparam = _uparam
            axes[i][j].fill_between(
                uparam, umetric_min, umetric_max,
                alpha=0.5, color=f'C{i}', ec='none'
            )
            # axes[i][j].plot(param, metric, '.', color=f'C{i}', markersize=1)
            for n, _indices in enumerate(indices):
                axes[i][j].plot(
                    param[_indices], metric[_indices], '.',
                    color=f'C{n}', markersize=3
                )
            for n, _indices in enumerate(indices_best):
                axes[i][j].plot(
                    param[_indices], metric[_indices], 'x',
                    color=f'C{n}', markersize=8
                )

            if i == 2:
                axes[i][j].set_xlabel(xlabel)
        axes[i][0].set_ylabel(ylabel)
    axes[2][1].set_xticks(np.arange(4) * 0.01 + 0.03)
    # axes[2][0].set_ylim(1.5, 13)
    plt.show()
