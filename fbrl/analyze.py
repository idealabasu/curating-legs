import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from fbrl.env import Env


def results():
    rs = np.load(os.path.join('data', 'fbrl_eval.npy'), allow_pickle=True)
    env = Env()

    # Make symmetric
    for r in rs:
        _metrics = np.array(r['metrics'])
        _cells = np.array(r['cells'])
        cells = []
        metrics = []
        for cell, metric in zip(_cells, _metrics):
            cell_m = np.array([
                2 * env.curriculum_x_max - cell[0],
                cell[1],
                2 * env.curriculum_fx_max - cell[2] - 1,
                cell[3]
            ])
            symmetric = np.any(np.all(cell_m == _cells, axis=1))
            if symmetric or True:
                cells.append(cell)
                metrics.append(metric)

        r['cells'] = np.array(cells)
        r['metrics'] = np.array(metrics)

        for metric_index in range(2):
            heatmap_shape = [
                2 * env.curriculum_x_max + 1,
                2 * env.curriculum_z_max + 1,
                2 * env.curriculum_fx_max,
                2 * env.curriculum_fz_max
            ]
            heatmap = np.empty(heatmap_shape)
            heatmap[:] = np.nan
            for cell, metric in zip(r['cells'], r['metrics']):
                # if metric[metric_index] < env.curriculum_score_th:
                #     continue
                heatmap[*cell] = metric[metric_index]

            heatmap = np.transpose(heatmap, axes=[1, 3, 0, 2])
            heatmap = heatmap.reshape(
                heatmap_shape[1] * heatmap_shape[3], -1
            )
            heatmap = np.flip(heatmap, axis=0)
            r[f'heatmap_{metric_index}'] = heatmap

    return rs


if __name__ == '__main__':
    rs = results()
    env = Env()

    stats = []
    for r in rs:
        leg_comb_index = r['leg_comb_index']
        leg_index_front = r['leg_index_front']
        leg_index_rear = r['leg_index_rear']
        stat = []
        for metric_index in range(2):
            heatmap = r[f'heatmap_{metric_index}']

            if metric_index == 0:
                metric_fn = np.nansum
            else:
                metric_fn = np.nanmean
            score = metric_fn(heatmap)
            diff_anchor = (
                metric_fn(heatmap[:, :12]) -
                metric_fn(heatmap[:, 12:])
            ) / (
                0.5 * metric_fn(heatmap[:, :12]) +
                0.5 * metric_fn(heatmap[:, 12:])
            )
            diff_heading = (
                metric_fn([heatmap[:, 0:4], heatmap[:, 8:12], heatmap[:, 16:20]]) -
                metric_fn(
                    [heatmap[:, 4:8], heatmap[:, 12:16], heatmap[:, 20:24]])
            ) / (
                0.5 * metric_fn(
                    [heatmap[:, 0:4],
                     heatmap[:, 8:12],
                     heatmap[:, 16:20]]
                ) +
                0.5 * metric_fn(
                    [heatmap[:, 4:8],
                     heatmap[:, 12:16],
                     heatmap[:, 20:24]]
                )
            )
            stat.append(score)
            stat.append(diff_anchor)
            stat.append(diff_heading)
        stats.append(stat)
    stats = np.array(stats).T

    sym_indices = np.array([0, 4, 7, 10, 12, 14])
    asym_indices = np.array([1, 2, 3, 5, 6, 8, 9, 11, 13])

    print('Total r_v: ', end='')
    print(', '.join([f'{s:.0f}' for s in stats[0, :]]))
    print(f'design: {np.argmax(stats[0, :])}, {np.argmin(stats[0, :])}')

    # print('r_v anchor diff: ', end='')
    # print(', '.join([f'{s:.1f}' for s in stats[1, :]]))
    # print(
    #     f'sym mean: {np.mean(np.abs(stats[1, sym_indices])):.2f}, '
    #     f'asym mean: {np.mean(np.abs(stats[1, asym_indices])):.2f}, '
    #     f'design: {np.argmax(np.abs(stats[1, :]))}'
    # )

    print('r_v heading diff: ', end='')
    print(', '.join([f'{s:.2f}' for s in stats[2, :]]))
    print(
        f'sym mean: {np.mean(np.abs(stats[2, sym_indices])):.2f}, '
        f'asym mean: {np.mean(np.abs(stats[2, asym_indices])):.2f}, '
        f'design: {np.argmax(np.abs(stats[2, :]))}'
    )

    print('Average COT: ', end='')
    print(', '.join([f'{s:.2f}' for s in stats[3, :]]))
    print(f'design: {np.argmin(stats[3, :])}')

    # print('COT anchor diff: ', end='')
    # print(', '.join([f'{s:.1f}' for s in stats[4, :]]))
    # print(
    #     f'sym mean: {np.mean(np.abs(stats[4, sym_indices])):.2f}, '
    #     f'asym mean: {np.mean(np.abs(stats[4, asym_indices])):.2f}, '
    #     f'design: {np.argmax(np.abs(stats[4, :]))}'
    # )

    print('COT heading diff: ', end='')
    print(', '.join([f'{s:.2f}' for s in stats[5, :]]))
    print(
        f'sym mean: {np.mean(np.abs(stats[5, sym_indices])):.2f}, '
        f'asym mean: {np.mean(np.abs(stats[5, asym_indices])):.2f}, '
        f'design: {np.argmax(np.abs(stats[5, :]))}'
    )

    for metric_index in range(2):
        v = []
        for r in rs:
            v.append(r['metrics'][:, metric_index])
        v = np.concatenate(v)
        vmin = np.nanmin(v)
        vmax = np.nanmax(v)

        h_pad = 0.25
        w_pad = 0.05
        w_border = 0.01
        h_cb = 0.6
        h_pad_cb = 0.05
        w_total = 5
        w = (w_total - w_border * 2 - w_pad * 4) / 5
        h_total = w * 3 + h_pad * 3 + h_cb + h_pad_cb
        plt.figure(figsize=(w_total, h_total))
        plt.subplots_adjust(
            left=w_border / w_total, right=1 - w_border / w_total,
            top=1, bottom=h_pad / h_total,
            wspace=w_pad / w, hspace=h_pad / w
        )
        for i, r in enumerate(rs):
            plt.subplot(3, 5, i + 1)
            c = plt.imshow(r[f'heatmap_{metric_index}'], vmin=vmin, vmax=vmax)
            pts = np.array([
                [4, 4],
                [4, 12],
                [4, 20],
                [12, 4],
                [12, 12],
                [12, 20],
                [20, 4],
                [20, 12],
                [20, 20]
            ]) - np.array([0.5, 0.5])
            plt.scatter(pts[:, 0], pts[:, 1], marker='o', color='k', s=1)
            # plt.axis('off')
            plt.plot([3.5, 3.5], [-0.5, 23.5], '--k', lw=0.4)
            plt.plot([11.5, 11.5], [-0.5, 23.5], '--k', lw=0.4)
            plt.plot([19.5, 19.5], [-0.5, 23.5], '--k', lw=0.4)
            plt.plot([7.5, 7.5], [-0.5, 23.5], '-k', lw=0.4)
            plt.plot([15.5, 15.5], [-0.5, 23.5], '-k', lw=0.4)
            plt.plot([23.5, 23.5], [-0.5, 23.5], '-k', lw=0.4)
            plt.plot([-0.5, 23.5], [7.5, 7.5], '-k', lw=0.4)
            plt.plot([-0.5, 23.5], [15.5, 15.5], '-k', lw=0.4)
            plt.plot([-0.5, 23.5], [23.5, 23.5], '-k', lw=0.4)
            plt.xticks([])
            plt.yticks([])

            # if r['leg_index_front'] == r['leg_index_rear']:
            #     label = f'$\\underline{{\\boldsymbol{{\delta_{{{i}}}}}}}$'
            # else:
            #     label = f'$\\boldsymbol{{\delta_{{{i}}}}}$'
            plt.text(
                0.5, -0.05,
                f'$\\boldsymbol{{\\delta_{{{i}}}}}$',
                ha='center', va='top', transform=plt.gca().transAxes
            )
        plt.colorbar(
            c, ax=plt.gcf().get_axes(),
            location='top', pad=h_pad_cb / h_total, fraction=h_cb / h_total,
            aspect=30, shrink=0.6,
            ticks=np.linspace(vmin, vmax, 5),
            format='%.1f',
            label=['Velocity reward', 'COT'][metric_index]
        )
        name = ['rv', 'cot'][metric_index]
    plt.show()
