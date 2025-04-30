import sys
import os
import time
from multiprocessing import Pool, set_start_method
import numpy as np
import matplotlib.pyplot as plt
import torch
from rl.agent import Agent
from rl.env import Env
from rl.analyze import key_metrics


def eval_single_leg_fully(leg_index, num_trials=100):
    env = Env()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(
        os.path.join('data', 'rl_checkpoint.pt'),
        map_location=device, weights_only=False
    )
    agent = Agent(
        np.array(env.observation_space.shape).prod(),
        np.array(env.action_space.shape).prod()
    ).to(device)
    agent.load_state_dict(checkpoint['agent'])
    agent.eval()

    index_start = 100
    num_steps = 200

    h_nrow = 4
    h_ncol = 2
    v, w = np.meshgrid(
        np.arange(-h_nrow, h_nrow + 1),
        np.arange(-h_ncol, h_ncol + 1)
    )
    cells = np.array([v.flatten(), w.flatten()]).T
    metrics = []

    for cell in cells:
        metric = []
        for _ in range(num_trials):
            env.curriculum_cells = [[cell]]
            obs, info = env.reset(options={'leg_index': leg_index})
            states = []
            for _ in range(num_steps):
                obs = torch.Tensor(obs).to(device)
                with torch.no_grad():
                    action = agent.get_deterministic_action(obs)
                obs, reward, terminated, truncated, info = env.step(
                    action.cpu().numpy()
                )
                states.append(env.get_states())
                if terminated or truncated:
                    break
            env.close()
            states = np.array(states)
            states = states[index_start:, :]
            t = states[:, 0]
            q = states[:, 3:11]
            dq = states[:, 11:19]
            g = states[:, 19:22]
            w = states[:, 22:25]
            a = states[:, 25:33]
            v = states[:, 36:39]
            tau = states[:, 39:47]

            p = np.sum((tau * dq).clip(min=0), axis=1)
            vx = v[:, 0]
            m = env.model.body('body').subtreemass[0]
            cot = np.mean(p) / m / 9.81 / np.abs(np.mean(vx))
            score = np.mean(np.array(env.rewards[0])[index_start:])
            tau = np.sum(np.abs(tau), axis=1)

            v_cmd = np.array([env.vx_cmd, 0, 0])
            w_cmd = np.array([0, 0, env.wz_cmd])
            metric.append([
                score,
                cot,
                *np.sqrt(np.mean((v - v_cmd)**2, axis=0)),  # RMSE
                *np.mean(np.abs(v - v_cmd), axis=0),  # MAE
                *np.abs(np.mean(v, axis=0) - v_cmd),  # Error of mean
                *np.std(v, axis=0),
                *np.sqrt(np.mean((w - w_cmd)**2, axis=0)),
                *np.mean(np.abs(w - w_cmd), axis=0),
                *np.abs(np.mean(w, axis=0) - w_cmd),
                *np.std(w, axis=0),
                np.mean(tau),
                np.std(tau)
            ])

        metric = np.mean(metric, axis=0)
        metrics.append(metric)

    return {
        'index': leg_index,
        'params': env.leg_params[leg_index],
        'shapes': env.leg_shapes[leg_index],
        'mass': m,
        'cells': cells,
        'metrics': metrics,
    }


def eval_single_leg(leg_index, num_trials=100):
    env = Env()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(
        os.path.join('data', 'rl_checkpoint.pt'),
        map_location=device, weights_only=False
    )
    agent = Agent(
        np.array(env.observation_space.shape).prod(),
        np.array(env.action_space.shape).prod()
    ).to(device)
    agent.load_state_dict(checkpoint['agent'])
    agent.eval()

    index_start = 100
    num_steps = 200

    candidates = [[0, 0]]
    valids = []
    metrics = []

    while len(candidates) > 0:
        cell = candidates.pop(0)
        metric = []
        for _ in range(num_trials):
            env.curriculum_cells = [[cell]]
            obs, info = env.reset(options={'leg_index': leg_index})
            states = []
            for _ in range(num_steps):
                obs = torch.Tensor(obs).to(device)
                with torch.no_grad():
                    action = agent.get_deterministic_action(obs)
                obs, reward, terminated, truncated, info = env.step(
                    action.cpu().numpy()
                )
                states.append(env.get_states())
                if terminated or truncated:
                    break
            env.close()
            states = np.array(states)
            states = states[index_start:, :]
            t = states[:, 0]
            q = states[:, 3:11]
            dq = states[:, 11:19]
            g = states[:, 19:22]
            w = states[:, 22:25]
            a = states[:, 25:33]
            v = states[:, 36:39]
            tau = states[:, 39:47]

            p = np.sum((tau * dq).clip(min=0), axis=1)
            vx = v[:, 0]
            m = env.model.body('body').subtreemass[0]
            cot = np.mean(p) / m / 9.81 / np.abs(np.mean(vx))
            score = np.mean(np.array(env.rewards[0])[index_start:])
            tau = np.sum(np.abs(tau), axis=1)

            v_cmd = np.array([env.vx_cmd, 0, 0])
            w_cmd = np.array([0, 0, env.wz_cmd])
            metric.append([
                score,
                cot,
                *np.sqrt(np.mean((v - v_cmd)**2, axis=0)),  # RMSE
                *np.mean(np.abs(v - v_cmd), axis=0),  # MAE
                *np.abs(np.mean(v, axis=0) - v_cmd),  # Error of mean
                *np.std(v, axis=0),
                *np.sqrt(np.mean((w - w_cmd)**2, axis=0)),
                *np.mean(np.abs(w - w_cmd), axis=0),
                *np.abs(np.mean(w, axis=0) - w_cmd),
                *np.std(w, axis=0),
                np.mean(tau),
                np.std(tau)
            ])

        metric = np.mean(metric, axis=0)

        if metric[0] > env.curriculum_score_th:
            valids.append(cell)
            metrics.append(metric)
            for dx, dy in [(0, 1), (1, 0,), (0, -1), (-1, 0)]:
                neighbour_cell = [cell[0] + dx, cell[1] + dy]
                if (
                    neighbour_cell not in candidates and
                    neighbour_cell not in valids
                ):
                    candidates.append(neighbour_cell)

    return {
        'index': leg_index,
        'params': env.leg_params[leg_index],
        'shapes': env.leg_shapes[leg_index],
        'mass': m,
        'cells': valids,
        'metrics': metrics,
    }


def eval_num_trials(num_trials):
    t0 = time.time()
    r = eval_single_leg(280, num_trials=num_trials)
    t = time.time() - t0
    return num_trials, r, t


if __name__ == "__main__":
    if 'l' in sys.argv:
        leg_indices = [int(i) for i in sys.argv[2].split(',')]
        if len(sys.argv) > 3:
            r_target = np.load(sys.argv[3], allow_pickle=True)
            name = f'{int(time.time())}_eval.npy'
            folder_name = os.path.join('logs', 'rl')
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
        else:
            r_target = None

        th = Env().curriculum_score_th
        with Pool(len(leg_indices)) as p:
            for i, r in enumerate(p.imap(eval_single_leg, leg_indices)):
                vx, wz, cot = key_metrics(r)
                print(f'{leg_indices[i]}: {vx:.3f}, {wz:.3f}, {cot:.3f}')

                if r_target is not None:
                    assert r_target[r['index']]['index'] == r['index']
                    r_target[r['index']] = r
                    np.save(os.path.join(folder_name, name), r_target)

    if 'f' in sys.argv:
        leg_indices = [int(i) for i in sys.argv[2].split(',')]
        name = f'{int(time.time())}_eval_fully.npy'
        folder_name = os.path.join('logs', 'rl')
        results = []
        th = Env().curriculum_score_th
        with Pool(len(leg_indices)) as p:
            for i, r in enumerate(p.imap(eval_single_leg_fully, leg_indices)):
                vx, wz, cot = key_metrics(r)
                print(f'{leg_indices[i]}: {vx:.3f}, {wz:.3f}, {cot:.3f}')
                results.append(r)
                np.save(
                    os.path.join(folder_name, name),
                    results
                )

    if 'n' in sys.argv:
        th = Env().curriculum_score_th
        num_trials = (np.arange(10) + 1) * 100
        areas = []
        times = []
        set_start_method('spawn')
        with Pool(len(num_trials)) as p:
            for n, r, t in p.imap(eval_num_trials, num_trials):
                areas.append(
                    len(r['cells']) +
                    (np.mean(np.array(r['metrics'])[:, 0]) - th)
                )
                times.append(t)
                print(n, areas[-1], t)
        plt.subplot(211)
        plt.plot(num_trials, areas)
        plt.ylabel('Area')
        plt.subplot(212)
        plt.plot(num_trials, times)
        plt.ylabel('Time (s)')
        plt.xlabel('Number of trials')
        plt.show()

    if 'a' in sys.argv:
        name = f'{int(time.time())}_eval.npy'
        folder_name = os.path.join('logs', 'rl')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        num_legs = Env().leg_params.shape[0]
        results = []
        set_start_method('spawn')
        t0 = time.time()
        with Pool(42) as p:
            for r in p.imap(eval_single_leg, np.arange(num_legs)):
                results.append(r)
                vx, wz, cot = key_metrics(r)
                print(
                    f'time: {time.time() - t0:.0f}, '
                    f'index: {r["index"]}, '
                    f'vx: {vx:.3f}, '
                    f'wz: {wz:.3f}, '
                    f'cot: {cot:.3f}'
                )
                np.save(
                    os.path.join(folder_name, name),
                    results
                )
