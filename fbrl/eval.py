import os
import time
from multiprocessing import Pool, set_start_method
import numpy as np
import torch
from rl.agent import Agent
from fbrl.env import Env


def eval_leg(leg_index):
    env = Env()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(
        os.path.join('data', 'fbrl_checkpoint.pt'),
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
    num_trials = 100

    cells = np.argwhere(checkpoint['curriculum_cells'][leg_index])
    metrics = []

    t0 = time.time()
    for i, cell in enumerate(cells):
        metric = []
        n = 0
        while n < num_trials:
            obs, info = env.reset(options={
                'leg_index': leg_index,
                'curriculum_cell': np.array([leg_index, *cell])
            })

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

            if len(states) < num_steps:
                print('Early break')
                continue

            n += 1
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
            vx_mean = np.abs(np.mean(vx))
            if vx_mean > 0.1:
                cot = np.mean(p) / m / 9.81 / vx_mean
            else:
                cot = np.nan
            score = np.mean(np.array(env.rewards[0])[index_start:])
            metric.append([
                score,
                cot,
                *np.mean(np.abs(tau), axis=0),
                *np.std(np.abs(tau), axis=0),
                *np.mean(q, axis=0),
                *np.std(q, axis=0),
            ])
        metrics.append(np.nanmean(metric, axis=0))
        print(
            f'leg: {leg_index}, '
            f'cell: {i + 1}/{len(cells)}, '
            f'CPS: {(i + 1) / (time.time() - t0):.3f}'
        )

    leg_index_front, leg_index_rear = env.leg_combs[leg_index]

    return {
        'leg_comb_index': leg_index,
        'leg_index_front': leg_index_front,
        'leg_index_rear': leg_index_rear,
        'leg_param_front': env.leg_params[leg_index_front],
        'leg_shape_front': env.leg_shapes[leg_index_front],
        'leg_param_rear': env.leg_params[leg_index_rear],
        'leg_shape_rear': env.leg_shapes[leg_index_rear],
        'mass': m,
        'cells': cells,
        'metrics': metrics,
    }


if __name__ == "__main__":
    name = f'{int(time.time())}_eval.npy'
    folder_name = os.path.join('logs', 'fbrl')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    num_legs = len(Env().leg_combs)
    results = []
    set_start_method('spawn')
    t0 = time.time()
    with Pool(num_legs) as p:
        for r in p.imap(eval_leg, np.arange(num_legs)):
            results.append(r)
            np.save(
                os.path.join(folder_name, name),
                results
            )
