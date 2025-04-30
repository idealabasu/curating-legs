import os
import shutil
import argparse
import time
from collections import deque
import numpy as np
from leg import opt

PARAMS_START = np.array([0.04, 0.02, 0.3, 0, 0.4])
PARAMS_STEP = np.array([0.02, 0.02, 0.3, 0.1, 0.4])
CANDIDATE_START = [0, 0, 0, 0, 0]
MAX_STEPS = 50000
KEYPOINT_FREQUENCY = 1000
TRACK_FREQUENCY = 10
DIRECTIONS = np.concatenate([np.eye(5), -np.eye(5)]).astype(int)


def search(name=None, track=False, checkpoint=None):
    if checkpoint is not None and name is None:
        actual_name = checkpoint['name']
    elif name is None:
        actual_name = f'{int(time.time())}'
    else:
        actual_name = f'{int(time.time())}_{name}'
    folder_name = os.path.join('logs', 'leg', actual_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if track:
        import wandb
        if checkpoint is not None and name is None:
            wandb.init(
                project='laminate-quadruped-leg',
                id=checkpoint['track_id'],
                resume='must'
            )
        else:
            wandb.init(
                project='laminate-quadruped-leg',
                name=actual_name,
                save_code=True,
            )
        wandb.define_metric('search/step')
        wandb.define_metric("search/*", step_metric="search/step")
        wandb.run.log_code(os.getcwd())

    if checkpoint is None:
        candidates = deque()
        candidates.append(CANDIDATE_START)
        valids = deque()
        legs = deque()
        step = 0
    else:
        candidates = checkpoint['candidates']
        valids = checkpoint['valids']
        legs = checkpoint['legs']
        step = checkpoint['step']

    step_init = step
    start_time = time.time()

    while len(candidates) > 0 and step < MAX_STEPS:
        candidate = candidates.pop()
        p = np.array(candidate) * PARAMS_STEP + PARAMS_START
        try:
            step += 1
            r = opt.optimize(p)
            cost, constraints = opt.obj_with_constraints(r.x, p)
            valid = np.sum(constraints) == 0
        except AssertionError:
            valid = False

        if valid:
            valids.append(candidate)
            for dir in DIRECTIONS:
                new_candidate_array = np.array(candidate) + dir
                new_candidate_list = list(new_candidate_array)
                if (
                    new_candidate_list not in valids and
                    new_candidate_list not in candidates and
                    np.all(new_candidate_array >= 0)
                ):
                    candidates.append(new_candidate_list)

            legs.append(np.concatenate([r.x, p]))
        else:
            candidates.insert(0, candidate)

        checkpoint = {
            'name': actual_name,
            'track_id': wandb.run.id if track else None,
            'step': step,
            'candidates': candidates,
            'valids': valids,
            'legs': legs,
            'params_start': PARAMS_START,
            'params_step': PARAMS_STEP
        }
        np.save(
            os.path.join(folder_name, 'checkpoint.npy'),
            checkpoint
        )

        if step % KEYPOINT_FREQUENCY == 0:
            checkpoint_name = f'checkpoint_{step}.npy'
            checkpoint_path = os.path.join(folder_name, checkpoint_name)
            np.save(checkpoint_path, checkpoint)
            if track:
                shutil.copyfile(
                    checkpoint_path,
                    os.path.join(wandb.run.dir, checkpoint_name)
                )
                wandb.save(checkpoint_name, policy='now')

        sps = (step - step_init) / (time.time() - start_time)
        print(', '.join([f'{_p:.2f}' for _p in p]))
        print(
            f'step: {step}, '
            f'num_valids: {len(valids)}, '
            f'num_candidates: {len(candidates)}, '
            f'SPS: {sps:.2f}'
        )

        if track and step % TRACK_FREQUENCY == 0:
            wandb.log({
                'search/step': step,
                'search/num_valids': len(valids),
                'search/num_candidates': len(candidates),
                'search/SPS': sps
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument(
        '--track',
        default=False,
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument('--checkpoint', type=str, default=None)
    args = vars(parser.parse_args())

    if args['checkpoint'] is not None:
        args['checkpoint'] = np.load(
            os.path.join('logs', args['checkpoint']),
            allow_pickle=True
        ).item()

    search(**args)
