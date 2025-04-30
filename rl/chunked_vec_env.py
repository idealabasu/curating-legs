import sys
import numpy as np
import gymnasium as gym
from gymnasium.vector.utils import write_to_shared_memory


class ChunkedVectorEnv(gym.vector.AsyncVectorEnv):
    def __init__(self, env_fn, num_async_vec_envs, num_sync_vec_envs):
        def make_sync_env():
            return gym.vector.SyncVectorEnv(
                [env_fn for i in range(num_sync_vec_envs)]
            )

        super().__init__(
            [make_sync_env for i in range(num_async_vec_envs)],
            worker=_worker_shared_memory
        )

        self.num_async_vec_envs = num_async_vec_envs
        self.num_sync_vec_envs = num_sync_vec_envs
        self.num_total_envs = num_async_vec_envs * num_sync_vec_envs

    def reset(self, seed=None, options=None):
        if not options:
            options = {}
        assert 'num_sync_vec_envs' not in options
        options['num_sync_vec_envs'] = self.num_sync_vec_envs
        self.action_space.seed(seed)

        obs, info = super().reset(
            seed=seed,
            options=options
        )
        return (
            obs.reshape(self.num_total_envs, -1),
            self._flatten_info(info)
        )

    def step(self, action):
        action = action.reshape(
            self.num_async_vec_envs, self.num_sync_vec_envs, -1
        )
        obs, reward, terminated, truncated, info = super().step(action)
        return (
            obs.reshape(self.num_total_envs, -1),
            reward.reshape(-1),
            terminated.reshape(-1),
            truncated.reshape(-1),
            self._flatten_info(info)
        )

    def _flatten_info(self, info):
        if not info:
            return info

        def get_pads(dtype, length):
            if dtype == 'bool':
                return [False] * length
            else:
                return [None] * length

        flattened_info = {}
        for i, single_info in enumerate(info['child_info']):
            if single_info is None:
                continue

            for k, v in single_info.items():
                if k not in flattened_info:
                    if i == 0:
                        flattened_info[k] = v
                    else:
                        flattened_info[k] = np.concatenate([
                            get_pads(v.dtype, i * self.num_sync_vec_envs),
                            v
                        ])
                else:
                    if i * self.num_sync_vec_envs == len(flattened_info[k]):
                        flattened_info[k] = np.concatenate([
                            flattened_info[k],
                            v
                        ])
                    else:
                        flattened_info[k] = np.concatenate([
                            flattened_info[k],
                            get_pads(
                                v.dtype,
                                i * self.num_sync_vec_envs -
                                len(flattened_info[k])
                            ),
                            v
                        ])

        for has_info in info['_child_info'][::-1]:
            if has_info:
                break

            for k, v in flattened_info.items():
                flattened_info[k] = np.concatenate([
                    v,
                    get_pads(v.dtype, self.num_sync_vec_envs)
                ])

        return flattened_info


def _worker_shared_memory(
    index, env_fn, pipe, parent_pipe, shared_memory, error_queue,
):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                if 'seed' in data:
                    data['seed'] = (
                        data['seed'] +
                        index * (data['options']['num_sync_vec_envs'] - 1)
                    )
                observation, info = env.reset(**data)
                # Add a key to enable info stacking
                if info:
                    info = {'child_info': info}
                write_to_shared_memory(
                    observation_space, index, observation, shared_memory
                )
                pipe.send(((None, info), True))

            elif command == "step":
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(data)
                # Add a key to enable info stacking
                if info:
                    info = {'child_info': info}
                # Since child env is sync vector env, no need to handle reset here.
                # if terminated or truncated:
                #     old_observation, old_info = observation, info
                #     observation, info = env.reset()
                #     info["final_observation"] = old_observation
                #     info["final_info"] = old_info
                write_to_shared_memory(
                    observation_space, index, observation, shared_memory
                )
                pipe.send(
                    ((None, reward, terminated, truncated, info), True))
            elif command == "seed":
                # Probably not used
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                # Pass attr to child env
                env.set_attr(name, value)
                pipe.send((None, True))
            elif command == "_check_spaces":
                pipe.send((
                    (
                        data[0] == observation_space,
                        data[1] == env.action_space
                    ),
                    True
                ))
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, `_call`, "
                    "`_setattr`, `_check_spaces`}."
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


if __name__ == "__main__":
    from rl.env import Env
    env = ChunkedVectorEnv(Env, 3, 2)
    obs, info = env.reset(seed=0)
    for i in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if info:
            print(i, info)
    env.close()
