import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from rl.agent import Agent
from rl.env import Env

if __name__ == "__main__":
    leg_index = int(sys.argv[1])
    vx_cmd = float(sys.argv[2])
    wz_cmd = float(sys.argv[3])
    record = 'r' in sys.argv

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(
        os.path.join('data', 'rl_checkpoint.pt'),
        map_location=device, weights_only=False
    )

    env = Env(render_mode='human' if not record else 'rgb_array')
    env.metadata['render_fps'] = 25 if not record else 100
    env.model_params['hfield_textrepeat'] = 4
    env.model_params['hfield_nr'] = 200
    env.model_params['hfield_r'] = 2

    agent = Agent(
        np.array(env.observation_space.shape).prod(),
        np.array(env.action_space.shape).prod()
    ).to(device)
    agent.load_state_dict(checkpoint['agent'])
    agent.eval()

    frames = []
    states = []
    obs, info = env.reset(options={
        'leg_index': int(leg_index),
        'vx_cmd': vx_cmd,
        'wz_cmd': wz_cmd
    })
    for i in range(200):
        obs = torch.Tensor(obs).to(device)
        with torch.no_grad():
            action = agent.get_deterministic_action(obs)
        obs, reward, terminated, truncated, info = env.step(
            action.cpu().numpy()
        )
        if record:
            frames.append(env.render())
        states.append(env.get_states())
        if terminated or truncated:
            print(info)
            break
    env.close()

    if record:
        from moviepy.editor import ImageSequenceClip
        video = ImageSequenceClip(frames, fps=100)
        video.write_videofile(
            f'env_{leg_index:d}_{vx_cmd:.1f}_{wz_cmd:.1f}.mp4'
        )

    states = np.array(states)
    t = states[:, 0]
    v = states[:, 36:39]
    w = states[:, 22:25]
    q = states[:, 3:11]
    dq = states[:, 11:19]
    g = states[:, 19:22]
    w = states[:, 22:25]
    a = states[:, 25:33]
    tau = states[:, 39:47]

    plt.figure()
    plt.subplot(211)
    plt.plot(t, q[:, ::2])
    plt.subplot(212)
    plt.plot(t, q[:, 1::2])

    # plt.figure()
    # plt.subplot(211)
    # plt.plot(t, dq[:, ::2])
    # plt.subplot(212)
    # plt.plot(t, dq[:, 1::2])

    # plt.figure()
    # plt.subplot(211)
    # plt.plot(t, g)
    # plt.subplot(212)
    # plt.plot(t, w)

    plt.figure()
    plt.subplot(211)
    plt.plot(t, v)
    plt.ylabel('v (m/s)')
    plt.subplot(212)
    plt.plot(t, w)
    plt.ylabel('w (rad/s)')
    plt.xlabel('Time (s)')

    indices = t > 1
    vx_mean = np.mean(v[:, 0][indices])
    wz_mean = np.mean(w[:, 2][indices])
    p_mean = np.mean(np.sum(
        (tau[indices] * dq[indices]).clip(min=0),
        axis=1
    ))
    cot = np.abs(p_mean / 0.5 / 9.81 / vx_mean)
    print(
        f'vx: {vx_mean: .2f}/{vx_cmd:.2f}, '
        f'wz: {wz_mean: .2f}/{wz_cmd:.2f}, '
        f'cot: {cot:.2f}'
    )
    plt.show()
