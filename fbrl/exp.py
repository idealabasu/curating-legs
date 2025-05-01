import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from rl.agent import Agent
from fbrl.env import Env

if __name__ == "__main__":
    leg_index = int(sys.argv[1])
    x = float(sys.argv[2])
    z = float(sys.argv[3])
    fx = float(sys.argv[4])
    fz = float(sys.argv[5])
    record = 'r' in sys.argv

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(
        os.path.join('data', 'fbrl_checkpoint.pt'),
        map_location=device, weights_only=False
    )

    env = Env(render_mode='human' if not record else 'rgb_array')
    env.metadata['render_fps'] = 50 if not record else 100
    env.model_params['hfield_texrepeat'] = 2
    env.model_params['hfield_nr'] = 100
    env.model_params['hfield_r'] = 1

    agent = Agent(
        np.array(env.observation_space.shape).prod(),
        np.array(env.action_space.shape).prod()
    ).to(device)
    agent.load_state_dict(checkpoint['agent'])
    agent.eval()

    frames = []
    states = []
    options = {
        'vx_cmd': 0.2,
        'wz_cmd': 0,
        'x': x,
        'z': z,
        'fx': fx,
        'fz': fz,
        'body_yaw': 0
    }
    if leg_index >= 0:
        options['leg_index'] = leg_index
    obs, info = env.reset(options=options)
    vx_cmd = env.vx_cmd
    wz_cmd = env.wz_cmd

    for i in range(500):
        obs = torch.Tensor(obs).to(device)
        with torch.no_grad():
            action = agent.get_deterministic_action(obs)
        obs, reward, terminated, truncated, info = env.step(
            action.cpu().numpy()
        )
        if record and i % 4 == 0:
            frames.append(env.render())
        states.append(env.get_states())
        if terminated or truncated:
            print(info)
            break
    env.close()

    if record:
        from moviepy.editor import ImageSequenceClip
        video = ImageSequenceClip(frames, fps=25)
        video.write_videofile(os.path.join(
            'logs',
            f'env_{env.leg_index:d}.mp4'
        ))

    states = np.array(states)
    t = states[:, 0]
    v = states[:, 36:39]
    w = states[:, 22:25]
    q = states[:, 3:11]
    dq = states[:, 11:19]
    g = states[:, 19:22]
    w = states[:, 22:25]
    a = states[:, 25:33]
    pos = states[:, 33:40]

    plt.figure()
    plt.subplot(211)
    plt.plot(t, q[:, ::2])
    plt.subplot(212)
    plt.plot(t, q[:, 1::2])

    plt.figure()
    plt.plot(t, pos[:, :3])

    plt.show()
