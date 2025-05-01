import os
import re
import numpy as np
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
from gymnasium import spaces
import mujoco
import glfw
from collections import deque
import leg.model


class Env(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(37,),
            dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(8,),
            dtype=np.float64
        )

        self.timestep = 1 / 100
        self.max_steps = 500

        with open(os.path.join('rl', 'model.xml'), 'r') as file:
            self.model_string = file.read()
        self.model_params = {
            'gravity_z': -9.81,
            'hfield_texrepeat': 10,
            'hfield_nr': 500,
            'hfield_r': 5,
            'floor_x': 0,
            'floor_y': 0,
            'floor_z': 0,
            'floor_roll': 0,
            'floor_pitch': 0,
            'floor_yaw': 0,
            'body_x': 0,
            'body_y': 0,
            'body_z': 0.07,
            'body_roll': 0,
            'body_pitch': 0,
            'body_yaw': 0,
            'friction_tan': np.array([0.4, 1.0]),
            'servo_k': 1.8221793693748647 * np.array([0.9, 1.1]),
            'servo_b': 0.05077375928980776 * np.array([0.9, 1.1]),
            'servo_I': 0.0005170412918172306 * np.array([0.9, 1.1]),
            'servo_f': 0.02229709671396319 * np.array([0.9, 1.1]),
            'servo_tmax': 0.5771893852055175 * np.array([0.9, 1.1]),
            'ps_b': 0,
            'ss_b': 0,
        }
        self.hfield_pad = 0.1
        self.hfiled_elevation_max = 0.01  # m
        self.hfiled_altitude_max = 5  # deg
        self.sub_indices = None
        self.del_indices = None

        # Design
        legs = np.array(np.load(
            os.path.join('data', 'leg_checkpoint.npy'),
            allow_pickle=True
        ).item()['legs'])
        self.leg_shapes = legs[:, :8]
        self.leg_params = legs[:, 8:]
        self.leg_param_min = np.amin(self.leg_params, axis=0)
        self.leg_param_max = np.amax(self.leg_params, axis=0)

        # Obs
        self.obs_max = 3
        self.q_noise_max = 0.05
        self.dq_noise_max = 1.0
        self.g_noise_max = 0.05
        self.w_noise_max = 0.2
        self.q_scale = 1
        self.dq_scale = 0.05
        self.v_scale = 2
        self.w_scale = 0.25
        self.action_scale = 0.5
        self.latency_min = 0.03
        self.latency_max = 0.07
        self.latency_noise_max = 0.005
        self.leg_param_offset = (self.leg_param_max + self.leg_param_min) / 2
        self.leg_param_scale = 2 / (self.leg_param_max - self.leg_param_min)

        # Reward
        self.reward_max = 3
        self.reward_wz_scale = 1
        self.reward_v_tau = 1 / 25

        # Curriculum
        self.curriculum_designs = None
        self.curriculum_designs_labels = None
        self.curriculum_cells = None
        self.curriculum_vx_step = 0.2
        self.curriculum_wz_step = self.reward_wz_scale * self.curriculum_vx_step
        self.curriculum_score_th = 2 * np.exp(-1 / self.reward_v_tau * (
            (self.curriculum_vx_step / 2)**2 +
            (1 / self.reward_wz_scale)**2 * (self.curriculum_wz_step / 2)**2
        ))

        # Render
        assert (
            render_mode is None or
            render_mode in self.metadata['render_modes']
        )
        self.render_mode = render_mode
        self.metadata['render_fps'] = 1 / self.timestep
        self.window = None
        self.viewport = mujoco.MjrRect(0, 0, 640, 360)
        self.fps_step_count = 0
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = 1
        self.cam.distance = 0.5
        self.cam.azimuth = 90
        self.cam.elevation = -30
        self.cam.lookat = (0, 0, 0)
        self.opt = mujoco.MjvOption()
        # self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1

        mujoco.set_mju_user_warning(self._mju_user_warning)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.action_space.seed(seed)

        # Buffers
        self.step_count = 0
        self.rewards = []
        self.last_dq = np.zeros(8)
        self.last_a = np.zeros(8)
        self.qh_history = np.zeros((50, 4))
        self.qh_history_ptr = 0
        self.obs_history = np.zeros((10, 22))
        self.obs_history_ptr = 0
        self._model_params = self.model_params.copy()

        # Leg design
        if options and 'leg_index' in options:
            self.leg_index = options['leg_index']
        else:
            self.leg_index = self.np_random.choice(self.leg_params.shape[0])
        self.leg_shape = self.leg_shapes[self.leg_index]
        self.leg_param = self.leg_params[self.leg_index]
        self._model_params['body_z'] = self.leg_param[0] + self.leg_param[1]
        self._model_params = {
            **self._model_params,
            **leg.model.mj_params(self.leg_shape, self.leg_param)
        }
        self.q_offset = np.array([0, self.leg_param[2] / 2] * 4)

        # Tilt floor and pose body
        if options and 'hfiled_altitude' in options:
            self.hfiled_altitude = options['hfiled_altitude']
        else:
            self.hfiled_altitude = self.np_random.uniform(
                -self.hfiled_altitude_max, self.hfiled_altitude_max
            )
        r = R.from_euler(
            'XYZ',
            [0, self.hfiled_altitude, 0],
            degrees=True
        )
        floor_rpy = r.as_euler('XYZ', degrees=True)
        self._model_params['floor_roll'] = floor_rpy[0]
        self._model_params['floor_pitch'] = floor_rpy[1]
        self._model_params['floor_yaw'] = floor_rpy[2]
        if options and 'hfield_elevation' in options:
            self.hfield_elevation = options['hfield_elevation']
        else:
            self.hfield_elevation = self.np_random.uniform(
                0, self.hfiled_elevation_max
            )
        mat = r.as_matrix()
        self.hfield_n = mat[:, 2]
        body_pos = (
            mat.reshape(3, 3) @
            np.array([0, 0, self._model_params['body_z'] + self.hfield_elevation])
        )
        self._model_params['body_x'] = body_pos[0]
        self._model_params['body_y'] = body_pos[1]
        self._model_params['body_z'] = body_pos[2]
        self._model_params['body_roll'] = floor_rpy[0]
        self._model_params['body_pitch'] = floor_rpy[1]
        if options and 'body_yaw' in options:
            self._model_params['body_yaw'] = options['body_yaw']
        else:
            self._model_params['body_yaw'] = self.np_random.uniform(-180, 180)

        # Domain randomization
        for k, v in self._model_params.items():
            if not isinstance(v, np.ndarray):
                continue
            self._model_params[k] = self.np_random.uniform(v[0], v[1])

        # Create model
        if self.sub_indices is None:
            for k, v in self._model_params.items():
                # Pad keys
                self.model_string = re.sub(
                    r'\b%s\b' % k, f'{k:30}',
                    self.model_string
                )
            self.sub_indices = []
            for k, v in self._model_params.items():
                # Record indices
                self.sub_indices.append([
                    m.start()
                    for m in re.finditer(r'\b%s\b' % k, self.model_string)
                ])
        if self.del_indices is None:
            self.del_indices = [
                [m.start(), m.end()]
                for m in re.finditer(
                    r'<body name=\"ps_..\">[\s\S]*?</body>[\s\S]*?</body>',
                    self.model_string
                )
            ]
            self.del_indices += [
                [m.start(), m.end()]
                for m in re.finditer(
                    r'<connect name=\"input_ps_coupler_..\"[\s\S]*?>',
                    self.model_string
                )
            ]
        model_string = list(self.model_string)
        for (k, v), indices in zip(self._model_params.items(), self.sub_indices):
            for i in indices:
                model_string[i:i + 30] = f'{str(v):30}'
        if self.leg_param[3] == 0:
            # Remove ps related bodies
            for indices in self.del_indices:
                model_string[indices[0]:indices[1]] = (
                    ' ' * (indices[1] - indices[0])
                )
        model_string = ''.join(model_string)
        self.model = mujoco.MjModel.from_xml_string(model_string, {})
        self.data = mujoco.MjData(self.model)
        self.floor_id = self.data.geom('floor').id
        self.foot_ids = [
            self.model.geom(f'foot_{l}').id
            for l in ['fl', 'fr', 'rl', 'rr']
        ]

        # Populate hfield
        hfield_shape = self.model.hfield("floor").data.shape
        hfield_size = self.model.hfield("floor").size
        map = self.np_random.uniform(0, 1, hfield_shape)
        self.model.hfield("floor").data = (
            map * self.hfield_elevation / hfield_size[2]
        )

        # Latency
        self.latency = self.np_random.uniform(
            self.latency_min, self.latency_max
        )

        # Curriculum
        if self.curriculum_designs is None:
            self.curriculum_designs = np.zeros((1, self.leg_params.shape[1]))
        if self.curriculum_designs_labels is None:
            self.curriculum_designs_labels = np.zeros(
                self.leg_params.shape[0]
            ).astype(int)
        if self.curriculum_cells is None:
            self.curriculum_cells = [[[0, 0]]]
        self.curriculum_design = self.curriculum_designs_labels[self.leg_index]
        self.curriculum_cell = list(self.np_random.choice(
            self.curriculum_cells[self.curriculum_design]
        ))
        if options and 'vx_cmd' in options:
            self.vx_cmd = options['vx_cmd']
        else:
            self.vx_cmd = self.np_random.uniform(
                (self.curriculum_cell[0] - 0.5) * self.curriculum_vx_step,
                (self.curriculum_cell[0] + 0.5) * self.curriculum_vx_step
            )
        if options and 'wz_cmd' in options:
            self.wz_cmd = options['wz_cmd']
        else:
            self.wz_cmd = self.np_random.uniform(
                (self.curriculum_cell[1] - 0.5) * self.curriculum_wz_step,
                (self.curriculum_cell[1] + 0.5) * self.curriculum_wz_step
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self.render()

        return observation, info

    def step(self, action):
        action = action.flatten()
        self.data.ctrl = action * self.action_scale + self.q_offset
        try:
            mujoco.mj_step(
                self.model, self.data,
                nstep=int(self.timestep / self.model.opt.timestep)
            )
            warned = False
        except ValueError:
            warned = True
        self.step_count += 1
        if self.render_mode == 'human':
            self.render()

        da = (action - self.last_a) / self.timestep
        self.last_a = action

        q = self._get_q()
        qh = q[[0, 2, 4, 6]]
        self.qh_history[self.qh_history_ptr] = qh
        self.qh_history_ptr += 1
        if self.qh_history_ptr >= self.qh_history.shape[0]:
            self.qh_history_ptr = 0
        qh_mean = np.mean(self.qh_history, axis=0)

        qk = q[[1, 3, 5, 7]]
        qk_off_limits = (
            (qk - self.leg_param[2]).clip(min=0) -  # upper bound
            (qk - 0).clip(max=0)  # lower bound
        )

        collisions = 0
        for pair in zip(self.data.contact.geom1, self.data.contact.geom2):
            if self.floor_id not in pair:
                # collision between geoms other than floor
                collisions += 1
            else:
                if all([foot_id not in pair for foot_id in self.foot_ids]):
                    # collision between floor and geoms other than foot
                    collisions += 1

        p = np.clip(
            self._get_tau() * self._get_dq(),
            0, None
        )

        reward = np.array([
            2 * np.exp(-1 / self.reward_v_tau * (
                (self._get_v()[0] - self.vx_cmd)**2 +
                (1 / self.reward_wz_scale)**2 *
                (self._get_w()[2] - self.wz_cmd)**2
            )),
            -1e-5 * np.sum(da**2),
            -10 * np.sum(self._get_n()[:2]**2),
            -10 * np.sum(qh_mean**2),
            -100 * np.sum(qk_off_limits**2),
            -1 * collisions,
            -0.5 * np.sum(self._get_tau()**2),
            -0.02 * np.sum(p),
            0
        ])
        reward = np.clip(reward, -self.reward_max, self.reward_max)
        reward[-1] = np.sum(reward)
        for i, r in enumerate(reward):
            if i > len(self.rewards) - 1:
                self.rewards.append(deque())
            self.rewards[i].append(r)

        terminated = warned
        bound = self.model.hfield("floor").size[0] - self.hfield_pad
        out_of_bound = (
            np.abs(self._get_pos()[0]) > bound or
            np.abs(self._get_pos()[1]) > bound
        )
        time_limited = self.step_count >= self.max_steps
        truncated = time_limited or out_of_bound

        observation = self._get_obs()
        info = self._get_info()
        if terminated or truncated:
            # Calculate mean rewards
            rewards = {}
            for k, v in zip(
                [
                    'v',
                    'da',
                    'nxy', 'qhm', 'qkol', 'col',
                    'tau', 'p',
                    'total'
                ],
                self.rewards
            ):
                rewards[k] = np.mean(v)
            rewards['length'] = self.step_count

            info = {
                **info,
                'reward': rewards,
                'curriculum': {
                    'score': rewards['v'] if not terminated else 0,
                    'design': self.curriculum_design,
                    'cell': self.curriculum_cell
                }
            }

        return observation, reward[-1], terminated, truncated, info

    def render(self):
        def mj_render():
            scn = mujoco.MjvScene(self.model, maxgeom=1000)
            con = mujoco.MjrContext(
                self.model, mujoco.mjtFontScale.mjFONTSCALE_100
            )
            mujoco.mjv_updateScene(
                self.model, self.data,
                self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, scn
            )
            mujoco.mjr_render(self.viewport, scn, con)
            # mujoco.mjr_text(
            #     mujoco.mjtFont.mjFONT_NORMAL,
            #     f't_sim: {self.data.time:.2f}',
            #     con,
            #     0, 0,
            #     0, 0, 0
            # )

            return con

        if self.render_mode == 'human':
            if self.window is None:
                glfw.init()
                self.window = glfw.create_window(
                    self.viewport.width, self.viewport.height,
                    'Quadruped',
                    None, None
                )
                (
                    self.viewport.width,
                    self.viewport.height
                ) = glfw.get_framebuffer_size(self.window)
                glfw.make_context_current(self.window)
                glfw.swap_interval(1)

            self.fps_step_count += 1
            if (
                self.fps_step_count * self.timestep >
                1 / self.metadata['render_fps'] - 1e-6
            ):
                mj_render()
                self.fps_step_count = 0
                glfw.swap_buffers(self.window)
                glfw.poll_events()

        if self.render_mode == 'rgb_array':
            gl_con = mujoco.GLContext(
                self.viewport.width, self.viewport.height
            )
            gl_con.make_current()
            con = mj_render()
            img = np.empty(
                (self.viewport.height, self.viewport.width, 3),
                dtype=np.uint8
            )
            mujoco.mjr_readPixels(
                rgb=img, depth=None, viewport=self.viewport, con=con
            )
            return np.flip(img, axis=0)

    def close(self):
        if self.window is not None:
            glfw.terminate()
            self.window = None

    def get_states(self):
        return np.concatenate([
            [self.data.time],
            [self.vx_cmd, self.wz_cmd],
            self._get_q(),
            self._get_dq(),
            self._get_g(),
            self._get_w(),
            self.last_a,
            self._get_pos(),
            self._get_v(),
            self._get_tau(),
        ])

    def _get_obs(self):
        q = self._get_q() - self.q_offset
        dq = self._get_dq()
        g = self._get_g()
        w = self._get_w()

        q += self.np_random.uniform(-self.q_noise_max, self.q_noise_max)
        dq += self.np_random.uniform(-self.dq_noise_max, self.dq_noise_max)
        g += self.np_random.uniform(-self.g_noise_max, self.g_noise_max)
        w += self.np_random.uniform(-self.w_noise_max, self.w_noise_max)

        self.obs_history[self.obs_history_ptr] = np.concatenate([
            q * self.q_scale,
            dq * self.dq_scale,
            g,
            w * self.w_scale,
        ])
        self.obs_history_ptr += 1
        if self.obs_history_ptr >= self.obs_history.shape[0]:
            self.obs_history_ptr = 0

        latency = self.latency + self.np_random.uniform(
            -self.latency_noise_max, self.latency_noise_max
        )
        index = int(latency / self.timestep)
        r = (latency - (index * self.timestep)) / self.timestep
        _index = self.obs_history_ptr - 1 - index
        _index_next = _index - 1
        if _index < 0:
            _index += self.obs_history.shape[0]
        if _index_next < 0:
            _index_next += self.obs_history.shape[0]
        obs_w_latency = (
            (1 - r) * self.obs_history[_index] +
            r * self.obs_history[_index_next]
        )

        obs = np.concatenate([
            [self.vx_cmd * self.v_scale, self.wz_cmd * self.w_scale],
            obs_w_latency,
            self.last_a,
            (self.leg_param - self.leg_param_offset) * self.leg_param_scale
        ])
        return np.clip(obs, -self.obs_max, self.obs_max)

    def _get_info(self):
        return {}

    def _get_pos(self):
        return self.data.sensor('body_framepos').data.copy()

    def _get_g(self):
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, self.data.joint('body').qpos[3:7])
        return mat.reshape(3, 3).T @ np.array([0, 0, -1])

    def _get_n(self):
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, self.data.joint('body').qpos[3:7])
        return mat.reshape(3, 3).T @ self.hfield_n

    def _get_q(self):
        return self.data.actuator_length.copy()

    def _get_dq(self):
        return self.data.actuator_velocity.copy()

    def _get_tau(self):
        # The servo is modelled as a spring damper.
        # The torque should be the net torque.
        return self.data.actuator_force - self._get_dq() * self._model_params['servo_b']

    def _get_w(self):
        return self.data.sensor('body_gyro').data.copy()

    def _get_v(self):
        return self.data.sensor('body_velocimeter').data.copy()

    def _mju_user_warning(self, e):
        raise ValueError(e)


if __name__ == "__main__":
    env = Env(render_mode='human')
    env.metadata['render_fps'] = 50
    env.model_params['hfield_texrepeat'] = 2
    env.model_params['hfield_nr'] = 100
    env.model_params['hfield_r'] = 1
    observation, info = env.reset()
    observations = []
    states = []
    for i in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        observations.append(observation)
        states.append(env.get_states())
        if terminated or truncated:
            print(info)
            break
    env.close()
    observations = np.array(observations)
    states = np.array(states)

    import matplotlib.pyplot as plt
    plt.plot(observations[:, 2])
    plt.plot(states[:, 3])
    plt.show()
