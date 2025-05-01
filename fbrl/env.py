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
import itertools
from multiprocessing import shared_memory


class Env(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(46,),
            dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(8,),
            dtype=np.float64
        )

        self.timestep = 1 / 100
        self.max_steps = 500

        with open(os.path.join('fbrl', 'model.xml'), 'r') as file:
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
            'load_m': 0.1,
            'load_r': 0.02,
            'load_z': 0.0295 + 0.02
        }
        self.hfield_pad = 0.1
        self.hfiled_elevation_max = 0.01  # m
        self.hfiled_altitude_max = 5  # deg
        self.sub_indices = None
        self.del_indices_front = None
        self.del_indices_rear = None

        # Design
        legs = np.array(np.load(
            os.path.join('data', 'leg_checkpoint.npy'),
            allow_pickle=True
        ).item()['legs'])
        self.leg_shapes = legs[:, :8]
        self.leg_params = legs[:, 8:]
        self.leg_param_min = np.amin(self.leg_params, axis=0)
        self.leg_param_max = np.amax(self.leg_params, axis=0)
        self.leg_indices = [119, 243, 134, 5, 398, 386]
        # self.leg_indices = [119, 243, 134]
        leg_combs = list(
            itertools.combinations_with_replacement(self.leg_indices, 2)
        )
        self.leg_combs = []
        i = 0
        for leg_index_front, leg_index_rear in leg_combs:
            leg_param_front = self.leg_params[leg_index_front]
            leg_param_rear = self.leg_params[leg_index_rear]
            leg_length_front = leg_param_front[0] + leg_param_front[1] / 2
            leg_length_rear = leg_param_rear[0] + leg_param_rear[1] / 2
            leg_length_diff = np.abs(leg_length_front - leg_length_rear)
            if leg_length_diff < 0.04 + 1e-3:
                # print(f'{i}, {leg_length_diff:.2f}')
                self.leg_combs.append([leg_index_front, leg_index_rear])
                i += 1
        self.leg_combs = np.array(self.leg_combs)

        self.leg_mj_params = {}
        for leg_index in self.leg_indices:
            self.leg_mj_params[leg_index] = leg.model.mj_params(
                self.leg_shapes[leg_index], self.leg_params[leg_index]
            )

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
        self.x_scale = 5  # -0.2, 0.2m
        self.z_scale = 5  # -0.2, 0.2m
        self.fx_scale = 0.2  # -5, 5N
        self.fz_scale = 0.2  # -5, 5N
        self.ft_noise_max_scale = 0.1

        # Velocity
        self.vx_max = 0.3
        self.reward_wz_scale = 1
        self.wz_max = self.vx_max * self.reward_wz_scale
        self.reward_v_tau = 1 / 25

        # Reward
        self.reward_max = 3

        # Curriculum
        self.curriculum_x_max = 1
        self.curriculum_z_max = 1
        self.curriculum_fx_max = 4
        self.curriculum_fz_max = 4
        self.curriculum_cells_shared = None  # not used during training
        self.curriculum_cells_shared_name = None
        self.curriculum_x_step = 0.1
        self.curriculum_z_step = 0.05
        self.curriculum_fx_step = 1
        self.curriculum_fz_step = 1
        self.curriculum_score_th = 2 * np.exp(
            -1 / self.reward_v_tau * (0.1**2 + 0.1**2)
        )

        # Render
        assert (
            render_mode is None or
            render_mode in self.metadata['render_modes']
        )
        self.render_mode = render_mode
        self.metadata['render_fps'] = 1 / self.timestep
        self.window = None
        self.viewport = mujoco.MjrRect(0, 0, 1280, 720)
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
        self.qh_history = np.zeros((100, 4))
        self.qh_history_ptr = 0
        self.obs_history = np.zeros((10, 22))
        self.obs_history_ptr = 0
        self._model_params = self.model_params.copy()

        # Leg design
        if options and 'leg_index' in options:
            self.leg_index = options['leg_index']
        else:
            self.leg_index = self.np_random.choice(len(self.leg_combs))
        self.leg_index_front, self.leg_index_rear = self.leg_combs[self.leg_index]
        self.leg_param_front = self.leg_params[self.leg_index_front]
        self.leg_param_rear = self.leg_params[self.leg_index_rear]

        self._model_params['body_z'] = np.maximum(
            self.leg_param_front[0] + self.leg_param_front[1],
            self.leg_param_rear[0] + self.leg_param_rear[1]
        )
        leg_mj_param_front = {}
        for k, v in self.leg_mj_params[self.leg_index_front].items():
            leg_mj_param_front[f'{k}_f'] = v
        leg_mj_param_rear = self.leg_mj_params[self.leg_index_rear]
        self._model_params = {
            **self._model_params,
            **leg_mj_param_front,
            **leg_mj_param_rear,
        }

        hip_spread = 0
        self.q_offset = np.array([
            hip_spread, self.leg_param_front[2] / 2,
            hip_spread, self.leg_param_front[2] / 2,
            -hip_spread, self.leg_param_rear[2] / 2,
            -hip_spread, self.leg_param_rear[2] / 2,
        ])

        # Curriculum
        if options and 'curriculum_cell' in options:
            self.curriculum_cell = options['curriculum_cell']
        else:
            if self.curriculum_cells_shared_name is None:
                curriculum_cells_shape = (
                    len(self.leg_combs),
                    2 * self.curriculum_x_max + 1,
                    2 * self.curriculum_z_max + 1,
                    2 * self.curriculum_fx_max,
                    2 * self.curriculum_fz_max
                )
                self.curriculum_cells_shared = shared_memory.SharedMemory(
                    create=True, size=int(np.prod(curriculum_cells_shape))
                )
                self.curriculum_cells_shared_name = self.curriculum_cells_shared.name
                curriculum_cells = np.ndarray(
                    curriculum_cells_shape,
                    dtype=np.bool_,
                    buffer=self.curriculum_cells_shared.buf
                )
                curriculum_cells[:] = False
                for leg_index in range(len(self.leg_combs)):
                    for x, y in [[0, 0], [-1, 0], [0, -1], [-1, -1]]:
                        curriculum_cells[
                            leg_index,
                            0 + self.curriculum_x_max,
                            0 + self.curriculum_z_max,
                            x + self.curriculum_fx_max,
                            y + self.curriculum_fz_max
                        ] = True
            curriculum_cells_shared = shared_memory.SharedMemory(
                name=self.curriculum_cells_shared_name
            )
            curriculum_cells = np.ndarray(
                (
                    len(self.leg_combs),
                    2 * self.curriculum_x_max + 1,
                    2 * self.curriculum_z_max + 1,
                    2 * self.curriculum_fx_max,
                    2 * self.curriculum_fz_max
                ),
                dtype=np.bool_,
                buffer=curriculum_cells_shared.buf
            )
            self.curriculum_cell = np.array([
                self.leg_index, *self.np_random.choice(
                    np.argwhere(curriculum_cells[self.leg_index] == True)
                )
            ])

        _curriculum_cell = self.curriculum_cell[1:] - np.array([
            self.curriculum_x_max,
            self.curriculum_z_max,
            self.curriculum_fx_max,
            self.curriculum_fz_max
        ])
        if options and 'x' in options:
            self.x = options['x']
        else:
            self.x = self.np_random.uniform(
                (_curriculum_cell[0] - 0.5) * self.curriculum_x_step,
                (_curriculum_cell[0] + 0.5) * self.curriculum_x_step
            )
        if options and 'z' in options:
            self.z = options['z']
        else:
            self.z = self.np_random.uniform(
                (_curriculum_cell[1] - 0.5) * self.curriculum_z_step,
                (_curriculum_cell[1] + 0.5) * self.curriculum_z_step
            )
        if options and 'fx' in options:
            self.fx = options['fx']
        else:
            self.fx = self.np_random.uniform(
                (_curriculum_cell[2]) * self.curriculum_fx_step,
                (_curriculum_cell[2] + 1) * self.curriculum_fx_step
            )
        if options and 'fz' in options:
            self.fz = options['fz']
        else:
            self.fz = self.np_random.uniform(
                (_curriculum_cell[3]) * self.curriculum_fz_step,
                (_curriculum_cell[3] + 1) * self.curriculum_fz_step
            )

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
        if self.del_indices_front is None:
            self.del_indices_front = [
                [m.start(), m.end()]
                for m in re.finditer(
                    r'<body name=\"ps_f.\">[\s\S]*?</body>[\s\S]*?</body>',
                    self.model_string
                )
            ]
            self.del_indices_front += [
                [m.start(), m.end()]
                for m in re.finditer(
                    r'<connect name=\"input_ps_coupler_f.\"[\s\S]*?>',
                    self.model_string
                )
            ]
        if self.del_indices_rear is None:
            self.del_indices_rear = [
                [m.start(), m.end()]
                for m in re.finditer(
                    r'<body name=\"ps_r.\">[\s\S]*?</body>[\s\S]*?</body>',
                    self.model_string
                )
            ]
            self.del_indices_rear += [
                [m.start(), m.end()]
                for m in re.finditer(
                    r'<connect name=\"input_ps_coupler_r.\"[\s\S]*?>',
                    self.model_string
                )
            ]
        model_string = list(self.model_string)
        for (k, v), indices in zip(self._model_params.items(), self.sub_indices):
            for i in indices:
                model_string[i:i + 30] = f'{str(v):30}'
        if self.leg_param_front[3] == 0:
            # Remove ps related bodies
            for indices in self.del_indices_front:
                model_string[indices[0]:indices[1]] = (
                    ' ' * (indices[1] - indices[0])
                )
        if self.leg_param_rear[3] == 0:
            # Remove ps related bodies
            for indices in self.del_indices_rear:
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

        # Velocity
        if options and 'vx_cmd' in options:
            self.vx_cmd = options['vx_cmd']
        else:
            self.vx_cmd = self.np_random.uniform(
                -self.vx_max, self.vx_max
            )

        if self.fx > 0:
            self.vx_cmd = -np.abs(self.vx_cmd)
        else:
            self.vx_cmd = np.abs(self.vx_cmd)

        if options and 'wz_cmd' in options:
            self.wz_cmd = options['wz_cmd']
        else:
            self.wz_cmd = self.np_random.uniform(
                -self.wz_max, self.wz_max
            )

        # Latency
        self.latency = self.np_random.uniform(
            self.latency_min, self.latency_max
        )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self.render()

        return observation, info

    def step(self, action):
        action = action.flatten()
        self.data.ctrl = action * self.action_scale + self.q_offset

        force = np.array([self.fx, 0, self.fz])
        force += np.linalg.norm(force) * self.np_random.uniform(
            -self.ft_noise_max_scale, self.ft_noise_max_scale, 3
        )
        torque = np.array([0, self.fx * self.z - self.fz * self.x, 0])
        torque += np.linalg.norm(torque) * self.np_random.uniform(
            -self.ft_noise_max_scale, self.ft_noise_max_scale, 3
        )
        robot_mat = self._get_robot_tf()[:3, :3]
        self.data.xfrc_applied[self.data.body('body').id] = np.array(
            [*robot_mat @ force, *robot_mat @ torque]
        )

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
        qh_diff = qh_mean[[0, 2]] - qh_mean[[1, 3]]

        qk = q[[1, 3, 5, 7]]
        qk_ub = np.array([
            self.leg_param_front[2], self.leg_param_front[2],
            self.leg_param_rear[2], self.leg_param_rear[2]
        ])
        qk_off_limits = (
            (qk - qk_ub).clip(min=0) -  # upper bound
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

        v = robot_mat.T @ self._get_v_w()
        w = robot_mat.T @ self._get_w_w()

        reward = np.array([
            2 * np.exp(-1 / self.reward_v_tau * (
                (v[0] - self.vx_cmd)**2 +
                (1 / self.reward_wz_scale)**2 *
                (w[2] - self.wz_cmd)**2
            )),
            -5 * v[2]**2,
            -1e-5 * np.sum(da**2),
            -10 * self._get_n()[1]**2,
            -5 * np.sum(qh_diff**2),
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
                    'dz', 'da',
                    'ny', 'qhd', 'qkol', 'col',
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
                    'cell': self.curriculum_cell,
                    'score': rewards['v'] if not terminated else 0
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

            if scn.ngeom >= scn.maxgeom:
                return
            scn.ngeom += 2
            force_id = scn.ngeom - 2
            origin_id = scn.ngeom - 1

            mujoco.mjv_initGeom(
                scn.geoms[force_id],
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3), np.zeros(3), np.zeros(9),
                np.array([1, 0, 1, 1]).astype(np.float32)
            )
            mujoco.mjv_initGeom(
                scn.geoms[origin_id],
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3), np.zeros(3), np.zeros(9),
                np.array([1, 0, 1, 1]).astype(np.float32)
            )

            robot_tf = self._get_robot_tf()
            _p1 = np.array([self.x, 0, self.z])
            _p1p = _p1 + np.array([0.001, 0, 0])
            _p2 = _p1 + np.array([self.fx, 0, self.fz]) * 0.1
            p1 = (robot_tf @ np.array([*_p1, 1]))[:3].flatten()
            p1p = (robot_tf @ np.array([*_p1p, 1]))[:3].flatten()
            p2 = (robot_tf @ np.array([*_p2, 1]))[:3].flatten()

            mujoco.mjv_connector(
                scn.geoms[force_id],
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                0.005, p1, p2
            )
            mujoco.mjv_connector(
                scn.geoms[origin_id],
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                0.01, p1, p1p
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

        if self.curriculum_cells_shared is not None:
            self.curriculum_cells_shared.close()
            self.curriculum_cells_shared.unlink()
            self.curriculum_cells_shared_name = None

    def get_states(self):
        robot_mat = self._get_robot_tf()[:3, :3]
        return np.concatenate([
            [self.data.time],
            [self.vx_cmd, self.wz_cmd],
            self._get_q(),
            self._get_dq(),
            self._get_g(),
            self._get_w(),
            self.last_a,
            self._get_pos(),
            robot_mat.T @ self._get_v_w(),
            self._get_tau(),
            robot_mat.T @ self._get_w_w(),
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
            (self.leg_param_front - self.leg_param_offset) * self.leg_param_scale,
            (self.leg_param_rear - self.leg_param_offset) * self.leg_param_scale,
            [
                self.x * self.x_scale,
                self.z * self.z_scale,
                self.fx * self.fx_scale,
                self.fz * self.fz_scale
            ]
        ])
        return np.clip(obs, -self.obs_max, self.obs_max)

    def _get_info(self):
        return {}

    def _get_robot_tf(self):
        body_mat = np.zeros(9)
        mujoco.mju_quat2Mat(body_mat, self.data.joint('body').qpos[3:7])
        body_mat = body_mat.reshape(3, 3)
        robot_z = self.hfield_n
        robot_y = np.cross(robot_z, body_mat[:, 0])
        robot_y /= np.linalg.norm(robot_y)
        robot_x = np.cross(robot_y, robot_z)
        robot_mat = np.array([robot_x, robot_y, robot_z]).T
        robot_tf = np.eye(4)
        robot_tf[:3, :3] = robot_mat.copy()
        robot_tf[:3, 3] = self.data.joint('body').qpos[:3].copy()

        return robot_tf

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

    def _get_w_w(self):
        return self.data.joint('body').qvel[3:].copy()

    def _get_v_w(self):
        return self.data.joint('body').qvel[:3].copy()

    def _mju_user_warning(self, e):
        raise ValueError(e)


if __name__ == "__main__":
    env = Env(render_mode='human')
    env.metadata['render_fps'] = 50
    env.model_params['hfield_texrepeat'] = 2
    env.model_params['hfield_nr'] = 100
    env.model_params['hfield_r'] = 1
    observation, info = env.reset(options={
        'leg_index': 1,
        'vx_cmd': 0.2,
        'x': 0.1,
        'z': 0.04,
        'fx': -1.0,
        'fz': -1.0,
        'body_yaw': 0,
        'hfiled_altitude': 0
    })
    observations = []
    states = []
    for i in range(500):
        action = env.action_space.sample() * 0
        # action = np.array([0, 0.5, 0, 0, 0, 0.5, 0, 0])
        # action = np.array([0, 0.5, 0, 0.5, 0, 0, 0, 0])
        observation, reward, terminated, truncated, info = env.step(action)
        observations.append(observation)
        states.append(env.get_states())
        if terminated or truncated:
            print(info)
            break
    env.close()
    observations = np.array(observations)
    states = np.array(states)
