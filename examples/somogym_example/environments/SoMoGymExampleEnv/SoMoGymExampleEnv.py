from collections import deque
import numpy as np
import os
import pybullet as p

from somo.sm_manipulator_definition import SMManipulatorDefinition
from somo.sm_continuum_manipulator import SMContinuumManipulator
from somo.utils import load_constrained_urdf

from environments import SomoEnv
from environments.SoMoGymExampleEnv.env_utils import box_start_funcs


class SoMoGymExampleEnv(SomoEnv.SomoEnv):
    def __init__(
        self,
        run_config=None,
        run_ID=None,
        render=False,
        debug=0,
        invert_hand=False,
    ):

        n_manipulators = 4

        # load the manipulator definition
        manipulator_def_file = os.path.join(
            os.path.dirname(__file__),
            "definitions/IHM_finger.yaml",  # RANDY: to change finger properties, adjust this yaml file
        )
        manipulator_def = SMManipulatorDefinition.from_file(manipulator_def_file)
        manipulator_def.urdf_filename = os.path.join(
            os.path.dirname(__file__),
            "definitions/" + manipulator_def.manipulator_name + ".urdf",
        )

        active_actuators_list = [([0, 0], [0, 1])] * n_manipulators
        manipulators = [
            SMContinuumManipulator(manipulator_def) for i in range(n_manipulators)
        ]

        # RANDY: finger positions are computed here
        scaler = (
            2 / 101.6
        )  # fingers are roughly 4in (=101.6mm) long in hardware, and 2m in simulation

        theta_code = (
            run_config['finger_arrangement'] * 1 / 2 * np.pi
        )  # always in [0,1/2pi] rad (or [0,90] degrees respectively)
        theta_rad = (
            -theta_code + 1 / 2 * np.pi
        )  # todo: switch angel necessary bcs definitions in randy's code are [0,90] to [90,0]
        theta_deg = theta_rad / np.pi * 180
        # all values below are given in mm first (e.g., 36mm), and then scaled to simulation dimensions
        h = 0 * scaler
        w1 = 36 * scaler
        r2 = 24.365 * scaler
        p1 = [-w1 / 2, -w1 / 2, h]
        p2 = [w1 / 2, -w1 / 2, h]
        p3 = [
            p1[0] + r2 * np.cos(1.25 * np.pi - theta_rad),
            p1[1] + r2 * np.sin(1.25 * np.pi - theta_rad),
            p1[2],
        ]
        p4 = [
            p2[0] + r2 * np.cos(1.75 * np.pi + theta_rad),
            p2[1] + r2 * np.sin(1.75 * np.pi + theta_rad),
            p2[2],
        ]
        p5 = [0, 34.421 * scaler, h]
        p6 = [0, -34.421 * scaler, h]

        start_positions = [
            p5,
            p3,
            p6,
            p4,
        ]  # RANDY: to change finger properties, adjust this yaml file
        start_orientations = [
            p.getQuaternionFromEuler(np.deg2rad([15, 0, 180])),
            p.getQuaternionFromEuler(
                np.deg2rad([15, 0, -theta_deg])
            ),  # todo: fix 0*; for debugging only
            p.getQuaternionFromEuler(np.deg2rad([15, 0, 0])),
            p.getQuaternionFromEuler(np.deg2rad([15, 0, theta_deg])),
        ]

        manipulator_base_constraints = ["static"] * n_manipulators

        # set some geometry params
        self.palm_start_pos = [0.0, 0.0, 1.25]  # todo: maybe don't hard-code
        self.palm_start_or = p.getQuaternionFromEuler([0, 0, 0])
        self.palm_friction = 0.15  # RANDY: make sure to tune palm and box friction

        # RANDY: make sure object weight and other properties are accurate
        self.box_start_pos = np.array(
            [0.0, 0.0, 2.25]
        )  # RANDY: make sure to change box position as necessary
        self.box_start_or = np.array([0, 0, 0])
        self.box_start_or_quat = p.getQuaternionFromEuler(list(self.box_start_or))
        self.box_pos = self.box_start_pos
        self.box_or = self.box_start_or

        self.box_friction_range = [1.2] * 2
        if "box_friction" in run_config and run_config["box_friction"] is not None:
            box_friction = run_config["box_friction"]
            if isinstance(box_friction, float):
                self.box_friction_range = [box_friction] * 2
            else:
                self.box_friction_range = box_friction[:2]

        self.get_box_start = None
        if "box_start_func" in run_config and run_config["box_start_func"] is not None:
            self.get_box_start = box_start_funcs.box_start_funcs[
                run_config["box_start_func"]
            ]

        action_shape = (8,)

        # todo: make this inversion prettier
        try:
            self.invert_hand = (
                invert_hand or run_config["invert_hand"]
            )  # if set to True, the hand is operated over a plane (i.e., a table) that prevents the box from falling out
        except KeyError:
            self.invert_hand = invert_hand

        if self.invert_hand:
            gravity = [0, 0, 196.2]
            ground_plane_height = 12.8
        else:
            gravity = [0, 0, -196.2]
            ground_plane_height = None

        super().__init__(
            manipulators=manipulators,
            start_positions=start_positions,
            start_orientations=start_orientations,
            manipulator_base_constraints=manipulator_base_constraints,
            active_actuators_list=active_actuators_list,
            action_shape=action_shape,
            run_config=run_config,
            gravity=gravity,  # scale factor 20 applied to gravity
            ground_plane_height=ground_plane_height,
            run_ID=run_ID,
            render=render,
            debug=debug,
        )

        # todo: consider adding a penalty for final error in x and y orientation
        # TODO: make sure to properly handle angle wrapping for box z rotation
        # in case we want to observe the unwrapped z rotation angle (or others);
        # careful, this  is only checked every env/action step, when technically it should be checked and updated every
        # pybullet step. as long as angle changes between action steps are small enough, this is fine tho. risk if angle
        # changes are too large across  env steps is that we miss a jump in angle that should trigger an unwrapping.
        self.previous_or = None  # todo: may not be needed

    def step(self, action):

        self.prev_box_pos = self.box_pos
        self.prev_box_or = self.box_or

        obs, reward, done, info = super().step(action=action)

        box_pos, box_or_quat = p.getBasePositionAndOrientation(self.box_id)
        box_or = np.array(p.getEulerFromQuaternion(box_or_quat))
        self.box_pos, self.box_or = np.array(box_pos), np.array(box_or)

        try:
            failure_penalty = self.run_config["failure_penalty"]
        except:
            failure_penalty = 0

        # For IHM, done means we failed.
        if done and failure_penalty:
            reward -= failure_penalty

        return (obs, reward, done, info)

    def is_done(self):
        boundary = 5
        done = np.linalg.norm(self.box_pos - self.box_start_pos) >= boundary
        return done


    # Get column labels for the observation data
    def get_observation_labels(self):
        flags = self.run_config["observation_flags"]
        labels = deque([])
        for key in flags:
            curr_shape = self.get_data_shape({key:flags[key]})
            for idx in range(curr_shape[0]):
                labels.append("%s_%d"%(key, idx))

        return list(labels)

    
    # Get the shape of observations
    def get_observation_shape(self):
        return self.get_data_shape(self.run_config["observation_flags"])


    # Get the shape of any type of supported data
    def get_data_shape(self, flags):
        # possible observation flags: box_pos, box_or, torques
        obs_dimension_per_sample = {
            "box_pos": 3,
            "box_or": 3,
            "box_velocity": 3,
            "positions": 3 * 4,
            "velocities": 3 * 4,
            "tip_pos": 3 * 4,
            "angles": 1 * 4,
            "curvatures": 1 * 4,
            "applied_input_torques": 2 * 4,
        }

        obs_len = 0
        for f in flags:
            num_pts = flags[f] if flags[f] is not None else 1
            obs_len += num_pts * obs_dimension_per_sample[f]
        return (obs_len,)


    def get_observation(self):
        obs_flags = self.run_config["observation_flags"]
        self.manipulator_states = self.get_manipulator_states(
            components=list(obs_flags.keys())
        )
        state = np.array([])

        if "box_pos" in obs_flags or "box_or" in obs_flags:
            box_pos, box_or_quat = p.getBasePositionAndOrientation(self.box_id)
            if "box_pos" in obs_flags:
                state = np.concatenate((state, np.array(box_pos)))
            if "box_or" in obs_flags:
                box_or = p.getEulerFromQuaternion(box_or_quat)
                state = np.concatenate((state, np.array(box_or)))

        if "box_velocity" in obs_flags:
            box_velocity = np.array(p.getBaseVelocity(bodyUniqueId=self.box_id)[0])
            state = np.concatenate((state, np.array(box_velocity)))

        if "positions" in obs_flags:
            positions = np.array(
                [state["positions"] for state in self.manipulator_states]
            )
            if obs_flags["positions"]:
                positions = np.array(
                    [
                        self.reduce_state_len(ps, obs_flags["positions"])
                        for ps in positions
                    ]
                )
            state = np.concatenate((state, positions.flatten()))

        if "velocities" in obs_flags:
            velocities = np.array(
                [state["velocities"] for state in self.manipulator_states]
            )
            if obs_flags["velocities"]:
                velocities = np.array(
                    [
                        self.reduce_state_len(vs, obs_flags["velocities"])
                        for vs in velocities
                    ]
                )
            state = np.concatenate((state, velocities.flatten()))

        if "tip_pos" in obs_flags:
            positions = np.array(
                [state["positions"] for state in self.manipulator_states]
            )
            tip_pos = positions[:, -1]
            state = np.concatenate((state, tip_pos.flatten()))

        if "angles" in obs_flags:
            # todo: this backbone angle stuff seems to have a problem... rethink how to read this - maybe get joint angles instead. try to get to angles that are 0 in one of the axes in case of planar manipulators
            angles = np.array([state["angles"] for state in self.manipulator_states])
            if obs_flags["angles"]:
                angles = np.array(
                    [
                        self.reduce_state_len(angs, obs_flags["angles"])
                        for angs in angles
                    ]
                )
            state = np.concatenate((state, angles.flatten()))

        if "curvatures" in obs_flags:
            curvatures = np.array(
                [state["curvatures"] for state in self.manipulator_states]
            )
            if obs_flags["curvatures"]:
                curvatures = np.array(
                    [
                        self.reduce_state_len(cs, obs_flags["curvatures"])
                        for cs in curvatures
                    ]
                )
            state = np.concatenate((state, curvatures.flatten()))

        if "applied_input_torques" in obs_flags:
            applied_input_torques = np.array(self.applied_torque)
            state = np.concatenate((state, applied_input_torques.flatten()))

        return state

    def get_logger_data(self):
        pass

    def get_reward(self, *args, **kwargs):
        if (
            self.stabilizing
        ):  # todo: move this to SomoEnv base class; also introduce attribute to set flexible stabilization time
            return 0

        # possible reward flags: z_rotation_step, z_rotation, x_rotation, y_rotation, position
        reward_flags = self.run_config["reward_flags"]

        box_pos, box_or_quat = p.getBasePositionAndOrientation(self.box_id)
        box_or = p.getEulerFromQuaternion(box_or_quat)
        box_pos, box_or = np.array(box_pos), np.array(box_or)

        reward = 0

        if "z_rotation_step" in reward_flags:
            z_rotation_step_reward = box_or[2] - self.prev_box_or[2]
            self.reward_component_info["z_rotation_step"] += z_rotation_step_reward
            reward += reward_flags["z_rotation_step"] * z_rotation_step_reward

        if "z_rotation" in reward_flags:
            z_rotation_reward = box_or[2] - self.box_start_or[2]
            self.reward_component_info["z_rotation"] += z_rotation_reward
            reward += reward_flags["z_rotation"] * z_rotation_reward

        if "x_rotation" in reward_flags:
            x_rotation_reward = abs(box_or[0] - self.box_start_or[0])
            self.reward_component_info["x_rotation"] += x_rotation_reward
            reward += reward_flags["x_rotation"] * x_rotation_reward

        if "y_rotation" in reward_flags:
            y_rotation_reward = abs(box_or[1] - self.box_start_or[1])
            self.reward_component_info["y_rotation"] += y_rotation_reward
            reward += reward_flags["y_rotation"] * y_rotation_reward

        if "position" in reward_flags:
            position_reward = np.sum(np.abs(box_pos - self.box_start_pos))
            self.reward_component_info["position"] += position_reward
            reward += reward_flags["position"] * position_reward

        return reward

    def modify_physics(self):

        # change the lateral friction of the fingers
        for manipulator in self.manipulators:
            contact_properties = {
                "lateralFriction": 3,
                "restitution": 0.7,
            }
            manipulator.set_contact_property(contact_properties)

        # define object to manipulate
        # todo: define meaningful weight and inertias for the box / other objects
        object_urdf_path = os.path.join(
            os.path.dirname(__file__), "definitions/additional_urdfs/cube.urdf"
        )

        if self.get_box_start is not None:
            self.box_start_pos, self.box_start_or = self.get_box_start(self)
            self.box_start_or_quat = p.getQuaternionFromEuler(list(self.box_start_or))

        self.box_id = p.loadURDF(  # todo: change to 'object_id'; throughout
            object_urdf_path,
            self.box_start_pos,  # todo: change to snake case
            self.box_start_or_quat,
            useFixedBase=0,
        )
        box_friction = np.random.uniform(*self.box_friction_range)
        p.changeDynamics(self.box_id, -1, lateralFriction=box_friction)

        # RANDY: change palm urdf here.
        palm_path = os.path.join(
            os.path.dirname(__file__), "definitions/additional_urdfs/palm.urdf"
        )

        self.palmId, self.palmConstraintId = load_constrained_urdf(
            palm_path,
            self.palm_start_pos,
            self.palm_start_or,
            physicsClient=self.physics_client,
        )

        p.changeDynamics(self.palmId, -1, lateralFriction=self.palm_friction)

        self.box_pos = self.box_start_pos
        self.box_or = self.box_start_or

    def get_cam_settings(self):
        opt_str = "--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0"
        # cam_width, cam_height = 1920, 1640
        cam_width, cam_height = (
            None,
            None,
        )
        if cam_width is not None and cam_height is not None:
            opt_str += " --width=%d --height=%d" % (cam_width, cam_height)

        cam_distance, cam_yaw, cam_pitch, cam_xyz_target = (
            3.0,
            45,
            -89,
            [0.0, 0.0, 1.5],
        )

        return opt_str, cam_distance, cam_yaw, cam_pitch, cam_xyz_target

    def check_success(self, binary=True):
        box_pos, box_or_quat = p.getBasePositionAndOrientation(self.box_id)
        box_or = p.getEulerFromQuaternion(box_or_quat)
        box_pos, box_or = np.array(box_pos), np.array(box_or)

        if binary:
            return np.degrees(box_or[2]) >= 45 and box_pos[2] > 0

        if box_pos[2] > 0:
            return np.degrees(box_or[2])
        return 0
