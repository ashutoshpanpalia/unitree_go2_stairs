# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from .unitree_go2_stairs_env_cfg import UnitreeGo2StairsEnvCfg


import gymnasium as gym
import torch
from isaaclab.sensors import ContactSensor, RayCaster

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

def define_velocity_markers() -> VisualizationMarkers:
    """
    Define markers to visualize current velocity (red) and target velocity (green) as arrows.
    """
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/velocityMarkers",
        markers={
            "current_vel": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.2, 0.2, 0.5),  # length scales the arrow
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # red
            ),
            "target_vel": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.2, 0.2, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # green
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


#(ref: anymal_c_env.py)

class UnitreeGo2StairsEnv(DirectRLEnv):
    cfg: UnitreeGo2StairsEnvCfg

    def __init__(self, cfg: UnitreeGo2StairsEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        #Initialize action as zeros and of size [num_env, action_space]
        #action space is computed from num of robot controllable joints defined in robot.py and not env_cfg.py
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        # Initialize X and Y linear velocity and yaw angular velocity commands(target velocity)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        #Initializing Torso(base), feets and legs
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*thigh")



    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        # add articulation to scene
        self.scene.articulations["robot"] = self._robot
        #add contact sensors to the scene(defined in env_cfg.py contact sensor on all robot)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # add terrain(plain as defined in env_cfg.py)
        #Also making terrain patchs all over and not using one plain ground
        # (in case we ever want todo rough terrain this is the way to go, patches and copies)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        #here we filter all the terrain patches and there collision/interaction in CPU
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


        #Viz initialization
        self.visualization_markers = define_velocity_markers()
        # Marker visualization setup
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)
        self.marker_offset[:, 2] = 0.5  # lift markers above ground

    def _visualize_markers(self):
        # Base locations
        base_pos = self._robot.data.root_pos_w + self.marker_offset

        # Velocities
        current_vel = self._robot.data.root_lin_vel_w[:, :2]
        target_vel  = self._commands[:, :2]

        # Angles
        angles_current = torch.atan2(current_vel[:, 1], current_vel[:, 0])
        angles_target  = torch.atan2(target_vel[:, 1], target_vel[:, 0])

        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)

        rot_current = math_utils.quat_from_angle_axis(angles_current, z_axis)
        rot_target  = math_utils.quat_from_angle_axis(angles_target, z_axis)

        # Speed-based scaling
        current_speed = torch.norm(current_vel, dim=1)
        target_speed  = torch.norm(target_vel, dim=1)

        max_speed = 2.0
        scale_current = torch.clamp(current_speed / max_speed, 0.05, 1.0)
        scale_target  = torch.clamp(target_speed / max_speed, 0.05, 1.0)

        # Build marker inputs
        positions = torch.vstack((base_pos, base_pos))
        orientations = torch.vstack((rot_current, rot_target))

        scales = torch.cat([
            scale_current.unsqueeze(1).repeat(1, 3),
            scale_target.unsqueeze(1).repeat(1, 3),
        ], dim=0)

        env_ids = torch.arange(self.num_envs, device=self.device)
        marker_ids = torch.cat([
            torch.zeros_like(env_ids),   # current_vel
            torch.ones_like(env_ids),    # target_vel
        ])

        self.visualization_markers.visualize(
            positions,
            orientations,
            scales=scales,
            marker_indices=marker_ids,
        )


    
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        


    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._processed_actions)
    
    def _get_observations(self) -> dict:
        #To compute action rate used in rewards      
        self._previous_actions = self._actions.clone()

        #For visualization Can be moved to some place which only gets activated during rendering
        self._visualize_markers()

        #Obs space is Linear velocity (3), Angular velocity (3), 
        #projected gravity/easy way in sim for orientation(3), target velocity(3),
        #Joint movement (12), Joint velocity (12), Action(12) = 48
        height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.4
            ).clip(-1.0, 1.0)
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    height_data,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        #Testing out height scanner data as training is not learning properly
        # scanner_z = self._height_scanner.data.pos_w[:, 2]
        # ground_z = self._height_scanner.data.ray_hits_w[:, :, 2].mean(dim=1)  # Average across rays
        # standing_height = (scanner_z - ground_z).mean()
        # print(f"Average standing height: {standing_height:.3f}m")



        return observations

    def _get_rewards(self) -> torch.Tensor:
        
        # linear velocity tracking
        #L2 loss plus mapping from [0 to inf) to [1 to 0]
        #making higher loss equals less reward
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        
        # yaw rate tracking
        #Same as velocity tracking loss and mapping
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        
        # z velocity tracking
        # Error to be multiplied with a negative cfg param,
        # #to penalize any movement in Z direction 
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        
        # angular velocity x/y
        # Same as Z velocity to penalize roll and pitch movement
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
 
        # joint torques
        # To be multiplied by a negative cfg to ensure lowest possible usage of torque 
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
 
        # joint acceleration
        # Same as joint torque logic
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
 
        # action rate 
        # To be multiplies with a negative cfg, to ensure least action rate, i.e. smooth
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        # feet air time
        # Encourage more feet air time, and more than 0.5 sec is better less thanthat is penalized
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.75) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )

        # undesired contacts
        #To penalize any contact with undesired bodies i.e. thighs, other undesired leads to episode termination
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)

        # flat orientation
        # To penalize tilt by multiplying the error to negative cfg 
        # projected gravity is a cheap sim way to get robot orientation
        
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        #Each error is multiplies by a 
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
       
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        #Termination citeria 1: Episode length exceeding
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        #TErmination criteria 2: Robot Base/torso touching the ground
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        #ref anymial_c_env.py
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands 
        # Vx Vy and Wz from [-1 to 1]
        #Command is gives at each episode
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        
        #Modifying just for video move Vx=1
        # self._commands[env_ids] = 0.0
        # self._commands[env_ids, 0] = 1.0 
        
        #Making the Yaw target to be zero to start learning to walk basic.
        # self._commands[env_ids, 2] = 0.0
        # self._commands[env_ids, 1] = 0.0

        #Since the robot has learned to to basic locomotion lets add yaw also


        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
       

