# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from unitree_go2_stairs.robots.unitree import UNITREE_GO2_CFG
 
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

STAIRS_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.50,
            step_height_range=(0.02, 0.1),
            step_width=0.3,
            platform_width=1.50,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.50,
            step_height_range=(0.02, 0.1),
            step_width=0.3,
            platform_width=1.50,
            border_width=1.0,
            holes=False,
        ),

        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.1, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
    },
)




@configclass
class UnitreeGo2StairsEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    # - spaces definition
    action_space = 12
    observation_space = 235 #48 #235
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )


    # robot(s)
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAIRS_TERRAIN_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # Contact sensors for feet (basic)
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # events
    #events: EventCfg = EventCfg()
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    # custom parameters/scales

    # action scale: 
    action_scale = 0.5

    # reset joint noise:
    lin_vel_reward_scale = 2.750
    yaw_rate_reward_scale = 0.75
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.1
    joint_torque_reward_scale = -2.5e-6
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.005
    feet_air_time_reward_scale = 1.0
    undesired_contact_reward_scale = -2.0
    # flat_orientation_reward_scale = -4.0

    flat_orientation_reward_scale = -0.0

    # lin_vel_reward_scale = 1.750
    # yaw_rate_reward_scale = 0.5
    # z_vel_reward_scale = -1.70
    # ang_vel_reward_scale = -0.05
    # joint_torque_reward_scale = -2.5e-5
    # joint_accel_reward_scale = -2.5e-7
    # action_rate_reward_scale = -0.005
    # feet_air_time_reward_scale = 0.750
    # undesired_contact_reward_scale = -1.50