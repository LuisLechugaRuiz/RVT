# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

# initial source: https://colab.research.google.com/drive/1HAqemP4cE81SQ6QO1-N85j5bF4C0qLs0?usp=sharing
# adapted to support loading from disk for faster initialization time

# Adapted from: https://github.com/stepjam/ARM/blob/main/arm/c2farm/launch_utils.py
import os
import torch
import pickle
import logging
import numpy as np
from typing import List, Tuple

import clip
import peract_colab.arm.utils as utils

from peract_colab.rlbench.utils import get_stored_demo
from yarr.utils.observation_type import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from rlbench.backend.observation import Observation
from rlbench.demo import Demo

from rvt.utils.peract_utils import LOW_DIM_SIZE, IMAGE_SIZE, CAMERAS
from rvt.libs.peract.helpers.demo_loading_utils import keypoint_discovery
from rvt.libs.peract.helpers.utils import extract_obs


def create_replay(
    action_chunk_size: int,
    batch_size: int,
    timesteps: int,
    disk_saving: bool,
    cameras: list,
    voxel_sizes,
    replay_size=3e5,
):

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = 3 + 1
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_emb_dim = 512
    joint_positions_size = 7

    # low_dim_state
    observation_elements = []
    observation_elements.append(
        ObservationElement("low_dim_state", (LOW_DIM_SIZE,), np.float32)
    )

    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            ObservationElement(
                "%s_rgb" % cname,
                (
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_depth" % cname,
                (
                    1,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_point_cloud" % cname,
                (
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement(
                "%s_camera_extrinsics" % cname,
                (
                    4,
                    4,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_camera_intrinsics" % cname,
                (
                    3,
                    3,
                ),
                np.float32,
            )
        )

    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    observation_elements.extend(
        [
            ReplayElement("trans_action_indicies", (trans_indicies_size,), np.int32),
            ReplayElement(
                "rot_grip_action_indicies", (rot_and_grip_indicies_size,), np.int32
            ),
            ReplayElement("ignore_collisions", (ignore_collisions_size,), np.int32),
            ReplayElement("gripper_pose", (gripper_pose_size,), np.float32),
            ReplayElement(
                "lang_goal_embs",
                (
                    max_token_seq_len,
                    lang_emb_dim,
                ),  # extracted from CLIP's language encoder
                np.float32,
            ),
            ReplayElement(
                "lang_goal", (1,), object
            ),  # language goal string for debugging and visualization
            ReplayElement("joint_positions", (joint_positions_size,), np.float32),
            ReplayElement("actions", (action_chunk_size, joint_positions_size), np.float32),
            ReplayElement("is_pad", (action_chunk_size,), bool),
        ]
    )

    replay_buffer = (
        UniformReplayBuffer(  # all tuples in the buffer have equal sample weighting
            disk_saving=disk_saving,
            batch_size=batch_size,
            timesteps=timesteps,
            replay_capacity=int(replay_size),
            action_shape=(8,),  # 3 translation + 4 rotation quaternion + 1 gripper open
            action_dtype=np.float32,
            reward_shape=(),
            reward_dtype=np.float32,
            update_horizon=1,
            observation_elements=observation_elements,
            # extra_replay_elements=extra_replay_elements,
        )
    )
    return replay_buffer


# discretize translation, rotation, gripper open, and ignore collision actions
def _get_action(
    obs_tp1: Observation,
    rlbench_scene_bounds: List[float],  # metric 3D bounds of the scene
    voxel_sizes: List[int],
    rotation_resolution: int,
):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    trans_indicies = []
    bounds = np.array(rlbench_scene_bounds)
    for depth, vox_size in enumerate(
        voxel_sizes
    ):  # only single voxelization-level is used in PerAct
        index = utils.point_to_voxel_index(obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return (
        trans_indicies,
        rot_and_grip_indicies,
        np.concatenate([obs_tp1.gripper_pose, np.array([grip])])
    )


# extract CLIP language features for goal string
def _clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(
        clip_model.dtype
    )  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x, emb


def _add_keypoints_to_replay(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    demo: Demo,
    episode_keypoints: List[int],
    action_chunk_size: int,
    rlbench_scene_bounds: List[float],
    voxel_sizes: List[int],
    rotation_resolution: int,
    description: str = "",
    clip_model=None,
    device="cpu",
):
    k = 0
    for i in range(len(demo)):
        if i == episode_keypoints[k] or i == 0:
            if k < len(episode_keypoints) - 1:
                if i != 0:
                    k += 1
                # Get keypoint action
                keypoint = episode_keypoints[k]
                obs_tp1 = demo[keypoint]
                (
                    trans_indicies,
                    rot_grip_indicies,
                    action,
                ) = _get_action(
                    obs_tp1,
                    rlbench_scene_bounds,
                    voxel_sizes,
                    rotation_resolution,
                )

        terminal = i == len(demo) - 1
        reward = float(terminal) * 1.0 if terminal else 0

        obs = demo[i]
        obs_dict = extract_obs(
            obs,
            CAMERAS,
            t=k,
            episode_length=25,
        )
        tokens = clip.tokenize([description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        with torch.no_grad():
            lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
        obs_dict["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()

        final_obs = {
            "trans_action_indicies": trans_indicies,
            "rot_grip_action_indicies": rot_grip_indicies,
            "gripper_pose": obs_tp1.gripper_pose,
            "lang_goal": np.array([description], dtype=object),
        }
        # -------------------
        # Get joint position actions
        last_index = min(i + action_chunk_size, episode_keypoints[k])
        if i < len(demo) - 1:
            target_actions = [i.joint_positions for i in demo[i+1:last_index]]
        else:
            target_actions = []
        is_pad = [False] * len(target_actions)
        # Ensure target_actions size is action_chunk_size
        difference = action_chunk_size - len(target_actions)
        if len(target_actions) > 0:
            target_actions += [target_actions[-1]] * difference
        else:
            target_actions = [demo[i].joint_positions] * difference
        # Padding to true as we are filling with duplicated data.
        is_pad += [True] * difference

        extra_obs = {
            "joint_positions": obs.joint_positions,
            "actions": target_actions,
            "is_pad": is_pad,
        }
        obs_dict.update(final_obs)
        obs_dict.update(extra_obs)

        timeout = False
        replay.add(
            task,
            task_replay_storage_folder,
            action,
            reward,
            terminal,
            timeout,
            **obs_dict
        )
    # final step
    obs_dict_tp1 = extract_obs(
        obs_tp1,
        CAMERAS,
        t=k,
        episode_length=25,
    )
    obs_dict_tp1["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()

    obs_dict_tp1.pop("wrist_world_to_cam", None)
    obs_dict_tp1.update(final_obs)
    obs_dict_tp1.update(extra_obs)
    replay.add_final(task, task_replay_storage_folder, **obs_dict_tp1)


def fill_replay(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    start_idx: int,
    num_demos: int,
    action_chunk_size: str,
    rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
    voxel_sizes: List[int],
    rotation_resolution: int,
    data_path: str,
    episode_folder: str,
    variation_desriptions_pkl: str,
    clip_model=None,
    device="cpu",
):
    print("Replay disk saving:", replay._disk_saving)
    print("Storage:", task_replay_storage_folder)
    print("data_path", data_path)
    disk_exist = False
    if replay._disk_saving:
        if os.path.exists(task_replay_storage_folder):
            print(
                "[Info] Replay dataset already exists in the disk: {}".format(
                    task_replay_storage_folder
                ),
                flush=True,
            )
            disk_exist = True
        else:
            logging.info("\t saving to disk: %s", task_replay_storage_folder)
            os.makedirs(task_replay_storage_folder, exist_ok=True)

    if disk_exist:
        replay.recover_from_disk(task, task_replay_storage_folder)
    else:
        print("Filling replay ...")
        for d_idx in range(start_idx, start_idx + num_demos):
            print("Filling demo %d" % d_idx)
            demo = get_stored_demo(data_path=data_path, index=d_idx)

            # get language goal from disk
            varation_descs_pkl_file = os.path.join(
                data_path, episode_folder % d_idx, variation_desriptions_pkl
            )
            with open(varation_descs_pkl_file, "rb") as f:
                descs = pickle.load(f)

            # extract keypoints
            episode_keypoints = keypoint_discovery(demo)
            demo, episode_keypoints = clean_samples(demo, episode_keypoints)
            _add_keypoints_to_replay(
                replay,
                task,
                task_replay_storage_folder,
                demo,
                episode_keypoints,
                action_chunk_size,
                rlbench_scene_bounds,
                voxel_sizes,
                rotation_resolution,
                description=descs[0],
                clip_model=clip_model,
                device=device,
            )

        # save TERMINAL info in replay_info.npy
        task_idx = replay._task_index[task]
        with open(
            os.path.join(task_replay_storage_folder, "replay_info.npy"), "wb"
        ) as fp:
            np.save(
                fp,
                replay._store["terminal"][
                    replay._task_replay_start_index[
                        task_idx
                    ] : replay._task_replay_start_index[task_idx]
                    + replay._task_add_count[task_idx].value
                ],
            )

        print("Replay filled with demos.")


def clean_samples(demo: Demo, keypoints: List[int], joint_diff_threshold: float = 0.02) -> Tuple[Demo, List[int]]:
    prev_joint_position = np.zeros(7,)
    indices_to_remove = []
    proximity_threshold = 10  # Minimum distance to a keypoint to not be removed

    for i, sample in enumerate(demo._observations):
        diff = sample.joint_positions - prev_joint_position
        is_close_to_keypoint = any(abs(k - i) < proximity_threshold for k in keypoints)

        if np.all(np.abs(diff) < joint_diff_threshold) and i not in keypoints and not is_close_to_keypoint:
            indices_to_remove.append(i)
        else:
            prev_joint_position = sample.joint_positions

    indices_to_remove.sort(reverse=True)

    for idx in indices_to_remove:
        # Adjust the keypoints that come after this index.
        keypoints = [k - 1 if k > idx else k for k in keypoints]

        # Remove the observation at this index.
        del demo._observations[idx]

    return demo, keypoints
