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
    lang_feat_dim = 1024
    lang_emb_dim = 512

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
        ]
    )

    extra_replay_elements = [
        ReplayElement("demo", (), bool),
        ReplayElement("keypoint_idx", (), int),
        ReplayElement("episode_idx", (), int),
        ReplayElement("keypoint_frame", (), int),
        ReplayElement("next_keypoint_frame", (), int),
        ReplayElement("sample_frame", (), int),
    ]

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
            extra_replay_elements=extra_replay_elements,
        )
    )
    return replay_buffer


def create_act_replay(
    batch_size: int,
    timesteps: int,
    disk_saving: bool,
    cameras: list,
    num_images: int = 5,
    replay_size=3e5,
):
    joint_positions_size = 7
    max_token_seq_len = 77
    lang_emb_dim = 512

    observation_elements = []
    observation_elements.append(
        ObservationElement("low_dim_state", (LOW_DIM_SIZE,), np.float32)
    )

    # rgb, intrinsics, extrinsics
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
    observation_elements.append(
        ObservationElement(
            "heatmap",
            (
                num_images,
                1,
                220, # TODO: Get from cfg
                220,
            ),
            np.float32,
        )
    )
    observation_elements.extend(
        [
            ReplayElement("joint_positions", (joint_positions_size,), np.float32),
            ReplayElement("actions", (20, joint_positions_size), np.float32),
            ReplayElement("is_pad", (20,), bool),
        ]
    )
    replay_buffer = (
        UniformReplayBuffer(  # all tuples in the buffer have equal sample weighting
            disk_saving=disk_saving,
            batch_size=batch_size,
            timesteps=timesteps,
            replay_capacity=int(replay_size),
            action_shape=(20, joint_positions_size), # Same as target_actions
            action_dtype=np.float32,
            reward_shape=(),
            reward_dtype=np.float32,
            update_horizon=1,
            observation_elements=observation_elements,
        )
    )
    return replay_buffer


# discretize translation, rotation, gripper open, and ignore collision actions
def _get_action(
    obs_tp1: Observation,
    obs_tm1: Observation,
    rlbench_scene_bounds: List[float],  # metric 3D bounds of the scene
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs_tm1.ignore_collisions)
    for depth, vox_size in enumerate(
        voxel_sizes
    ):  # only single voxelization-level is used in PerAct
        index = utils.point_to_voxel_index(obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return (
        trans_indicies,
        rot_and_grip_indicies,
        ignore_collisions,
        np.concatenate([obs_tp1.gripper_pose, np.array([grip])]),
        attention_coordinates,
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


# add individual data points to a replay
def _add_keypoints_to_replay(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    episode_idx: int,
    sample_frame: int,
    inital_obs: Observation,
    demo: Demo,
    episode_keypoints: List[int],
    cameras: List[str],
    rlbench_scene_bounds: List[float],
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    next_keypoint_idx: int,
    description: str = "",
    clip_model=None,
    device="cpu",
):
    prev_action = None
    obs = inital_obs
    for k in range(
        next_keypoint_idx, len(episode_keypoints)
    ):  # confused here, it seems that there are many similar samples in the replay
        keypoint = episode_keypoints[k]
        obs_tp1 = demo[keypoint]
        obs_tm1 = demo[max(0, keypoint - 1)]
        (
            trans_indicies,
            rot_grip_indicies,
            ignore_collisions,
            action,
            attention_coordinates,
        ) = _get_action(
            obs_tp1,
            obs_tm1,
            rlbench_scene_bounds,
            voxel_sizes,
            rotation_resolution,
            crop_augmentation,
        )

        terminal = k == len(episode_keypoints) - 1
        reward = float(terminal) * 1.0 if terminal else 0

        obs_dict = extract_obs(
            obs,
            CAMERAS,
            t=k - next_keypoint_idx,
            prev_action=prev_action,
            episode_length=25,
        )
        tokens = clip.tokenize([description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        with torch.no_grad():
            lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
        obs_dict["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()

        prev_action = np.copy(action)

        if k == 0:
            keypoint_frame = -1
        else:
            keypoint_frame = episode_keypoints[k - 1]
        others = {
            "demo": True,
            "keypoint_idx": k,
            "episode_idx": episode_idx,
            "keypoint_frame": keypoint_frame,
            "next_keypoint_frame": keypoint,
            "sample_frame": sample_frame,
        }
        final_obs = {
            "trans_action_indicies": trans_indicies,
            "rot_grip_action_indicies": rot_grip_indicies,
            "gripper_pose": obs_tp1.gripper_pose,
            "lang_goal": np.array([description], dtype=object),
        }

        others.update(final_obs)
        others.update(obs_dict)

        timeout = False
        replay.add(
            task,
            task_replay_storage_folder,
            action,
            reward,
            terminal,
            timeout,
            **others
        )
        obs = obs_tp1
        sample_frame = keypoint

    # final step
    obs_dict_tp1 = extract_obs(
        obs_tp1,
        CAMERAS,
        t=k + 1 - next_keypoint_idx,
        prev_action=prev_action,
        episode_length=25,
    )
    obs_dict_tp1["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()

    obs_dict_tp1.pop("wrist_world_to_cam", None)
    obs_dict_tp1.update(final_obs)
    replay.add_final(task, task_replay_storage_folder, **obs_dict_tp1)


def fill_replay(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    start_idx: int,
    num_demos: int,
    demo_augmentation: bool,
    demo_augmentation_every_n: int,
    cameras: List[str],
    rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
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
            next_keypoint_idx = 0
            for i in range(len(demo) - 1):
                if not demo_augmentation and i > 0:
                    break
                if i % demo_augmentation_every_n != 0:  # choose only every n-th frame
                    continue

                obs = demo[i]
                desc = descs[0]
                # if our starting point is past one of the keypoints, then remove it
                while (
                    next_keypoint_idx < len(episode_keypoints)
                    and i >= episode_keypoints[next_keypoint_idx]
                ):
                    next_keypoint_idx += 1
                if next_keypoint_idx == len(episode_keypoints):
                    break
                _add_keypoints_to_replay(
                    replay,
                    task,
                    task_replay_storage_folder,
                    d_idx,
                    i,
                    obs,
                    demo,
                    episode_keypoints,
                    cameras,
                    rlbench_scene_bounds,
                    voxel_sizes,
                    rotation_resolution,
                    crop_augmentation,
                    next_keypoint_idx=next_keypoint_idx,
                    description=desc,
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


# Duplicating to avoid modifying source code. TODO: Merge both logics into a single function.
def fill_act_replay(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    start_idx: int,
    num_demos: int,
    data_path: str,
    action_chunk_size: str,
    episode_folder: str,
    variation_desriptions_pkl: str,
    clip_model,
    rvt_agent,
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
        print("Filling act replay ...")
        for d_idx in range(start_idx, start_idx + num_demos):
            print("Filling demo %d" % d_idx)
            demo = get_stored_demo(data_path=data_path, index=d_idx)

            # get language goal from disk
            varation_descs_pkl_file = os.path.join(
                data_path, episode_folder % d_idx, variation_desriptions_pkl
            )
            with open(varation_descs_pkl_file, "rb") as f:
                descs = pickle.load(f)
            description = descs[0]

            # extract keypoints
            episode_keypoints = keypoint_discovery(demo)
            demo, episode_keypoints = clean_samples(demo, episode_keypoints)
            _add_joint_positions(
                replay,
                task,
                task_replay_storage_folder,
                demo,
                episode_keypoints,
                action_chunk_size,
                description,
                clip_model,
                rvt_agent,
                device,
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


def _add_joint_positions(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    demo: Demo,
    episode_keypoints: List[int],
    action_chunk_size: int,
    description: str,
    clip_model,
    rvt_agent,
    device: str,
):
    k = 0
    for i in range(len(demo) - 1):
        calculate_heatmap = False
        if i == 0:
            # Get heatmap at first observation
            calculate_heatmap = True
        if i == episode_keypoints[k]:
            if k < len(episode_keypoints) - 1:
                k += 1
                # Get heatmap on new keypoint
                calculate_heatmap = True

        obs = demo[i]
        obs_dict = extract_camera_data(obs, CAMERAS)

        obs_dict['low_dim_state'] = get_low_dim_state(obs, t=k, episode_length=25)

        # Fill actions_chunk
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
            "is_pad": is_pad
        }
        obs_dict.update(extra_obs)

        if calculate_heatmap:
            observation = obs_dict.copy()
            tokens = clip.tokenize([description]).numpy()
            token_tensor = torch.from_numpy(tokens).unsqueeze(0).to(device)
            observation["lang_goal_tokens"] = token_tensor
            observation = {
                k: torch.tensor(v).to(device) if isinstance(v, np.ndarray) else v.to(device)
                for k, v in observation.items()
                if isinstance(v, (torch.Tensor, np.ndarray))
            }
            observation = expand_obs_dict(observation, CAMERAS)
            rvt_result = rvt_agent.act(step=None, observation=observation)
            heatmap = rvt_result.observation_elements["heatmap"]
        obs_dict["heatmap"] = heatmap

        terminal = k == len(episode_keypoints) - 1
        reward = float(terminal) * 1.0 if terminal else 0
        timeout = False
        replay.add(
            task,
            task_replay_storage_folder,
            action=target_actions,
            reward=reward,
            terminal=terminal,
            timeout=timeout,
            **obs_dict
        )
    # Handling the final observation:
    obs_dict = extract_camera_data(demo[-1], CAMERAS)
    final_obs = {
        "joint_positions": demo[-1].joint_positions,
        "actions": [demo[-1].joint_positions] * action_chunk_size,
        "is_pad": [True] + [False] * (action_chunk_size - 1)
    }
    final_obs["heatmap"] = heatmap
    final_obs['low_dim_state'] = get_low_dim_state(obs, t=k, episode_length=25)
    obs_dict.update(final_obs)
    replay.add_final(task, task_replay_storage_folder, **obs_dict)


def clean_samples(demo: Demo, keypoints: List[int], joint_diff_threshold: float = 0.02) -> Tuple[Demo, List[int]]:
    prev_joint_position = np.zeros(7,)
    indices_to_remove = []
    proximity_threshold = 10  # Minimum distance to a keypoint to not be removed

    # print(f"Last keypoint before adjustment: {demo[keypoints[-1]].joint_positions}")
    # print(f"Before Adjusting Keypoints: {keypoints}, demo length: {len(demo._observations)}")
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
    # print(f"Last keypoint after adjustment: {demo[keypoints[-1]].joint_positions}")
    # print(f"After Adjusting Keypoints: {keypoints}, demo length: {len(demo._observations)}")

    return demo, keypoints


def extract_camera_data(
    obs: Observation,
    cameras,
):
    obs_dict = vars(obs)
    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
    camera_obs_dict = {}
    obs_dict = {k: np.transpose(v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
                for k, v in obs_dict.items() if isinstance(v, (np.ndarray, list))}
    for camera_name in cameras:
        camera_obs_dict['%s_rgb' % camera_name] = obs_dict['%s_rgb' % camera_name]
        camera_obs_dict['%s_point_cloud' % camera_name] = obs_dict['%s_point_cloud' % camera_name]
        camera_obs_dict['%s_camera_extrinsics' % camera_name] = obs.misc['%s_camera_extrinsics' % camera_name]
        camera_obs_dict['%s_camera_intrinsics' % camera_name] = obs.misc['%s_camera_intrinsics' % camera_name]

    return camera_obs_dict


def get_low_dim_state(obs: Observation, t: int, episode_length: int):
    if obs.gripper_joint_positions is not None:
        obs.gripper_joint_positions = np.clip(
            obs.gripper_joint_positions, 0., 0.04)

    robot_state = np.array([
        obs.gripper_open,
        *obs.gripper_joint_positions])

    low_dim_state = np.array(robot_state, dtype=np.float32)

    # add timestep to low_dim_state
    time = (1. - (t / float(episode_length - 1))) * 2. - 1.
    low_dim_state = np.concatenate([low_dim_state, [time]]).astype(np.float32)

    return low_dim_state


def expand_obs_dict(
    obs_dict,
    cameras,
):
    obs_dict["low_dim_state"] = obs_dict["low_dim_state"].unsqueeze(0).unsqueeze(0).float()
    for camera_name in cameras:
        obs_dict['%s_rgb' % camera_name] = obs_dict['%s_rgb' % camera_name].unsqueeze(0).unsqueeze(0).float()
        obs_dict['%s_point_cloud' % camera_name] = obs_dict['%s_point_cloud' % camera_name].unsqueeze(0).unsqueeze(0).float()
    return obs_dict
