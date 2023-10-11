# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import sys
import shutil
import torch
import clip

from rvt.libs.peract.helpers.utils import extract_obs
from rvt.utils.rvt_utils import ForkedPdb
from rvt.utils.dataset import create_act_replay, create_replay, fill_act_replay, fill_replay
from rvt.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    EPISODE_FOLDER,
    VARIATION_DESCRIPTIONS_PKL,
    DEMO_AUGMENTATION_EVERY_N,
    ROTATION_RESOLUTION,
    VOXEL_SIZES,
)
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer


def get_dataset(
    tasks,
    action_chunk_size,
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_TEST,
    TRAIN_REPLAY_STORAGE_DIR,
    TEST_REPLAY_STORAGE_DIR,
    data_folder,
    NUM_TRAIN,
    NUM_VAL,
    refresh_replay,
    device,
    num_workers,
    only_train,
    sample_distribution_mode="transition_uniform",
):

    train_replay_buffer = create_replay(
        action_chunk_size,
        batch_size=BATCH_SIZE_TRAIN,
        timesteps=1,
        disk_saving=True,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES,
    )
    if not only_train:
        test_replay_buffer = create_replay(
            action_chunk_size,
            batch_size=BATCH_SIZE_TEST,
            timesteps=1,
            disk_saving=True,
            cameras=CAMERAS,
            voxel_sizes=VOXEL_SIZES,
        )

    # load pre-trained language model
    try:
        clip_model, _ = clip.load("RN50", device="cpu")  # CLIP-ResNet50
        clip_model = clip_model.to(device)
        clip_model.eval()
    except RuntimeError:
        print("WARNING: Setting Clip to None. Will not work if replay not on disk.")
        clip_model = None

    for task in tasks:  # for each task
        # print("---- Preparing the data for {} task ----".format(task), flush=True)
        EPISODES_FOLDER_TRAIN = f"train/{task}/all_variations/episodes"
        EPISODES_FOLDER_VAL = f"val/{task}/all_variations/episodes"
        data_path_train = os.path.join(data_folder, EPISODES_FOLDER_TRAIN)
        data_path_val = os.path.join(data_folder, EPISODES_FOLDER_VAL)
        train_replay_storage_folder = f"{TRAIN_REPLAY_STORAGE_DIR}/{task}"
        test_replay_storage_folder = f"{TEST_REPLAY_STORAGE_DIR}/{task}"

        # if refresh_replay, then remove the existing replay data folder
        if refresh_replay:
            print("[Info] Remove exisitng replay dataset as requested.", flush=True)
            if os.path.exists(train_replay_storage_folder) and os.path.isdir(
                train_replay_storage_folder
            ):
                shutil.rmtree(train_replay_storage_folder)
                print(f"remove {train_replay_storage_folder}")
            if os.path.exists(test_replay_storage_folder) and os.path.isdir(
                test_replay_storage_folder
            ):
                shutil.rmtree(test_replay_storage_folder)
                print(f"remove {test_replay_storage_folder}")

        # print("----- Train Buffer -----")
        fill_replay(
            replay=train_replay_buffer,
            task=task,
            task_replay_storage_folder=train_replay_storage_folder,
            start_idx=0,
            num_demos=NUM_TRAIN,
            action_chunk_size=action_chunk_size,
            rlbench_scene_bounds=SCENE_BOUNDS,
            voxel_sizes=VOXEL_SIZES,
            rotation_resolution=ROTATION_RESOLUTION,
            data_path=data_path_train,
            episode_folder=EPISODE_FOLDER,
            variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
            clip_model=clip_model,
            device=device,
        )

        if not only_train:
            # print("----- Test Buffer -----")
            fill_replay(
                replay=test_replay_buffer,
                task=task,
                task_replay_storage_folder=test_replay_storage_folder,
                start_idx=0,
                num_demos=NUM_VAL,
                action_chunk_size=action_chunk_size,
                rlbench_scene_bounds=SCENE_BOUNDS,
                voxel_sizes=VOXEL_SIZES,
                rotation_resolution=ROTATION_RESOLUTION,
                data_path=data_path_val,
                episode_folder=EPISODE_FOLDER,
                variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
                clip_model=clip_model,
                device=device,
            )

    # delete the CLIP model since we have already extracted language features
    del clip_model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    # wrap buffer with PyTorch dataset and make iterator
    train_wrapped_replay = PyTorchReplayBuffer(
        train_replay_buffer,
        sample_mode="random",
        num_workers=num_workers,
        sample_distribution_mode=sample_distribution_mode,
    )
    train_dataset = train_wrapped_replay.dataset()

    if only_train:
        test_dataset = None
    else:
        test_wrapped_replay = PyTorchReplayBuffer(
            test_replay_buffer,
            sample_mode="enumerate",
            num_workers=num_workers,
        )
        test_dataset = test_wrapped_replay.dataset()
    return train_dataset, test_dataset


def get_act_dataset(
    tasks,
    batch_size,
    replay_storage_dir,
    data_folder,
    num_demos,
    refresh_replay,
    num_workers,
    training,
    sample_distribution_mode="transition_uniform",
):
    replay_buffer = create_act_replay(
        batch_size=batch_size,
        timesteps=1,
        disk_saving=True,
        cameras=CAMERAS,
    )
    for task in tasks:
        if training:
            folder = "train"
        else:
            folder = "test"
        EPISODES_FOLDER = f"{folder}/{task}/all_variations/episodes"
        data_path = os.path.join(data_folder, EPISODES_FOLDER)
        replay_storage_folder = f"{replay_storage_dir}/{task}"

        # if refresh_replay, then remove the existing replay data folder
        if refresh_replay:
            print("[Info] Remove existing replay dataset as requested.", flush=True)
            if os.path.exists(replay_storage_folder) and os.path.isdir(
                replay_storage_folder
            ):
                shutil.rmtree(replay_storage_folder)
                print(f"remove {replay_storage_folder}")

        fill_act_replay(
            replay=replay_buffer,
            task=task,
            task_replay_storage_folder=replay_storage_folder,
            start_idx=0,
            num_demos=num_demos,
            data_path=data_path,
            action_chunk_size=20,  # TODO: Tune
        )

    # wrap buffer with PyTorch dataset and make iterator
    wrapped_replay = PyTorchReplayBuffer(
        replay_buffer,
        sample_mode='random',
        num_workers=num_workers,
        sample_distribution_mode=sample_distribution_mode,
    )
    dataset = wrapped_replay.dataset()

    return dataset
