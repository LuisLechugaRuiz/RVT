from collections import defaultdict
import numpy as np
import torch
import os
import pickle

from peract_colab.arm.utils import stack_on_channel
from rvt.utils.get_dataset import get_act_dataset
from rvt.utils.peract_utils import (
    CAMERAS,
)


class ACTDataset:
    def __init__(
        self,
        tasks,
        batch_size,
        replay_storage_dir,
        data_folder,
        num_demos,
        num_workers,
        training,
        training_iterations,
        ckpt_dir,
    ):
        super(ACTDataset).__init__()
        self.dataset = get_act_dataset(
            tasks,
            batch_size,
            replay_storage_dir,
            data_folder,
            num_demos,
            False,
            num_workers,
            training=training,
        )
        self.norm_stats = self.get_norm_stats(
            dataset=self.dataset.dataset,
            training_iterations=training_iterations,
            ckpt_dir=ckpt_dir,
        )

    def get_data(self, sample):
        qpos = torch.from_numpy(sample["gripper_pose"].squeeze(1))
        target_pose = torch.from_numpy(sample["target_pose"].squeeze(1))
        action = torch.from_numpy(sample["target_actions"].squeeze(1))
        is_pad = torch.from_numpy(sample["is_pad"].squeeze(1))
        # new axis for different cameras
        all_cam_images = []
        for cam_name in CAMERAS:
            rgb = torch.from_numpy(sample["%s_rgb" % cam_name].squeeze(1))
            all_cam_images.append(rgb)
        # construct observations
        image_data = torch.stack(all_cam_images, axis=1)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        qpos = (qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        target_pose = (
            target_pose - self.norm_stats["target_pose_mean"]
        ) / self.norm_stats["target_pose_std"]
        action = (action - self.norm_stats["action_mean"]) / self.norm_stats[
            "action_std"
        ]

        return image_data, qpos, target_pose, action, is_pad

    @classmethod
    def get_norm_stats(self, dataset, training_iterations, ckpt_dir):
        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        if os.path.exists(stats_path):
            with open(stats_path, "rb") as f:
                return pickle.load(f)

        all_gripper_pose = []
        all_target_pose = []
        all_actions = []
        norm_stats = {}
        data_iter = iter(dataset)
        print("Training iterations", training_iterations)
        for i in range(training_iterations):
            raw_batch = next(data_iter)
            all_gripper_pose.append(torch.tensor(raw_batch["gripper_pose"].squeeze(1)))
            all_target_pose.append(torch.tensor(raw_batch["target_pose"].squeeze(1)))
            all_actions.append(torch.tensor(raw_batch["target_actions"].squeeze(1)))

        # normalize action data
        all_gripper_pose = torch.stack(all_gripper_pose)
        all_target_pose = torch.stack(all_target_pose)
        all_actions = torch.stack(all_actions)
        action_mean = all_actions.mean(dim=[0, 1], keepdim=True).squeeze()
        norm_stats["action_mean"] = action_mean
        action_std = all_actions.std(dim=[0, 1], keepdim=True).squeeze()
        action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping
        norm_stats["action_std"] = action_std

        # normalize gripper_pose data
        qpos_mean = all_gripper_pose.mean(dim=[0, 1], keepdim=True).squeeze()
        norm_stats["qpos_mean"] = qpos_mean
        qpos_std = all_gripper_pose.std(dim=[0, 1], keepdim=True).squeeze()
        qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping
        norm_stats["qpos_std"] = qpos_std

        # normalize target_pose data
        target_pose_mean = all_target_pose.mean(dim=[0, 1], keepdim=True).squeeze()
        norm_stats["target_pose_mean"] = target_pose_mean
        target_pose_std = all_target_pose.std(dim=[0, 1], keepdim=True).squeeze()
        target_pose_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping
        norm_stats["target_pose_std"] = target_pose_std
        print("target pose mean:", target_pose_mean)

        # save dataset stats
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        with open(stats_path, "wb") as f:
            pickle.dump(norm_stats, f)

        return norm_stats
