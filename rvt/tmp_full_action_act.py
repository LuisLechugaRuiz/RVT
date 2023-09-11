import torch
import numpy as np

import peract_colab.arm.utils as utils
from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.scene import Scene
import rvt.mvt.aug_utils as aug_utils
from rvt.utils.peract_utils import CAMERAS


class MoveArmThenGripperACT(ActionMode):
    """The arm action is first applied, followed by the gripper action using act policy."""

    def __init__(self, arm_action_mode, gripper_mode, act_executor):
        super(MoveArmThenGripperACT, self).__init__(arm_action_mode, gripper_mode)
        self.act_executor = act_executor
        self.threshold = 0.05

    def action(self, scene: Scene, action: np.ndarray):
        obs = scene.get_observation()
        obs_dict = vars(obs)
        obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
        print("DEBUG INIT TARGET", action[:7])
        target_action = self.to_action(pose=action[:7], gripper_open=action[8])
        print("TARGET ACTION:", target_action)
        target_reached = False
        try:
            while not target_reached:
                image_data = self.get_image(obs_dict)
                qpos = self.to_action(pose=obs_dict["gripper_pose"], gripper_open=obs_dict["gripper_open"])
                print("QPOS:", qpos)
                action_chunk = self.act_executor.step(
                    qpos, image_data, target_action
                )
                pred_action = action_chunk[0].cpu().numpy()
                print("PRED ACTION:", pred_action)
                self.arm_action_mode.action(scene, pred_action[:6], ignore_collisions=bool(action[8]))
                self.act_executor.iterate()  # Increase iteration to roll out executed poses. TODO: Remove and do it every step.
                ee_action = np.array(pred_action[7])
                self.gripper_action_mode.action(scene, ee_action)
                if (
                    self.euclidean_distance(qpos, target_action)
                    < self.threshold
                ):
                    success, _ = scene.task.success()
                    target_reached = success
        except Exception as e:
            print(f"Exception caught: {e}")        

    def euclidean_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def get_image(self, obs_dict):
        all_cam_images = []
        obs_dict = {
            k: np.transpose(v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
            for k, v in obs_dict.items()
            if isinstance(v, (np.ndarray, list))
        }
        for cam_name in CAMERAS:
            rgb = torch.from_numpy(obs_dict["%s_rgb" % cam_name])
            all_cam_images.append(rgb)
        # construct observations
        image_data = torch.stack(all_cam_images, axis=0)
        image_data = image_data / 255.0
        return image_data

    def to_action(self, pose, gripper_open):
        quat = utils.normalize_quaternion(pose[3:])
        if quat[-1] < 0:
            quat = -quat
        euler_rot = aug_utils.quaternion_to_euler_rad(quat)
        return np.concatenate([pose[:3], euler_rot, np.array([int(gripper_open)])])

    def action_shape(self, scene: Scene):
        return (9,)
