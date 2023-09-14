import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.scene import Scene
from rvt.utils.peract_utils import CAMERAS


class MoveArmThenGripperACT(ActionMode):
    """The arm action is first applied, followed by the gripper action using act policy."""

    def __init__(self, arm_action_mode, gripper_mode, act_executor):
        super(MoveArmThenGripperACT, self).__init__(arm_action_mode, gripper_mode)
        self.act_executor = act_executor
        self.key_points = {}
        self.last_action = np.zeros(9,)
        self.threshold = 0.02
        self.images = 0  # TODO: Remove when we draw_circle_on_image.
        self.keypoints = 0 # TODO: Remove later!
        self.step_iterations = 20

    def action(self, scene: Scene, action: np.ndarray):
        target_reached = False

        # DEBUG
        obs = scene.get_observation()
        obs_dict = self.get_obs_dict(obs)
        gripper_pose = obs_dict["gripper_pose"]
        print(f"IT: {self.keypoints}, CURRENT POSE: {gripper_pose}, NEW KEYPOINT: {action}")
        print("DEBUG - Gripper open", obs_dict["gripper_open"])
        self.keypoints += 1

        # Only fill key point if action is not equal to last, this is due to the stuck condition, remove it after.
        if self.euclidean_distance(self.last_action[:7], action[:7]) > 0.1:
            self.fill_key_point(action[:3], scene.get_observation())
        first_action = True  # TODO :REMOVE
        stuck_iterations = 0
        actions = 0
        prev_qpos = np.zeros(7,)
        while not target_reached:
            if actions % self.step_iterations == 0:
                obs = scene.get_observation()
                obs_dict = self.get_obs_dict(obs)
                image_data = self.get_image(obs)
                qpos = obs_dict["joint_positions"]
                # print("QPOS:", qpos)
                action_chunk = self.act_executor.step(
                    qpos, image_data
                ).cpu().numpy()
            # print("PRED ACTION:", pred_action[0])
            # TODO: REMOVE
            pred_action = action_chunk[0]
            action_chunk = action_chunk[1:]
            if first_action:
                print("Current position:", qpos)
                print("First action:", pred_action)
                print("NEXT ACTIONS:", action_chunk)
                first_action = False
            self.arm_action_mode.action(scene, pred_action)
            self.act_executor.iterate()  # Increase iteration to roll out executed poses. TODO: Should we remove this and call the model every step?
            # print("Euclidean distance is:", self.euclidean_distance(obs_dict["gripper_pose"], action[:7]))
            # print("Current gripper pose:", obs_dict["gripper_pose"])
            # print("Target action:", action[:7])
            if (
                self.euclidean_distance(obs_dict["gripper_pose"][:3], action[:3])
                < self.threshold
            ):
                print("TARGET REACHED!, last action was:", pred_action)
                gripper_pose = obs_dict["gripper_pose"]
                diff = self.euclidean_distance(obs_dict["gripper_pose"][:3], action[:3])
                print(f"IT: {self.keypoints}, CURRENT POSE: {gripper_pose}, KEYPOINT: {action}, DIFF: {diff}")
                target_reached = True
            if (self.euclidean_distance(qpos, prev_qpos)) < 0.02:
                stuck_iterations += 1
                if stuck_iterations > 20:
                    print("IS STUCK!!")
                    target_reached = True  # Leave the loop, retrieve new action point or fail after timeout.
            else:
                stuck_iterations = 0
            actions += 1
            prev_qpos = qpos
        # ee_action = np.atleast_1d(action[7]) TODO: Check if remove.
        # self.gripper_action_mode.action(scene, ee_action)
        self.last_action = action

    def euclidean_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def get_obs_dict(self, obs):
        obs_dict = vars(obs)
        obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
        return obs_dict

    def get_image(self, obs):
        all_cam_images = []
        obs_dict = self.get_obs_dict(obs)
        obs_dict = {
            k: np.transpose(v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
            for k, v in obs_dict.items()
            if isinstance(v, (np.ndarray, list))
        }

        for cam_name in CAMERAS:
            keypoint_channel = np.zeros_like(obs_dict['%s_rgb' % cam_name][0])
            if self.key_points[cam_name]:
                px, py = self.key_points[cam_name]
                keypoint_channel[int(py), int(px)] = 1.0
            rgba_image = np.concatenate([obs_dict['%s_rgb' % cam_name], keypoint_channel[np.newaxis, :, :]], axis=0)
            rgba = torch.from_numpy(rgba_image).float()
            all_cam_images.append(rgba)
        # construct observations
        image_data = torch.stack(all_cam_images, axis=0)
        image_data[:, :3, :, :] = image_data[:, :3, :, :] / 255.0
        return image_data

    def fill_key_point(self, keypoint, obs):
        obs_dict = self.get_obs_dict(obs)
        for cam_name in CAMERAS:
            height, width = obs_dict['%s_rgb' % cam_name].shape[0:2]
            pixels = self._point_to_pixel_index(
                keypoint,
                obs.misc['%s_camera_extrinsics' % cam_name],
                obs.misc['%s_camera_intrinsics' % cam_name],
                width,
                height)
            self.key_points[cam_name] = pixels
            if pixels:
                px, py = pixels
                path = "data/pictures"
                if not os.path.exists(path):
                    os.mkdir(path)
                self.draw_circle_on_image(obs_dict['%s_rgb' % cam_name], px, py, save_path=f"{path}/image{self.images}")
                self.images += 1

    # Duplicated from dataset -> TODO: Get from file when refactor.
    def _point_to_pixel_index(
            self,
            point: np.ndarray,
            extrinsics: np.ndarray,
            intrinsics: np.ndarray,
            image_width: int,
            image_height: int):

        point = np.array([point[0], point[1], point[2], 1])
        world_to_cam = np.linalg.inv(extrinsics)
        point_in_cam_frame = world_to_cam.dot(point)
        px, py, pz = point_in_cam_frame[:3]

        # Check if the point is behind the camera
        if pz <= 0:
            return None

        px = 2 * intrinsics[0, 2] - int(-intrinsics[0, 0] * (px / pz) + intrinsics[0, 2])
        py = 2 * intrinsics[1, 2] - int(-intrinsics[1, 1] * (py / pz) + intrinsics[1, 2])

        # Clip px and py to the image bounds
        px = np.clip(px, 0, image_width-1)
        py = np.clip(py, 0, image_height-1)

        return px, py

    # TODO: Remove, only for debugging.
    def draw_circle_on_image(self, image_np, px, py, save_path):
        # If image data is uint8 [0, 255], convert to float [0, 1] for display
        if image_np.dtype == np.uint8:
            image_np = image_np.astype(float) / 255

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(image_np)

        # Define the radius for the circle
        radius = 5

        # Create a circle with the specified center and radius
        circle = patches.Circle((px, py), radius, color='red', fill=False)

        # Add the circle to the axis
        ax.add_patch(circle)

        # Remove axis
        ax.axis('off')

        # Save the image without displaying it
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # Close the figure

    def action_shape(self, scene: Scene):
        return (9,)
