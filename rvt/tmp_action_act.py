import numpy as np

from rlbench.action_modes.action_mode import ArmActionMode
from rlbench.backend.scene import Scene


class EndEffectorPoseViaACT(ArmActionMode):
    """Used to run ACT policy"""

    def action(self, scene: Scene, action: np.ndarray, ignore_collisions):
        path = scene.robot.arm.get_linear_path(
            position=action[:3],
            euler=action[3:],
            ignore_collisions=ignore_collisions,
        )
        local_point_reached = False
        while not local_point_reached:
            local_point_reached = path.step()
            scene.step()

    def action_shape(self, scene: Scene) -> tuple:
        return (6,)
