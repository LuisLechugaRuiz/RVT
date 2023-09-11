import torch
import numpy as np


class ACTExecutor:
    def __init__(self, policy, norm_stats, state_dim, num_queries):
        self.policy = policy
        self.norm_stats = norm_stats
        self.prev_actions = torch.zeros([num_queries, state_dim]).cuda()
        self.num_queries = num_queries
        self.exp_weights = self.calculate_exp_weights()
        self.iteration = 0

    def _pre_process(self, qpos):
        return (qpos - self.norm_stats["qpos_mean"].cuda()) / self.norm_stats["qpos_std"].cuda()

    def _post_process(self, action):
        return action * self.norm_stats["action_std"].cuda() + self.norm_stats["action_mean"].cuda()

    def calculate_exp_weights(self):
        k = 0.01
        exp_weights = np.exp(-k * np.arange(self.num_queries))
        exp_weights = exp_weights / exp_weights.sum()
        return torch.from_numpy(exp_weights).float().cuda().unsqueeze(-1)

    def _temporal_ensembling(self, all_actions):
        all_actions = all_actions.squeeze(0)
        if self.iteration > 0:
            # Roll both actions and weights
            self.prev_actions = torch.roll(self.prev_actions, shifts=-self.iteration, dims=0)
            self.exp_weights = torch.roll(self.exp_weights, shifts=-self.iteration, dims=0)

            # Replace the last 'iteration' actions with the new actions
            self.prev_actions[-self.iteration:] = all_actions[:self.iteration]
            self.iteration = 0
        else:
            # Just add the new actions to the buffer if no iteration is performed
            self.prev_actions += all_actions

        weighted_actions = self.prev_actions * self.exp_weights
        raw_action = weighted_actions.sum(dim=0, keepdim=True)

        return raw_action

    def step(self, qpos, image, target_pose):
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
        target_pose = torch.from_numpy(target_pose).float().cuda().unsqueeze(0)
        image = image.cuda().unsqueeze(0)
        qpos = self._pre_process(qpos)
        all_actions = self.policy(qpos, image, target_pose)
        raw_action = self._temporal_ensembling(all_actions)
        raw_action = raw_action.squeeze(0).cuda()
        action = self._post_process(raw_action)
        return action

    def iterate(self):
        self.iteration += 1
