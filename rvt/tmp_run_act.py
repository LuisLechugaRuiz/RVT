import torch
import numpy as np


class ACTExecutor:
    # def __init__(self, policy, norm_stats, state_dim, num_queries): -> TODO: Enable when norm
    def __init__(self, policy, state_dim, num_queries, max_size=100):
        self.policy = policy
        # self.norm_stats = norm_stats -> TODO: Enable when norm
        self.prev_actions = torch.zeros([num_queries, state_dim]).cuda()
        self.num_queries = num_queries
        self.max_size = max_size
        self.all_time_actions = torch.zeros([max_size * num_queries, state_dim]).cuda()
        self.iteration = 0

    def _pre_process(self, qpos):
        return (qpos - self.norm_stats["qpos_mean"].cuda()) / self.norm_stats["qpos_std"].cuda()

    def _post_process(self, action):
        return action * self.norm_stats["action_std"].cuda() + self.norm_stats["action_mean"].cuda()

    def _temporal_ensembling(self, all_actions):
        all_actions = all_actions.squeeze(0)

        # Insert new actions into the rolling vector
        start_index = (self.iteration % self.max_size) * self.num_queries
        self.all_time_actions[start_index:start_index+self.num_queries] = all_actions

        # Extract actions for the current step
        indices = torch.arange(0, self.iteration * self.num_queries + 1, step=self.num_queries)
        actions_for_curr_step = self.all_time_actions[indices]

        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        print("DEBUG ACTIONS SHAPE:", actions_for_curr_step.shape)

        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
        print("DEBUG EXP WEIGHTS SHAPE:", exp_weights.shape)

        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
        print("DEBUG RAW ACTION:", raw_action)
        return raw_action

    def step(self, qpos, image):
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
        image = image.cuda().unsqueeze(0)
        # qpos = self._pre_process(qpos) TODO: Enable normalization
        all_actions = self.policy(qpos, image)
        # TODO: Enable after verification
        # raw_action = self._temporal_ensembling(all_actions)
        raw_action = all_actions.squeeze(0).cuda()
        # action = self._post_process(raw_action) TODO: Enable normalization
        action = raw_action
        return action

    def iterate(self):
        self.iteration += 1
