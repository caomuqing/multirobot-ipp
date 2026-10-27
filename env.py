import os
import numpy as np
from itertools import product
from utils.graph_controller import GraphController
from utils.target_controller import VTSPGaussian
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from gaussian_process import GaussianProcessWrapper
from arguments import arg
from stamp_mapper_multi import STAMP_FR_GPF_IK


def _mapper_cls():
    """Covariance-representation ablation switch: 'diag' (default) or 'full'."""
    if getattr(arg, 'mapper_impl', 'diag') == 'full':
        from stamp_mapper_fullcov import STAMP_FR_GPF_IK_FullCov
        return STAMP_FR_GPF_IK_FullCov
    return STAMP_FR_GPF_IK
import time

def add_t(X, t: float):
    # 给 (x, y) 数组添加时间维度
    return np.concatenate((X, np.zeros((X.shape[0], 1)) + t), axis=1)

class Env:
    def __init__(self, graph_size, k_size, budget_size=None, target_size=None, start=None, obstacles=None):
        self.graph_size = graph_size
        self.k_size = k_size
        self.budget_init = budget_size
        self.budget = budget_size
        self.n_targets = target_size
        self.obstacles = obstacles

        # NEW for multi-agent
        self.num_agents = arg.num_agents  # 读ma数量
        # 指定初始位置
        if start is None:
            self.start_positions = np.random.rand(self.num_agents, 2)
        else:
            self.start_positions = np.array(start)
            if self.start_positions.shape[0] != self.num_agents:
                raise ValueError("给定 start 的数量和 num_agents 不符！")
        self.curr_t = 0.0
        # 记录每个 agent 拜访各目标的时间
        self.visit_t = [[] for _ in range(self.n_targets)]

        # Get elevated regions from arguments
        elevated_regions = getattr(arg, 'elevated_regions', [])
        flight_altitude = getattr(arg, 'mapper_flight_altitude', 0.1)
        
        self.graph_ctrl = GraphController(self.graph_size, self.start_positions[0:1], self.k_size, 
                                         self.obstacles, elevated_regions=elevated_regions, 
                                         flight_altitude=flight_altitude)
        self.node_coords, self.graph = self.graph_ctrl.generate_graph()
        self.elevated_regions = elevated_regions  # Store for visualization
        
        self.underlying_distrib = None
        self.ground_truth = None
        self.high_info_idx = None

        # GP
        self.gp_wrapper = None
        self.node_feature = None

        # evaluation metrics
        self.RMSE = None
        self.JS, self.JS_list = None, None
        self.KL, self.KL_list = None, None
        self.cov_trace = None
        self.unc = None
        self.unc_list = [[] for _ in range(self.n_targets)]  # List of n lists, one per target
        # Per-agent-belief counterpart: posterior std at target cells averaged over
        # the agents' OWN mappers (decentralized mode). Measures local belief
        # quality, which the centralized reference metric above cannot see.
        self.unc_list_agents = [[] for _ in range(self.n_targets)]
        self.unc_sum, self.unc_sum_list = None, None
        # Opt-in eval diagnostics (camera-ready round): accuracy/calibration of the
        # belief vs. the measurement-encoded truth, sensing/communication counters,
        # and a per-measurement pose log. Off by default -> no behaviour change.
        self.extra_metrics = os.environ.get('KFGP_EVAL_EXTRA_METRICS', '0') == '1'
        self.log_measurements = bool(os.environ.get('KFGP_EVAL_EPISODE_LOG'))
        self._reset_diagnostics()

        # 新加的ma核心状态
        # current_node_indices[i] 表示第 i 个 agent 当前所在节点
        # dist_residuals[i] 表示 agent i 在下一次采样距离到达之前，还剩多少路程
        # sample_positions[i] 表示 agent i 的当前位置（float坐标，用于插值观测）
        # routes[i] 表示 agent i 的历史轨迹（节点 id）
        # random speed factors随机速度因子（动态调整移动速度）
        self.current_node_indices = [1 for _ in range(self.num_agents)]
        self.dist_residuals = [0.0 for _ in range(self.num_agents)]
        self.sample_positions = self.start_positions.copy()
        self.routes = [[] for _ in range(self.num_agents)]
        self.random_speed_factors = np.random.rand(self.num_agents)
        # ==============================================

        self.frame_files=[]
        # Live plotting persistent figure
        self.live_fig = None
        # Field mapping display figures
        self.field_display_fig = None
        self.field_variance_fig = None
        # Consistent colorbar limits for field mapping
        self.field_vmin, self.field_vmax = 0.0, 1.0  # Fixed range for field values
        self.variance_vmin, self.variance_vmax = 0.0, 1.0  # Fixed range for variance
        
        # Field mapping
        self.decentralized_sensing = getattr(arg, 'decentralized_sensing', False)
        self.comm_exchange_interval = getattr(arg, 'comm_exchange_interval', 10)
        self.ci_fusion_omega = getattr(arg, 'ci_fusion_omega', 0.5)
        self.step_counter = 0  # Track steps for communication exchange
        
        if arg.enable_field_mapping:
            if self.decentralized_sensing:
                # Decentralized: each agent has its own field mapper
                self.field_mappers = []
                for i in range(self.num_agents):
                    mapper = _mapper_cls()(
                        x_boundary=1.0,
                        y_boundary=1.0,
                        x_resolution=arg.mapper_resolution,
                        y_resolution=arg.mapper_resolution,
                        f_prior=arg.mapper_f_prior,
                        v_prior=arg.mapper_v_prior,
                        l=arg.mapper_length_scale,
                        elevated_regions=self.elevated_regions,
                        flight_altitude=arg.mapper_flight_altitude,
                        temporal_alpha=0.97
                    )
                    self.field_mappers.append(mapper)
                # Create a separate centralized mapper for reward computation only
                self.centralized_field_mapper = _mapper_cls()(
                    x_boundary=1.0,
                    y_boundary=1.0,
                    x_resolution=arg.mapper_resolution,
                    y_resolution=arg.mapper_resolution,
                    f_prior=arg.mapper_f_prior,
                    v_prior=arg.mapper_v_prior,
                    l=arg.mapper_length_scale,
                    elevated_regions=self.elevated_regions,
                    flight_altitude=arg.mapper_flight_altitude
                )
                self.field_mapper = self.field_mappers[0]  # For compatibility, reference first mapper
                print(f"Decentralized field mappers initialized: {self.num_agents} agents, {arg.mapper_resolution}x{arg.mapper_resolution} grid each (with separate centralized mapper for rewards)")
            else:
                # Centralized: single shared field mapper
                self.field_mapper = _mapper_cls()(
                    x_boundary=1.0,  # Same as COMPASS's unit square
                    y_boundary=1.0,
                    x_resolution=arg.mapper_resolution,
                    y_resolution=arg.mapper_resolution,
                    f_prior=arg.mapper_f_prior,
                    v_prior=arg.mapper_v_prior,
                    l=arg.mapper_length_scale,
                    elevated_regions=self.elevated_regions,
                    flight_altitude=arg.mapper_flight_altitude
                )
                self.field_mappers = None
                self.centralized_field_mapper = None  # Not needed in centralized mode
                print(f"Centralized field mapper initialized: {arg.mapper_resolution}x{arg.mapper_resolution} grid")
            # Track current step's observed cells (not cumulative)
            self.current_step_observed_cells = set()
        else:
            self.field_mapper = None
            self.field_mappers = None
            self.centralized_field_mapper = None
            self.current_step_observed_cells = set()

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # 构建随机目标
        self.underlying_distrib = VTSPGaussian(n_targets=self.n_targets)
        self.ground_truth = self.get_ground_truth()
        self.high_info_idx = self.get_high_info_idx() if arg.high_info_thre else None

        # 初始化 GP
        self.gp_wrapper = GaussianProcessWrapper(self.n_targets, self.node_coords)
        self.curr_t = 0.0
        self.budget = self.budget_init
        self.visit_t = [[] for _ in range(self.n_targets)]

        # 如果有先验测量
        if arg.prior_measurement:
            node_prior = self.underlying_distrib.mean
            # 每个目标都加上先验(这里仅在第0个agent位置上测量也行)，也可以分别放在 agent 各处
            self.gp_wrapper.add_init_measures(add_t(node_prior, self.curr_t))
            self.gp_wrapper.update_gps()

        # 评估量
        self.RMSE = self.gp_wrapper.eval_avg_RMSE(self.ground_truth, self.curr_t)
        self.cov_trace = self.gp_wrapper.eval_avg_cov_trace(self.curr_t, self.high_info_idx)
        # Initialize unc_list as list of n lists (one per target)
        self.unc_list = [[] for _ in range(self.n_targets)]
        self.unc_list_agents = [[] for _ in range(self.n_targets)]
        self._reset_diagnostics()
        # For compatibility, compute initial unc from GP wrapper
        gp_unc, gp_unc_list = self.gp_wrapper.eval_avg_unc(self.curr_t, self.high_info_idx, return_all=True)
        self.unc = gp_unc  # Initial value, will be updated from field mapper
        self.unc_sum, self.unc_sum_list = self.gp_wrapper.eval_avg_unc_sum(gp_unc_list, self.high_info_idx, return_all=True)
        self.JS, self.JS_list = self.gp_wrapper.eval_avg_JS(self.ground_truth, self.curr_t, return_all=True)
        self.KL, self.KL_list = self.gp_wrapper.eval_avg_KL(self.ground_truth, self.curr_t, return_all=True)

        # Reset field mappers for new episode
        self.step_counter = 0
        if arg.enable_field_mapping:
            if self.decentralized_sensing and self.field_mappers is not None:
                # Reinitialize all agent mappers for decentralized mode
                for i in range(self.num_agents):
                    self.field_mappers[i] = _mapper_cls()(
                        x_boundary=1.0,
                        y_boundary=1.0,
                        x_resolution=arg.mapper_resolution,
                        y_resolution=arg.mapper_resolution,
                        f_prior=arg.mapper_f_prior,
                        v_prior=arg.mapper_v_prior,
                        l=arg.mapper_length_scale,
                        elevated_regions=self.elevated_regions,
                        flight_altitude=arg.mapper_flight_altitude,
                        temporal_alpha=0.97
                    )
                # Reinitialize centralized mapper for reward computation
                self.centralized_field_mapper = _mapper_cls()(
                    x_boundary=1.0,
                    y_boundary=1.0,
                    x_resolution=arg.mapper_resolution,
                    y_resolution=arg.mapper_resolution,
                    f_prior=arg.mapper_f_prior,
                    v_prior=arg.mapper_v_prior,
                    l=arg.mapper_length_scale,
                    elevated_regions=self.elevated_regions,
                    flight_altitude=arg.mapper_flight_altitude
                )
                self.field_mapper = self.field_mappers[0]
            elif self.field_mapper is not None:
                # Reinitialize single mapper for centralized mode
                self.field_mapper = _mapper_cls()(
                    x_boundary=1.0,
                    y_boundary=1.0,
                    x_resolution=arg.mapper_resolution,
                    y_resolution=arg.mapper_resolution,
                    f_prior=arg.mapper_f_prior,
                    v_prior=arg.mapper_v_prior,
                    l=arg.mapper_length_scale,
                    elevated_regions=self.elevated_regions,
                    flight_altitude=arg.mapper_flight_altitude
                )
                self.centralized_field_mapper = None  # Not needed in centralized mode
            self.current_step_observed_cells = set()

        # Update node features from field mapper
        self.node_feature = self._update_node_feature_from_field_mapper()

        # ma状态重置
        # 随机选择每个 agent 在 graph 中最近的那个节点为起点
        for i in range(self.num_agents):
            # 找到离 self.start_positions[i] 最近的 node
            dists = np.linalg.norm(self.node_coords - self.start_positions[i], axis=1)
            self.current_node_indices[i] = int(np.argmin(dists))
            self.dist_residuals[i] = 0.0
            self.sample_positions[i] = self.start_positions[i].copy()
            self.routes[i] = [self.current_node_indices[i]]
            self.random_speed_factors[i] = np.random.rand()

        return self.node_coords, self.graph, self.node_feature, self.budget

    # ------------------------------------------------------------------
    # Opt-in eval diagnostics (KFGP_EVAL_EXTRA_METRICS / KFGP_EVAL_EPISODE_LOG).
    # Read-only w.r.t. the beliefs and the RNG: they never change an episode.
    # ------------------------------------------------------------------
    ACC_KEYS = ('rmse_all', 'abserr_tgt', 'nees_tgt', 'nees_all', 'cov2_tgt', 'cov2_all')

    def _reset_diagnostics(self):
        self.acc_ref = {k: [] for k in self.ACC_KEYS}
        self.acc_agents = {k: [] for k in self.ACC_KEYS}
        self.sense_counts = {'steps': 0, 'meas': 0, 'cells': 0, 'pos': 0,
                             'drop_events': 0, 'neg_dropped': 0}
        self.meas_log = []

    def _record_measurement(self, agent_id, mapper, m, Hm, agent_pos, agent_heading, targets_observed):
        """Per-measurement counters: cells assimilated by update_map (the
        non-obstructed cells take_measurement returned), positive cells, and the
        negative footprint cells take_measurement discarded because a target
        was in view (it then returns the positive cells only)."""
        n_assim = int(sum(1 for c in Hm if not mapper.obstructed_grids_mask[c]))
        n_pos = int(np.sum(np.asarray(m) > 0.0))
        n_drop = int(getattr(mapper, 'last_negatives_dropped', 0))
        sc = self.sense_counts
        sc['meas'] += 1
        sc['cells'] += n_assim
        sc['pos'] += n_pos
        sc['drop_events'] += int(n_drop > 0)
        sc['neg_dropped'] += n_drop
        if self.log_measurements:
            self.meas_log.append([self.curr_t, agent_id, agent_pos[0], agent_pos[1], agent_heading,
                                  n_assim, n_pos, n_drop] + [int(b) for b in targets_observed]
                                 + list(np.asarray(self.underlying_distrib.mean, dtype=float).ravel()))

    @staticmethod
    def _belief_accuracy(mapper, target_cells):
        """Accuracy/calibration of a grid belief against the measurement-encoded
        truth y (2.0 in a cell containing a target, 0.0 elsewhere; prior mean 0)."""
        mu = np.asarray(mapper.state_value, dtype=float)
        var = np.maximum(np.asarray(mapper.state_var, dtype=float), 1e-12)
        y = np.zeros_like(mu)
        y[target_cells] = 2.0
        err = mu - y
        z2 = err ** 2 / var
        inside = np.abs(err) <= 2.0 * np.sqrt(var)
        valid = ~np.asarray(mapper.obstructed_grids_mask, dtype=bool)
        tc = np.asarray(target_cells, dtype=int)  # one entry per target, as avgunc
        return {
            'rmse_all': float(np.sqrt(np.mean(err[valid] ** 2))),
            'abserr_tgt': float(np.mean(np.abs(err[tc]))),
            'nees_tgt': float(np.mean(z2[tc])),
            'nees_all': float(np.mean(z2[valid])),
            'cov2_tgt': float(np.mean(inside[tc])),
            'cov2_all': float(np.mean(inside[valid])),
        }

    def _record_step_diagnostics(self, ref_mapper, target_mean):
        # Same target->cell assignment as the uncertainty metric (nearest centre).
        sv = ref_mapper.state_vector
        target_cells = [int(np.argmin(np.linalg.norm(sv - target_mean[t], axis=1)))
                        for t in range(self.n_targets)]
        ref = self._belief_accuracy(ref_mapper, target_cells)
        if self.decentralized_sensing and self.field_mappers is not None:
            per_agent = [self._belief_accuracy(mp, target_cells) for mp in self.field_mappers]
            agents = {k: float(np.mean([d[k] for d in per_agent])) for k in self.ACC_KEYS}
        else:
            agents = ref
        for k in self.ACC_KEYS:
            self.acc_ref[k].append(ref[k])
            self.acc_agents[k].append(agents[k])
        self.sense_counts['steps'] += 1

    def step(self, next_node_indices, global_step=0, eval_speed=None):
        """
        ma-step：核心仍然是单智能体的循环插值移动思想，并引入重复观测惩罚
        "CTDE"
        
        args:
            next_node_indices: 长度为 num_agents 的列表，每个元素为对应 agent 下一目标节点的索引
            global_step: 全局步数（用于计算 curriculum 系数 alpha）
            eval_speed: 如果非 None，则表示评估模式下使用固定速度；否则采用训练模式速度
        
        return:
            reward: 团队奖励（基于各目标不确定度下降的加权和，其中对同一目标重复观测按比例惩罚）
            done: 是否预算耗尽（True 表示 episode 结束）
            node_feature: 更新后的节点特征（来自 GP 更新）
            budget: 剩余预算
            metrics: 记录的一些时序指标（例如预算消耗等，仅在评估模式下使用）
        """
        # print("step")
        # print(next_node_indices)
        # # print agent node indices
        # for i in range(self.num_agents):
        #     print(f"Agent {i} current node index: {self.current_node_indices[i]}")
        time_start = time.time()
        sample_length = 0.1  # 固定采样间隔
        alpha = min(global_step // 1000 * 0.1, 1) if arg.curriculum else 1
        metrics = {'budget': [], 'rmse': [], 'jsd': [], 'jsdall': [], 'jsdstd': [],
                'unc': [], 'uncall': [], 'uncstd': []}
        
        # Save initial curr_t to compute dt_step after all agents finish moving
        curr_t_start = self.curr_t
        
        # 保存旧的不确定度列表用于计算奖励 (for backward compatibility with old reward methods)
        # Note: unc_list is now a list of lists, one per target
        old_unc_list = [np.mean(unc_list) if len(unc_list) > 0 else 0.0 for unc_list in self.unc_list] if self.unc_list is not None else [0.0] * self.n_targets
        self._obs_counts = [0 for _ in range(self.n_targets)]# 初始化本次 step 的观测计数（每个目标被观测的 agent 数），用于重复观测惩罚
        
        # Initialize timing variables
        field_mapper_time_measurement = 0.0
        field_mapper_time_update = 0.0
        field_mapper_time_process = 0.0
        field_mapper_time_node_feature = 0.0
        field_mapper_time_reward = 0.0
        gp_time_after_obs = 0.0
        gp_time_end = 0.0
        gp_time_eval_unc = 0.0
        
        # Save old field map variance for reward computation
        # In decentralized mode, use centralized_field_mapper; otherwise use field_mapper
        reward_mapper = self.centralized_field_mapper if (self.decentralized_sensing and hasattr(self, 'centralized_field_mapper') and self.centralized_field_mapper is not None) else self.field_mapper
        old_field_variance = None
        if reward_mapper is not None:
            old_field_variance = np.diag(reward_mapper.covariance).copy()  # Get all grid cell variances
        
        # Reset current step's observed cells at the start of each step
        if self.field_mapper is not None:
            self.current_step_observed_cells = set()
        
        # 对每个 agent 计算从当前节点到目标节点的剩余距离和运动方向
        remain_lengths = []
        directions = []
        previous_headings = []  # Store previous headings for initial measurement
        for i in range(self.num_agents):
            curr_idx = self.current_node_indices[i]
            nxt_idx = next_node_indices[i]
            vec = self.node_coords[nxt_idx] - self.node_coords[curr_idx]
            d_len = np.linalg.norm(vec)
            remain_lengths.append(d_len)
            if d_len > 1e-9:
                directions.append(vec / d_len)
            else:
                directions.append(np.zeros(2))
            
            # Calculate previous heading from route history
            if len(self.routes[i]) > 1:
                # Use direction from previous node to current node
                prev_node_idx = self.routes[i][-2] if len(self.routes[i]) >= 2 else curr_idx
                prev_to_curr_vec = self.node_coords[curr_idx] - self.node_coords[prev_node_idx]
                if np.linalg.norm(prev_to_curr_vec) > 1e-9:
                    previous_headings.append(np.arctan2(prev_to_curr_vec[1], prev_to_curr_vec[0]))
                else:
                    previous_headings.append(0.0)  # Default heading
            else:
                previous_headings.append(0.0)  # Default heading if no history
        
        # Mark agents as done_move if next node is same as current node
        done_move = []
        for i in range(self.num_agents):
            if self.current_node_indices[i] == next_node_indices[i]:
                done_move.append(True)
            else:
                done_move.append(False)
        
        # Take initial measurement at current node location using previous heading (before movement)
        # if self.field_mapper is not None:
        #     target_mean = self.underlying_distrib.mean
        #     target_positions = [target_mean[t] for t in range(self.n_targets)]
            
        #     for i in range(self.num_agents):
        #         if not done_move[i]:  # Only take measurement if agent will move
        #             agent_pos = self.node_coords[self.current_node_indices[i]].copy()
        #             agent_heading = previous_headings[i]
                    
        #             # Take measurement at current node
        #             t_start = time.time()
        #             m, Hm, distances = self.field_mapper.take_measurement(
        #                 agent_pos,
        #                 target_positions,
        #                 agent_heading=agent_heading,
        #                 camera_pitch=arg.mapper_camera_pitch,
        #                 camera_hfov=arg.mapper_camera_hfov,
        #                 camera_vfov=arg.mapper_camera_vfov,
        #                 flight_altitude=arg.mapper_flight_altitude,
        #                 fov_size=arg.mapper_fov_size
        #             )
        #             field_mapper_time_measurement += time.time() - t_start
                    
        #             # Track cells observed
        #             self.current_step_observed_cells.update(Hm)
                    
        #             # Update map
        #             if len(Hm) > 0:
        #                 t_start = time.time()
        #                 self.field_mapper.update_map(
        #                     m, Hm,
        #                     noise_std=arg.mapper_noise_std,
        #                     regularization=arg.mapper_regularization,
        #                     max_condition_number=arg.mapper_max_condition_number,
        #                     distances=distances,
        #                     noise_distance_scale=arg.mapper_noise_distance_scale
        #                 )
        #                 field_mapper_time_update += time.time() - t_start
        
        # 开始循环插值移动，直到所有 agent 到达目标节点
        while not all(done_move):
            # 计算每个 agent 尚需达到采样间隔的距离
            next_lengths = []
            for i in range(self.num_agents):
                if done_move[i]:
                    next_lengths.append(np.inf)
                else:
                    next_lengths.append(max(sample_length - self.dist_residuals[i], 0.0))
            # 取所有未到达 agent 中最小的采样距离
            min_next = min(next_lengths)
        
            # 每个 agent 实际本次移动距离 = min(min_next, remain_lengths[i])
            move_dists = []
            for i in range(self.num_agents):
                if done_move[i]:
                    move_dists.append(0.0)
                else:
                    move_dists.append(min(min_next, remain_lengths[i]))
            real_move = min(move_dists)
            if real_move < 1e-9:
                for i in range(self.num_agents):
                    if remain_lengths[i] < 1e-9:
                        done_move[i] = True
                break
        
            # 计算时间步长 steplen
            if eval_speed is not None:
                steplen = eval_speed * real_move
            else:
                # 训练模式下，采用所有 agent 随机速度因子均值
                mean_factor = np.mean(self.random_speed_factors)
                steplen = 0.2 * real_move * alpha * mean_factor # 初始设置是0.1
        
            # 更新全局时间和预算
            self.curr_t += real_move
            self.budget -= real_move
        
            # 目标整体移动
            self.underlying_distrib.step(steplen)
        
            # 更新各 agent 位置、剩余距离和采样残差
            for i in range(self.num_agents):
                if not done_move[i]:
                    self.sample_positions[i] += directions[i] * real_move
                    remain_lengths[i] -= real_move
                    self.dist_residuals[i] += real_move
                    
                    # Determine which mapper to use for this agent
                    if self.decentralized_sensing and self.field_mappers is not None:
                        agent_mapper = self.field_mappers[i]
                    else:
                        agent_mapper = self.field_mapper
                    
                    # Take measurement when reaching sample_length interval (0.1 distance)
                    # Use heading towards next node
                    if abs(self.dist_residuals[i] - sample_length) < 1e-6:
                        if agent_mapper is not None:
                            agent_pos = self.sample_positions[i].copy()
                            target_mean = self.underlying_distrib.mean
                            target_positions = [target_mean[t] for t in range(self.n_targets)]
                            
                            # Heading towards next node
                            agent_heading = np.arctan2(directions[i][1], directions[i][0])
                            
                            # Take measurement at sample interval
                            t_start = time.time()
                            m, Hm, distances, targets_observed = agent_mapper.take_measurement(
                                agent_pos,
                                target_positions,
                                agent_heading=agent_heading,
                                camera_pitch=arg.mapper_camera_pitch,
                                camera_hfov=arg.mapper_camera_hfov,
                                camera_vfov=arg.mapper_camera_vfov,
                                flight_altitude=arg.mapper_flight_altitude,
                                fov_size=arg.mapper_fov_size
                            )
                            field_mapper_time_measurement += time.time() - t_start
                            
                            # Track cells observed
                            self.current_step_observed_cells.update(Hm)
                            if self.extra_metrics or self.log_measurements:
                                self._record_measurement(i, agent_mapper, m, Hm, agent_pos,
                                                         agent_heading, targets_observed)
                            
                            # Update visit_t for observed targets
                            for t in range(self.n_targets):
                                if targets_observed[t]:
                                    self.visit_t[t].append(self.curr_t)
                                    self._obs_counts[t] += 1
                            
                            # Update map
                            if len(Hm) > 0:
                                t_start = time.time()
                                agent_mapper.update_map(
                                    m, Hm,
                                    noise_std=arg.mapper_noise_std,
                                    regularization=arg.mapper_regularization,
                                    max_condition_number=arg.mapper_max_condition_number,
                                    distances=distances,
                                    noise_distance_scale=arg.mapper_noise_distance_scale
                                )
                                field_mapper_time_update += time.time() - t_start
                                
                                # Also update centralized mapper for reward computation in decentralized mode
                                if self.decentralized_sensing and hasattr(self, 'centralized_field_mapper') and self.centralized_field_mapper is not None:
                                    self.centralized_field_mapper.update_map(
                                        m, Hm,
                                        noise_std=arg.mapper_noise_std,
                                        regularization=arg.mapper_regularization,
                                        max_condition_number=arg.mapper_max_condition_number,
                                        distances=distances,
                                        noise_distance_scale=arg.mapper_noise_distance_scale
                                    )
                        
                        self.dist_residuals[i] = 0.0
                    
                    # Take measurement when reaching next node
                    if remain_lengths[i] < 1e-9:
                        if agent_mapper is not None:
                            # Agent has reached next node, take measurement at current position
                            agent_pos = self.sample_positions[i].copy()
                            target_mean = self.underlying_distrib.mean
                            target_positions = [target_mean[t] for t in range(self.n_targets)]
                            
                            # Heading towards next node (same as movement direction)
                            agent_heading = np.arctan2(directions[i][1], directions[i][0])
                            
                            # Take measurement at next node
                            t_start = time.time()
                            m, Hm, distances, targets_observed = agent_mapper.take_measurement(
                                agent_pos,
                                target_positions,
                                agent_heading=agent_heading,
                                camera_pitch=arg.mapper_camera_pitch,
                                camera_hfov=arg.mapper_camera_hfov,
                                camera_vfov=arg.mapper_camera_vfov,
                                flight_altitude=arg.mapper_flight_altitude,
                                fov_size=arg.mapper_fov_size
                            )
                            field_mapper_time_measurement += time.time() - t_start
                            
                            # Track cells observed
                            self.current_step_observed_cells.update(Hm)
                            if self.extra_metrics or self.log_measurements:
                                self._record_measurement(i, agent_mapper, m, Hm, agent_pos,
                                                         agent_heading, targets_observed)
                            
                            # Update visit_t for observed targets
                            for t in range(self.n_targets):
                                if targets_observed[t]:
                                    self.visit_t[t].append(self.curr_t)
                                    self._obs_counts[t] += 1
                            
                            # Update map
                            if len(Hm) > 0:
                                t_start = time.time()
                                agent_mapper.update_map(
                                    m, Hm,
                                    noise_std=arg.mapper_noise_std,
                                    regularization=arg.mapper_regularization,
                                    max_condition_number=arg.mapper_max_condition_number,
                                    distances=distances,
                                    noise_distance_scale=arg.mapper_noise_distance_scale
                                )
                                field_mapper_time_update += time.time() - t_start
                                
                                # Also update centralized mapper for reward computation in decentralized mode
                                if self.decentralized_sensing and hasattr(self, 'centralized_field_mapper') and self.centralized_field_mapper is not None:
                                    self.centralized_field_mapper.update_map(
                                        m, Hm,
                                        noise_std=arg.mapper_noise_std,
                                        regularization=arg.mapper_regularization,
                                        max_condition_number=arg.mapper_max_condition_number,
                                        distances=distances,
                                        noise_distance_scale=arg.mapper_noise_distance_scale
                                    )
                        
                        done_move[i] = True
                # 已到达目标的 agent 保持不动
            
            # 每次观测后更新 GP - DISABLED
            # gp_time_start = time.time()
            # self.gp_wrapper.update_gps()
            # gp_time_after_obs = time.time() - gp_time_start
            gp_time_after_obs = 0.0
            
            # 记录评估指标（如果在评估模式）
            if eval_speed is not None:
                metrics['budget'].append(self.budget_init - self.budget)
                # 此处可以扩展记录 RMSE、JSD、UNC 等指标
        
        # Compute path penalty BEFORE updating current_node_indices
        path_penalty = np.mean([
            (np.linalg.norm(self.node_coords[self.current_node_indices[i]] - self.node_coords[next_node_indices[i]])**1.5)
            for i in range(self.num_agents)
        ])
        
        # Compute collision penalty
        collision_penalty = 0.0
        collision_count = 0
        
        # Check 1: Multiple agents have the same next node
        next_node_counts = {}
        for i in range(self.num_agents):
            next_node = next_node_indices[i]
            if next_node in next_node_counts:
                next_node_counts[next_node].append(i)
            else:
                next_node_counts[next_node] = [i]
        
        # Count collisions from multiple agents going to same next node
        for next_node, agent_list in next_node_counts.items():
            if len(agent_list) > 1:
                # All pairs of agents colliding at this node
                collision_count += len(agent_list) * (len(agent_list) - 1) // 2
        
        # Check 2: Agent's next node is same as another agent's current node
        for i in range(self.num_agents):
            next_node_i = next_node_indices[i]
            for j in range(self.num_agents):
                if i != j:
                    current_node_j = self.current_node_indices[j]
                    if next_node_i == current_node_j:
                        collision_count += 1
        
        # Collision penalty: scale with number of collisions
        if collision_count > 0:
            collision_penalty = float(collision_count)
        
        # 所有 agent 均到达目标节点，更新各 agent 当前节点和轨迹
        for i in range(self.num_agents):
            self.current_node_indices[i] = next_node_indices[i]
            self.routes[i].append(next_node_indices[i])
            # Match sample_positions to node coordinates when agent finishes moving
            self.sample_positions[i] = self.node_coords[next_node_indices[i]].copy()
        
        # Compute dt_step as the change in curr_t after all agents finish moving
        dt_step = self.curr_t - curr_t_start
        
        # Compute alpha for process_update: alpha = (1+sqrt(3)/3*dt_step)*exp(-sqrt(3)/3*dt_step)
        sqrt3_over_3 = np.sqrt(3) / 3.0
        alpha_temporal = (1.0 + sqrt3_over_3 * dt_step) * np.exp(-sqrt3_over_3 * dt_step)
        alpha_factor = 0.99
        alpha_temporal = alpha_factor * alpha_temporal
        # Ablation hook: fixed alpha override (e.g. 1.0 disables the process update)
        alpha_override = getattr(arg, 'temporal_alpha_override', None)
        if alpha_override is not None:
            alpha_temporal = float(alpha_override)
        # Field mapper process update and decentralized communication
        field_mapper_time_process = 0.0
        field_mapper_time_fusion = 0.0
        
        if self.decentralized_sensing and self.field_mappers is not None:
            # Decentralized mode: each agent does process update on its own mapper
            t_start = time.time()
            for mapper in self.field_mappers:
                mapper.process_update(alpha=alpha_temporal)
            field_mapper_time_process = time.time() - t_start
            
            # Also do process update on centralized mapper (use same alpha_temporal)
            if hasattr(self, 'centralized_field_mapper') and self.centralized_field_mapper is not None:
                self.centralized_field_mapper.process_update(alpha=alpha_temporal)
            
            # Increment step counter
            self.step_counter += 1
            
            # Periodic communication exchange via CI fusion
            if self.step_counter % self.comm_exchange_interval == 0:
                t_start = time.time()
                # Collect belief packets from all agents
                belief_packets = [mapper.get_belief_packet() for mapper in self.field_mappers]

                # Measured communication volume: each agent broadcasts its packet
                # to the N-1 teammates once per exchange.
                packet_bytes = sum(p["mu"].nbytes + p["var"].nbytes for p in belief_packets)
                self.comm_bytes_total = getattr(self, 'comm_bytes_total', 0) + packet_bytes * (self.num_agents - 1)
                self.comm_exchanges = getattr(self, 'comm_exchanges', 0) + 1
                
                # Each agent fuses with all other agents' beliefs
                # Using pairwise CI fusion with equal weights
                for i in range(self.num_agents):
                    for j in range(self.num_agents):
                        if i != j:
                            # Fuse agent i's belief with agent j's belief
                            self.field_mappers[i].fuse_belief_ci(
                                belief_packets[j]["mu"],
                                belief_packets[j]["var"],
                                omega=self.ci_fusion_omega
                            )
                field_mapper_time_fusion = time.time() - t_start
            
            # Update reference mapper for compatibility (not used for reward computation)
            self.field_mapper = self.field_mappers[0]
        elif self.field_mapper is not None:
            # Centralized mode: single mapper process update
            t_start = time.time()
            self.field_mapper.process_update(alpha=alpha_temporal)
            field_mapper_time_process = time.time() - t_start
        
        # 最后再更新一次 GP - DISABLED
        # gp_time_start = time.time()
        # self.gp_wrapper.update_gps()
        # gp_time_end = time.time() - gp_time_start
        gp_time_end = 0.0

        # 在 step() 计算完新的节点特征后:
        field_mapper_time_node_feature = 0.0
        t_start = time.time()
        self.node_feature = self._update_node_feature_from_field_mapper()
        field_mapper_time_node_feature = time.time() - t_start

        
        # 计算新的不确定度 - DISABLED (keep old values)
        # gp_time_start = time.time()
        # new_unc, new_unc_list = self.gp_wrapper.eval_avg_unc(self.curr_t, self.high_info_idx, return_all=True)
        # gp_time_eval_unc = time.time() - gp_time_start
        gp_time_eval_unc = 0.0
        new_unc, new_unc_list = self.unc, self.unc_list  # Keep old values since GP is not updated
        
        # 重要！！！计算奖励：对每个目标，若被观察到，则奖励 = (old_unc - new_unc) / (观测次数)，再累加
        r = 0.0

        # Version1
        # info_gain = sum(max(old_unc_list[t] - new_unc_list[t], 0.0) / max(self._obs_counts[t], 1) for t in range(self.n_targets))
        # penalty = np.mean([np.linalg.norm(self.node_coords[self.current_node_indices[i]] - self.node_coords[next_node_indices[i]]) for i in range(self.num_agents)])
        # reward = 3.0 * info_gain - 0.05 * penalty

        # Version2
        # info_gain = sum(max(old_unc_list[t] - new_unc_list[t], 0.0) / max(self._obs_counts[t], 1) for t in range(self.n_targets))
        # coverage_penalty = sum(self._obs_counts[t]**2 for t in range(self.n_targets)) / self.num_agents
        # path_penalty = np.mean([
        #     np.linalg.norm(self.node_coords[self.current_node_indices[i]] - self.node_coords[next_node_indices[i]])
        #     for i in range(self.num_agents)
        # ])

        # reward = 3.0 * info_gain - 0.1 * coverage_penalty - 0.05 * path_penalty

        # Version3 - initial one
        # for t in range(self.n_targets):
        #     if self._obs_counts[t] > 0:
        #         r += max(old_unc_list[t] - new_unc_list[t], 0.0) / self._obs_counts[t]
        # reward = 5.0 * r - 0.1
        
        # Version5 - Field map based reward
        # collision_coeff = 0.1
        # episode_ratio = min(global_step / 20000.0, 1.0)
        # info_coeff = 0.15 - 0.10 * episode_ratio         # 早期 ~3.0 => 后期 ~1.5
        # coverage_coeff = 0.05 + 0.2 * episode_ratio     # 早期 ~0.1 => 后期 ~0.4
        # path_coeff = 0.02 + 0.05 * episode_ratio       # 早期 0.05 => 后期 0.1
        
        # Version6 - Field map based FIXED COEFFICIENT reward
        collision_coeff = 0.0
        info_coeff = 2.0
        coverage_coeff = 0.40
        path_coeff = 0.1
        
        field_mapper_time_reward = 0.0
        if reward_mapper is not None and old_field_variance is not None:
            # Get new field map variance after updates
            t_start = time.time()
            new_field_variance = np.diag(reward_mapper.covariance)
            
            # Info gain: same as COMPASS_ORIG - per-target uncertainty reduction divided by observation count
            # Coverage penalty: same as COMPASS_ORIG - penalizes repeated observations
            # First, find target cells and compute info gain
            target_mean = self.underlying_distrib.mean
            info_gain = 0.0
            for t in range(self.n_targets):
                target_pos = target_mean[t]
                # Find grid cell containing this target
                target_cell_idx = None
                min_dist = float('inf')
                for i, cell_center in enumerate(reward_mapper.state_vector):
                    dist = np.linalg.norm(cell_center - target_pos)
                    if dist < min_dist:
                        min_dist = dist
                        target_cell_idx = i
                
                if target_cell_idx is not None and target_cell_idx < len(new_field_variance):
                    # Compute raw_gain using sqrt of variance (standard deviation reduction)
                    old_std = np.sqrt(old_field_variance[target_cell_idx])
                    new_std = np.sqrt(new_field_variance[target_cell_idx])
                    raw_gain = old_std - new_std
                    
                    # Apply same formula as COMPASS_ORIG
                    if raw_gain > 0:
                        raw_gain_clamped = min(raw_gain, 0.2)  # max=0.2
                        c = max(self._obs_counts[t], 1)
                        info_gain += (raw_gain_clamped / c)
                    
                    # Add new_field_variance[target_cell_idx] to unc_list[target_idx]
                    uncertainty_value = new_std
                    self.unc_list[t].append(uncertainty_value)
                    # print(f"target {t} Uncertainty value: {uncertainty_value}")

                    # Per-agent-belief uncertainty at the same cell: mean over the
                    # agents' own mappers (equals the reference value when centralized).
                    if self.decentralized_sensing and self.field_mappers is not None:
                        agent_std = float(np.mean([
                            np.sqrt(max(mp.state_var[target_cell_idx], 0.0))
                            for mp in self.field_mappers
                        ]))
                    else:
                        agent_std = float(uncertainty_value)
                    self.unc_list_agents[t].append(agent_std)

            if self.extra_metrics:
                self._record_step_diagnostics(reward_mapper, target_mean)

            # Compute coverage penalty using same formula as COMPASS_ORIG
            coverage_penalty = 0.0
            for t in range(self.n_targets):
                new_unc_mean = np.mean(self.unc_list[t]) if isinstance(self.unc_list[t], list) and len(self.unc_list[t]) > 0 else 0.0
                penalty_factor = (1.0 - new_unc_mean)   # 不确定度越低 => penalty越大
                coverage_penalty += (self._obs_counts[t]**2) * penalty_factor
            coverage_penalty /= self.num_agents  # normalize by number of agents (same as COMPASS_ORIG)
            field_mapper_time_reward = time.time() - t_start
        else:
            # Fallback to old method if field mapper not available
            # Note: unc_list is now a list of lists, need to compute mean for each target
            info_gain = 0.0
            for t in range(self.n_targets):
                # old_unc_list[t] is now a scalar (mean of list), new_unc_list[t] needs to be computed from list
                old_unc_mean = old_unc_list[t] if isinstance(old_unc_list[t], (int, float)) else (np.mean(old_unc_list[t]) if isinstance(old_unc_list[t], list) and len(old_unc_list[t]) > 0 else 0.0)
                new_unc_mean = np.mean(self.unc_list[t]) if isinstance(self.unc_list[t], list) and len(self.unc_list[t]) > 0 else 0.0
                raw_gain = (old_unc_mean - new_unc_mean)
                if raw_gain > 0:
                    raw_gain_clamped = min(raw_gain, 0.2)  # max=0.2
                    c = max(self._obs_counts[t], 1)
                    info_gain += (raw_gain_clamped / c)

            coverage_penalty = 0.0
            for t in range(self.n_targets):
                new_unc_mean = np.mean(self.unc_list[t]) if isinstance(self.unc_list[t], list) and len(self.unc_list[t]) > 0 else 0.0
                penalty_factor = (1.0 - new_unc_mean)   # 不确定度越低 => penalty越大
                coverage_penalty += (self._obs_counts[t]**2) * penalty_factor
            coverage_penalty /= self.num_agents

        # Path penalty was computed earlier (before updating current_node_indices)
        # Collision penalty was computed earlier (before updating current_node_indices)
        
        # Update self.unc as mean of all uncertainty values across all targets
        # This is done after reward computation, regardless of whether field mapper was used
        all_unc_values = []
        for unc_list_target in self.unc_list:
            if isinstance(unc_list_target, list):
                all_unc_values.extend(unc_list_target)
        if len(all_unc_values) > 0:
            self.unc = np.mean(all_unc_values)
        # If unc_list is empty, keep previous self.unc value (don't reset to 0.0)

        reward = info_coeff * info_gain \
                - coverage_coeff * coverage_penalty \
                - path_coeff * path_penalty \
                - collision_coeff * collision_penalty
        
        # Print reward components
        print(f"Reward Components - "
              f"Info Gain: {info_gain:.6f} (coeff={info_coeff:.3f}, weighted={info_coeff * info_gain:.6f}), "
              f"Coverage Penalty: {coverage_penalty:.6f} (coeff={coverage_coeff:.3f}, weighted={coverage_coeff * coverage_penalty:.6f}), "
              f"Path Penalty: {path_penalty:.6f} (coeff={path_coeff:.3f}, weighted={path_coeff * path_penalty:.6f}), "
              f"Collision Penalty: {collision_penalty:.1f} (coeff={collision_coeff:.3f}, weighted={collision_coeff * collision_penalty:.6f}), "
              f"Final Reward: {reward:.6f}")
        # 更新环境评估量 - GP updates DISABLED
        # Note: unc_list is now updated from field mapper during reward computation
        # self.unc is computed as mean of all values in unc_list (updated above)
        # For backward compatibility, we don't update unc_list here anymore
        # self.RMSE = self.gp_wrapper.eval_avg_RMSE(self.ground_truth, self.curr_t)
        # self.JS, self.JS_list = self.gp_wrapper.eval_avg_JS(self.ground_truth, self.curr_t, return_all=True)
        # self.KL, self.KL_list = self.gp_wrapper.eval_avg_KL(self.ground_truth, self.curr_t, return_all=True)
        # self.cov_trace = self.gp_wrapper.eval_avg_cov_trace(self.curr_t, self.high_info_idx)
        # self.unc_sum, self.unc_sum_list = self.gp_wrapper.eval_avg_unc_sum(self.unc_list, self.high_info_idx, return_all=True)
        # self.node_feature = self.gp_wrapper.update_node_feature(self.curr_t)
        self.ground_truth = self.get_ground_truth()
        self.high_info_idx = self.get_high_info_idx() if arg.high_info_thre else None
        
        # 判断是否结束
        done = (self.budget <= 0.0)
        time_end = time.time()
        
        # Compute and print timing breakdown
        total_field_mapper_time = field_mapper_time_measurement + field_mapper_time_update + field_mapper_time_process + field_mapper_time_node_feature + field_mapper_time_reward
        total_gp_wrapper_time = gp_time_after_obs + gp_time_end + gp_time_eval_unc
        total_step_time = time_end - time_start
        
        # print(f"Step time: {total_step_time:.4f}s | "
        #       f"Field Mapper: {total_field_mapper_time:.4f}s "
        #       f"(measure: {field_mapper_time_measurement:.4f}s, "
        #       f"update: {field_mapper_time_update:.4f}s, "
        #       f"process: {field_mapper_time_process:.4f}s, "
        #       f"node_feat: {field_mapper_time_node_feature:.4f}s, "
        #       f"reward: {field_mapper_time_reward:.4f}s) | "
        #       f"GP Wrapper: {total_gp_wrapper_time:.4f}s "
        #       f"(after_obs: {gp_time_after_obs:.4f}s, "
        #       f"end: {gp_time_end:.4f}s, "
        #       f"eval_unc: {gp_time_eval_unc:.4f}s)")
        
        return reward, done, self.node_feature, self.budget, metrics

    def _update_node_feature_from_field_mapper(self):
        """
        Update node features from field mapper.
        In decentralized mode, returns per-agent features: [num_agents, num_nodes, 5]
        In centralized mode, returns shared features: [num_nodes, 5]
        Features: [field_value, field_variance, future_field_value, future_field_variance, occupancy]
        """
        total_timer = time.time()
        num_nodes = len(self.node_coords)
        
        if self.field_mapper is None:
            # Fallback: return zeros if field mapper is not enabled
            if self.decentralized_sensing and self.field_mappers is not None:
                # Return per-agent zeros in decentralized mode
                node_feature_base = np.zeros((num_nodes, 4), dtype=float)
                per_agent_features = []
                for ag_idx in range(self.num_agents):
                    occupancy = np.zeros((num_nodes, 1), dtype=float)
                    occupancy[self.current_node_indices[ag_idx], 0] = 1.0
                    node_feature = np.concatenate((node_feature_base, occupancy), axis=1)
                    per_agent_features.append(node_feature)
                return np.array(per_agent_features)  # [num_agents, num_nodes, 5]
            else:
                # Centralized mode
                node_feature = np.zeros((num_nodes, 4), dtype=float)
                occupancy = np.zeros((num_nodes, 1), dtype=float)
                for idx in self.current_node_indices:
                    occupancy[idx, 0] = 1.0
                node_feature = np.concatenate((node_feature, occupancy), axis=1)
                return node_feature  # [num_nodes, 5]
        
        if self.decentralized_sensing and self.field_mappers is not None:
            # Decentralized mode: each agent uses its own field mapper
            per_agent_features = []
            for ag_idx in range(self.num_agents):
                mapper = self.field_mappers[ag_idx]
                
                # Get current field values and variances from this agent's mapper
                field_values, field_variances = mapper.get_field_values_and_variances_at_points(
                    self.node_coords, num_process_updates=0
                )
                
                # Get predicted field values and variances
                future_field_values, future_field_variances = mapper.get_field_values_and_variances_at_points(
                    self.node_coords, num_process_updates=0, alpha=0.97
                )
                
                # Stack features: [value, variance, future_value, future_variance]
                node_feature = np.column_stack([
                    field_values,
                    field_variances,
                    future_field_values,
                    future_field_variances
                ])
                
                # Add occupancy (physical occupancy, same for all agents)
                occupancy = np.zeros((num_nodes, 1), dtype=float)
                occupancy[self.current_node_indices[ag_idx], 0] = 1.0
                node_feature = np.concatenate((node_feature, occupancy), axis=1)
                
                per_agent_features.append(node_feature)
            
            return np.array(per_agent_features)  # [num_agents, num_nodes, 5]
        
        else:
            # Centralized mode: single shared field mapper
            field_values, field_variances = self.field_mapper.get_field_values_and_variances_at_points(
                self.node_coords, num_process_updates=0
            )
            future_field_values, future_field_variances = self.field_mapper.get_field_values_and_variances_at_points(
                self.node_coords, num_process_updates=0, alpha=0.90
            )
            
            # Stack features: [value, variance, future_value, future_variance]
            node_feature = np.column_stack([
                field_values,
                field_variances,
                future_field_values,
                future_field_variances
            ])
            
            # Add occupancy
            occupancy = np.zeros((node_feature.shape[0], 1), dtype=float)
            for idx in self.current_node_indices:
                occupancy[idx, 0] = 1.0
            
            node_feature = np.concatenate((node_feature, occupancy), axis=1)
            return node_feature  # [num_nodes, 5]

    def _agent_observe(self, agent_id):
        """
        当某个 agent 达到采样间隔或到达目标节点时，执行一次观测，
        同时记录该 agent 对每个目标是否有效观测（FOV=0.1）。
        """
        target_mean = self.underlying_distrib.mean  # 目标当前真实位置 (n_targets, 2)
        agent_pos = self.sample_positions[agent_id]
        for t in range(self.n_targets):
            dist_to_target = np.linalg.norm(agent_pos - target_mean[t])
            if dist_to_target < 0.6:
                measure_coord = target_mean[t].reshape(-1, 2)
                measure_value = 1.0
                self.visit_t[t].append(self.curr_t)
                self._obs_counts[t] += 1  # 记录该目标被该 agent 观测到
            else:
                measure_coord = agent_pos.reshape(-1, 2)
                measure_value = 0.0
            self.gp_wrapper.GPs[t].add_observed_point(add_t(measure_coord, self.curr_t), measure_value)


    def get_ground_truth(self):
        x1 = np.linspace(0, 1, 40)
        x2 = np.linspace(0, 1, 40)
        x1x2 = np.array(list(product(x1, x2)))
        ground_truth = self.underlying_distrib.fn(x1x2)
        return ground_truth

    def get_high_info_idx(self):
        """
        Returns indices of grid points closest to each target's location.
        Each target gets exactly one grid point index (the closest one).
        """
        from itertools import product
        # Create the same grid as used in GP (40x40)
        x1 = np.linspace(0, 1, 40)
        x2 = np.linspace(0, 1, 40)
        grid = np.array(list(product(x1, x2)))  # Shape: (1600, 2)
        
        high_info_idx = []
        target_mean = self.underlying_distrib.mean  # Shape: (n_targets, 2)
        
        for i in range(self.n_targets):
            target_pos = target_mean[i]  # Shape: (2,)
            # Find the closest grid point to this target
            distances = np.linalg.norm(grid - target_pos, axis=1)
            closest_idx = np.argmin(distances)
            high_info_idx.append(np.array([closest_idx]))
        
        return high_info_idx
   

    def plot(self, routes, n, step, path, budget_list, rew_list, div_list):
        # 将 div_list 转换为 numpy 数组，并确保与 budget_list 长度匹配
        div_list = np.array(div_list, dtype=float)
        budget_array = np.array(budget_list, dtype=float)
        if div_list.ndim == 1:
            div_list = div_list[:, np.newaxis]
        if div_list.size == 0 or budget_array.size == 0:
            print("[COMPASS][plot] Warning: empty div_list or budget_list, skip plotting JS curves.")
            div_list = np.zeros((1, max(self.n_targets, 1)))
            budget_array = np.array([0.0])
        min_len = min(div_list.shape[0], budget_array.shape[0])
        if div_list.shape[0] != budget_array.shape[0]:
            print(
                f"[COMPASS][plot] Warning: budget_list length ({budget_array.shape[0]}) "
                f"!= div_list rows ({div_list.shape[0]}). Truncating to {min_len}."
            )
        div_list = div_list[:min_len]
        budget_array = budget_array[:min_len]
        y_pred_sum = []
        _live = getattr(arg, 'live_plot', False)
        if not _live:
            plt.switch_backend('agg')
            fig = plt.figure(figsize=(self.n_targets * 2.8 + 3.6, 6))
        else:
            try:
                plt.ion()
            except Exception:
                pass
            if self.live_fig is None:
                self.live_fig = plt.figure(figsize=(self.n_targets * 2.8 + 3.6, 6), num=1)
            else:
                plt.figure(1)  # Explicitly switch to main figure (number 1)
                self.live_fig.clf()
            fig = self.live_fig
            plt.figure(1)  # Ensure we're on figure 1 (main plot)
        
        target_cmap = ['r', 'g', 'b', 'm', 'y', 'c', 'lightcoral', 'lightgreen', 'lightblue', 'orange', 'gold', 'pink']
        assert len(target_cmap) >= self.n_targets
        target_mean = self.underlying_distrib.mean

        # 将所有 agent 当前的位置构成数组，便于传给 GP.plot 绘制视野圈等
        agent_locs = np.array([self.node_coords[idx] for idx in self.current_node_indices])

        # 绘制每个 target 的图像（每个 GP 绘制一个子图）
        # Ensure we're on the main figure (1) before plotting - CRITICAL for subplots
        plt.figure(1)
        for i, gp in enumerate(self.gp_wrapper.GPs):
            # Ensure we're still on main figure (1) before each GP.plot() call
            plt.figure(1)
            y_pred = gp.plot(self.ground_truth, target_id=i, target_num=self.n_targets,
                            target_loc=target_mean, all_pred=y_pred_sum,
                            high_idx=self.high_info_idx, agent_loc=agent_locs)
            y_pred_sum.append(y_pred)

        # Ensure we're still on the main figure (1) after GP plots
        plt.figure(1)
        
        # 定义每个 agent 的颜色
        agent_colors = ['cyan', 'magenta', 'yellow', 'blue', 'green', 'red']
        while len(agent_colors) < self.num_agents:
            agent_colors.append('black')

        # Ensure we're on figure 1 before plotting trajectories
        plt.figure(1)
        # 绘制每个 agent 的轨迹
        for ag_idx, route in enumerate(routes):
            # route 为单个 agent 的节点索引列表
            points = [self.graph_ctrl.find_point_from_node(nd) for nd in route]
            x = [pt[0] for pt in points]
            y = [pt[1] for pt in points]
            # 绘制轨迹线段，alpha 随轨迹位置逐渐减弱以突出最新部分
            for i in range(len(x) - 1):
                alpha = max(0.02 * (i - len(x)) + 1, 0.1)
                plt.plot(x[i:i+2], y[i:i+2], color=agent_colors[ag_idx], linewidth=2, zorder=5, alpha=alpha)
            # 绘制起始和末端标记
            plt.scatter(x[0], y[0], color=agent_colors[ag_idx], s=30, zorder=10, marker='o', edgecolors='k')
            plt.scatter(x[-1], y[-1], color=agent_colors[ag_idx], s=18, zorder=10, label=f'Robot {ag_idx}')
        
        plt.legend()
        
        # Ensure we're on figure 1 before adding JSDiv subplot
        plt.figure(1)
        # 添加 JSDiv 子图：这里使用 subplot 的方式，将画布分成两行，第二行绘制 JSDiv 曲线
        # 此处我们假设 subplot 的总列数为 self.n_targets+1（与单智能体版本一致）
        ax1 = plt.subplot(2, self.n_targets+1, 2 * self.n_targets + 2)
        plt.grid(linestyle='--')
        plt.xlim(0, self.budget_init)
        plt.ylim(0, 1.4)
        # div_list 的每一列代表一个 target 的 JS divergence 曲线
        for target_div in range(div_list.shape[1]):
            plt.plot(budget_array, div_list[:, target_div], alpha=0.5, color=target_cmap[target_div % len(target_cmap)])
        plt.ylabel('JSDiv')
        last_reward = rew_list[-1] if isinstance(rew_list, (list, tuple)) and len(rew_list) > 0 else 0.0
        plt.title('{:g}/{:g} Reward:{:.3f}'.format(self.budget_init - self.budget, self.budget_init, last_reward))
        
        plt.tight_layout()
        
        # Update main figure first
        if _live:
            try:
                plt.figure(1)  # Ensure we're on main figure (1)
                fig.canvas.draw_idle()
                plt.show(block=False)
                plt.pause(0.001)
            except Exception as _:
                pass
        
        # Save field map figure (figure 2) instead of main figure (figure 1)
        if self.field_mapper is not None:
            # Create/update field mapping plot (figure 2 with 3 subplots)
            self.plot_field_map(target_mean, target_cmap)
            
            # Save figure 2
            if self.field_display_fig is not None:
                plt.figure(2)  # Ensure we're on figure 2 before saving
                self.field_display_fig.savefig(f"{path}/{n}_{step}_samples.png", dpi=150)
                if not _live:
                    plt.close(self.field_display_fig)
                    self.field_display_fig = None
        else:
            # Fallback: save main figure if field mapper not available
            plt.figure(1)  # Ensure we're on main figure (1) before saving
            fig.savefig(f"{path}/{n}_{step}_samples.png", dpi=150)
        frame = '{}/{}_{}_samples.png'.format(path, n, step, self.graph_size)
        self.frame_files.append(frame)
        if not _live:
            plt.close(fig)

    def save_belief_trajectory_only(self, routes, save_path):
        plt.switch_backend('agg')
        plt.figure(figsize=(self.n_targets * 2.8 + 3.6, 6))
        target_cmap = ['r', 'g', 'b', 'm', 'y', 'c', 'lightcoral', 'lightgreen', 'lightblue', 'orange', 'gold', 'pink']
        assert len(target_cmap) >= self.n_targets
        target_mean = self.underlying_distrib.mean
        agent_locs = np.array([self.node_coords[idx] for idx in self.current_node_indices])
        
        y_pred_sum = []
        for i, gp in enumerate(self.gp_wrapper.GPs):
            y_pred = gp.plot(self.ground_truth, target_id=i, target_num=self.n_targets,
                            target_loc=target_mean, all_pred=y_pred_sum,
                            high_idx=self.high_info_idx, agent_loc=agent_locs)
            y_pred_sum.append(y_pred)
        
        agent_colors = ['cyan', 'magenta', 'yellow', 'blue', 'green', 'red']
        while len(agent_colors) < self.num_agents:
            agent_colors.append('black')

        for ag_idx, route in enumerate(routes):
            points = [self.graph_ctrl.find_point_from_node(nd) for nd in route]
            x = [pt[0] for pt in points]
            y = [pt[1] for pt in points]
            for i in range(len(x) - 1):
                alpha = max(0.02 * (i - len(x)) + 1, 0.1)
                plt.plot(x[i:i+2], y[i:i+2], color=agent_colors[ag_idx], linewidth=2, zorder=5, alpha=alpha)
            plt.scatter(x[0], y[0], color=agent_colors[ag_idx], s=30, zorder=10, marker='o', edgecolors='k')
            plt.scatter(x[-1], y[-1], color=agent_colors[ag_idx], s=18, zorder=10, label=f'Robot {ag_idx}')

        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def _get_visualization_mapper(self):
        """
        Get the field mapper to use for visualization.
        In decentralized mode, returns the mapper for the robot specified by visualize_robot_id.
        In centralized mode, returns the single shared mapper.
        """
        if self.decentralized_sensing and self.field_mappers is not None:
            robot_id = getattr(arg, 'visualize_robot_id', 0)
            # Clamp robot_id to valid range
            robot_id = max(0, min(robot_id, len(self.field_mappers) - 1))
            return self.field_mappers[robot_id], robot_id
        else:
            return self.field_mapper, -1  # -1 indicates centralized mode

    def plot_field_map(self, target_mean, target_cmap):
        """Plot the field map if field mapping is enabled"""
        if self.field_mapper is None:
            return
        
        _live = getattr(arg, 'live_plot', False)
        
        # Get the mapper for visualization
        viz_mapper, viz_robot_id = self._get_visualization_mapper()
        if viz_mapper is None:
            return
            
        # Create or reuse the field mapping figure (number 2) with 3 subplots
        if self.field_display_fig is None:
            if not _live:
                plt.switch_backend('agg')
            self.field_display_fig = plt.figure(figsize=(18, 6), num=2)
            if _live:
                try:
                    plt.ion()  # Turn on interactive mode
                except Exception:
                    pass
        else:
            plt.figure(2)  # Switch to field map figure (number 2)
            plt.clf()  # Clear the figure content
        
        # Ensure we're on the field display figure (2)
        plt.figure(2)
        
        # Agent colors
        agent_colors = ['cyan', 'magenta', 'yellow', 'blue', 'green', 'red']
        while len(agent_colors) < self.num_agents:
            agent_colors.append('black')
        
        # ========== Subplot 1: Field Map ==========
        plt.subplot(1, 3, 1)
        
        # Get field map data from visualization mapper
        field_map = viz_mapper.get_map_grid()
        X, Y = viz_mapper.get_cell_coordinates()
        
        # Plot field map with consistent colorbar limits
        im = plt.imshow(field_map, extent=[0, 1, 0, 1], origin='lower',
                       cmap='viridis', alpha=0.7, aspect='equal',
                       vmin=self.field_vmin, vmax=self.field_vmax)

        # Plot graph nodes and edges
        self._plot_graph()

        # Plot elevated regions
        self._plot_elevated_regions()

        # Overlay traversed paths (per-robot color)
        self._plot_robot_paths(agent_colors)

        # Add colorbar
        cbar = plt.colorbar(im, label='Field Value')
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label('Field Value', fontsize=16)
        plt.xlabel('X', fontsize=16)
        plt.ylabel('Y', fontsize=16)
        plt.tick_params(axis='both', labelsize=14)
        # Title includes robot ID in decentralized mode
        title_suffix = f' (Robot {viz_robot_id})' if viz_robot_id >= 0 else ' (Centralized)'
        plt.title(f'Field Map{title_suffix}', fontsize=18)
        plt.grid(True, alpha=0.3)
        # Lock view to workspace so FOV extending beyond [0,1] does not shift frames
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        if not getattr(arg, 'clean_field_plots', False):
            # Overlay all agent positions (highlight visualization robot)
            for ag_idx in range(self.num_agents):
                agent_pos = self.sample_positions[ag_idx]
                marker_size = 150 if ag_idx == viz_robot_id else 100
                edge_color = 'black' if ag_idx == viz_robot_id else 'none'
                plt.scatter(agent_pos[0], agent_pos[1], c=agent_colors[ag_idx], s=marker_size,
                          marker='*', zorder=10, edgecolors=edge_color, linewidths=2)

            # Overlay target positions
            if target_mean is not None and len(target_mean) > 0:
                for i, mean in enumerate(target_mean):
                    plt.scatter(mean[0], mean[1], c=target_cmap[i % len(target_cmap)], s=50,
                              marker='s', label=f'Target {i}', zorder=10)

        # Overlay each robot's projected camera FOV polygon (gated independently)
        self._plot_sensor_fov(agent_colors)

        # Legend outside the axes (below the plot), shared for paths + overlays
        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
                       ncol=max(1, self.num_agents), fontsize=11,
                       frameon=True, borderaxespad=0.)

        # ========== Subplot 2: Field Variance ==========
        plt.subplot(1, 3, 2)
        
        # Get variance map data from visualization mapper
        variance_map = viz_mapper.get_variance_grid()
        X_var, Y_var = viz_mapper.get_cell_coordinates()
        
        # Plot variance map with consistent colorbar limits
        im_var = plt.imshow(variance_map, extent=[0, 1, 0, 1], origin='lower',
                       cmap='Reds', alpha=0.7, aspect='equal',
                       vmin=self.variance_vmin, vmax=self.variance_vmax)

        # Plot graph nodes and edges
        self._plot_graph()

        # Plot elevated regions
        self._plot_elevated_regions()

        # Overlay traversed paths (per-robot color)
        self._plot_robot_paths(agent_colors)

        # Add colorbar
        cbar_var = plt.colorbar(im_var, label='Variance (Uncertainty)')
        cbar_var.ax.tick_params(labelsize=16)
        cbar_var.set_label('Variance (Uncertainty)', fontsize=16)
        plt.xlabel('X', fontsize=16)
        plt.ylabel('Y', fontsize=16)
        plt.tick_params(axis='both', labelsize=14)
        plt.title(f'Field Variance{title_suffix}', fontsize=18)
        plt.grid(True, alpha=0.3)
        # Lock view to workspace so FOV extending beyond [0,1] does not shift frames
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        if not getattr(arg, 'clean_field_plots', False):
            # Overlay all agent positions (highlight visualization robot)
            for ag_idx in range(self.num_agents):
                agent_pos = self.sample_positions[ag_idx]
                marker_size = 150 if ag_idx == viz_robot_id else 100
                edge_color = 'black' if ag_idx == viz_robot_id else 'none'
                plt.scatter(agent_pos[0], agent_pos[1], c=agent_colors[ag_idx], s=marker_size,
                          marker='*', zorder=10, edgecolors=edge_color, linewidths=2)

            # Overlay target positions
            if target_mean is not None and len(target_mean) > 0:
                for i, mean in enumerate(target_mean):
                    plt.scatter(mean[0], mean[1], c=target_cmap[i % len(target_cmap)], s=50,
                              marker='s', label=f'Target {i}', zorder=10)

        # Overlay each robot's projected camera FOV polygon (gated independently)
        self._plot_sensor_fov(agent_colors)

        # Legend outside the axes (below the plot), shared for paths + overlays
        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
                       ncol=max(1, self.num_agents), fontsize=11,
                       frameon=True, borderaxespad=0.)

        # ========== Subplot 3: All Targets (GP Predictions) ==========
        plt.subplot(1, 3, 3)
        
        # Get GP predictions for all targets (similar to GP.plot() "All targets" subplot)
        if self.gp_wrapper is not None and len(self.gp_wrapper.GPs) > 0:
            # Get predictions from all GPs
            all_pred = []
            for gp in self.gp_wrapper.GPs:
                if gp.y_pred_at_grid is not None:
                    y_pred = gp.y_pred_at_grid.reshape(40, 40)
                    all_pred.append(y_pred)
            
            if len(all_pred) > 0:
                # Take max across all targets
                all_pred = np.asarray(all_pred).max(axis=0)
                
                # Create grid coordinates (same as GP.plot())
                grid = np.array(list(product(np.linspace(0, 1, 40), np.linspace(0, 1, 40))))
                X0p = grid[:, 0].reshape(40, 40)
                X1p = grid[:, 1].reshape(40, 40)
                
                # Plot "All targets" prediction
                plt.pcolormesh(X0p, X1p, all_pred, shading='auto', vmin=0, vmax=1)
                plt.xlim((0, 1))
                plt.ylim((0, 1))
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title('All targets')
                plt.grid(True, alpha=0.3)
                
                # Plot graph nodes and edges
                self._plot_graph()
                
                # Plot elevated regions
                self._plot_elevated_regions()
                
                # Overlay high-info areas if available
                if self.high_info_idx is not None:
                    high_info_area = [grid[self.high_info_idx[i]] for i in range(self.n_targets)]
                    for i in range(self.n_targets):
                        if len(high_info_area[i]) > 0:
                            plt.scatter(high_info_area[i][:, 0], high_info_area[i][:, 1], 
                                      s=0.5, c=target_cmap[i % len(target_cmap)], alpha=0.5)
                
                # Overlay all agent positions with FOV circles
                agent_locs = np.array([self.node_coords[idx] for idx in self.current_node_indices])
                for ag_idx, loc in enumerate(agent_locs):
                    # Draw agent position
                    plt.scatter(loc[0], loc[1], c=agent_colors[ag_idx], s=100, 
                              marker='*', label=f'Robot {ag_idx}', zorder=10)
                    # Draw FOV circle (r=0.08, same as GP.plot())
                    fov = Circle(tuple(loc), 0.08, color='y', fill=False)
                    plt.gca().add_patch(fov)
                
                # Overlay target positions
                if target_mean is not None and len(target_mean) > 0:
                    for i, mean in enumerate(target_mean):
                        plt.scatter(mean[0], mean[1], c=target_cmap[i % len(target_cmap)], s=50, 
                                  marker='s', label=f'Target {i}' if i == 0 else "", zorder=10)
                
                plt.legend()
            else:
                plt.text(0.5, 0.5, 'GP predictions not available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('All targets')
        else:
            plt.text(0.5, 0.5, 'GP wrapper not available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('All targets')

        # Reserve a fixed band at the bottom for the per-subplot legends so every
        # saved frame has identical dimensions (required for MP4 assembly).
        self.field_display_fig.subplots_adjust(
            left=0.04, right=0.98, top=0.92, bottom=0.22, wspace=0.30
        )
        if _live:
            try:
                self.field_display_fig.canvas.draw_idle()
                plt.show(block=False)
                plt.pause(0.001)
            except Exception:
                pass

    def _plot_elevated_regions(self):
        """Helper function to plot elevated regions on current subplot"""
        from matplotlib.patches import Polygon
        flight_altitude = getattr(arg, 'mapper_flight_altitude', 0.1)
        
        for region in self.elevated_regions:
            vertices = np.array(region['vertices'])
            height = region['height']
            
            # Close the polygon by adding first vertex at the end
            closed_vertices = np.vstack([vertices, vertices[0]])
            
            # Plot regions above flight altitude in red (obstacles)
            if height > flight_altitude:
                # Plot polygon boundary
                plt.plot(closed_vertices[:, 0], closed_vertices[:, 1], 
                        'r-', linewidth=2, alpha=0.8, zorder=9)
                
                # Fill polygon with semi-transparent red
                poly = Polygon(vertices, closed=True, facecolor='red', 
                             edgecolor='darkred', alpha=0.3, zorder=8)
                plt.gca().add_patch(poly)
                
                # Add text label with altitude at centroid
                centroid = np.mean(vertices, axis=0)
                plt.text(centroid[0], centroid[1], f'h={height:.2f}', 
                        ha='center', va='center', fontsize=8, color='darkred',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                        zorder=10)
            
            # Plot regions below or equal to flight altitude in green (safe regions)
            else:
                # Plot polygon boundary
                plt.plot(closed_vertices[:, 0], closed_vertices[:, 1], 
                        'g-', linewidth=2, alpha=0.8, zorder=9)
                
                # Fill polygon with semi-transparent green
                poly = Polygon(vertices, closed=True, facecolor='green', 
                             edgecolor='darkgreen', alpha=0.2, zorder=8)
                plt.gca().add_patch(poly)
                
                # Add text label with altitude at centroid
                centroid = np.mean(vertices, axis=0)
                plt.text(centroid[0], centroid[1], f'h={height:.2f}', 
                        ha='center', va='center', fontsize=8, color='darkgreen',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                        zorder=10)

    def _plot_sensor_fov(self, agent_colors):
        """Overlay each robot's projected camera FOV (trapezoid on the ground) on the current subplot.
        Gated by arg.show_sensor_fov. Uses the same geometry as the mapper's take_measurement."""
        if not getattr(arg, 'show_sensor_fov', False):
            return
        if self.node_coords is None or self.routes is None:
            return

        pitch_deg = getattr(arg, 'mapper_camera_pitch', None)
        hfov_deg  = getattr(arg, 'mapper_camera_hfov',  None)
        vfov_deg  = getattr(arg, 'mapper_camera_vfov',  None)
        altitude  = getattr(arg, 'mapper_flight_altitude', None)
        if pitch_deg is None or hfov_deg is None or vfov_deg is None or altitude is None:
            return

        pitch = np.deg2rad(pitch_deg)
        half_hfov = np.deg2rad(hfov_deg) / 2.0
        half_vfov = np.deg2rad(vfov_deg) / 2.0

        for ag_idx in range(self.num_agents):
            agent_pos = self.sample_positions[ag_idx]

            # Heading: derived from the last route segment (same convention as env.step)
            heading = 0.0
            route = self.routes[ag_idx] if ag_idx < len(self.routes) else []
            if route is not None and len(route) >= 2:
                vec = self.node_coords[route[-1]] - self.node_coords[route[-2]]
                if np.linalg.norm(vec) > 1e-9:
                    heading = np.arctan2(vec[1], vec[0])

            # Compute trapezoid in camera frame (matches stamp_mapper_multi._get_cells_in_camera_fov)
            if pitch < np.pi / 2 - 1e-6:
                pitch_min = max(0.0, pitch - half_vfov)
                pitch_max = min(np.pi / 2 - 1e-6, pitch + half_vfov)
                near_dist = altitude * np.tan(pitch_min) if pitch_min > 0 else 0.0
                far_dist  = altitude * np.tan(pitch_max)
                near_width = 2 * near_dist * np.tan(half_hfov) / np.cos(pitch) if near_dist > 0 else 0.0
                far_width  = 2 * far_dist  * np.tan(half_hfov) / np.cos(pitch)
                verts_cam = [
                    [near_dist, -near_width / 2],
                    [near_dist,  near_width / 2],
                    [far_dist,   far_width  / 2],
                    [far_dist,  -far_width  / 2],
                ]
            else:
                # Nadir-looking: rectangle
                hw = altitude * np.tan(half_hfov)
                hh = altitude * np.tan(half_vfov)
                verts_cam = [[-hh, -hw], [-hh, hw], [hh, hw], [hh, -hw]]

            # Rotate by heading and translate to agent_pos, then close the polygon
            cos_h, sin_h = np.cos(heading), np.sin(heading)
            R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
            verts_world = np.array([R @ np.array(v) + agent_pos for v in verts_cam])
            verts_world = np.vstack([verts_world, verts_world[0]])

            color = agent_colors[ag_idx % len(agent_colors)]
            plt.plot(verts_world[:, 0], verts_world[:, 1],
                     color=color, linewidth=0.2, alpha=0.9, zorder=9)
            plt.fill(verts_world[:, 0], verts_world[:, 1],
                     color=color, alpha=0.12, zorder=4)

    def _plot_robot_paths(self, agent_colors):
        """Overlay each robot's traversed path on the current subplot.
        Gated by arg.show_robot_paths. Uses self.routes (per-agent node indices).
        Adds one labeled handle per agent so all robots appear in the legend."""
        if not getattr(arg, 'show_robot_paths', False):
            return
        if self.node_coords is None or self.routes is None:
            return
        for ag_idx, route in enumerate(self.routes):
            if route is None or len(route) < 1:
                continue
            color = agent_colors[ag_idx % len(agent_colors)]
            pts = np.array([self.node_coords[idx] for idx in route])
            if len(pts) >= 2:
                plt.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2.0,
                         alpha=0.9, zorder=6, label=f'Robot {ag_idx}')
            else:
                plt.scatter(pts[-1, 0], pts[-1, 1], color=color, s=80,
                            marker='*', edgecolors='k', linewidths=1.0,
                            zorder=8, label=f'Robot {ag_idx}')
            plt.scatter(pts[0, 0], pts[0, 1], color=color, s=40, marker='o',
                        edgecolors='k', linewidths=1.0, zorder=7)
            if len(pts) >= 2:
                plt.scatter(pts[-1, 0], pts[-1, 1], color=color, s=80,
                            marker='*', edgecolors='k', linewidths=1.0,
                            zorder=8)

    def _plot_graph(self):
        """Helper function to plot graph nodes and edges on current subplot"""
        if not getattr(arg, 'show_graph_nodes_edges', True):
            return
        if self.node_coords is None or self.graph is None:
            return
        
        # Plot edges first (so nodes appear on top)
        edges_dict = self.graph
        for from_node_str, neighbors in edges_dict.items():
            from_idx = int(from_node_str)
            if from_idx >= len(self.node_coords):
                continue
            from_pos = self.node_coords[from_idx]
            for to_node_str in neighbors.keys():
                to_idx = int(to_node_str)
                if to_idx >= len(self.node_coords):
                    continue
                to_pos = self.node_coords[to_idx]
                # Draw edge as a gray line
                plt.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]], 
                        c='gray', alpha=0.3, linewidth=0.5, zorder=1)
        
        # Plot nodes as small gray circles
        plt.scatter(self.node_coords[:, 0], self.node_coords[:, 1], 
                   c='gray', s=10, alpha=0.6, zorder=2, marker='o')

    def plot_field_variance(self, target_mean, target_cmap):
        """Plot the field variance if field mapping is enabled"""
        if self.field_mapper is None:
            return
        
        _live = getattr(arg, 'live_plot', False)
        if not _live:
            return
        
        # Get the mapper for visualization
        viz_mapper, viz_robot_id = self._get_visualization_mapper()
        if viz_mapper is None:
            return
            
        # Create or reuse the field variance figure (number 3)
        if self.field_variance_fig is None:
            self.field_variance_fig = plt.figure(figsize=(6, 5), num=3)
            try:
                plt.ion()  # Turn on interactive mode
            except Exception:
                pass
        else:
            plt.figure(3)  # Switch to variance figure (number 3)
            plt.clf()  # Clear the figure content
        
        # Ensure we're on the variance figure (3)
        plt.figure(3)
        
        # Get variance map data from visualization mapper
        variance_map = viz_mapper.get_variance_grid()
        X, Y = viz_mapper.get_cell_coordinates()
        
        # Plot variance map with consistent colorbar limits
        im = plt.imshow(variance_map, extent=[0, 1, 0, 1], origin='lower', 
                       cmap='Reds', alpha=0.7, aspect='equal',
                       vmin=self.variance_vmin, vmax=self.variance_vmax)
        
        # Plot graph nodes and edges
        self._plot_graph()
        
        # Plot elevated regions
        self._plot_elevated_regions()
        
        # Add colorbar
        plt.colorbar(im, label='Variance (Uncertainty)')
        plt.xlabel('X')
        plt.ylabel('Y')
        # Title includes robot ID in decentralized mode
        title_suffix = f' (Robot {viz_robot_id})' if viz_robot_id >= 0 else ' (Centralized)'
        plt.title(f'Field Variance{title_suffix}')
        plt.grid(True, alpha=0.3)
        
        # Overlay all agent positions (highlight visualization robot)
        agent_colors = ['cyan', 'magenta', 'yellow', 'blue', 'green', 'red']
        while len(agent_colors) < self.num_agents:
            agent_colors.append('black')
        
        for ag_idx in range(self.num_agents):
            agent_pos = self.sample_positions[ag_idx]
            marker_size = 150 if ag_idx == viz_robot_id else 100
            edge_color = 'black' if ag_idx == viz_robot_id else 'none'
            plt.scatter(agent_pos[0], agent_pos[1], c=agent_colors[ag_idx], s=marker_size, 
                      marker='*', label=f'Robot {ag_idx}', zorder=10, edgecolors=edge_color, linewidths=2)
        
        # Overlay target positions
        if target_mean is not None and len(target_mean) > 0:
            for i, mean in enumerate(target_mean):
                plt.scatter(mean[0], mean[1], c=target_cmap[i % len(target_cmap)], s=50, 
                          marker='s', label=f'Target {i}' if i == 0 else "", zorder=10)
        
        # Overlay current step's observed areas (not cumulative)
        if len(self.current_step_observed_cells) > 0:
            current_step_mask_var = np.zeros(
                (viz_mapper.y_resolution, viz_mapper.x_resolution), 
                dtype=float
            )
            for cell_idx in self.current_step_observed_cells:
                # Convert cell index to grid coordinates
                grid_y = cell_idx // viz_mapper.x_resolution
                grid_x = cell_idx % viz_mapper.x_resolution
                if 0 <= grid_y < viz_mapper.y_resolution and 0 <= grid_x < viz_mapper.x_resolution:
                    current_step_mask_var[grid_y, grid_x] = 1.0
            plt.contour(X, Y, current_step_mask_var, levels=[0.5], colors='white', 
                      linewidths=2, alpha=0.8)
        
        plt.legend()
        plt.tight_layout()
        try:
            self.field_variance_fig.canvas.draw_idle()
            plt.show(block=False)
            plt.pause(0.001)
        except Exception:
            pass

    def close_display(self):
        """Close the display windows"""
        if self.live_fig is not None:
            plt.close(self.live_fig)
            self.live_fig = None
        if self.field_display_fig is not None:
            plt.close(self.field_display_fig)
            self.field_display_fig = None
        if self.field_variance_fig is not None:
            plt.close(self.field_variance_fig)
            self.field_variance_fig = None


if __name__ == '__main__':
    pass

