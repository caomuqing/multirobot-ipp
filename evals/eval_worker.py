import copy
import os
import imageio
import numpy as np
import time
import torch
import torch.nn.functional as F

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import Env
from network import AttentionNet
from arguments import arg_eval


class WorkerEval:
    def __init__(self, meta_id, local_net, global_step, device='cuda', greedy=True, save_image=False, config=None):
        self.device = device
        self.greedy = greedy
        self.meta_id = meta_id
        self.global_step = global_step
        self.save_image = save_image

        if config:
            arg_eval.graph_size, arg_eval.history_size, arg_eval.target_size, arg_eval.target_speed = config
        print(f'#node {arg_eval.graph_size}, #history {arg_eval.history_size}, #tgt {arg_eval.target_size}, speed {arg_eval.target_speed}')

        self.num_agents = arg_eval.num_agents
        self.env_speed = arg_eval.target_speed
        if os.environ.get('KFGP_EVAL_GRAPH_SEED', '0') == '1':
            # Opt-in: the graph and start positions are drawn in Env.__init__,
            # BEFORE reset() seeds the RNG, i.e. from the Ray worker's unseeded
            # state. Seeding here makes them a function of the episode seed, so
            # two runs (or the two codebases) see the same graph per episode.
            # Offset stream: reset() reseeds with global_step and draws the
            # targets first, so reusing global_step here would place graph
            # nodes/start positions exactly on the target waypoints.
            np.random.seed(1_000_000 + global_step)
        self.env = Env(
            graph_size=arg_eval.graph_size,
            k_size=arg_eval.k_size,
            budget_size=arg_eval.budget_size,
            target_size=arg_eval.target_size
        )
        self.local_net = local_net
        self.avgpool = torch.nn.AvgPool1d(kernel_size=arg_eval.history_stride, stride=arg_eval.history_stride, ceil_mode=True)
        self.perf_metrics = None
        self.planning_time = 0
        self.decentralized_sensing = getattr(arg_eval, 'decentralized_sensing', False)
        # Planner ablation: 'MODEL' (default) uses the learned policy; 'AUCTION'
        # runs the classical auction planner (identical to the COMPASS-side
        # baseline) on top of THIS mapper's belief — isolates representation
        # from policy (reviewer-requested attribution ablation).
        self.eval_method = os.environ.get('KFGP_EVAL_METHOD', 'MODEL')
        self._auction_targets = [-1] * self.num_agents
        self._coverage_cycle = None
        self._coverage_ptrs = None
        self._coverage_adj = None
        # Opt-in per-episode replay log (graph, routes, targets, measurement poses)
        self.episode_log_dir = os.environ.get('KFGP_EVAL_EPISODE_LOG')

    def _bfs_next_step(self, start, goal, adjacency):
        """Next-hop neighbour of `start` on the shortest path to `goal` (BFS
        over the KNN adjacency list). Mirrors the COMPASS-side baseline."""
        if start == goal:
            return goal
        from collections import deque
        parent = {start: -1}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nb in adjacency[node]:
                nb = int(nb)
                if nb == node:
                    continue
                if nb not in parent:
                    parent[nb] = node
                    if nb == goal:
                        step = goal
                        while parent[step] != start:
                            step = parent[step]
                        return step
                    queue.append(nb)
        return int(adjacency[start][0])

    def _auction_next_nodes(self, edge_inputs_list, current_index):
        """AUCTION baseline on the grid GP-Kalman belief: top-10R most
        uncertain nodes as candidates, utility = unc - 0.5*dist, greedy
        conflict-free persistent assignment, BFS routing, re-auction on
        arrival. Identical rules to the COMPASS-side implementation; only the
        belief supplying per-node uncertainty differs."""
        node_coords = self.env.node_coords
        _, node_vars = self.env.field_mapper.get_field_values_and_variances_at_points(
            node_coords, num_process_updates=0)
        node_unc = np.sqrt(np.maximum(np.asarray(node_vars, dtype=float), 0.0))

        curr_nodes = [int(current_index[0, ag, 0].item()) for ag in range(self.num_agents)]

        agents_needing_target = [
            ag for ag in range(self.num_agents)
            if self._auction_targets[ag] == -1
            or self._auction_targets[ag] == curr_nodes[ag]
        ]

        if agents_needing_target:
            num_candidates = self.num_agents * 10
            candidate_nodes = np.argsort(node_unc)[-num_candidates:]

            bids = []
            for ag in agents_needing_target:
                for cand in candidate_nodes:
                    cand = int(cand)
                    if cand == curr_nodes[ag]:
                        continue
                    dist = float(np.linalg.norm(node_coords[curr_nodes[ag]] - node_coords[cand]))
                    bids.append((node_unc[cand] - 0.5 * dist, ag, cand))
            bids.sort(key=lambda x: x[0], reverse=True)

            occupied = {
                self._auction_targets[ag]
                for ag in range(self.num_agents)
                if ag not in agents_needing_target and self._auction_targets[ag] != -1
            }
            assigned = set()
            for _, ag, cand in bids:
                if ag in assigned or cand in occupied:
                    continue
                self._auction_targets[ag] = cand
                assigned.add(ag)
                occupied.add(cand)

            for ag in agents_needing_target:
                if ag not in assigned:
                    valid = [n for n in edge_inputs_list[curr_nodes[ag]] if n != curr_nodes[ag]]
                    self._auction_targets[ag] = int(np.random.choice(valid)) if valid else int(edge_inputs_list[curr_nodes[ag]][0])

        next_nodes = []
        for ag in range(self.num_agents):
            nxt = self._bfs_next_step(curr_nodes[ag], self._auction_targets[ag], edge_inputs_list)
            next_nodes.append(int(nxt))
        return next_nodes

    def _coverage_setup(self):
        """COVERAGE baseline (belief-free), ported from COMPASS
        evals/new_eval_worker.py (_compute_coverage_cycle) with the same seeded
        RNGs. Two COMPASS environment details are reproduced so a matched
        episode yields the same routes and measurement poses in both codebases:
        routing runs on COMPASS's KNN graph (node itself + k-1 nearest, KNN
        order; this env's graph has k non-self neighbours), and start poses are
        snapped to the start node (COMPASS Env.reset does; this env does not)."""
        from sklearn.neighbors import NearestNeighbors
        coords = self.env.node_coords
        _, knn_idx = NearestNeighbors(n_neighbors=arg_eval.k_size).fit(coords).kneighbors(coords)
        self._coverage_adj = [[int(n) for n in row] for row in knn_idx]
        for ag in range(self.num_agents):
            self.env.sample_positions[ag] = coords[self.env.current_node_indices[ag]].copy()

        n = self.env.graph_size + 1
        n_downsample = min(100, n)
        rng_fps = np.random.default_rng(seed=self.global_step)
        selected = rng_fps.choice(n, size=n_downsample, replace=False).tolist()
        rng_tour = np.random.default_rng(seed=self.global_step + 1)
        tour_sub = rng_tour.permutation(n_downsample).tolist()
        tour = [selected[i] for i in tour_sub]
        tour_coords = coords[tour]
        self._coverage_cycle = tour
        self._coverage_ptrs = [
            int(np.argmin(np.linalg.norm(tour_coords - coords[self.env.current_node_indices[ag]], axis=1)))
            for ag in range(self.num_agents)
        ]

    def _coverage_next_nodes(self):
        """Advance each robot's pointer past reached waypoints, take one BFS hop
        toward the current waypoint, and never stay in place (COMPASS
        _get_forced_move_action: random non-self neighbour)."""
        cycle = self._coverage_cycle
        next_nodes = []
        for ag in range(self.num_agents):
            curr = self.env.current_node_indices[ag]
            neighbors = self._coverage_adj[curr]
            ptr = self._coverage_ptrs[ag]
            while cycle[ptr] == curr:
                ptr = (ptr + 1) % len(cycle)
            self._coverage_ptrs[ag] = ptr
            next_hop = self._bfs_next_step(curr, cycle[ptr], self._coverage_adj)
            aidx = neighbors.index(next_hop) if next_hop in neighbors else 0
            if neighbors[aidx] == curr:
                valid = [i for i, nb in enumerate(neighbors) if nb != curr]
                aidx = np.random.choice(valid) if valid else 0
            next_nodes.append(int(neighbors[aidx]))
        return next_nodes

    def _extra_episode_metrics(self):
        """Episode means of the opt-in diagnostics (KFGP_EVAL_EXTRA_METRICS=1)."""
        env = self.env
        out = {'episode': self.global_step}
        for k in env.ACC_KEYS:
            out[f'{k}_ref'] = float(np.mean(env.acc_ref[k])) if env.acc_ref[k] else np.nan
            out[f'{k}_agents'] = float(np.mean(env.acc_agents[k])) if env.acc_agents[k] else np.nan
        sc = env.sense_counts
        denom = max(sc['steps'], 1) * self.num_agents
        out['decision_steps'] = sc['steps']
        out['meas_per_robot_step'] = sc['meas'] / denom
        out['cells_per_robot_step'] = sc['cells'] / denom
        out['detections_per_robot_step'] = sc['pos'] / denom
        out['frac_meas_neg_dropped'] = sc['drop_events'] / sc['meas'] if sc['meas'] else np.nan
        out['neg_dropped_per_drop_event'] = sc['neg_dropped'] / sc['drop_events'] if sc['drop_events'] else np.nan
        n_ex = getattr(env, 'comm_exchanges', 0)
        out['comm_exchanges'] = n_ex
        out['comm_bytes_per_exchange'] = getattr(env, 'comm_bytes_total', 0) / n_ex if n_ex else 0.0
        if env.field_mappers is not None:
            pkt = env.field_mappers[0].get_belief_packet()
            out['packet_bytes'] = pkt['mu'].nbytes + pkt['var'].nbytes
            out['packet_itemsize'] = pkt['mu'].itemsize
        return out

    def _save_episode_log(self, perf_metrics):
        env = self.env
        os.makedirs(self.episode_log_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(self.episode_log_dir, f'ep{self.global_step:03d}.npz'),
            node_coords=env.node_coords,
            start_positions=env.start_positions,
            routes=np.array(env.routes),
            target_traj=np.array(env.underlying_distrib.trajectories),
            meas_log=np.array(env.meas_log),
            visit_counts=np.array([len(v) for v in env.visit_t]),
            adjacency=np.array(self._coverage_adj if self._coverage_adj is not None else []),
            coverage_cycle=np.array(self._coverage_cycle if self._coverage_cycle is not None else []),
            avgunc=perf_metrics.get('avgunc', np.nan),
        )

    def run_episode(self, curr_eval):
        perf_metrics = {}
        node_coords, graph, node_feature, budget = self.env.reset(seed=self.global_step)
        self._auction_targets = [-1] * self.num_agents
        if self.eval_method == 'COVERAGE':
            self._coverage_setup()
        # New structure: node_feature already contains [field_value, field_variance, future_field_value, future_field_variance, occupancy]
        # node_coords are handled separately via spatio_pos_encoding, so don't concatenate them
        # Initialize histories based on sensing mode
        if self.decentralized_sensing:
            # Decentralized mode: per-agent histories
            per_agent_node_features = self.env._update_node_feature_from_field_mapper()
            # per_agent_node_features shape: [num_agents, num_nodes, 5]
            per_agent_histories_unpooled = []
            for ag_idx in range(self.num_agents):
                agent_node_inputs = torch.Tensor(per_agent_node_features[ag_idx]).unsqueeze(0).to(self.device)
                agent_history = agent_node_inputs.repeat(arg_eval.history_size, 1, 1)
                per_agent_histories_unpooled.append(agent_history)
            # For backward compatibility, use agent 0's history as placeholder (not used for decisions)
            node_history = per_agent_histories_unpooled[0]
            history_pool_inputs = self.avgpool(node_history.permute(1, 2, 0)).permute(2, 0, 1).unsqueeze(0)
        else:
            # Centralized mode: shared history
            node_inputs = torch.Tensor(node_feature).unsqueeze(0).to(self.device)
            node_history = node_inputs.repeat(arg_eval.history_size, 1, 1)
            history_pool_inputs = self.avgpool(node_history.permute(1, 2, 0)).permute(2, 0, 1).unsqueeze(0)

        # Build edge_inputs_list and pad/truncate to k_size to match worker.py
        edge_inputs_list = []
        for node_idx_int in range(arg_eval.graph_size + 1):  # +1 for the extra node
            node_idx_str = str(node_idx_int)
            neighbors_data = graph.get(node_idx_str, {})
            nb_list = []
            if isinstance(neighbors_data, dict):
                nb_list = [int(n_str) for n_str in neighbors_data.keys()]
            elif isinstance(neighbors_data, (list, tuple)):
                nb_list = [int(n) for n in neighbors_data]
            
            # Pad or truncate to k_size (matching worker.py logic)
            if len(nb_list) < arg_eval.k_size:
                nb_list += [node_idx_int] * (arg_eval.k_size - len(nb_list))  # Pad with self-loop
            else:
                nb_list = nb_list[:arg_eval.k_size]  # Truncate to k_size
            edge_inputs_list.append(nb_list)
        
        spatio_pos_encoding = self.graph_pos_encoding(edge_inputs_list)
        spatio_pos_encoding = torch.from_numpy(spatio_pos_encoding).float().unsqueeze(0).to(self.device)
        edge_inputs = torch.tensor(edge_inputs_list).unsqueeze(0).to(self.device)

        dt_history = torch.zeros((1, arg_eval.history_size, 1), device=self.device)
        dt_pool_inputs = self.avgpool(dt_history.permute(0, 2, 1)).permute(0, 2, 1)

        all_dist = [self.calc_distance_to_nodes(idx) for idx in self.env.current_node_indices]
        all_dist = np.stack(all_dist, axis=1)
        min_dist = np.min(all_dist, axis=1).reshape(-1, 1)
        min_dist[min_dist > 1.5] = 1.5
        dist_inputs = torch.tensor(min_dist, dtype=torch.float32).unsqueeze(0).to(self.device)

        current_index = []
        for ag in range(self.num_agents):
            current_index.append([self.env.current_node_indices[ag]])
        current_index = torch.tensor([current_index], device=self.device)

        spatio_mask = torch.zeros((1, arg_eval.graph_size + 1, arg_eval.k_size), dtype=torch.bool).to(self.device)
        temporal_mask = torch.tensor([1])

        routes = [[idx] for idx in current_index[0, :, 0].tolist()]
        rmse_list = [self.env.RMSE]
        jsd_list = [self.env.JS]
        unc_list = [self.env.unc]
        budget_list = [0]
        rew_list = []
        unc_stddev_list = [np.std(self.env.unc_list)]
        jsd_stddev_list = [np.std(self.env.JS_list)]

        for step in range(1024):
            if self.save_image:
                # 这里先构造 div_array，保证其为二维数组
                div_array = np.array(self.env.JS_list) if self.env.JS_list is not None else np.zeros((len(budget_list), arg_eval.target_size))
                if div_array.ndim == 1:
                    div_array = div_array.reshape(-1, 1)
                self.env.plot(
                    routes, 
                    self.global_step, 
                    step, 
                    arg_eval.result_path, 
                    budget_list, 
                    rew_list, 
                    div_array
                )

            time_start = time.time()
            with torch.no_grad():
                if self.decentralized_sensing:
                    # Decentralized mode: process each agent separately
                    logp_list_per_agent = []
                    for ag_idx in range(self.num_agents):
                        # Get this agent's history
                        agent_history_unpooled = per_agent_histories_unpooled[ag_idx]
                        agent_history_pooled = self.avgpool(agent_history_unpooled.permute(1, 2, 0)).permute(2, 0, 1).unsqueeze(0)
                        
                        # Single agent input
                        agent_current_index = current_index[:, ag_idx:ag_idx+1, :]  # [1, 1, 1]
                        
                        # Process this agent through network
                        agent_logp, agent_value = self.local_net(
                            agent_history_pooled,
                            edge_inputs,
                            dist_inputs,
                            dt_pool_inputs,
                            agent_current_index,
                            spatio_pos_encoding,
                            temporal_mask,
                            spatio_mask
                        )
                        logp_list_per_agent.append(agent_logp.squeeze(1))  # [1, K_size]
                    
                    # Stack results: [1, num_agents, K_size]
                    logp_list = torch.stack(logp_list_per_agent, dim=1)
                else:
                    # Centralized mode: process all agents together
                    logp_list, shared_value = self.local_net(
                        history_pool_inputs,
                        edge_inputs,
                        dist_inputs,
                        dt_pool_inputs,
                        current_index,
                        spatio_pos_encoding,
                        temporal_mask,
                        spatio_mask
                    )
            logp_exp = logp_list.exp().squeeze(0)  # (num_agents, k_size)
            k_size   = edge_inputs.shape[2]
            actions  = [0] * self.num_agents

            if arg_eval.no_collision:
                # Conflict-free: most confident agent picks first; others take
                # their next-best node not already claimed.
                agent_order = sorted(
                    range(self.num_agents),
                    key=lambda ag: logp_exp[ag].max().item(),
                    reverse=True,
                )
                claimed_nodes = set()
                for ag in agent_order:
                    curr_nd = current_index[0, ag, 0].item()
                    if self.greedy:
                        sorted_aidx = torch.argsort(logp_exp[ag], descending=True).tolist()
                    else:
                        probs = logp_exp[ag]
                        if not torch.all(torch.isfinite(probs)) or torch.sum(probs) <= 0:
                            probs = torch.ones_like(probs) / probs.shape[-1]
                        sorted_aidx = torch.multinomial(probs, probs.shape[0], replacement=False).tolist()

                    chosen_aidx = None
                    for aidx in sorted_aidx:
                        if aidx >= k_size:
                            continue
                        nxt_nd = int(edge_inputs[0, curr_nd, aidx].item())
                        if nxt_nd not in claimed_nodes:
                            chosen_aidx = aidx
                            claimed_nodes.add(nxt_nd)
                            break

                    if chosen_aidx is None:  # all nodes claimed — allow overlap
                        chosen_aidx = next((i for i in sorted_aidx if i < k_size), 0)
                    actions[ag] = chosen_aidx

            else:
                # Original independent action selection
                for ag in range(self.num_agents):
                    if self.greedy:
                        aidx = torch.argmax(logp_exp[ag], dim=-1).item()
                    else:
                        probs = logp_exp[ag]
                        if not torch.all(torch.isfinite(probs)) or torch.sum(probs) <= 0:
                            probs = torch.ones_like(probs) / probs.shape[-1]
                        aidx = torch.multinomial(probs, 1).squeeze(-1).item()
                    actions[ag] = aidx

            next_node_indices = []
            for ag in range(self.num_agents):
                curr_nd = current_index[0, ag, 0].item()
                aidx = actions[ag]
                if aidx >= k_size:
                    aidx = k_size - 1
                nxt_nd = edge_inputs[0, curr_nd, aidx].item()
                next_node_indices.append(int(nxt_nd))

            if self.eval_method == 'AUCTION':
                # Planner ablation: override the policy's choice with the
                # auction planner acting on the same belief.
                next_node_indices = self._auction_next_nodes(edge_inputs_list, current_index)
            elif self.eval_method == 'COVERAGE':
                next_node_indices = self._coverage_next_nodes()

            time_end = time.time()
            self.planning_time += (time_end - time_start)

            reward, done, node_feature, remain_budget, metrics = self.env.step(next_node_indices, global_step=self.global_step, eval_speed=self.env_speed)
            rew_list.append(reward)
            budget_list.append(self.env.budget_init - remain_budget)

            rmse_list.append(self.env.RMSE)
            jsd_list.append(self.env.JS)
            unc_list.append(self.env.unc)
            unc_stddev_list.append(np.std(self.env.unc_list))
            jsd_stddev_list.append(np.std(self.env.JS_list))

            for ag in range(self.num_agents):
                routes[ag].append(next_node_indices[ag])

            if done:
                if self.save_image:
                    div_array = np.array(self.env.JS_list) if self.env.JS_list is not None else np.zeros((len(budget_list), arg_eval.target_size))
                    if div_array.ndim == 1:
                        div_array = div_array.reshape(-1, 1)
                    self.env.plot(
                        routes, 
                        self.global_step, 
                        step + 1, 
                        arg_eval.result_path, 
                        budget_list, 
                        rew_list, 
                        div_array
                    )
                    self.make_gif(arg_eval.result_path, self.global_step)
                perf_metrics['minnvisit'] = np.min([len(v) for v in self.env.visit_t]) if self.env.visit_t else np.nan
                perf_metrics['avgnvisit'] = np.mean([len(v) for v in self.env.visit_t]) if self.env.visit_t else np.nan
                perf_metrics['stdnvisit'] = np.std([len(v) for v in self.env.visit_t]) if self.env.visit_t else np.nan
                
                # Compute gap visit metrics
                gap_visit_means = []
                for vt_list_item in self.env.visit_t:
                    if isinstance(vt_list_item, list) and len(vt_list_item) > 1:
                        gaps = np.diff(vt_list_item)
                        if len(gaps) > 0:
                            gap_visit_means.append(np.mean(gaps))
                if gap_visit_means:
                    perf_metrics['avggapvisit'] = np.mean(gap_visit_means)
                    perf_metrics['stdgapvisit'] = np.std(gap_visit_means)
                else:
                    perf_metrics['avggapvisit'] = np.nan
                    perf_metrics['stdgapvisit'] = np.nan
                
                perf_metrics['avgrmse'] = np.mean(rmse_list)
                perf_metrics['avgjsd'] = np.mean(jsd_list)
                # Compute avgunc and stdunc from flattened unc_list (which is now a list of lists, one per target)
                if self.env.unc_list is not None and isinstance(self.env.unc_list, list):
                    all_unc_values = []
                    for unc_list_target in self.env.unc_list:
                        if isinstance(unc_list_target, list):
                            all_unc_values.extend(unc_list_target)
                    if len(all_unc_values) > 0:
                        perf_metrics['avgunc'] = np.mean(all_unc_values)
                        perf_metrics['stdunc'] = np.std(all_unc_values)
                    else:
                        perf_metrics['avgunc'] = np.mean(unc_list) if unc_list else np.nan
                        perf_metrics['stdunc'] = np.std(unc_list) if unc_list else np.nan
                else:
                    perf_metrics['avgunc'] = np.mean(unc_list) if unc_list else np.nan
                    perf_metrics['stdunc'] = np.std(unc_list) if unc_list else np.nan
                # Per-agent-belief uncertainty (falls back to the reference value
                # when the env did not populate it, e.g. centralized mode)
                agent_vals = []
                for lst in (getattr(self.env, 'unc_list_agents', None) or []):
                    if isinstance(lst, list):
                        agent_vals.extend(lst)
                perf_metrics['avgunc_agents'] = (
                    np.mean(agent_vals) if agent_vals else perf_metrics['avgunc']
                )
                perf_metrics['stdjsd'] = np.mean(jsd_stddev_list)
                if self.env.extra_metrics:
                    perf_metrics.update(self._extra_episode_metrics())
                if self.episode_log_dir:
                    self._save_episode_log(perf_metrics)
                print(f'\033[92mmeta{self.meta_id:02}:\033[0m episode {curr_eval} done at {step} steps, avg JS {perf_metrics["avgjsd"]:.4g}')
                break

            for ag in range(self.num_agents):
                current_index[0, ag, 0] = next_node_indices[ag]

            # Update histories based on sensing mode
            if self.decentralized_sensing:
                # Decentralized mode: update per-agent histories
                per_agent_node_features = self.env._update_node_feature_from_field_mapper()
                for ag_idx in range(self.num_agents):
                    agent_node_inputs = torch.Tensor(per_agent_node_features[ag_idx]).unsqueeze(0).to(self.device)
                    per_agent_histories_unpooled[ag_idx] = torch.cat(
                        (per_agent_histories_unpooled[ag_idx], agent_node_inputs), dim=0
                    )[-arg_eval.history_size:, :, :]
                # Update placeholder for backward compatibility (not used for decisions)
                node_history = per_agent_histories_unpooled[0]
                history_pool_inputs = self.avgpool(node_history.permute(1, 2, 0)).permute(2, 0, 1).unsqueeze(0)
            else:
                # Centralized mode: update shared history
                # New structure: node_feature already contains all needed features (5 dims)
                # node_coords are handled separately via spatio_pos_encoding
                new_node_inputs = torch.Tensor(node_feature).unsqueeze(0).to(self.device)
                node_history = torch.cat((node_history, new_node_inputs), dim=0)[-arg_eval.history_size :, :, :]
                history_pool_inputs = self.avgpool(node_history.permute(1, 2, 0)).permute(2, 0, 1).unsqueeze(0)

            dt_history += (budget_list[-1] - budget_list[-2]) / (1.993 * 3)
            dt_history = torch.cat((dt_history, torch.zeros((1, 1, 1), device=self.device)), dim=1)[:, -arg_eval.history_size:, :]
            dt_pool_inputs = self.avgpool(dt_history.permute(0, 2, 1)).permute(0, 2, 1)

            all_dist = [self.calc_distance_to_nodes(idx) for idx in self.env.current_node_indices]
            all_dist = np.stack(all_dist, axis=1)
            min_dist = np.min(all_dist, axis=1).reshape(-1, 1)
            min_dist[min_dist > 1.5] = 1.5
            dist_inputs = torch.tensor(min_dist, dtype=torch.float32).unsqueeze(0).to(self.device)

            spatio_mask = torch.zeros((1, arg_eval.graph_size + 1, arg_eval.k_size), dtype=torch.bool).to(self.device)
            temporal_mask = torch.tensor([(len(budget_list) - 1) // arg_eval.history_stride + 1])

        print(f'episode {self.global_step} planning time: {self.planning_time / len(routes[0]):.3f}/step')
        return perf_metrics

    def graph_pos_encoding(self, edge_inputs_list):
        A_matrix = np.zeros((arg_eval.graph_size + 1, arg_eval.graph_size + 1))
        D_matrix = np.zeros((arg_eval.graph_size + 1, arg_eval.graph_size + 1))
        for i in range(arg_eval.graph_size + 1):
            neighbors = edge_inputs_list[i]
            for j in neighbors:
                if j != i:
                    A_matrix[i][j] = 1.0
            deg = len([n for n in neighbors if n != i])
            deg = max(deg, 1)
            D_matrix[i][i] = 1 / np.sqrt(deg)
        L = np.eye(arg_eval.graph_size + 1) - D_matrix @ A_matrix @ D_matrix
        eigen_values, eigen_vector = np.linalg.eig(L)
        idx = eigen_values.argsort()
        eigen_values, eigen_vector = eigen_values[idx], np.real(eigen_vector[:, idx])
        eigen_vector = eigen_vector[:, 1:32 + 1]
        return eigen_vector

    def calc_distance_to_nodes(self, current_idx):
        all_dist = []
        current_coord = self.env.node_coords[current_idx]
        for point_coord in self.env.node_coords:
            d_current_to_point = self.env.graph_ctrl.calc_distance(current_coord, point_coord)
            all_dist.append(d_current_to_point)
        return np.asarray(all_dist).reshape(-1, 1)

    def make_gif(self, path, n):
        os.makedirs(path, exist_ok=True)
        writer_filename = f"{path}/{n}_cov_trace_{self.env.cov_trace:.4g}.mp4"
        try:
            with imageio.get_writer(writer_filename, fps=5, format='FFMPEG', codec='libx264') as writer:
                if not self.env.frame_files:
                    print(f"Warning: No frames found to create GIF for episode {n}")
                    return
                for frame_path in self.env.frame_files:
                    image = imageio.v2.imread(frame_path)
                    writer.append_data(image)
            print(f'GIF/MP4 saved to {writer_filename}')
            for filename in self.env.frame_files:
                if os.path.exists(filename):
                    os.remove(filename)
            self.env.frame_files = []
        except Exception as e:
            print(f"Error creating GIF/MP4: {e}")


if __name__ == '__main__':
    save_img = False
    if save_img and not os.path.exists(arg_eval.gifs_path):
        os.makedirs(arg_eval.gifs_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    localNetwork = AttentionNet(arg_eval.embedding_dim).to(device)
    checkpoint = torch.load(f'../{arg_eval.model_path}/checkpoint.pth', map_location=device)
    localNetwork.load_state_dict(checkpoint['model'])
    worker = WorkerEval(meta_id=0, local_net=localNetwork, global_step=2, device=device, save_image=save_img)
    worker.run_episode(curr_eval=0)
