import csv
import os
import ray
import torch
import numpy as np
import time
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network import AttentionNet
from eval_worker import WorkerEval
from arguments import arg_eval


def main(config=None):
    """
    多智能体评估驱动函数。
    """
    if config:
        arg_eval.graph_size, arg_eval.history_size, arg_eval.target_size, arg_eval.target_speed = config

    os.makedirs(arg_eval.result_path, exist_ok=True)
    device = torch.device('cuda') if arg_eval.use_gpu_driver and torch.cuda.is_available() else torch.device('cpu')
    local_device = torch.device('cuda') if arg_eval.use_gpu_runner and torch.cuda.is_available() else torch.device('cpu')

    # 加载全局网络权重
    global_network = AttentionNet(arg_eval.embedding_dim).to(device)
    checkpoint = torch.load(f'{arg_eval.model_path}/checkpoint.pth', map_location=device)
    global_network.load_state_dict(checkpoint['model'])
    print(f'Loading model: {arg_eval.run_name}...')

    # 初始化 Ray
    if not ray.is_initialized():
        ray.init()

    # 创建若干 meta runners，用于并行评估
    meta_runners = [Runner.remote(i) for i in range(arg_eval.num_meta)]

    # 如果 driver 所在设备与 runner 不同，需要把权重先 to(local_device)
    if device != local_device:
        weights = global_network.to(local_device).state_dict()
        global_network.to(device)  # 再把网络移回 driver 的原始 device
    else:
        weights = global_network.state_dict()

    curr_test = 1
    metric_names = [
        'avgjsd', 'avgunc', 'minnvisit', 'avgrmse',
        'stdunc', 'stdjsd', 'avgnvisit', 'stdnvisit',
        'avggapvisit', 'stdgapvisit'
    ]
    # 存放所有评估结果
    perf_metrics = {n: [] for n in metric_names}

    eval_num_list = []
    budget_list = []
    rmse_list = []
    jsd_list = []
    unc_list = []
    avgjsd_list = []
    avgunc_list = []
    stdjsd_list = []
    stdunc_list = []
    minvisit_list = []
    avgvisit_list = []
    stdvisit_list = []
    avggapvisit_list = []
    stdgapvisit_list = []
    avgunc_agents_list = []
    # Opt-in: per-episode rows of every scalar metric (incl. the diagnostics and
    # the episode seed) to a separate *_metric_ext.csv; the standard CSV is unchanged.
    ext_rows = [] if os.environ.get('KFGP_EVAL_EXTRA_METRICS', '0') == '1' else None

    try:
        while True:
            jobList = []
            # 下发评估任务
            for i, meta_agent in enumerate(meta_runners):
                jobList.append(meta_agent.job.remote(weights, curr_test, config))
                curr_test += 1

            # 等待所有 meta 完成
            done_id, jobList = ray.wait(jobList, num_returns=arg_eval.num_meta)
            done_jobs = ray.get(done_id)

            # 收集结果
            for job in done_jobs:
                metrics, eval_meta_id = job

                # 仅当 metrics 不为空时，记录
                if not metrics:
                    continue

                # 这里的 eval_meta_id 并不一定等于 curr_test-1，可根据需要记录
                eval_num_list.append(eval_meta_id)
                if ext_rows is not None:
                    ext_rows.append({k: v for k, v in metrics.items() if np.isscalar(v)})

                # 依次提取需要的指标
                if 'avgjsd' in metrics:
                    avgjsd_list.append(metrics['avgjsd'])
                if 'avgunc' in metrics:
                    avgunc_list.append(metrics['avgunc'])
                if 'stdjsd' in metrics:
                    stdjsd_list.append(metrics['stdjsd'])
                if 'stdunc' in metrics:
                    stdunc_list.append(metrics['stdunc'])
                if 'minnvisit' in metrics:
                    minvisit_list.append(metrics['minnvisit'])
                if 'avgnvisit' in metrics:
                    avgvisit_list.append(metrics['avgnvisit'])
                if 'stdnvisit' in metrics:
                    stdvisit_list.append(metrics['stdnvisit'])
                if 'avggapvisit' in metrics:
                    avggapvisit_list.append(metrics['avggapvisit'])
                if 'stdgapvisit' in metrics:
                    stdgapvisit_list.append(metrics['stdgapvisit'])
                avgunc_agents_list.append(metrics.get('avgunc_agents', metrics.get('avgunc', np.nan)))

                # 将通用指标存储起来
                for n in metric_names:
                    if n in metrics and metrics[n] is not None:
                        perf_metrics[n].append(metrics[n])

                # 轨迹数据
                if 'budget_list' in metrics:
                    budget_list.extend(metrics['budget_list'])
                if 'rmse_list' in metrics:
                    rmse_list.extend(metrics['rmse_list'])
                if 'jsd_list' in metrics:
                    jsd_list.extend(metrics['jsd_list'])
                if 'unc_list' in metrics:
                    unc_list.extend(metrics['unc_list'])

            # 到达指定评估次数则退出
            if curr_test > arg_eval.num_eval:
                break

        # 评估结束，统计最终结果
        perf_data = []
        for n in metric_names:
            if len(perf_metrics[n]) > 0:
                perf_data.append(np.nanmean(perf_metrics[n]))
            else:
                perf_data.append(np.nan)

        idx = np.argsort(eval_num_list)
        avgjsd_list = np.array(avgjsd_list)[idx] if avgjsd_list else []
        avgunc_list = np.array(avgunc_list)[idx] if avgunc_list else []
        stdjsd_list = np.array(stdjsd_list)[idx] if stdjsd_list else []
        stdunc_list = np.array(stdunc_list)[idx] if stdunc_list else []
        minvisit_list = np.array(minvisit_list)[idx] if minvisit_list else []
        avgvisit_list = np.array(avgvisit_list)[idx] if avgvisit_list else []
        stdvisit_list = np.array(stdvisit_list)[idx] if stdvisit_list else []
        avgunc_agents_list = np.array(avgunc_agents_list)[idx] if avgunc_agents_list else []
        budget_list = np.array(budget_list)
        rmse_list = np.array(rmse_list)
        jsd_list = np.array(jsd_list)
        unc_list = np.array(unc_list)

        print(
            f'Graph {arg_eval.graph_size}, History {arg_eval.history_size}, '
            f'#T {arg_eval.target_size}, Budget {arg_eval.budget_size}, K {arg_eval.k_size}, results:'
        )
        for i, name in enumerate(metric_names):
            print(name, ':\t', perf_data[i])

        # 根据需要保存 csv
        if arg_eval.save_results:
            os.makedirs(arg_eval.result_path + '/metric', exist_ok=True)
            os.makedirs(arg_eval.result_path + '/traj', exist_ok=True)

            csv_filename_metric = (
                f'{arg_eval.result_path}/metric/'
                f'tgt{arg_eval.target_size}_speed{round(1/arg_eval.target_speed)}'
                f'_b{arg_eval.budget_size}_g{arg_eval.graph_size}_h{arg_eval.history_size}_metric.csv'
            )
            csv_filename_traj = (
                f'{arg_eval.result_path}/traj/'
                f'tgt{arg_eval.target_size}_speed{round(1/arg_eval.target_speed)}'
                f'_b{arg_eval.budget_size}_g{arg_eval.graph_size}_h{arg_eval.history_size}_traj.csv'
            )

            # 写 metric
            new_file = not os.path.exists(csv_filename_metric)
            field_names = ['avgjsd', 'avgunc', 'stdjsd', 'stdunc', 'minnvisit', 'avgnvisit', 'stdnvisit', 'avggapvisit', 'stdgapvisit', 'avgunc_agents']
            with open(csv_filename_metric, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if new_file:
                    writer.writerow(field_names)
                metric_data = np.column_stack([
                    avgjsd_list, avgunc_list, stdjsd_list,
                    stdunc_list, minvisit_list, avgvisit_list, stdvisit_list,
                    avggapvisit_list, stdgapvisit_list, avgunc_agents_list
                ])
                writer.writerows(metric_data)

            if ext_rows:
                csv_filename_ext = csv_filename_metric.replace('_metric.csv', '_metric_ext.csv')
                ext_rows.sort(key=lambda r: r.get('episode', 0))
                keys = ['episode'] + sorted({k for r in ext_rows for k in r} - {'episode'})
                new_file = not os.path.exists(csv_filename_ext)
                with open(csv_filename_ext, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=keys, restval='nan')
                    if new_file:
                        writer.writeheader()
                    writer.writerows(ext_rows)

            # 写 traj
            new_file = not os.path.exists(csv_filename_traj)
            field_names = ['budget', 'unc', 'jsd', 'rmse']
            with open(csv_filename_traj, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if new_file:
                    writer.writerow(field_names)
                # 如果没有对应数据，可能需要判空
                length = min(len(budget_list), len(unc_list), len(jsd_list), len(rmse_list))
                traj_data = np.column_stack([
                    budget_list[:length],
                    unc_list[:length],
                    jsd_list[:length],
                    rmse_list[:length],
                ])
                writer.writerows(traj_data)

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_runners:
            ray.kill(a)
    finally:
        if ray.is_initialized():
            ray.shutdown()


@ray.remote(num_cpus=1, num_gpus=len(arg_eval.cuda_devices) / arg_eval.num_meta if arg_eval.use_gpu_runner else 0)
class Runner:
    """
    在 eval 中对应的多智能体 Runner
    """
    def __init__(self, meta_id):
        self.meta_id = meta_id
        self.device = torch.device('cuda' if arg_eval.use_gpu_runner and torch.cuda.is_available() else 'cpu')
        self.local_net = AttentionNet(arg_eval.embedding_dim).to(self.device)

    def set_weights(self, weights):
        self.local_net.load_state_dict(weights)

    def job(self, global_weights, eval_number, config=None):
        """
        评估任务：接收全局权重，执行 WorkerEval。
        """
        print(f'\033[92mmeta{self.meta_id:02}:\033[0m eval {eval_number} starts')
        self.set_weights(global_weights)

        # 是否存图
        save_img = True if arg_eval.save_img_gap != 0 and (eval_number % arg_eval.save_img_gap == 0) else False
        worker = WorkerEval(
            meta_id=self.meta_id,
            local_net=self.local_net,
            global_step=eval_number,
            device=self.device,
            greedy=arg_eval.eval_greedy,
            save_image=save_img,
            config=config
        )
        metrics = worker.run_episode(eval_number)
        # 返回 (metrics, meta_id) 或 (metrics, eval_number)
        # 这里按需要返回识别信息
        return metrics, self.meta_id


if __name__ == '__main__':
    ray.init()
    print(f'#Evals: {arg_eval.num_eval}, #meta: {arg_eval.num_meta}')
    main()
    # 或者批量测试
    # for test_graph in [400]:
    #     for test_history in [200]:
    #         for test_tgt_number in [4]:
    #             for test_tgt_speed in [1/10]:
    #                 main(config=(test_graph, test_history, test_tgt_number, test_tgt_speed))
    #                 print('==================================')
