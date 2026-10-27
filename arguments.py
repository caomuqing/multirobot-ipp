# arguments.py
import math

class Arguments:
    def __init__(self):
        self.num_agents = 3
        self.use_gpu_runner = True
        self.use_gpu_driver = True
        self.cuda_devices = [0] # Make sure this matches your setup
        self.episode_steps = 256
        self.num_meta = 16 # Number of parallel Ray runners
        self.num_minibatch = 16 # Number of minibatches to split buffer into for PPO updates
        self.buffer_size = int(self.num_meta * self.episode_steps)
        self.minibatch_size = int(self.buffer_size // self.num_minibatch)
        self.update_epochs = 4 # Number of PPO epochs over the collected buffer
        self.curriculum = True  # curriculum learning for environment parameters

        self.lr = 1e-4 # Learning rate
        self.lr_decay_step = 64 # Episodes after which LR decays
        self.gamma = 0.99 # Discount factor for rewards
        self.gae_lambda = 0.95 # Lambda for Generalized Advantage Estimation
        self.clip_coef = 0.2 # PPO clipping coefficient

        self.embedding_dim = 128
        self.high_info_thre = math.exp(-0.5)
        self.adaptive_kernel = False

        self.budget_size = (20, 40) # Range for budget randomization
        self.graph_size = (100, 201)   # Range for graph size randomization
        self.history_size = (50, 101)  # Range for history sequence length randomization
        self.k_size = 10  # KNN - number of neighboring nodes
        self.target_size = (6, 10) # Range for number of targets randomization
        self.history_stride = 5
        self.prior_measurement = True

        self.summary_window = 1 # Episodes to average for TensorBoard/W&B logging
        self.run_name = 'kfgp_train'  # checkpoints go to models/<run_name>, logs to runs/<run_name>
        self.model_path = f'models/{self.run_name}'
        self.train_path = f'runs/{self.run_name}'
        self.gifs_path = f'gifs/{self.run_name}'
        self.load_model = False # Set to True to resume from models/<run_name>
        self.use_wandb = False
        if self.use_wandb:
            self.project_name = 'STAMP_IPPO' # Changed project name
            self.wandb_notes = 'Independent PPO with shared parameters'
            self.wandb_id = None # Set to a specific ID to resume a W&B run, or None/'' for new
        self.save_img_gap = 1000
        self.save_files = True
        self.live_plot = False # If True, show live plots during episodes (interactive)
        self.show_graph_nodes_edges = False  # If True, show graph nodes and edges in plots
        self.clean_field_plots = False  # If True, hide agents/targets/sensor-FOV overlays on Field Map & Field Variance subplots (eval visualization)
        self.show_robot_paths = True  # If True, overlay each robot's traversed path (colored per-robot) on Field Map & Field Variance subplots
        self.show_sensor_fov = True  # If True, overlay each robot's projected camera FOV polygon (per-robot color) on Field Map & Field Variance subplots
        
        # Field mapping parameters
        self.enable_field_mapping = True  # Enable field mapping functionality
        self.mapper_resolution = 32  # Resolution of the field map (32x32)
        self.mapper_f_prior = 0.0  # Prior field value
        self.mapper_v_prior = 1.0  # Prior variance
        self.mapper_length_scale = 0.06  # GP length scale
        self.mapper_fov_size = 0.2  # Field of view size for measurements (deprecated, use camera parameters)
        self.mapper_noise_std = 0.1  # Base measurement noise standard deviation
        self.mapper_noise_distance_scale = 50.0  # Distance scaling factor for noise (noise = base_noise * (1 + scale * distance))
        self.mapper_regularization = 0.0  # Regularization for numerical stability
        self.mapper_max_condition_number = 1e8  # Maximum allowed condition number
        
        # Camera parameters for realistic sensor model
        self.mapper_camera_pitch =55.0  # Camera pitch angle in degrees (0-90, 0=horizontal, 90=vertical down)
        self.mapper_camera_hfov = 40.0  # Horizontal field of view in degrees
        self.mapper_camera_vfov = 45.0  # Vertical field of view in degrees
        self.mapper_flight_altitude = 0.03  # Flight altitude (normalized units, same as environment scale)
        
        # Decentralized sensing parameters
        self.decentralized_sensing = True  # If True, each agent has its own field mapper
        self.comm_exchange_interval = 3  # How often agents exchange beliefs (every k steps)
        self.ci_fusion_omega = 0.5  # Weight for Covariance Intersection fusion (0.5 = equal weight)
        self.visualize_robot_id = 2  # Which robot's field map to visualize (0-indexed, only used in decentralized mode)
        
        # Elevated regions (obstacles) parameters
        # Format: List of dictionaries, each with 'vertices' (list of [x,y] coordinates in clockwise order) and 'height' (float)
        # Vertices must be in clockwise order for convex polygons
        # Only regions with height > mapper_flight_altitude will act as obstacles
        
        # Example 1: Single square region
        # self.elevated_regions = [
        #     {'vertices': [[0.2, 0.3], [0.4, 0.3], [0.4, 0.5], [0.2, 0.5]], 'height': 0.15}
        # ]
        
        # Example 2: Triangle and square regions
        # self.elevated_regions = [
        #     # Triangle region (3 vertices in clockwise order)
        #     {'vertices': [[0.3, 0.2], [0.5, 0.2], [0.4, 0.4]], 'height': 0.12},
        #     # Square region (4 vertices in clockwise order)
        #     {'vertices': [[0.6, 0.6], [0.8, 0.6], [0.8, 0.8], [0.6, 0.8]], 'height': 0.18}
        # ]
        
        # Example 3: Multiple triangles and squares
        # self.elevated_regions = [
        #     # Triangle 1: Bottom-left
        #     {'vertices': [[0.1, 0.1], [0.25, 0.1], [0.175, 0.25]], 'height': 0.15},
        #     # Square 1: Center
        #     {'vertices': [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]], 'height': 0.12},
        #     # Triangle 2: Top-right
        #     {'vertices': [[0.7, 0.7], [0.9, 0.7], [0.8, 0.9]], 'height': 0.20},
        #     # Square 2: Top-left
        #     {'vertices': [[0.1, 0.7], [0.3, 0.7], [0.3, 0.9], [0.1, 0.9]], 'height': 0.12}
        # ]
        
        # Default: no elevated regions
        self.elevated_regions = []


class ArgumentsEval(Arguments):
    def __init__(self):
        super().__init__()
        self.high_info_thre = 'change in arguments' # Placeholder
        self.prior_measurement = 'change in arguments' # Placeholder

        self.run_name = 'decentralized_nocollision_hgt003' # Policy in models/<run_name> to evaluate
        self.model_path = f'models/{self.run_name}'
        self.result_path = self.run_name
        self.cuda_devices = [0]
        self.num_meta = 16
        self.num_eval = 100

        self.budget_size = 30
        self.graph_size = 200
        self.history_size = 100
        self.k_size = 10
        self.target_size = 6
        self.target_speed = 1/20.0 # Ensure float division
        self.history_stride = 5

        self.save_results = True
        self.save_img_gap = 1
        self.live_plot = False  # If True, show live plots during episodes (interactive)
        self.eval_greedy = True  # Set to False for exploration
        self.no_collision = True  # Prevent agents from moving to the same node
        # Field mapping parameters (inherited from Arguments, can override if needed)
        # self.enable_field_mapping = True
        # self.mapper_resolution = 32
        # self.mapper_f_prior = 0.0
        # self.mapper_v_prior = 1.0
        # self.mapper_length_scale = 0.1
        # self.mapper_fov_size = 0.2
        # self.mapper_noise_std = 0.1
        # self.mapper_regularization = 0.0
        # self.mapper_max_condition_number = 1e8


arg = Arguments()
arg_eval = ArgumentsEval()