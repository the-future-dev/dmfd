import tqdm
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import pickle
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from softgym.registered_env import env_arg_dict
from envs.env import SoftGymEnvSB3
from sb3.utils import str2bool, set_seed_everywhere, update_env_kwargs, make_dir
from softgym.utils.visualization import save_numpy_as_gif
from torchvision import transforms
from collections import defaultdict, deque

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion

reward_scales = defaultdict(lambda: 1.0)
clip_obs      = defaultdict(lambda: None)

parser = argparse.ArgumentParser()

## general arguments
parser.add_argument('--is_eval', type=str2bool, default=False, help="evaluation or training mode")
parser.add_argument('--is_image_based', type=str2bool, default=False, help="state-based or image-based observations")
parser.add_argument('--enable_img_transformations', type=str2bool, default=False, help="Whether to enable image transformations")
parser.add_argument('--eval_videos', type=str2bool, default=False, help="whether or not to save evaluation video per episode")
parser.add_argument('--eval_gif_size',  default=512, type=int, help="evaluation GIF width and height size")
parser.add_argument('--model_save_dir', type=str, default='./data/diffusion', help="directory for saving trained model weights")
parser.add_argument('--saved_rollouts', type=str, default=None, help="directory to load saved expert demonstrations from")
parser.add_argument('--seed', type=int, default=1234, help="torch seed value")

## training arguments
parser.add_argument('--train_data_ratio', type=float, default=0.95, help="ratio for training data for train-test split")
parser.add_argument('--max_train_epochs', type=int, default=5000, help="ending epoch for training")
parser.add_argument(
    '--max_train_steps',
    type=int,
    default=None,
    help="maximum number of gradient steps for fair dataset-size comparisons; if set, overrides max_train_epochs"
)

## validation arguments
parser.add_argument('--eval_interval', type=int, default=10, help="evaluation_interval")

## test arguments
parser.add_argument('--test_checkpoint', type=str, default='./checkpoints/epoch_0.pth', help="checkpoint file for evaluation")
parser.add_argument('--eval_over_five_seeds', default=False, type=str2bool, help="evaluation over 5 random seeds (100 episodes per seed)")

## arguments used in both validation and test
parser.add_argument('--num_eval_eps', type=int, default=50, help="number of episodes to run during evaluation")

## logs
parser.add_argument('--wandb', action='store_true', help="learning curves logged on weights and biases")
parser.add_argument('--name', default=None, type=str, help='[optional] set experiment name. Useful to resume experiments.')

## diffusion model arguments
parser.add_argument('--lrate', type=float, default=1e-4, help="initial learning rate for the policy network update")
parser.add_argument('--beta1', type=float, default=0.95, help="betas for Adam Optimizer")
parser.add_argument('--beta2', type=float, default=0.999, help="betas for Adam Optimizer")
parser.add_argument('--batch_size', type=int, default=256, help="batch size for model training")
parser.add_argument('--scheduler_step_size', type=int, default=5, help="step size for optimizer scheduler")
parser.add_argument('--scheduler_gamma', type=float, default=0.99, help="decay rate for optimizer scheduler")
parser.add_argument('--discount_factor', type=float, default=0.99, help="discount factor for calculating discounted rewards")
parser.add_argument('--observation_size', type=int, default=36, help="dimension of the observation space")
parser.add_argument('--action_size', type=int, default=8, help="dimension of the action space")

## diffusion specific
parser.add_argument('--horizon', type=int, default=16, help="diffusion model horizon")
parser.add_argument('--n_obs_steps', type=int, default=2, help="number of observation steps")
parser.add_argument('--n_action_steps', type=int, default=8, help="number of action steps to predict")
parser.add_argument('--num_inference_steps', type=int, default=100, help="number of diffusion inference steps")
parser.add_argument('--obs_as_global_cond', type=str2bool, default=True, help="use observations as global conditioning")
parser.add_argument('--obs_as_local_cond', type=str2bool, default=False, help="use observations as local conditioning")
parser.add_argument('--pred_action_steps_only', type=str2bool, default=False, help="predict only action steps")
parser.add_argument('--use_ema', type=str2bool, default=True, help="use EMA model for evaluation")
parser.add_argument('--model_type', choices=['unet', 'transformer'], default='unet',
                    help="Model type for state-based diffusion policy: unet or transformer")

## image-based model specific
parser.add_argument('--env_img_size', type=int, default=128, help='Environment (observation) image size')
parser.add_argument('--cnn_channels', nargs='+', type=int, default=[32, 64, 128, 256], help="CNN channels for image encoder")
parser.add_argument('--cnn_kernels', nargs='+', type=int, default=[3, 3, 3, 3], help="CNN kernel sizes for image encoder")
parser.add_argument('--cnn_strides', nargs='+', type=int, default=[2, 2, 2, 2], help="CNN strides for image encoder")
parser.add_argument('--latent_dim', type=int, default=512, help="Latent dimension for image features")

## environment arguments
parser.add_argument('--env_name', default='ClothFold')
parser.add_argument('--env_kwargs_render', default=True, type=bool)  # Turn off rendering can speed up training
parser.add_argument('--env_kwargs_camera_name', default='default_camera', type=str)
parser.add_argument('--env_kwargs_observation_mode', default='key_point', type=str)  # Should be in ['key_point', 'cam_rgb', 'point_cloud']
parser.add_argument('--env_kwargs_num_variations', default=1000, type=int)

## NEW: match run_bc interface
parser.add_argument(
    '--load_ob_image_mode',
    default='direct',
    choices=['direct', 'separate_folder'],
    help='direct: load all images in memory; separate_folder: only load mini-batch images'
)

parser.add_argument('--lr_scheduler',
                    choices=['step', 'cosine'],
                    default='step',
                    help="learning rate scheduler type (step or cosine)")
parser.add_argument('--lr_warmup_steps',
                    type=int,
                    default=500,
                    help="number of warm‐up epochs before cosine decay")

# --- NEW: expose all scheduler hyperparameters as CLI flags ---
parser.add_argument('--scheduler_num_train_timesteps',
                    type=int,
                    default=100,
                    help="diffusion scheduler: number of train timesteps")
parser.add_argument('--scheduler_beta_start',
                    type=float,
                    default=0.0001,
                    help="diffusion scheduler: beta start")
parser.add_argument('--scheduler_beta_end',
                    type=float,
                    default=0.02,
                    help="diffusion scheduler: beta end")
parser.add_argument('--scheduler_beta_schedule',
                    type=str,
                    default="squaredcos_cap_v2",
                    help="diffusion scheduler: beta schedule")
parser.add_argument('--scheduler_variance_type',
                    type=str,
                    default="fixed_small",
                    help="diffusion scheduler: variance type")
parser.add_argument('--scheduler_clip_sample',
                    type=str2bool,
                    default=True,
                    help="diffusion scheduler: clip sample?")
parser.add_argument('--scheduler_prediction_type',
                    type=str,
                    default="epsilon",
                    help="diffusion scheduler: prediction type")

# --- NEW: expose policy architecture hyperparameters ---
parser.add_argument('--channel_cond',
                    type=str2bool,
                    default=False,
                    help="image policy: use image channels as conditioning?")
parser.add_argument('--cond_predict_scale',
                    type=str2bool,
                    default=True,
                    help="policy: condition predicts scale?")
parser.add_argument('--diffusion_step_embed_dim',
                    type=int,
                    default=256,
                    help="UNet: diffusion step embedding dimension")
parser.add_argument('--unet_down_dims',
                    nargs='+',
                    type=int,
                    default=[256, 512, 1024],
                    help="UNet: list of down‐sampling channel sizes")
parser.add_argument('--unet_kernel_size',
                    type=int,
                    default=5,
                    help="UNet: convolutional kernel size")
parser.add_argument('--unet_n_groups',
                    type=int,
                    default=8,
                    help="UNet: number of group‐norm groups")
parser.add_argument('--oa_step_convention',
                    type=str2bool,
                    default=True,
                    help="lowdim UNet policy: use obs‐action step convention?")

parser.add_argument('--resume',
                    action='store_true',
                    help="if set, resume training from the latest checkpoint")

# --- Transformer-specific hyperparameters ---
parser.add_argument('--transformer_n_emb',
                    type=int,
                    default=256,
                    help="Transformer embedding dimension (n_emb)")
parser.add_argument('--transformer_n_layer',
                    type=int,
                    default=8,
                    help="Number of Transformer encoder layers (n_layer)")
parser.add_argument('--transformer_n_head',
                    type=int,
                    default=4,
                    help="Number of attention heads (n_head)")
parser.add_argument('--transformer_p_drop_emb',
                    type=float,
                    default=0.0,
                    help="Embedding dropout rate (p_drop_emb)")
parser.add_argument('--transformer_p_drop_attn',
                    type=float,
                    default=0.3,
                    help="Attention dropout rate (p_drop_attn)")
parser.add_argument('--transformer_causal_attn',
                    type=str2bool,
                    default=True,
                    help="Use causal attention? (causal_attn)")
parser.add_argument('--transformer_time_as_cond',
                    type=str2bool,
                    default=True,
                    help="Condition on time token? (time_as_cond)")
parser.add_argument('--transformer_n_cond_layers',
                    type=int,
                    default=0,
                    help="Number of Transformer layers for conditioning (n_cond_layers)")
parser.add_argument('--crop_shape',
                    nargs=2,
                    type=int,
                    default=[76, 76],
                    help="Crop height and width for transformer image policy; use 0 0 to disable")

# New arguments for optimizer setup
parser.add_argument('--transformer_weight_decay',
                    type=float,
                    default=1e-6,
                    help="Weight decay for transformer parameters")
parser.add_argument('--encoder_weight_decay',
                    type=float,
                    default=1e-4,
                    help="Weight decay for encoder parameters")
parser.add_argument('--weight_decay',
                    type=float,
                    default=1e-6,
                    help="Generic weight decay for other policy types")

# --- NEW: flag to choose image encoder ---
parser.add_argument(
    '--visual_encoder',
    type=str,
    choices=['ResNet18Conv', 'DrQCNN'],
    default='ResNet18Conv',
    help="Which image encoder to use for transformer-hybrid policy"
)

args = parser.parse_args()

# set env_specific parameters
env_name = args.env_name
obs_mode = args.env_kwargs_observation_mode
args.scale_reward = reward_scales[env_name]
args.clip_obs = clip_obs[env_name] if obs_mode == 'key_point' else None
args.env_kwargs = env_arg_dict[env_name]
args.__dict__ = update_env_kwargs(args.__dict__)  # Update env_kwargs

# Set is_image_based based on observation_mode if not explicitly set
if args.env_kwargs['observation_mode'] == 'cam_rgb':
    args.is_image_based = True

symbolic = args.env_kwargs['observation_mode'] != 'cam_rgb'
args.encoder_type = 'identity' if symbolic else 'pixel'
args.max_steps = 200
env_kwargs = {
    'env': args.env_name,
    'symbolic': symbolic,
    'seed': args.seed,
    'max_episode_length': args.max_steps,
    'action_repeat': 1,
    'bit_depth': 8,
    'image_dim': None,
    'env_kwargs': args.env_kwargs,
    'normalize_observation': False,
    'scale_reward': args.scale_reward,
    'clip_obs': args.clip_obs,
    'obs_process': None,
}
now = datetime.now().strftime("%m.%d.%H.%M")
args.folder_name = f'{args.env_name}_Diffusion_{now}' if not args.name else args.name

# fix random seed
set_seed_everywhere(args.seed)

class Demonstrations(Dataset):
    def __init__(self, file_path, is_image_based=False, img_transform=None, horizon=None):
        self.is_image_based = is_image_based
        self.img_transform = img_transform
        self.horizon = horizon
        # load_file now returns a list of episode‐dicts
        self.trajectories = self.load_file(file_path)
        print(f"Loaded {len(self.trajectories)} trajectories")
        
        # Resize image observations to match env_img_size if needed
        if is_image_based and args.env_img_size != 32:
            print(f"Resizing demonstration images to {args.env_img_size}x{args.env_img_size}...")
            for traj in self.trajectories:
                # Check original size
                orig_shape = traj['obs'].shape
                if orig_shape[-1] != args.env_img_size or orig_shape[-2] != args.env_img_size:
                    # Resize using F.interpolate - shape is [T, C, H, W]
                    traj['obs'] = torch.nn.functional.interpolate(
                        traj['obs'], 
                        size=(args.env_img_size, args.env_img_size),
                        mode='bilinear',
                        align_corners=False
                    )
            print(f"Resized images from {orig_shape[-2]}x{orig_shape[-1]} to {args.env_img_size}x{args.env_img_size}")
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        data = self.trajectories[idx]
        obs    = data['obs']
        action = data['action']

        # if the raw trajectory is longer than horizon, randomly crop a window
        if self.horizon is not None and obs.size(0) > self.horizon:
            L = obs.size(0)
            start = torch.randint(0, L - self.horizon + 1, (1,)).item()
            obs    = obs   [start : start + self.horizon]
            action = action[start : start + self.horizon]

        # --- Fix for transformer‐hybrid policy: Robomimic expects H×W×C
        if self.is_image_based and args.model_type == 'transformer':
            # obs: [T, C, H, W] -> [T, H, W, C]
            obs = obs.permute(0, 2, 3, 1).contiguous()

        return {'obs': obs, 'action': action}
    
    def load_file(self, file_path):
        """
        Copy-pasted (and lightly adapted) from run_bc.py:
        - Handles both state- and image-based pickles
        - Respects load_ob_image_mode
        - Returns list of {'obs': Tensor[T,...], 'action': Tensor[T,...]}
        """
        print('loading all data to RAM before training....')
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # extract obs trajectories
        if self.is_image_based:
            if args.load_ob_image_mode == 'direct':
                ob_trajs = data['ob_img_trajs']
                # from (E, T, H, W, C) → (E, T, C, H, W)
                ob_trajs = np.transpose(ob_trajs, (0, 1, 4, 2, 3))
            else:
                ob_trajs = data['ob_img_trajs']
        else:
            if 'ob_trajs' in data:
                ob_trajs = data['ob_trajs']
            elif 'obs_trajs' in data:
                ob_trajs = data['obs_trajs']
            else:
                raise KeyError("No 'ob_trajs' or 'obs_trajs' in pickle")
        
        action_trajs = data['action_trajs']
        
        # build list of episode dicts
        trajectories = []
        for obs_seq, act_seq in zip(ob_trajs, action_trajs):
            obs_tensor = torch.tensor(obs_seq, dtype=torch.float32)
            action_tensor = torch.tensor(act_seq, dtype=torch.float32)
            trajectories.append({
                'obs': obs_tensor, 
                'action': action_tensor
            })
        
        print('finished loading data.')
        return trajectories

class Evaluation:
    def __init__(self):
        # initialize SoftGym environment
        self.env = SoftGymEnvSB3(**env_kwargs)
        # set evaluation device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # prepare a folder for saving per-run videos if requested
        if args.eval_videos:
            eval_base = (
                os.path.join(args.model_save_dir, args.folder_name)
                if not args.is_eval
                else os.path.dirname(args.test_checkpoint)
            )
            self.eval_video_path = make_dir(os.path.join(eval_base, "videos"))

    def evaluate(self, policy, num_episodes=50, seed=None, save_video=False, epoch=None):
        if seed is not None:
            set_seed_everywhere(seed)

        total_reward = 0.0
        total_length = 0
        total_normalized_performance_final = []

        saved_gif_path = None
        first_episode = True

        for episode in tqdm.tqdm(range(num_episodes), desc="Evaluating"):
            obs = self.env.reset()

            # --- initialize a deque to hold the last n_obs_steps obs ---
            if args.is_image_based:
                init_img = self.env.get_image(args.env_img_size, args.env_img_size)
                init_img = np.transpose(init_img, (2, 0, 1))  # HWC→CHW
                init_tensor = torch.tensor(init_img, dtype=torch.float32).to(self.device)
                memory = deque([init_tensor.clone() for _ in range(args.n_obs_steps)],
                               maxlen=args.n_obs_steps)
            else:
                init_obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
                memory = deque([init_obs.clone() for _ in range(args.n_obs_steps)],
                               maxlen=args.n_obs_steps)
            episode_reward = 0
            episode_length = 0
            episode_normalized_perf = []
            frames = []

            if save_video and first_episode:
                frames.append(self.env.get_image(args.eval_gif_size, args.eval_gif_size))

            policy.reset()

            for step in range(args.max_steps):
                if args.is_image_based:
                    # --- EDIT 2a: get new frame, append to buffer, then stack last n_obs_steps frames ---
                    img = self.env.get_image(args.env_img_size, args.env_img_size)
                    img = np.transpose(img, (2, 0, 1)) # HWC -> CHW
                    img_tensor = torch.tensor(img, dtype=torch.float32).to(self.device)
                    memory.append(img_tensor)
                    obs_seq = torch.stack(list(memory), dim=0)       # (n_obs_steps, C, H, W)
                    obs_tensor = obs_seq.unsqueeze(0)               # (1, n_obs_steps, C, H, W)

                    # --- FIX: Permute obs tensor for Transformer Hybrid model ---
                    if args.model_type == 'transformer':
                        # (B, T, C, H, W) -> (B, T, H, W, C) expected by robomimic encoder
                        obs_tensor_permuted = obs_tensor.permute(0, 1, 3, 4, 2).contiguous() 
                        obs_dict_input = {'obs': obs_tensor_permuted}
                    else:
                        obs_dict_input = {'obs': obs_tensor} # Original CHW for UNet
                else:
                    # --- EDIT 2b: get new state, append to buffer, then stack last n_obs_steps states ---
                    obs_vec = torch.tensor(obs, dtype=torch.float32).to(self.device)
                    memory.append(obs_vec)
                    obs_seq = torch.stack(list(memory), dim=0)       # (n_obs_steps, obs_dim)
                    obs_tensor = obs_seq.unsqueeze(0)               # (1, n_obs_steps, obs_dim)
                    obs_dict_input = {'obs': obs_tensor}

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict_input)
                    action = action_dict['action'][0, 0].cpu().numpy()

                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                episode_normalized_perf.append(info['normalized_performance'])

                if save_video and first_episode:
                    frames.append(self.env.get_image(args.eval_gif_size, args.eval_gif_size))

                if done:
                    break

            # final performance for this episode
            ep_perf = episode_normalized_perf[-1]
            total_normalized_performance_final.append(ep_perf)
            total_reward += episode_reward
            total_length += episode_length

            print(f'Episode {episode}, Final Normalized Performance: {ep_perf:.4f}, '
                  f'Reward: {episode_reward:.2f}, Length: {episode_length}')

            if save_video and first_episode:
                # handle missing epoch (e.g. in main_testing where epoch=None)
                epoch_str = str(epoch + 1) if epoch is not None else "test"
                gif_path = os.path.join(
                    self.eval_video_path,
                    f"epoch_{epoch_str}_perf_{ep_perf:.4f}.gif"
                )
                save_numpy_as_gif(np.array(frames), gif_path)
                saved_gif_path = gif_path
                first_episode = False

        # compute statistics
        avg_perf = np.mean(total_normalized_performance_final)
        std_perf = np.std(total_normalized_performance_final)
        avg_reward = total_reward / num_episodes
        avg_length = total_length / num_episodes

        print("\nEvaluation Summary:")
        print(f"Average Final Normalized Performance: {avg_perf:.4f}")
        print(f"Std   Final Normalized Performance: {std_perf:.4f}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Episode Length: {avg_length:.2f}")

        return avg_perf, std_perf, avg_reward, avg_length, saved_gif_path

    def evaluate_five_seeds(self, policy):
        seeds = [100, 200, 300, 400, 500]  # Use different seeds
        all_final_normalized_perf = []     # Store final performance across all seeds/eps

        policy.eval()

        for seed in seeds:
            print(f"\nEvaluating with seed {seed}")
            set_seed_everywhere(seed)

            # Evaluate for 20 episodes per seed
            for ep in tqdm.tqdm(range(20), desc=f"Seed {seed} Episodes"):
                episode_normalized_perf = []

                # initialize a deque to hold the last n_obs_steps observations
                if args.is_image_based:
                    # reset the real env, not the broken wrapper
                    self.env.unwrapped.reset()
                    init_img = self.env.get_image(args.env_img_size, args.env_img_size)
                    init_img = np.transpose(init_img, (2, 0, 1))  # HWC → CHW
                    init_tensor = torch.tensor(init_img, dtype=torch.float32).to(self.device)
                    memory = deque([init_tensor.clone() for _ in range(args.n_obs_steps)],
                                   maxlen=args.n_obs_steps)
                else:
                    obs = self.env.reset()
                    init_obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
                    memory = deque([init_obs.clone() for _ in range(args.n_obs_steps)],
                                   maxlen=args.n_obs_steps)

                policy.reset()

                for step in range(args.max_steps):
                    # build an (n_obs_steps, …) tensor from the deque
                    if args.is_image_based:
                        img = self.env.get_image(args.env_img_size, args.env_img_size)
                        img = np.transpose(img, (2, 0, 1)) # HWC -> CHW
                        img_tensor = torch.tensor(img, dtype=torch.float32).to(self.device)
                        memory.append(img_tensor)
                        obs_seq = torch.stack(list(memory), dim=0)       # (n_obs_steps, C, H, W)
                        obs_tensor = obs_seq.unsqueeze(0)               # (1, n_obs_steps, C, H, W)

                        # --- FIX: Permute obs tensor for Transformer Hybrid model ---
                        if args.model_type == 'transformer':
                            # (B, T, C, H, W) -> (B, T, H, W, C) expected by robomimic encoder
                            obs_tensor_permuted = obs_tensor.permute(0, 1, 3, 4, 2).contiguous()
                            obs_dict_input = {'obs': obs_tensor_permuted}
                        else:
                            obs_dict_input = {'obs': obs_tensor} # Original CHW for UNet
                    else:
                        obs_vec = torch.tensor(obs, dtype=torch.float32).to(self.device)
                        memory.append(obs_vec)
                        obs_seq = torch.stack(list(memory), dim=0)       # (n_obs_steps, obs_dim)
                        obs_tensor = obs_seq.unsqueeze(0)               # (1, n_obs_steps, obs_dim)
                        obs_dict_input = {'obs': obs_tensor}

                    # Get action from policy
                    with torch.no_grad():
                        action_dict = policy.predict_action(obs_dict_input)
                        action = action_dict['action'][0, 0].cpu().numpy()

                    # Execute action in environment
                    obs, reward, done, info = self.env.step(action)
                    episode_normalized_perf.append(info['normalized_performance'])

                    if done:
                        break

                # Store final normalized performance for this episode
                ep_normalized_perf_final = episode_normalized_perf[-1]
                print(f'Seed {seed} Episode {ep}, Final Normalized Performance: {ep_normalized_perf_final:.4f}')
                all_final_normalized_perf.append(ep_normalized_perf_final)


        # Calculate and print statistics over all 100 episodes (5 seeds * 20 episodes)
        all_final_normalized_perf = np.array(all_final_normalized_perf)

        # Save results to .npy file, mimicking run_bc structure
        ckpt_file_path = args.test_checkpoint
        # Construct npy filename based on checkpoint name, placed in the same directory
        run_name = os.path.basename(os.path.dirname(os.path.dirname(ckpt_file_path))) # e.g., ClothFold_Diffusion_...
        npy_file_path = os.path.join(os.path.dirname(ckpt_file_path), f'{run_name}_eval_5seeds.npy')
        print(f"Saving 5-seed evaluation results to: {npy_file_path}")
        np.save(npy_file_path, all_final_normalized_perf)

        print('\n!!!!!!! Final Normalized Performance Statistics (100 episodes over 5 seeds) !!!!!!!')
        print(f'Mean: {np.mean(all_final_normalized_perf):.4f}')
        print(f'Std: {np.std(all_final_normalized_perf):.4f}')
        print(f'Median: {np.median(all_final_normalized_perf):.4f}')
        print(f'25th Percentile: {np.percentile(all_final_normalized_perf, 25):.4f}')
        print(f'75th Percentile: {np.percentile(all_final_normalized_perf, 75):.4f}')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        if args.wandb:
            wandb.log({
                "final_eval_5seeds/info_normalized_performance_mean":   np.mean(all_final_normalized_perf),
                "final_eval_5seeds/info_normalized_performance_std":    np.std(all_final_normalized_perf),
                "final_eval_5seeds/info_normalized_performance_median": np.median(all_final_normalized_perf),
                "final_eval_5seeds/info_normalized_performance_25th":   np.percentile(all_final_normalized_perf, 25),
                "final_eval_5seeds/info_normalized_performance_75th":   np.percentile(all_final_normalized_perf, 75),
            })

def create_diffusion_policy(args):
    # Create the noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.scheduler_num_train_timesteps,
        beta_start=args.scheduler_beta_start,
        beta_end=args.scheduler_beta_end,
        beta_schedule=args.scheduler_beta_schedule,
        variance_type=args.scheduler_variance_type,
        clip_sample=args.scheduler_clip_sample,
        prediction_type=args.scheduler_prediction_type
    )
    
    if args.is_image_based and args.model_type == 'transformer':
        # -------------------------------------------------------------------
        # Automatically derive shape_meta by instantiating a dummy env
        env_spec = SoftGymEnvSB3(**env_kwargs)
        obs_space = env_spec.observation_space   # could be gym.spaces.Dict or Box

        # build initial shape_meta using dummy env's observation space
        shape_meta = {
            'obs': {},
            'action': {'shape': list(env_spec.action_space.shape)}
        }
        if hasattr(obs_space, 'spaces'):
            # Handle Dict observation space if necessary (less likely for cam_rgb)
            for key, space in obs_space.spaces.items():
                obs_type = 'rgb' if len(space.shape) == 3 else 'low_dim' # Basic type inference
                # Use actual image size for rgb observations
                if obs_type == 'rgb':
                    shape = [args.env_img_size, args.env_img_size, space.shape[-1]] # Use args.env_img_size
                else:
                    shape = list(space.shape)
                shape_meta['obs'][key] = {
                    'shape': shape,
                    'type': obs_type
                }
        else:
            # Handle Box observation space (most likely case for cam_rgb)
            obs_shape = obs_space.shape
            obs_type = 'rgb' if len(obs_shape) == 3 else 'low_dim'
            # Use actual image size for rgb observations
            if obs_type == 'rgb':
                # Ensure channel dim is last if present
                if obs_shape[-1] == 3 or obs_shape[-1] == 1:
                    c = obs_shape[-1]
                elif obs_shape[0] == 3 or obs_shape[0] == 1: # Handle CHW case if somehow occurs
                    c = obs_shape[0]
                else: # Default to 3 channels if unsure
                    c = 3
                shape = [args.env_img_size, args.env_img_size, c] # Use args.env_img_size
            else:
                shape = list(obs_shape)
            # Assuming the key is 'obs' if it's a flat Box space
            shape_meta['obs']['obs'] = {
                'shape': shape,
                'type': obs_type
            }
        del env_spec   # cleanup

        # --- Log the final shape_meta being used ---
        print("\n============= Final shape_meta for Policy =============")
        for key in sorted(shape_meta['obs']):
            print(f"  {key} : {shape_meta['obs'][key]}")
        print("=======================================================\n")
        # --------------------------------------------

        # interpret "0 0" or any non-positive dims as disabling the crop
        if len(args.crop_shape) == 2 and args.crop_shape[0] > 0 and args.crop_shape[1] > 0:
            crop_shape_val = (args.crop_shape[0], args.crop_shape[1])
        else:
            crop_shape_val = None

        policy = DiffusionTransformerHybridImagePolicy(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            horizon=args.horizon,
            n_action_steps=args.n_action_steps,
            n_obs_steps=args.n_obs_steps,
            num_inference_steps=args.num_inference_steps,
            crop_shape=crop_shape_val,
            visual_encoder=args.visual_encoder,
            n_layer=args.transformer_n_layer,
            n_cond_layers=args.transformer_n_cond_layers,
            n_head=args.transformer_n_head,
            n_emb=args.transformer_n_emb,
            p_drop_emb=args.transformer_p_drop_emb,
            p_drop_attn=args.transformer_p_drop_attn,
            causal_attn=args.transformer_causal_attn,
            time_as_cond=args.transformer_time_as_cond,
            obs_as_cond=args.obs_as_global_cond,
            pred_action_steps_only=args.pred_action_steps_only,
        )
    elif args.is_image_based:
        # -------------------------------------------------------------------
        # Image-based policy (UNet)
        policy = DiffusionUnetImagePolicy(
            noise_scheduler=noise_scheduler,
            horizon=args.horizon,
            image_size=args.env_img_size,
            action_dim=args.action_size,
            n_action_steps=args.n_action_steps,
            n_obs_steps=args.n_obs_steps,
            num_inference_steps=args.num_inference_steps,
            channel_cond=args.channel_cond,
            cond_predict_scale=args.cond_predict_scale,
            cnn_channels=args.cnn_channels,
            cnn_kernels=args.cnn_kernels,
            cnn_strides=args.cnn_strides,
            obs_as_global_cond=args.obs_as_global_cond,
            latent_dim=args.latent_dim
        )
    else:
        # state-based policy (lowdim)
        if args.model_type == 'transformer':
            # --- Transformer-based branch ---
            cond_dim = args.observation_size if args.obs_as_global_cond else 0
            input_dim = args.action_size if args.obs_as_global_cond \
                        else (args.action_size + args.observation_size)
            model = TransformerForDiffusion(
                input_dim=input_dim,
                output_dim=args.action_size,
                horizon=args.horizon,
                n_obs_steps=args.n_obs_steps,
                cond_dim=cond_dim,
                obs_as_cond=args.obs_as_global_cond,
                n_emb=args.transformer_n_emb,
                n_layer=args.transformer_n_layer,
                n_head=args.transformer_n_head,
                p_drop_emb=args.transformer_p_drop_emb,
                p_drop_attn=args.transformer_p_drop_attn,
                causal_attn=args.transformer_causal_attn,
                time_as_cond=args.transformer_time_as_cond,
                n_cond_layers=args.transformer_n_cond_layers
            )
            policy = DiffusionTransformerLowdimPolicy(
                model=model,
                noise_scheduler=noise_scheduler,
                horizon=args.horizon,
                obs_dim=args.observation_size,
                action_dim=args.action_size,
                n_action_steps=args.n_action_steps,
                n_obs_steps=args.n_obs_steps,
                num_inference_steps=args.num_inference_steps,
                obs_as_cond=args.obs_as_global_cond,
                pred_action_steps_only=args.pred_action_steps_only
            )
        else:
            # Calculate input dimension based on conditioning
            if args.obs_as_local_cond or args.obs_as_global_cond:
                input_dim = args.action_size
            else:
                input_dim = args.observation_size + args.action_size
            
            # Calculate conditioning dimensions
            local_cond_dim = args.observation_size if args.obs_as_local_cond else None
            global_cond_dim = args.observation_size * args.n_obs_steps if args.obs_as_global_cond else None
            
            # Create the UNet model
            model = ConditionalUnet1D(
                input_dim=input_dim,
                local_cond_dim=local_cond_dim,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=args.diffusion_step_embed_dim,
                down_dims=args.unet_down_dims,
                kernel_size=args.unet_kernel_size,
                n_groups=args.unet_n_groups,
                cond_predict_scale=args.cond_predict_scale
            )
            
            # Create the diffusion policy
            policy = DiffusionUnetLowdimPolicy(
                model=model,
                noise_scheduler=noise_scheduler,
                horizon=args.horizon,
                obs_dim=args.observation_size,
                action_dim=args.action_size,
                n_action_steps=args.n_action_steps,
                n_obs_steps=args.n_obs_steps,
                num_inference_steps=args.num_inference_steps,
                obs_as_local_cond=args.obs_as_local_cond,
                obs_as_global_cond=args.obs_as_global_cond,
                pred_action_steps_only=args.pred_action_steps_only,
                oa_step_convention=args.oa_step_convention
            )
    
    # Create EMA model if needed
    if args.use_ema:
        # pick the correct sub-module for EMA:
        if args.is_image_based and args.model_type == 'transformer':
            # hybrid image-transformer: the diffusion network is policy.model
            ema_target = policy.model
        elif args.is_image_based:
            # image-unet policy: the UNet is stored in policy.nets['action_model']
            ema_target = policy.nets['action_model']
        else:
            # state-based branch: the model you passed into the policy
            ema_target = model

        ema_model = EMAModel(
            model=ema_target,
            update_after_step=0,
            inv_gamma=1.0,
            power=0.75,
            min_value=0.0,
            max_value=0.9999
        )
    else:
        ema_model = None
    
    return policy, ema_model

def main_training():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Print training configuration
    print("\n" + "="*60)
    print("DIFFUSION POLICY TRAINING CONFIGURATION")
    print("="*60)
    # collect only the args that actually drive training
    config = {
        "env_name": args.env_name,
        "observation_mode": args.env_kwargs["observation_mode"],
        "is_image_based": args.is_image_based,
        "model_type": args.model_type,
        "horizon": args.horizon,
        "n_obs_steps": args.n_obs_steps,
        "n_action_steps": args.n_action_steps,
        "num_inference_steps": args.num_inference_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lrate,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "lr_scheduler": args.lr_scheduler,
        "scheduler_step_size": args.scheduler_step_size,
        "scheduler_gamma": args.scheduler_gamma,
        "lr_warmup_steps": args.lr_warmup_steps,
        "max_train_epochs": args.max_train_epochs,
        "max_train_steps": args.max_train_steps,
        "seed": args.seed,
        "use_ema": args.use_ema,
        "obs_as_global_cond": args.obs_as_global_cond,
        "obs_as_local_cond": args.obs_as_local_cond,
        "pred_action_steps_only": args.pred_action_steps_only,
        "cond_predict_scale": args.cond_predict_scale,
        "train_data_ratio": args.train_data_ratio,
        "eval_interval": args.eval_interval,
        "wandb": args.wandb,
    }

    # add only the policy‐type & modality‐specific params
    if args.is_image_based:
        img_cfg = {"env_img_size": args.env_img_size}
        if args.enable_img_transformations:
            img_cfg["enable_img_transformations"] = True
        if args.model_type == "unet":
            img_cfg.update({
                "cnn_channels": args.cnn_channels,
                "cnn_kernels": args.cnn_kernels,
                "cnn_strides": args.cnn_strides,
                "latent_dim": args.latent_dim,
            })
        elif args.model_type == "transformer":
            img_cfg.update({
                "transformer_n_emb": args.transformer_n_emb,
                "transformer_n_layer": args.transformer_n_layer,
                "transformer_n_head": args.transformer_n_head,
                "transformer_p_drop_emb": args.transformer_p_drop_emb,
                "transformer_p_drop_attn": args.transformer_p_drop_attn,
                "transformer_causal_attn": args.transformer_causal_attn,
                "transformer_time_as_cond": args.transformer_time_as_cond,
                "transformer_n_cond_layers": args.transformer_n_cond_layers,
                "visual_encoder": args.visual_encoder,
                "transformer_weight_decay": args.transformer_weight_decay,
                "encoder_weight_decay": args.encoder_weight_decay,
            })
            if args.crop_shape[0] > 0 and args.crop_shape[1] > 0:
                img_cfg["crop_shape"] = tuple(args.crop_shape)
        config.update(img_cfg)
    else:
        ld_cfg = {
            "observation_size": args.observation_size,
            "action_size": args.action_size,
            "diffusion_step_embed_dim": args.diffusion_step_embed_dim,
            "oa_step_convention": args.oa_step_convention,
            "weight_decay": args.weight_decay,
        }
        if args.model_type == "unet":
            ld_cfg.update({
                "unet_down_dims": args.unet_down_dims,
                "unet_kernel_size": args.unet_kernel_size,
                "unet_n_groups": args.unet_n_groups,
            })
        elif args.model_type == "transformer":
            ld_cfg.update({
                "transformer_n_emb": args.transformer_n_emb,
                "transformer_n_layer": args.transformer_n_layer,
                "transformer_n_head": args.transformer_n_head,
                "transformer_p_drop_emb": args.transformer_p_drop_emb,
                "transformer_p_drop_attn": args.transformer_p_drop_attn,
                "transformer_causal_attn": args.transformer_causal_attn,
                "transformer_time_as_cond": args.transformer_time_as_cond,
                "transformer_n_cond_layers": args.transformer_n_cond_layers,
            })
        config.update(ld_cfg)

    # prune any None values and pretty-print
    config = {k: v for k, v in config.items() if v is not None}
    max_key_len = max(len(k) for k in config) if config else 0
    for key in sorted(config):
        print(f"  {key.ljust(max_key_len)} : {config[key]}")
    print("="*60 + "\n")
    
    # Set up image transformations if needed
    img_transform = None
    if args.is_image_based and args.enable_img_transformations:
        img_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ])
    
    # Create policy and EMA model
    policy, ema_model = create_diffusion_policy(args)
    policy = policy.to(device)

    # Ensure we always have the real LinearNormalizer, and move it to device
    if not hasattr(policy, 'normalizer') or policy.normalizer is None:
        policy.normalizer = LinearNormalizer()
    policy.normalizer.to(device)

    # Make sure the network isn't accidentally frozen
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            param.requires_grad_(True)

    # Collect and inspect the actual sub-network parameters we want to train
    if args.is_image_based:
        if args.model_type == 'transformer':
            # image-based transformer policy: use the underlying transformer model
            param_list = list(policy.model.parameters())
        else:
            # image-based UNet policy: parameters in nets['action_model']
            param_list = list(policy.nets['action_model'].parameters())
    else:
        # state-based policy (lowdim): parameters in .model
        param_list = list(policy.model.parameters())

    total_params     = sum(p.numel() for p in param_list)
    trainable_params = sum(p.numel() for p in param_list if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Collect parameters and setup optimizer
    if args.is_image_based and args.model_type == 'transformer':
        # Use policy's specific optimizer setup for hybrid transformer
         # TODO: Add parser arguments for these:
        transformer_wd = args.transformer_weight_decay # e.g., 1e-6
        encoder_wd = args.encoder_weight_decay     # e.g., 1e-4 
        
        optimizer = policy.get_optimizer(
            transformer_weight_decay=transformer_wd,
            obs_encoder_weight_decay=encoder_wd,
            learning_rate=args.lrate,
            betas=(args.beta1, args.beta2)
        )
        print("Using policy's get_optimizer for TransformerHybridImagePolicy.")
        # Get param_list just for reporting counts if needed
        param_list = list(policy.model.parameters()) + list(policy.obs_encoder.parameters())
    
    else:
         # Existing logic for other policy types (double-check if they need similar treatment)
         if args.is_image_based and args.model_type == 'unet':
             # DiffusionUnetImagePolicy might store nets differently
             param_list = list(policy.nets['action_model'].parameters()) # Assuming UNet model is here
             # Also need image encoder params if separate? Check policy structure.
             # DiffusionUnetImagePolicy seems to have integrated encoder? Check its __init__
             # Looks like DiffusionUnetImagePolicy has encoder within nets['vision_encoder']
             if 'vision_encoder' in policy.nets:
                   param_list += list(policy.nets['vision_encoder'].parameters())

         elif not args.is_image_based:
             # State-based: Original logic seems okay (uses policy.model)
             param_list = list(policy.model.parameters())
         else:
             # Fallback or error
             raise NotImplementedError("Parameter collection logic missing for this policy type")

         print(f"Using generic AdamW optimizer setup.")
         optimizer = optim.AdamW(
             param_list,
             lr=args.lrate,
             betas=(args.beta1, args.beta2),
             eps=1e-8,
             weight_decay=args.weight_decay # Need a generic weight decay arg? Or assume 1e-6?
             # Might need to add --weight_decay arg to parser
         )

    total_params = sum(p.numel() for grp in optimizer.param_groups for p in grp['params'])
    trainable_params = sum(p.numel() for grp in optimizer.param_groups for p in grp['params'] if p.requires_grad)
    print(f"Total Parameters (Optimizer Groups): {total_params:,}")
    print(f"Trainable Parameters (Optimizer Groups): {trainable_params:,}")
    
    # Create learning rate scheduler
    if args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma
        )
    elif args.lr_scheduler == 'cosine':
        import math
        def lr_lambda(epoch):
            # linear warmup
            if epoch < args.lr_warmup_steps:
                return float(epoch) / float(max(1, args.lr_warmup_steps))
            # cosine decay after warmup
            progress = float(epoch - args.lr_warmup_steps) \
                       / float(max(1, args.max_train_epochs - args.lr_warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")
    
    # Load data
    assert args.saved_rollouts is not None, "Must provide path to demonstrations"
    print(f"Loading demonstrations from: {args.saved_rollouts}")
    dataset = Demonstrations(
        args.saved_rollouts,
        is_image_based=args.is_image_based,
        img_transform=img_transform,
        horizon=args.horizon
    )

    # Normalizer: fit on data (unless resuming), exactly as we do for state-based
    if not args.resume:
        print("Fitting normalizer on demonstration data...")
        all_action = torch.cat([traj['action'] for traj in dataset.trajectories], dim=0) # Use cat for T*B, A
        fit_data = {'action': all_action}
        if not args.is_image_based:
             # Only include obs for state-based
             all_obs = torch.cat([traj['obs'] for traj in dataset.trajectories], dim=0) # Use cat for T*B, Do
             fit_data['obs'] = all_obs
        
        # Fit normalizer (adjust obs/action dims automatically)
        policy.normalizer.fit(fit_data, last_n_dims=1, mode='limits') # Or whatever mode is appropriate
        # Ensure normalizer state is transferred if needed, although fit usually sets it directly
    else:
        print("Skipping normalizer fit; loaded from checkpoint.")

    # Split into train and validation sets
    train_size = int(len(dataset) * args.train_data_ratio)
    val_size   = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Sample a batch to log shapes
    sample_batch = next(iter(train_loader))
    print("\nData shapes:")
    for k, v in sample_batch.items():
        print(f"  {k}: {v.shape}")
    
    #resumee can reference resume_ckpt when initializing WandB
    start_epoch = 0
    resume_ckpt = None
    if args.resume:
        # locate checkpoint folder
        resume_model_dir = os.path.join(args.model_save_dir, args.folder_name, 'checkpoints')
        if not os.path.isdir(resume_model_dir):
            raise ValueError(f"Checkpoint dir not found: {resume_model_dir}")
        ckpt_files = [f for f in os.listdir(resume_model_dir)
                      if f.startswith('epoch_') and f.endswith('.pth')]
        if not ckpt_files:
            raise ValueError(f"No checkpoint files in {resume_model_dir}")
        # pick the latest epoch
        def _epoch_num(f):
            try:
                return int(f.split('_')[1].split('.pth')[0])
            except:
                return -1
        latest_epoch, latest_file = max(
            [(_epoch_num(f), f) for f in ckpt_files],
            key=lambda x: x[0]
        )
        ckpt_path = os.path.join(resume_model_dir, latest_file)
        print(f"Resuming from checkpoint: {ckpt_path}")
        resume_ckpt = torch.load(ckpt_path, map_location=device)
        # restore model, optimizer, scheduler
        policy.load_state_dict(resume_ckpt['model_state_dict'])
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
        # restore normalizer if present
        if hasattr(policy, 'normalizer') and resume_ckpt.get('normalizer'):
            policy.normalizer.load_state_dict(resume_ckpt['normalizer'])
        # restore EMA if used
        if args.use_ema and resume_ckpt.get('ema_state_dict'):
            ema_model.averaged_model.load_state_dict(resume_ckpt['ema_state_dict'])
        # pick up from next epoch
        start_epoch = resume_ckpt['epoch'] + 1

    # Initialize WandB if enabled
    if args.wandb:
        project_id =  "priv-diff" # "cloth-diff"
        if args.resume and resume_ckpt is not None and resume_ckpt.get('wandb_run_id'):
            # re-attach to the same run
            wandb.init(
                project=project_id,
                id=resume_ckpt['wandb_run_id'],
                resume="must"
            )
        else:
            # fresh run
            wandb.init(
                project=project_id,
                name=args.folder_name,
                config=vars(args)
            )
        args.folder_name = wandb.run.name
        wandb.watch(policy, log="all", log_freq=100)
        wandb.log({
            "description/total_parameters": total_params,
            "description/trainable_parameters": trainable_params
        }, step=0)

    # Create directories for saving models
    model_dir = os.path.join(args.model_save_dir, args.folder_name, 'checkpoints')
    make_dir(model_dir)
    print(f"Model checkpoints will be saved to: {model_dir}")
    
    # Create evaluator
    evaluator = Evaluation()
    
    # initialize total gradient-step counter (and resume if checkpoint present)
    if args.resume and resume_ckpt is not None:
        total_steps = resume_ckpt.get('total_steps', 0)
        print(f"Resuming from total_steps = {total_steps}")
    else:
        total_steps = 0

    for epoch in range(start_epoch, args.max_train_epochs):
        print(f"Epoch {epoch+1}/{args.max_train_epochs}")
        
        # Training phase
        policy.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm.tqdm(train_loader)):
            # Move batch to device
            for k, v in batch.items():
                batch[k] = v.to(device)
            
            # Forward / backward / step
            optimizer.zero_grad()
            loss = policy.compute_loss(batch)
            loss.backward()
            optimizer.step()
            # Count gradient step and optionally stop
            total_steps += 1
            if args.max_train_steps is not None and total_steps >= args.max_train_steps:
                print(f"Reached max_train_steps={args.max_train_steps}, stopping training early")
                break
            train_loss += loss.item()
        
        # Scheduler update
        scheduler.step()

        # Break out of epoch loop if step-limit reached
        if args.max_train_steps is not None and total_steps >= args.max_train_steps:
            break

        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase - only compute every 10th epoch
        avg_val_loss = 0.0
        if epoch % 10 == 0:
            policy.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    # Move batch to device
                    for k, v in batch.items():
                        batch[k] = v.to(device)
                    
                    # Forward pass and loss computation
                    loss = policy.compute_loss(batch)
                    val_loss += loss.item()
            
            # Calculate average validation loss
            avg_val_loss = val_loss / len(val_loader)
            print(f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        else:
            print(f"Train Loss: {avg_train_loss:.6f}")
        
        # Log metrics
        metrics = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'gradient_steps': total_steps
        }
        
        if epoch % 10 == 0:
            metrics['val_loss'] = avg_val_loss
        
        # --- Logging to WandB ---
        if args.wandb:
            log_data_base = {
                "train/train_loss": avg_train_loss,
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "train/gradient_steps": total_steps,
            }
            
            if epoch % 10 == 0:
                log_data_base["validation/val_loss"] = avg_val_loss

            # Evaluation and Checkpointing
            if (epoch + 1) % args.eval_interval == 0:
                 # --- Replace model with EMA model for evaluation if used ---
                original_state_dict = None
                if args.use_ema:
                    ema_params = ema_model.averaged_model.state_dict()
                    if args.is_image_based and args.model_type == 'transformer':
                        # hybrid transformer image policy keeps its diffusion net in policy.model
                        original_state_dict = policy.model.state_dict()
                        policy.model.load_state_dict(ema_params)
                    elif args.is_image_based:
                        # UNet image policy stores its decoder in nets['action_model']
                        original_state_dict = policy.nets['action_model'].state_dict()
                        policy.nets['action_model'].load_state_dict(ema_params)
                    else:
                        # state-based (lowdim) policies attach the model directly
                        original_state_dict = policy.model.state_dict()
                        policy.model.load_state_dict(ema_params)

                # Evaluate policy
                # Now returns: avg_normalized_performance_final, avg_reward, avg_ep_length, saved_gif_path
                avg_normalized_perf, std_normalized_perf, avg_reward, avg_ep_length, saved_gif_path = evaluator.evaluate(
                    policy,
                    num_episodes=args.num_eval_eps,
                    save_video=args.eval_videos,
                    epoch=epoch # Pass epoch for GIF naming
                )

                print(f"Validation Eval: Norm Perf Mean: {avg_normalized_perf:.4f}, Std: {std_normalized_perf:.4f}, "
                      f"Avg Reward: {avg_reward:.2f}, Avg Ep Length: {avg_ep_length:.2f}")

                # Log evaluation metrics to WandB
                log_data_eval = {
                    "validation/info_normalized_performance_mean": avg_normalized_perf,
                    "validation/info_normalized_performance_std": std_normalized_perf,
                    "validation/avg_rews": avg_reward,
                    "validation/avg_ep_length": avg_ep_length,
                }
                log_data_base.update(log_data_eval) # Add eval metrics to base log data

                # Add video log if available
                if args.eval_videos and saved_gif_path:
                     log_data_base["validation/eval_video"] = wandb.Video(saved_gif_path, fps=10, format="gif")

                # --- Restore original weights ---
                if args.use_ema and original_state_dict is not None:
                    if args.is_image_based and args.model_type == 'transformer':
                        policy.model.load_state_dict(original_state_dict)
                    elif args.is_image_based:
                        policy.nets['action_model'].load_state_dict(original_state_dict)
                    else:
                        policy.model.load_state_dict(original_state_dict)

                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss if epoch % 10 == 0 else None,
                    'normalizer': policy.normalizer.state_dict() if hasattr(policy, 'normalizer') else None,
                    'wandb_run_id': wandb.run.id if args.wandb else None,
                    'wandb_run_name': args.folder_name,
                    'total_steps': total_steps,
                }

                if args.use_ema:
                    checkpoint['ema_state_dict'] = ema_model.averaged_model.state_dict() # Save EMA params

                torch.save(checkpoint, os.path.join(model_dir, f'epoch_{epoch+1}.pth'))
                print(f"Saved checkpoint at epoch {epoch+1}")

            # Log combined data to WandB at the end of the epoch
            wandb.log(log_data_base, step=epoch + 1)

    # --- Final evaluation after training loop ---
    print("\nRunning final evaluation over 5 seeds (100 episodes total)...")
    original_state_dict = None
    if args.use_ema:
        ema_params = ema_model.averaged_model.state_dict()
        if args.is_image_based and args.model_type == 'transformer':
            original_state_dict = policy.model.state_dict()
            policy.model.load_state_dict(ema_params)
        elif args.is_image_based:
            original_state_dict = policy.nets['action_model'].state_dict()
            policy.nets['action_model'].load_state_dict(ema_params)
        else:
            original_state_dict = policy.model.state_dict()
            policy.model.load_state_dict(ema_params)

    # Evaluate across 5 seeds (20 eps each)
    evaluator.evaluate_five_seeds(policy)

    if args.wandb:
        wandb.finish()


def main_testing():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create policy
    policy, _ = create_diffusion_policy(args) # We don't need the EMA object here
    policy = policy.to(device)

    # Load checkpoint
    print(f"Loading checkpoint from: {args.test_checkpoint}")
    checkpoint = torch.load(args.test_checkpoint, map_location=device)
    policy.load_state_dict(checkpoint['model_state_dict'])

    # --- Load Normalizer ---
    if hasattr(policy, 'normalizer') and checkpoint.get('normalizer') is not None:
        print("Loading normalizer state from checkpoint.")
        policy.normalizer.load_state_dict(checkpoint['normalizer'])
        # Ensure normalizer is on the correct device
        policy.normalizer.to(device)
    elif hasattr(policy, 'normalizer'):
         print("Warning: Policy has a normalizer, but no normalizer state found in checkpoint.")
    # ----------------------

    # Load EMA averaged_model parameters directly into the policy network if EMA was used during training
    if args.use_ema and 'ema_state_dict' in checkpoint:
        print("Loading EMA parameters from checkpoint.")
        if args.is_image_based:
            policy.nets['action_model'].load_state_dict(checkpoint['ema_state_dict'])
        else:
            policy.model.load_state_dict(checkpoint['ema_state_dict'])
    elif args.use_ema:
        print("Warning: --use_ema is True, but no 'ema_state_dict' found in checkpoint. Using standard model weights.")


    # Set policy to evaluation mode
    policy.eval()

    # Create evaluator
    evaluator = Evaluation()

    # Run evaluation
    if args.eval_over_five_seeds:
        print("Running evaluation over 5 seeds (100 episodes total)...")
        evaluator.evaluate_five_seeds(policy) # This function handles its own printing and saving
    else:
        print(f"Running evaluation for {args.num_eval_eps} episodes...")
        # Evaluate and print results
        avg_normalized_perf, std_normalized_perf, avg_reward, avg_ep_length, saved_gif_path = evaluator.evaluate(
            policy,
            num_episodes=args.num_eval_eps,
            save_video=args.eval_videos
        )
        # Final print handled within evaluate method, but can add a summary here if needed
        print("\nSingle-seed Evaluation Complete.")
        # print(f"Final Eval: Norm Perf: {avg_normalized_perf:.4f}, Avg Reward: {avg_reward:.2f}, Avg Ep Length: {avg_ep_length:.2f}")


if __name__ == "__main__":
    if args.is_eval:
        main_testing()
    else:
        main_training() 