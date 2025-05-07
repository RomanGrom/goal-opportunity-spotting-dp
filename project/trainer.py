import torch
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from pathlib import Path
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

from custom_callback import *
from custom_networks import *
from football_gym import *



# Learning rate function
def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

def compare_models(model1, model2):
    for (name1, param1), (name2, param2) in zip(model1.policy.state_dict().items(), model2.policy.state_dict().items()):
        if not torch.equal(param1, param2):
            print(f"❌ Parameter {name1} is different!")
            return False
    print("✅ All model parameters are identical!")
    return True



class Trainer:

    def __init__(self, cfg):
        self.cfg = cfg            
        self.output_path = Path(HydraConfig.get().run.dir)

        # CUDA / CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Paralel environments
        self.envs = SubprocVecEnv([make_football_env(0) for i in range(cfg.train.num_envs)])

        self.model = None


    def setup(self):
        policy_kwargs = dict(
            net_arch=self.cfg.model.net_arch
        )

        # Our cnn policy and hyperparameters
        self.model = PPO(CustomSharedCnnPolicy,
                         self.envs, verbose=2,
                         policy_kwargs=policy_kwargs,
                         device=self.device,
                          n_steps=self.cfg.train.n_steps,
                          batch_size=self.cfg.train.batch_size,
                          gamma=self.cfg.train.gamma,
                          ent_coef=self.cfg.train.ent_coef,
                          learning_rate=self.cfg.train.lr,
                          n_epochs=self.cfg.train.n_epochs,
                          clip_range=self.cfg.train.clip_range,
                          clip_range_vf=self.cfg.train.clip_range_vf,
                          max_grad_norm=self.cfg.train.max_grad_norm,
                          vf_coef=self.cfg.train.vf_coef)

    def train(self):
        eval_env = SubprocVecEnv([make_football_env(0)])
        eval_callback = WandbEvalCallback(
            eval_env=eval_env, 
            save_path=self.output_path,
            save_freq=4096,  
            eval_freq=4096,
            video_freq=1500 * 16 * 15,
            log_path=self.output_path,
            deterministic=True,
            n_eval_episodes=1
        )
        print(self.model.learning_rate)

        self.model.learn(self.cfg.train.total_timesteps, callback=eval_callback, progress_bar=True)

    def load_model(self, path):
        self.model2 = PPO.load(path, self.envs)
        
        self.model.policy.load_state_dict(self.model2.policy.state_dict())
        self.model.policy.optimizer.load_state_dict(self.model2.policy.optimizer.state_dict())

        compare_models(self.model, self.model2)

        print(f"Model {path} loaded.")


