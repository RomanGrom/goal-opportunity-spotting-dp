import os
import cv2
import wandb
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import EvalCallback
import torch

class WandbEvalCallback(EvalCallback):
    def __init__(self, eval_env, save_path, save_freq, video_freq, *args, **kwargs):
        super(WandbEvalCallback, self).__init__(eval_env, *args, **kwargs)
        self.save_freq = save_freq
        self.video_freq = video_freq
        self.save_path = save_path
        self.best_mean_reward = -1000
        self.last_model_path = os.path.join(save_path, 'last_model')
        self.best_model_path = os.path.join(save_path, 'best_model')
        self.video_frames = []
        self.capture_video = False

    def _on_training_start(self) -> None:
        # Initialize WandB project
        wandb.init(project="gfootball_eva", reinit=True, dir=".scratch")

    def _on_step(self) -> bool:
        eval_result = super()._on_step()

        if self.capture_video and self.locals['dones'][0]:
            self._save_video()
            self.capture_video = False

        if self.video_freq > 0 and self.num_timesteps % self.video_freq == 0:
            self.capture_video = True
            self.video_frames = []
            print("Zacinam nahravat video")
        
        if self.capture_video:
            # Choose env
            obs = self.locals['new_obs'][0]
            obs_rgb = obs[:3]
            frame = np.transpose(obs_rgb, (1, 2, 0))
            frame = np.array(frame, dtype=np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            obs_tensor = torch.as_tensor(self.locals['new_obs'], dtype=torch.float32, device=self.model.device)

            action_probs = self.model.policy.get_distribution(obs_tensor).distribution.probs[0].cpu().detach().numpy()

            # get chosen action and value
            selected_action = self.locals['actions'][0]
            value_estimate = self.model.policy.predict_values(obs_tensor)[0].cpu().item()

            # Creating a bar plot for action probabilities
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.bar(range(len(action_probs)), action_probs, color='skyblue')
            ax.set_ylim(0, 1)
            ax.set_title("Action Probabilities")
            ax.set_xlabel("Actions")
            ax.set_ylabel("Probability")
            plt.xticks(range(len(action_probs)), range(len(action_probs)))

            ax.text(0.5, 0.85, f"Selected Action: {selected_action}", 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.6))

            ax.text(0.5, 0.75, f"Value: {value_estimate:.3f}", 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=12, color='blue', bbox=dict(facecolor='white', alpha=0.6))

            # convert to image
            fig.canvas.draw()
            histogram_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            histogram_img = histogram_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            histogram_img = cv2.resize(histogram_img, (300, 224), interpolation=cv2.INTER_AREA)
            plt.close(fig)

            # merge frame with plot
            combined_frame = np.hstack((frame_bgr, histogram_img))
            self.video_frames.append(combined_frame)

        return True

    
    def _on_rollout_end(self) -> None:
        # Log evaluation statistics to WandB after evaluation is completed
        if self.n_eval_episodes > 0:
            print(f"Logging mean reward {self.last_mean_reward} over {self.n_eval_episodes} episodes")
            critic_loss = self.model.logger.name_to_value.get("train/value_loss", 0) 
            wandb.log({
                "eval/mean_reward": self.last_mean_reward,
                "eval/num_episodes": self.n_eval_episodes,
                "timesteps": self.num_timesteps,
                "critic_loss": critic_loss
            })

            # Save the last model
            self.model.save(self.last_model_path)
            print(f"Model saved at {self.last_model_path}")

            # Check if we have a new best mean reward and save the model if so
            if self.last_mean_reward > self.best_mean_reward:
                self.best_mean_reward = self.last_mean_reward
                self.model.save(self.best_model_path)
                print(f"New best model saved with mean reward: {self.best_mean_reward}")


    def _on_training_end(self) -> None:
        if wandb.run:
            wandb.finish()

    def _save_video(self):
        if len(self.video_frames) == 0:
            return
        
        print(f"POCET FRAMOV: {len(self.video_frames)}")
        
        video_path = os.path.join(self.save_path, f'training_video_{self.num_timesteps // (16*1500)}.mp4')
        height, width, _ = self.video_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
        
        for frame in self.video_frames:
            out.write(frame)
        
        out.release()
        print(f"Saved training video at {video_path}")



    