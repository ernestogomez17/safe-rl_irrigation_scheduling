import gymnasium as gym
import numpy as np
import torch
from models.safe_rl.worker.replay_buffer import ReplayBuffer
from models.safe_rl.policy.base_policy import Policy
from models.safe_rl.util.logger import EpochLogger
from models.safe_rl.util.torch_util import to_tensor


class Collector:
    r'''
    Collect data based on the policy and env, and store the interaction data to data buffer.
    '''
    def __init__(self,
                 env: gym.Env,
                 policy: Policy,
                 logger: EpochLogger,
                 batch_size=100,
                 timeout_steps=1000,
                 buffer_size=2e5,
                 warmup_steps=1000,
                 **kwargs) -> None:
        self.env = env
        self.policy = policy
        self.logger = logger
        self.batch_size = batch_size
        self.timeout_steps = timeout_steps

        obs_dim = env.observation_space.shape[-1]
        act_dim = env.action_space.shape[-1]

        env_dict = {
            'obs': {'shape': (obs_dim,), 'dtype': np.float32},
            'act': {'shape': (act_dim,), 'dtype': np.float32},
            'rew': {'shape': (), 'dtype': np.float32},
            'obs2': {'shape': (obs_dim,), 'dtype': np.float32},
            'done': {'shape': (), 'dtype': bool},
            'indicator': {'shape': (), 'dtype': np.float32},
            'S_STAR': {'shape': (), 'dtype': np.float32},
            'SFC': {'shape': (), 'dtype': np.float32},
            'SW': {'shape': (), 'dtype': np.float32}
        }

        # Use regular replay buffer with uniform sampling
        self.off_policy_buffer = ReplayBuffer(buffer_size, env_dict)
        print("Using regular replay buffer: uniform sampling")

        ######### Warmup phase to collect data with random policy #########
        steps = 0
        while steps < warmup_steps:
            epoch_steps, _, _ = self.work(warmup=True)  # Unpack only the needed value
            steps += epoch_steps

        ######### Train the policy with warmup samples #########
        for i in range(warmup_steps // 2):
            # Sample a batch of random data from the main buffer
            data = self.get_sample()
            self.policy.learn_on_batch(data)

    def work(self, warmup=False, sample_episode_num=5):
        '''
        Interact with the environment to collect data for multiple episodes
        '''
        total_epoch_steps = 0
        total_rewards = []
        total_violations = []
        
        # Run multiple training episodes
        for episode_idx in range(sample_episode_num):
            obs, info = self.env.reset()
            ep_reward, ep_len, ep_indicator, ep_violations = 0, 0, 0, 0
            
            episode_steps = 0
            terminal_freq = 0
            done_freq = 0
            
            for i in range(self.timeout_steps):
                if warmup:
                    action = self.env.action_space.sample()
                else:
                    action, _ = self.policy.act(obs, deterministic=False, with_logprob=False)
                
                obs_next, reward, indicator, terminated, truncated, info = self.env.step(action)

                # Ensure expected indicators are in info, report if not
                expected_keys = ['S_STAR indicator', 'SFC indicator', 'SW indicator']
                for key in expected_keys:
                    if key not in info or np.isnan(info[key]):
                        print(f"Error: {key} missing or NaN in info returned from env.step()")

                done = terminated or truncated

                if done:
                    done_freq += 1

                data = {
                    'obs': np.array(obs, dtype=np.float32),
                    'act': np.array([action], dtype=np.float32),
                    'rew': np.array([reward], dtype=np.float32),
                    'obs2': np.array(obs_next, dtype=np.float32),
                    'done': np.array([float(done)], dtype=np.float32),
                    'indicator': np.array([indicator], dtype=np.float32),
                    'S_STAR': np.array([info.get('S_STAR indicator')], dtype=np.float32),
                    'SFC': np.array([info.get('SFC indicator')], dtype=np.float32),
                    'SW': np.array([info.get('SW indicator')], dtype=np.float32)
                }

                self.off_policy_buffer.add(data)

                ep_reward += reward
                ep_indicator += indicator
                ep_violations += (1 - indicator)
                ep_len += 1
                episode_steps += 1
                total_epoch_steps += 1
                
                obs = obs_next
                
                if done:
                    terminal_freq += 1
                    break
            
            # Log this episode's results
            self.logger.store(
                EpRet=ep_reward, 
                EpIndicator=ep_indicator, 
                EpNumViolations=ep_violations,
                EpLen=ep_len, 
                Terminal=terminal_freq, 
                Done=done_freq, 
                tab="worker"
            )
            
            # Track rewards and violations for each episode
            total_rewards.append(ep_reward)
            total_violations.append(ep_violations)
        
        # Return the total steps and the mean rewards and violations
        # This way, your training loop gets meaningful aggregate statistics
        mean_reward = np.mean(total_rewards) if total_rewards else 0
        mean_violations = np.mean(total_violations) if total_violations else 0
        
        return total_epoch_steps, mean_reward, mean_violations

    def eval(self, eval_episode_num=5):
        '''
        Run evaluation episodes with different seeds.
        '''
        # Save just the last history for plotting
        final_history = {}
        
        # Run multiple evaluation episodes with different seeds
        for eval_episode_num_idx in range(eval_episode_num):
            seed = 50 + eval_episode_num_idx  # Different deterministic seed for each episode
            obs, info = self.env.reset(seed=seed, options={'training_mode': False})
            
            ep_reward = 0
            ep_indicator = 0
            ep_violations = 0
            ep_len = 0
            
            for _ in range(self.timeout_steps):
                action, _ = self.policy.act(obs, deterministic=True, with_logprob=False)
                obs_next, reward, indicator, terminated, truncated, info = self.env.step(action)
                
                ep_reward += reward
                ep_indicator += indicator
                ep_violations += (1 - indicator)
                ep_len += 1
                
                obs = obs_next
                if terminated or truncated:
                    break
            
            # Store each episode's results
            self.logger.store(TestEpRet=ep_reward,
                            TestEpLen=ep_len,
                            TestEpIndicator=ep_indicator,
                            TestEpNumViolations=ep_violations,
                            tab="eval")
            
            # Only keep history from the last episode for plotting
            if eval_episode_num_idx == eval_episode_num - 1:
                final_history = {
                    'It': self.env.history_It.copy() if hasattr(self.env, 'history_It') else [],
                    'Rain': self.env.history_Rain.copy() if hasattr(self.env, 'history_Rain') else [],
                    'st': self.env.history_st.copy() if hasattr(self.env, 'history_st') else [],
                    'ET_o': self.env.history_ET_o.copy() if hasattr(self.env, 'history_ET_o') else [],
                    'ETmax': self.env.history_ETmax.copy() if hasattr(self.env, 'history_ETmax') else [],
                    'Kc': self.env.history_Kc.copy() if hasattr(self.env, 'history_Kc') else [],
                    'rho': self.env.history_rho.copy() if hasattr(self.env, 'history_rho') else [],
                }
        
        # Return history data from final episode
        return (final_history.get('It', []), 
                final_history.get('Rain', []),
                final_history.get('st', []),
                final_history.get('ET_o', []),
                final_history.get('ETmax', []),
                final_history.get('Kc', []),
                final_history.get('rho', []))

    def get_sample(self):
        data = to_tensor(self.off_policy_buffer.sample(self.batch_size))

        # Squeezing single-dimensional entries
        data["rew"] = torch.squeeze(data["rew"])
        data["done"] = torch.squeeze(data["done"])
        data["indicator"] = torch.squeeze(data["indicator"])
        data["S_STAR"] = torch.squeeze(data["S_STAR"])
        data["SFC"] = torch.squeeze(data["SFC"])
        data["SW"] = torch.squeeze(data["SW"])

        return data

    def clear_buffer(self):
        self.off_policy_buffer.clear()