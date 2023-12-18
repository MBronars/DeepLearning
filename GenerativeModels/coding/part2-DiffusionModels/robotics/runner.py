from .environment import PushTEnv
from .dataset import PushTDataset
from diffusion_model import DiffusionModel
import torch
from collections import deque
import numpy as np
from tqdm import tqdm


class EnvRunner:
    def __init__(self):
        # Initialize parameters (do not change these)
        self.action_dim = 2
        self.state_dim = 5
        self.max_timesteps = 250
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define H_a, H_s and num_actions (you can vary these during your experiments)
        self.action_horizon = 16
        self.state_horizon = 2
        self.num_actions = 9

        ################################################################################
        # TODO: Fill in input shape and conditioning dim for the diffusion model       #
        #       Hint: conditioning dim is flattened                                    #
        #       Hint: first dim of input shape (tuple) is time                         # 
        ################################################################################
        self.input_shape = None
        self.cond_dim = None
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################

        self.policy = DiffusionModel(
            input_shape = self.input_shape, 
            condition_dim = self.cond_dim, 
            sequential = True,
            denoising_steps = 100,
            )
    
    def load_data(self, dataset_path, batch_size = 256):
        data = PushTDataset(
            dataset_path = dataset_path,
            action_horizon = self.action_horizon,
            state_horizon = self.state_horizon,
        )
        self.data = data
    
    def train_policy(self, epochs = 10):
        self.policy.train(self.data.data_loader, epochs)

    def run_rollout(self):
        env = PushTEnv()
        env.seed(100000)
        initial_state = env.reset()

        # initialize action and state deques
        state_deque = deque([initial_state] * self.state_horizon, maxlen = self.state_horizon)
        action_deque = deque()

        step_idx = 0
        imgs = [env.render(mode='rgb_array')]
        done = False

        max_index = None
        max_reward = -1

        with tqdm(total=self.max_timesteps, desc="Eval PushTStateEnv") as pbar:
            # loop until max timesteps reached or done (block pushing is successful)
            while not done:
                #############################################################
                # TODO: Fill in the code to generate actions                #
                #                                                           #
                #     Actions and states are normalized for the policy      #
                #       - Normalize the states passed to the policy         #
                #         * self.data.normalize_data(state(s), 'state')     #
                #       - Unnormalize the generated actions                 #
                #         * self.data.unnormalize_data(action(s), 'action') #
                #                                                           #
                #     Hint: Do not generate actions every time step         #
                #     Hint: make use of state_deque and action_deque        #
                #############################################################
                action = None
                #############################################################
                #                        END OF YOUR CODE                   #         
                #############################################################
                
                # take one action step in the environment, get the resulting state, reward, and done
                state, reward, done, _ = env.step(action)
                state_deque.append(state)
                imgs.append(env.render(mode='rgb_array'))
                if reward > max_reward:
                    max_reward = reward
                    max_index = step_idx
                # stop if max timesteps reached
                if step_idx > self.max_timesteps:
                    done = True
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)

        if max_reward <= 0:
            max_images = imgs[-10:]
        else:
            max_images = imgs[max_index-10:max_index + 1]
        print("Max reward: {}".format(max_reward))
        return imgs, max_images