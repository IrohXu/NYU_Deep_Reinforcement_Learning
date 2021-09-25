"""
[CSCI-GA 3033-090] Special Topics: Deep Reinforcement Learning

Homework - 1, DAgger
Deadline: Sep 17, 2021 11:59 PM.

Complete the code template provided in dagger.py, with the right 
code in every TODO section, to implement DAgger. Attach the completed 
file in your submission.
"""

import tqdm
import hydra
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as T

import numpy as np
import matplotlib.pyplot as plt

from reacher_env import ReacherDaggerEnv
from utils import weight_init, ExpertBuffer


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # TODO define your own network
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=76800, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Linear(in_features=1024, out_features=2)
        )

        # self.final = nn.Tanh()

        self.apply(weight_init)

    def forward(self, x):
        # Normalize
        x = x / 255.0 - 0.5
        # TODO pass it forward through your network.
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def initialize_model_and_optim(cfg):
    # TODO write a function that creates a model and associated optimizer
    # given the config object.

    model = CNN()
    model.to(cfg.device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    return model, optimizer

    # pass 


class Workspace:
    def __init__(self, cfg):
        self._work_dir = os.getcwd()
        print(f'workspace: {self._work_dir}')

        self.cfg = cfg

        self.device = torch.device(cfg.device)
        self.train_env = ReacherDaggerEnv()
        self.eval_env = ReacherDaggerEnv()

        self.expert_buffer = ExpertBuffer(cfg.experience_buffer_len,    # 150000
                                          self.train_env.observation_space.shape,
                                          self.train_env.action_space.shape)
        
        self.model, self.optimizer = initialize_model_and_optim(cfg)

        # TODO: define a loss function
        # self.loss_function = nn.MSELoss()    #  L2 Loss
        self.loss_function = nn.SmoothL1Loss()   # Huber Loss

        self.transforms = T.Compose([
            # T.RandomResizedCrop(size=(60, 80), scale=(0.95, 1.0)),
            T.Resize(size=(60, 80))
        ])
        self.eval_transforms = T.Compose([
            T.Resize(size=(60, 80))
        ])

    def eval(self):
        # A function that evaluates the 
        # Set the DAgger model to evaluation
        self.model.eval()

        avg_eval_reward = 0.
        avg_episode_length = 0.
        for _ in range(self.cfg.num_eval_episodes):
            eval_reward = 0.
            ep_length = 0.
            obs_np = self.eval_env.reset()
            # Need to be moved to torch from numpy first
            obs = torch.from_numpy(obs_np).float().to(self.device).unsqueeze(0)
            t_obs = self.eval_transforms(obs)
            with torch.no_grad():
                action = self.model(t_obs)
            done = False
            while not done:
                # Need to be moved to numpy from torch
                action = action.squeeze().detach().cpu().numpy()
                obs, reward, done, info = self.eval_env.step(action)
                obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
                t_obs = self.eval_transforms(obs)
                with torch.no_grad():
                    action = self.model(t_obs)
                eval_reward += reward
                ep_length += 1.
            avg_eval_reward += eval_reward
            avg_episode_length += ep_length
        avg_eval_reward /= self.cfg.num_eval_episodes
        avg_episode_length /= self.cfg.num_eval_episodes
        return avg_eval_reward, avg_episode_length


    def model_training_step(self):
        # A function that optimizes the model self.model using the optimizer 
        # self.optimizer using the experience  stored in self.expert_buffer.
        # Number of optimization step should be self.cfg.num_training_steps.

        # Set the model to training.
        self.model.train()
        # For num training steps, sample data from the training data.
        avg_loss = 0.
        for _ in range(self.cfg.num_training_steps):
            # TODO write the training code.
            # Hint: use the self.transforms to make sure the image observation is of the right size.
            batch_obs_np, batch_action = self.expert_buffer.sample(batch_size = self.cfg.batch_size)
            batch_obs = torch.from_numpy(batch_obs_np).float().to(self.device)
            batch_action = torch.from_numpy(batch_action).float().to(self.device)
            t_batch_obs = self.transforms(batch_obs)
            self.optimizer.zero_grad()
            
            pred_action = self.model(t_batch_obs)
            loss = self.loss_function(pred_action, batch_action)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()

            # pass
        avg_loss /= self.cfg.num_training_steps
        return avg_loss


    def run(self):
        train_loss, eval_reward, episode_length = 1., 0, 0
        iterable = tqdm.trange(self.cfg.total_training_episodes)

        # Add information for Question 2's plotting
        expert_queries = []
        eval_rewards = []

        beta = 0.995 # beta for Dagger sudo-code

        beta_decay = lambda x : beta ** x

        for ep_num in iterable:
            iterable.set_description('Collecting exp')
            # Set the DAGGER model to evaluation
            self.model.eval()
            ep_train_reward = 0.
            ep_length = 0.

            # TODO write the training loop.
            # 1. Roll out your current model on the environment.
            # 2. On each step, after calling either env.reset() or env.step(), call 
            #    env.get_expert_action() to get the expert action for the current 
            #    state of the environment.
            # 3. Store that observation alongside the expert action in the buffer.
            # 4. When you are training, use the stored obs and expert action.

            # Hints:
            # 1. You will need to convert your obs to a torch tensor before passing it
            #    into the model.
            # 2. You will need to convert your action predicted by the model to a numpy
            #    array before passing it to the environment.
            # 3. Make sure the actions from your model are always in the (-1, 1) range.
            # 4. Both the environment observation and the expert action needs to be a
            #    numpy array before being added to the environment.
            # 5. Use the self.transforms to make sure the image observation is of the right size.
            
            # TODO training loop here.

            obs_np = self.train_env.reset()
            obs = torch.from_numpy(obs_np).float().to(self.device).unsqueeze(0)
            t_obs = self.transforms(obs)
            expert_action = self.train_env.get_expert_action()
            self.expert_buffer.insert(obs_np, expert_action)
            with torch.no_grad():
                action = self.model(t_obs)
            done = False
            while not done:
                # Need to be moved to numpy from torch
                action = action.squeeze().detach().cpu().numpy()

                action = (1-beta_decay(ep_num)) * action + beta_decay(ep_num) * expert_action

                obs_np, reward, done, info = self.train_env.step(action)
                obs = torch.from_numpy(obs_np).float().to(self.device).unsqueeze(0)
                t_obs = self.eval_transforms(obs)
                expert_action = self.train_env.get_expert_action()

                # print((action, expert_action))

                self.expert_buffer.insert(obs_np, expert_action)
                with torch.no_grad():
                    action = self.model(t_obs)
                ep_train_reward += reward
                ep_length += 1.

            train_reward = ep_train_reward
            train_episode_length = ep_length

            if (ep_num + 1) % self.cfg.train_every == 0:
                # Reinitialize model every time we are training
                iterable.set_description('Training model')
                # TODO train the model and set train_loss to the appropriate value.
                # Hint: in the DAgger algorithm, when do we initialize a new model?

                self.model, self.optimizer = initialize_model_and_optim(self.cfg)
                train_loss = self.model_training_step()

            if (ep_num + 1) % self.cfg.eval_every == 0:
                # Evaluation loop
                iterable.set_description('Evaluating model')
                eval_reward, episode_length = self.eval()

                expert_queries.append(self.train_env.expert_calls)
                eval_rewards.append(eval_reward)

                # print((self.train_env.expert_calls, eval_reward))

            iterable.set_postfix({
                'Train loss': train_loss,
                'Train reward': train_reward,
                'Eval reward': eval_reward
            })
        
        expert_queries = np.array(expert_queries)
        eval_rewards = np.array(eval_rewards)

        return expert_queries, eval_rewards


@hydra.main(config_path='.', config_name='train')
def main(cfg):
    # In hydra, whatever is in the train.yaml file is passed on here
    # as the cfg object. To access any of the parameters in the file,
    # access them like cfg.param, for example the learning rate would
    # be cfg.lr
    workspace = Workspace(cfg)
    expert_queries, eval_rewards = workspace.run()

    np.savetxt("plot.out",(expert_queries, eval_rewards)) 
    print("Plot information is saved.")


if __name__ == '__main__':
    main()


