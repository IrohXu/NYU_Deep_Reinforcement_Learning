"""
[CSCI-GA 3033-090] Special Topics: Deep Reinforcement Learning

Homework - 2

Plot for question 1,2,3,4,5
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plot_dir = './plot'

# Question 1
Breakout_DQN = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_DQN.csv'), sep=',')
plt.plot(Breakout_DQN['step'], Breakout_DQN['episode_reward'])
plt.legend(["DQN"])
plt.xlabel('Steps')
plt.ylabel('Evaluation Rewards')
plt.title('Steps vs Eval Rewards for Breakout')
plt.savefig(os.path.join(plot_dir, 'eval_Breakout_q1.png'))
plt.close()

Pong_DQN = pd.read_csv(os.path.join(plot_dir, 'eval_Pong_DQN.csv'), sep=',')
plt.plot(Pong_DQN['step'], Pong_DQN['episode_reward'])
plt.legend(["DQN"])
plt.xlabel('Steps')
plt.ylabel('Evaluation Rewards')
plt.title('Steps vs Eval Rewards for Pong')
plt.savefig(os.path.join(plot_dir, 'eval_Pong_q1.png'))
plt.close()

SpaceInvaders_DQN = pd.read_csv(os.path.join(plot_dir, 'eval_SpaceInvaders_DQN.csv'), sep=',')
plt.plot(SpaceInvaders_DQN['step'], SpaceInvaders_DQN['episode_reward'])
plt.legend(["DQN"])
plt.xlabel('Steps')
plt.ylabel('Evaluation Rewards')
plt.title('Steps vs Eval Rewards for SpaceInvaders')
plt.savefig(os.path.join(plot_dir, 'eval_SpaceInvaders_q1.png'))
plt.close()


# Question 2
# Parameter selection
Breakout_DQN = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_DQN.csv'), sep=',')
Breakout_DDQN1 = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_DDQN_tau=1.csv'), sep=',')
Breakout_DDQN2 = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_DDQN_tau=0.1.csv'), sep=',')
Breakout_DDQN3 = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_DDQN_tau=0.01.csv'), sep=',')
plt.plot(Breakout_DQN['step'], Breakout_DQN['episode_reward'])
plt.plot(Breakout_DDQN1['step'], Breakout_DDQN1['episode_reward'])
plt.plot(Breakout_DDQN2['step'], Breakout_DDQN2['episode_reward'])
plt.plot(Breakout_DDQN3['step'], Breakout_DDQN3['episode_reward'])
plt.legend(["DQN", "Double DQN tau=1.0", "Double DQN tau=0.1", "Double DQN tau=0.01"])
plt.xlabel('Steps')
plt.ylabel('Evaluation Rewards')
plt.title('Steps vs Eval Rewards for Breakout')
plt.savefig(os.path.join(plot_dir, 'eval_Breakout_q2_tau.png'))
plt.close()

# Breakout
Breakout_DQN = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_DQN.csv'), sep=',')
Breakout_DDQN = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_DDQN.csv'), sep=',')
plt.plot(Breakout_DQN['step'], Breakout_DQN['episode_reward'])
plt.plot(Breakout_DDQN['step'], Breakout_DDQN['episode_reward'])
plt.legend(["DQN", "Double DQN"])
plt.xlabel('Steps')
plt.ylabel('Evaluation Rewards')
plt.title('Steps vs Eval Rewards for Breakout')
plt.savefig(os.path.join(plot_dir, 'eval_Breakout_q2.png'))
plt.close()

# Pong
Pong_DQN = pd.read_csv(os.path.join(plot_dir, 'eval_Pong_DQN.csv'), sep=',')
Pong_DDQN = pd.read_csv(os.path.join(plot_dir, 'eval_Pong_DDQN.csv'), sep=',')
plt.plot(Pong_DQN['step'], Pong_DQN['episode_reward'])
plt.plot(Pong_DDQN['step'], Pong_DDQN['episode_reward'])
plt.legend(["DQN", "Double DQN"])
plt.xlabel('Steps')
plt.ylabel('Evaluation Rewards')
plt.title('Steps vs Eval Rewards for Pong')
plt.savefig(os.path.join(plot_dir, 'eval_Pong_q2.png'))
plt.close()

# SpaceInvaders
SpaceInvaders_DQN = pd.read_csv(os.path.join(plot_dir, 'eval_SpaceInvaders_DQN.csv'), sep=',')
SpaceInvaders_DDQN = pd.read_csv(os.path.join(plot_dir, 'eval_SpaceInvaders_DDQN.csv'), sep=',')
plt.plot(SpaceInvaders_DQN['step'], SpaceInvaders_DQN['episode_reward'])
plt.plot(SpaceInvaders_DDQN['step'], SpaceInvaders_DDQN['episode_reward'])
plt.legend(["DQN", "Double DQN"])
plt.xlabel('Steps')
plt.ylabel('Evaluation Rewards')
plt.title('Steps vs Eval Rewards for SpaceInvaders')
plt.savefig(os.path.join(plot_dir, 'eval_SpaceInvaders_q2.png'))
plt.close()


# Question 3

# Breakout
Breakout_DQN = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_DQN.csv'), sep=',')
Breakout_DDQN = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_DDQN.csv'), sep=',')
Breakout_PER = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_PER.csv'), sep=',')
plt.plot(Breakout_DQN['step'], Breakout_DQN['episode_reward'])
plt.plot(Breakout_DDQN['step'], Breakout_DDQN['episode_reward'])
plt.plot(Breakout_PER['step'], Breakout_PER['episode_reward'])
plt.legend(["DQN", "Double DQN", "Double DQN with PER"])
plt.xlabel('Steps')
plt.ylabel('Evaluation Rewards')
plt.title('Steps vs Eval Rewards for Breakout')
plt.savefig(os.path.join(plot_dir, 'eval_Breakout_q3.png'))
plt.close()

# Pong
Pong_DQN = pd.read_csv(os.path.join(plot_dir, 'eval_Pong_DQN.csv'), sep=',')
Pong_DDQN = pd.read_csv(os.path.join(plot_dir, 'eval_Pong_DDQN.csv'), sep=',')
Pong_PER = pd.read_csv(os.path.join(plot_dir, 'eval_Pong_PER.csv'), sep=',')
plt.plot(Pong_DQN['step'], Pong_DQN['episode_reward'])
plt.plot(Pong_DDQN['step'], Pong_DDQN['episode_reward'])
plt.plot(Pong_PER['step'], Pong_PER['episode_reward'])
plt.legend(["DQN", "Double DQN", "Double DQN with PER"])
plt.xlabel('Steps')
plt.ylabel('Evaluation Rewards')
plt.title('Steps vs Eval Rewards for Pong')
plt.savefig(os.path.join(plot_dir, 'eval_Pong_q3.png'))
plt.close()

# SpaceInvaders
SpaceInvaders_DQN = pd.read_csv(os.path.join(plot_dir, 'eval_SpaceInvaders_DQN.csv'), sep=',')
SpaceInvaders_DDQN = pd.read_csv(os.path.join(plot_dir, 'eval_SpaceInvaders_DDQN.csv'), sep=',')
SpaceInvaders_PER = pd.read_csv(os.path.join(plot_dir, 'eval_SpaceInvaders_PER.csv'), sep=',')
plt.plot(SpaceInvaders_DQN['step'], SpaceInvaders_DQN['episode_reward'])
plt.plot(SpaceInvaders_DDQN['step'], SpaceInvaders_DDQN['episode_reward'])
plt.plot(SpaceInvaders_PER['step'], SpaceInvaders_PER['episode_reward'])
plt.legend(["DQN", "Double DQN", "Double DQN with PER"])
plt.xlabel('Steps')
plt.ylabel('Evaluation Rewards')
plt.title('Steps vs Eval Rewards for SpaceInvaders')
plt.savefig(os.path.join(plot_dir, 'eval_SpaceInvaders_q3.png'))
plt.close()

# Question 4
# Breakout
Breakout_DQN = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_DQN.csv'), sep=',')
Breakout_DDQN = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_DDQN.csv'), sep=',')
Breakout_PER = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_PER.csv'), sep=',')
Breakout_dueling = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_dueling.csv'), sep=',')
plt.plot(Breakout_DQN['step'], Breakout_DQN['episode_reward'])
plt.plot(Breakout_DDQN['step'], Breakout_DDQN['episode_reward'])
plt.plot(Breakout_PER['step'], Breakout_PER['episode_reward'])
plt.plot(Breakout_dueling['step'], Breakout_dueling['episode_reward'])
plt.legend(["DQN", "Double DQN", "Double DQN with PER", "Dueling DQN with PER"])
plt.xlabel('Steps')
plt.ylabel('Evaluation Rewards')
plt.title('Steps vs Eval Rewards for Breakout')
plt.savefig(os.path.join(plot_dir, 'eval_Breakout_q4.png'))
plt.close()

# Pong
Pong_DQN = pd.read_csv(os.path.join(plot_dir, 'eval_Pong_DQN.csv'), sep=',')
Pong_DDQN = pd.read_csv(os.path.join(plot_dir, 'eval_Pong_DDQN.csv'), sep=',')
Pong_PER = pd.read_csv(os.path.join(plot_dir, 'eval_Pong_PER.csv'), sep=',')
Pong_dueling = pd.read_csv(os.path.join(plot_dir, 'eval_Pong_dueling.csv'), sep=',')
plt.plot(Pong_DQN['step'], Pong_DQN['episode_reward'])
plt.plot(Pong_DDQN['step'], Pong_DDQN['episode_reward'])
plt.plot(Pong_PER['step'], Pong_PER['episode_reward'])
plt.plot(Pong_dueling['step'], Pong_dueling['episode_reward'])
plt.legend(["DQN", "Double DQN", "Double DQN with PER", "Dueling DQN with PER"])
plt.xlabel('Steps')
plt.ylabel('Evaluation Rewards')
plt.title('Steps vs Eval Rewards for Pong')
plt.savefig(os.path.join(plot_dir, 'eval_Pong_q4.png'))
plt.close()

# SpaceInvaders
SpaceInvaders_DQN = pd.read_csv(os.path.join(plot_dir, 'eval_SpaceInvaders_DQN.csv'), sep=',')
SpaceInvaders_DDQN = pd.read_csv(os.path.join(plot_dir, 'eval_SpaceInvaders_DDQN.csv'), sep=',')
SpaceInvaders_PER = pd.read_csv(os.path.join(plot_dir, 'eval_SpaceInvaders_PER.csv'), sep=',')
SpaceInvaders_dueling = pd.read_csv(os.path.join(plot_dir, 'eval_SpaceInvaders_dueling.csv'), sep=',')
plt.plot(SpaceInvaders_DQN['step'], SpaceInvaders_DQN['episode_reward'])
plt.plot(SpaceInvaders_DDQN['step'], SpaceInvaders_DDQN['episode_reward'])
plt.plot(SpaceInvaders_PER['step'], SpaceInvaders_PER['episode_reward'])
plt.plot(SpaceInvaders_dueling['step'], SpaceInvaders_dueling['episode_reward'])
plt.legend(["DQN", "Double DQN", "Double DQN with PER", "Dueling DQN with PER"])
plt.xlabel('Steps')
plt.ylabel('Evaluation Rewards')
plt.title('Steps vs Eval Rewards for SpaceInvaders')
plt.savefig(os.path.join(plot_dir, 'eval_SpaceInvaders_q4.png'))
plt.close()

# Question 5 Bonus
# Breakout
Breakout_dueling = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_dueling.csv'), sep=',')
Breakout_DrQ_intensity = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_DrQ_intensity.csv'), sep=',')
Breakout_DrQ_reflect_crop = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_DrQ_reflect_crop.csv'), sep=',')
Breakout_DrQ_crop_intensity = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_DrQ_crop_intensity.csv'), sep=',')
Breakout_DrQ_zero_crop = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_DrQ_zero_crop.csv'), sep=',')
Breakout_DrQ_rotate = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_DrQ_rotate.csv'), sep=',')
Breakout_DrQ_h_flip = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_DrQ_h_flip.csv'), sep=',')
Breakout_DrQ_v_flip = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_DrQ_v_flip.csv'), sep=',')
Breakout_DrQ_all = pd.read_csv(os.path.join(plot_dir, 'eval_Breakout_DrQ_all.csv'), sep=',')
plt.plot(Breakout_dueling['step'], Breakout_dueling['episode_reward'])
plt.plot(Breakout_DrQ_intensity['step'], Breakout_DrQ_intensity['episode_reward'])
plt.plot(Breakout_DrQ_reflect_crop['step'], Breakout_DrQ_reflect_crop['episode_reward'])
plt.plot(Breakout_DrQ_crop_intensity['step'], Breakout_DrQ_crop_intensity['episode_reward'])
plt.plot(Breakout_DrQ_zero_crop['step'], Breakout_DrQ_zero_crop['episode_reward'])
plt.plot(Breakout_DrQ_rotate['step'], Breakout_DrQ_rotate['episode_reward'])
plt.plot(Breakout_DrQ_h_flip['step'], Breakout_DrQ_h_flip['episode_reward'])
plt.plot(Breakout_DrQ_v_flip['step'], Breakout_DrQ_v_flip['episode_reward'])
plt.plot(Breakout_DrQ_all['step'], Breakout_DrQ_all['episode_reward'])
plt.legend(["Baseline", "DrQ intensity", "DrQ reflect_crop", "DrQ crop_intensity", "DrQ zero_crop",  "DrQ rotate", "DrQ h_flip", "DrQ v_flip", "DrQ all"])
plt.xlabel('Steps')
plt.ylabel('Evaluation Rewards')
plt.title('Steps vs Eval Rewards for Breakout')
plt.savefig(os.path.join(plot_dir, 'eval_Breakout_DrQ.png'))
plt.close()

# Pong
Pong_dueling = pd.read_csv(os.path.join(plot_dir, 'eval_Pong_dueling.csv'), sep=',')
Pong_DrQ_crop_intensity = pd.read_csv(os.path.join(plot_dir, 'eval_Pong_DrQ_crop_intensity.csv'), sep=',')
Pong_DrQ_all = pd.read_csv(os.path.join(plot_dir, 'eval_Pong_DrQ_all.csv'), sep=',')
plt.plot(Pong_dueling['step'], Pong_dueling['episode_reward'])
plt.plot(Pong_DrQ_crop_intensity['step'], Pong_DrQ_crop_intensity['episode_reward'])
plt.plot(Pong_DrQ_all['step'], Pong_DrQ_all['episode_reward'])
plt.legend(["Baseline", "DrQ crop_intensity", "DrQ all"])
plt.xlabel('Steps')
plt.ylabel('Evaluation Rewards')
plt.title('Steps vs Eval Rewards for Pong')
plt.savefig(os.path.join(plot_dir, 'eval_Pong_DrQ.png'))
plt.close()

# SpaceInvaders
SpaceInvaders_dueling = pd.read_csv(os.path.join(plot_dir, 'eval_SpaceInvaders_dueling.csv'), sep=',')
SpaceInvaders_DrQ_crop_intensity = pd.read_csv(os.path.join(plot_dir, 'eval_SpaceInvaders_DrQ_crop_intensity.csv'), sep=',')
SpaceInvaders_DrQ_all = pd.read_csv(os.path.join(plot_dir, 'eval_SpaceInvaders_DrQ_all.csv'), sep=',')
plt.plot(SpaceInvaders_dueling['step'], SpaceInvaders_dueling['episode_reward'])
plt.plot(SpaceInvaders_DrQ_crop_intensity['step'], SpaceInvaders_DrQ_crop_intensity['episode_reward'])
plt.plot(SpaceInvaders_DrQ_all['step'], SpaceInvaders_DrQ_all['episode_reward'])
plt.legend(["Baseline", "DrQ crop_intensity", "DrQ all"])
plt.xlabel('Steps')
plt.ylabel('Evaluation Rewards')
plt.title('Steps vs Eval Rewards for SpaceInvaders')
plt.savefig(os.path.join(plot_dir, 'eval_SpaceInvaders_DrQ.png'))
plt.close()