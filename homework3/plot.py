"""
    This file will run draw plot for question 1 and question 2
"""

import os
import matplotlib.pyplot as plt
import numpy as np

READ_DIR = './plot'

for file in os.listdir(READ_DIR):
    if(file.split('.')[-1] != 'csv'):
        continue
    file_name = file.split('.')[0]
    print(file_name)
    model_type = file_name.split('_')[0]
    seed = file_name.split('_')[1]
    environment = file_name.split('_')[2]

    if environment == 'Pendulum-v0':
        threshold = -400
    elif environment == 'BipedalWalker-v3':
        threshold = 125
    elif environment == 'LunarLanderContinuous-v2':
        threshold = 100
    else:
        raise("[Error] Wrong environment")

    table = np.genfromtxt(os.path.join(READ_DIR, file), delimiter=',')
    length = table.shape[0]
    threshold_draw = [threshold for i in range(length)]
    threshold_draw = np.array(threshold_draw)
    steps = table[...,0]
    returns = table[...,1]
    losses = table[...,2]

    plt.plot(steps, returns)
    plt.plot(steps, threshold_draw)
    plt.xlabel('Steps')
    plt.ylabel('Return')
    plt.title('Steps vs Return')
    plt.savefig(os.path.join(READ_DIR, file.split('.')[0] + '.png'))
    plt.close()
