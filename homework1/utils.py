import numpy as np
import torch.nn as nn

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

# def weight_init(m):
#     if isinstance(m, nn.Linear):
#         nn.init.orthogonal_(m.weight.data)
#         if hasattr(m.bias, 'data'):
#             m.bias.data.fill_(0.0)
#     elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#         gain = nn.init.calculate_gain('relu')
#         nn.init.orthogonal_(m.weight.data, gain)
#         if hasattr(m.bias, 'data'):
#             m.bias.data.fill_(0.0)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d):
        weights_init_kaiming(m)
    elif isinstance(m, nn.BatchNorm2d):
        weights_init_kaiming(m)
        

class ExpertBuffer:
    """
    An expert buffer class. This class stores the demonstration (obs, action) set from the expert.
    While this class is completely created for you, you should still pay attention to how it's coded.
    Inspired by https://github.com/denisyarats/pytorch_sac
    """
    def __init__(self, max_length, img_shape, action_shape):
        # Creates a buffer to hold all the expert demonstrations
        self._obs_buffer = np.empty(shape=(max_length, *img_shape), dtype=np.float64)
        self._expert_action_buffer = np.empty(shape=(max_length, *action_shape), dtype=np.float64)
        self._current_index = 0
        self._is_full = False
        self.capacity = max_length

    def insert(self, obs, action):
        # Insert an image observation along with the expert action in the buffer.
        insert_idx = self._current_index
        np.copyto(self._obs_buffer[insert_idx], obs)
        np.copyto(self._expert_action_buffer[insert_idx], action)
        self._current_index = (self._current_index + 1) % self.capacity
        if self._current_index == 0:
            self._is_full = True

    def __len__(self):
        return self.capacity if self._is_full else self._current_index

    def sample(self, batch_size=256):
        # Sample a batch of size batch_size of observations and expert actions.
        current_length = self.__len__()
        batch_indices = np.random.randint(low=0, high=current_length, size=batch_size)

        batch_obs = self._obs_buffer[batch_indices]
        batch_action = self._expert_action_buffer[batch_indices]
        return batch_obs, batch_action


class SafeBuffer:
    """
    Safe Buffer for SafeDAgger
    """
    def __init__(self, max_length, input_shape):
        # Creates a buffer to hold all the expert demonstrations
        self._input_buffer = np.empty(shape=(max_length, *input_shape), dtype=np.float64)
        self._safe_buffer = np.empty(shape=(max_length, 1), dtype=np.float64)
        self._current_index = 0
        self._is_full = False
        self.capacity = max_length

    def insert(self, input, target):
        # Insert an image observation along with the expert action in the buffer.
        insert_idx = self._current_index
        np.copyto(self._input_buffer[insert_idx], input)
        np.copyto(self._safe_buffer[insert_idx], target)
        self._current_index = (self._current_index + 1) % self.capacity
        if self._current_index == 0:
            self._is_full = True

    def __len__(self):
        return self.capacity if self._is_full else self._current_index

    def sample(self, batch_size=256):
        # Sample a batch of size batch_size of observations and expert actions.
        current_length = self.__len__()
        batch_indices = np.random.randint(low=0, high=current_length, size=batch_size)

        batch_x = self._input_buffer[batch_indices]
        batch_y = self._safe_buffer[batch_indices]
        return batch_x, batch_y

