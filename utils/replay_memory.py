import random
from collections import deque

# Implementation of a class that represent experience replay memory
class ReplayMemory:
    def __init__(self, replay_memory_size):
        self.memory = deque()
        self.replay_memory_size = replay_memory_size

    def add(self, state, action, reward, next_state, is_terminal):

        if len(self.memory) >= self.replay_memory_size:
            self.memory.popleft()

        self.memory.append((state,action,reward,next_state, is_terminal))

    def sample(self, mini_batch_size):
        minibatch = random.sample(self.memory, mini_batch_size)

        states = [element[0] for element in minibatch]
        actions = [element[1] for element in minibatch]
        rewards = [element[2] for element in minibatch]
        next_states = [element[3] for element in minibatch]
        terminals = [element[4] for element in minibatch]

        return states, actions, rewards, next_states, terminals

    def length(self):
        return len(self.memory)