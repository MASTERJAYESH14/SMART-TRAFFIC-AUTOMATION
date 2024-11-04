import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import pandas as pd
from traffic_env import TrafficEnv

# DQN Network
def build_model(input_dim, output_dim):
    model = models.Sequential()
    model.add(layers.Dense(64, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(output_dim, activation='linear'))
    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.001))
    return model

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        # Initialize main and target networks
        self.model = build_model(state_size, action_size)
        self.target_model = build_model(state_size, action_size)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action
        q_values = self.model.predict(state[np.newaxis])[0]
        return np.argmax(q_values)  # Best action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Prepare batches for training
        states = np.array(states)
        next_states = np.array(next_states)
        targets = self.model.predict(states)
        target_next = self.target_model.predict(next_states)

        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.amax(target_next[i])
            targets[i][actions[i]] = target

        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name + ".weights.h5")  # Load with .weights.h5

    def save(self, name):
        self.model.save_weights(name + ".weights.h5")  # Ensure the filename ends with .weights.h5


if __name__ == "__main__":
    # Load traffic data and initialize environment
    traffic_data = pd.read_csv("combined_intersections.csv", header=None)
    env = TrafficEnv(traffic_data)

    # Initialize DQN agent
    state_size = env.state.shape[0]  # Number of lanes
    action_size = state_size  # Each lane can have one action (cars to pass)
    agent = DQNAgent(state_size=state_size, action_size=action_size)

    episodes = 1000
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            actions = [0] * action_size  # Initialize all actions to 0
            actions[action] = max(1, int(state[action] * 0.75))  # Set cars to pass for the selected lane

            # Take action in environment
            next_state, reward, done, _ = env.step(actions)
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            # Train agent with experience replay
            agent.replay()

        # Update target model
        agent.update_target_model()

        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        # Save the model every 50 episodes
        if e % 50 == 0:
            agent.save(f"dqn_model_tf_{e}")
