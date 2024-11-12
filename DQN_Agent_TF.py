import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import pandas as pd
from traffic_env import TrafficEnv

def build_model(input_dim, output_dim):
    # Utilize GPU if available
    with tf.device('/GPU:0'):
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(output_dim, activation='linear')  # Output size: action_size + state_size (for green times)
        ])
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.001))
    return model

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_decay=0.975, epsilon_min=0.01, batch_size=32, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        # The model now predicts both the lane to select and the green time predictions
        self.model = build_model(state_size, action_size + state_size)  # action_size for lane selection, state_size for green times
        self.target_model = build_model(state_size, action_size + state_size)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        available_lanes = [i for i in range(self.state_size) if state[i] > 0]

        if len(available_lanes) == 0:
            print("All lanes are empty. Choosing no-op action.")
            return [random.randrange(self.action_size), [0 for _ in range(self.state_size)], [0 for _ in range(self.state_size)]]

        if np.random.rand() <= self.epsilon:
            action = random.choice(available_lanes)
            cars_to_pass = [random.randint(1, state[i]) if state[i] > 0 else 0 for i in range(self.state_size)]
            predicted_green_times = np.clip([random.uniform(0, 120) for _ in range(self.state_size)], 0, 120)
            return [action, cars_to_pass, predicted_green_times]

        q_values = self.model.predict(state[np.newaxis])[0]
        available_lanes_q_values = [(q_values[i], i) for i in range(self.state_size) if state[i] > 0]
        best_action = max(available_lanes_q_values)[1]

        cars_to_pass = [min(int(q_values[i + self.state_size]), state[i]) if state[i] > 0 else 0 for i in range(self.state_size)]
        predicted_green_times = np.clip(q_values[self.action_size:], 0, 120)

        return [best_action, cars_to_pass, predicted_green_times]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.array(states)
        next_states = np.array(next_states)
        # Utilize GPU if available
        with tf.device('/GPU:0'):
            targets = self.model.predict(states)
            target_next = self.target_model.predict(next_states)

            for i in range(self.batch_size):
                target = rewards[i]
                if not dones[i]:
                    target += self.gamma * np.amax(target_next[i][:self.action_size])
                targets[i][actions[i][0]] = target  # Update lane selection Q-value

                # Ensure the targets array has the correct shape
                cars_to_pass_shape = self.action_size + self.state_size
                if targets[i].shape[0] >= cars_to_pass_shape + self.state_size:
                    # Update cars to pass predictions
                    for j in range(self.state_size):
                        targets[i][self.action_size + j] = actions[i][1][j]
                    # Update green time predictions
                    for j in range(self.state_size):
                        targets[i][self.action_size + self.state_size + j] = actions[i][2][j]

            self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name + ".weights.h5")

    def save(self, name):
        self.model.save_weights(name + ".weights.h5")

# Main training loop
if __name__ == "__main__":
    traffic_data = pd.read_csv("output_file2.csv", header=None)
    env = TrafficEnv(traffic_data)

    state_size = env.state.shape[0]
    action_size = state_size  # Number of lanes
    agent = DQNAgent(state_size=state_size, action_size=action_size)

    episodes = 1000
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, cars_to_pass, predicted_green_times = agent.act(state)  # Unpack all three values
            actions = cars_to_pass  # Use the predicted number of cars to pass as actions

            # Take action in environment
            next_state, reward, done, _ = env.step(actions, predicted_green_times)
            agent.remember(state, [action, cars_to_pass, predicted_green_times], reward, next_state, done)  # Update to match action structure

            state = next_state
            total_reward += reward

            agent.replay()

        agent.update_target_model()

        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        if e % 20 == 0:
            agent.save(f"dqn_model_tf_{e}")
