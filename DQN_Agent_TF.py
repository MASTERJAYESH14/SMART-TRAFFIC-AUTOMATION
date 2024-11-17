import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import LearningRateScheduler
import pandas as pd
from traffic_env import TrafficEnv

def build_model(input_dim, output_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_dim, activation='linear')  # Output size: action_size + state_size (for green times)
    ])
    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.001))
    return model

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def predict_cars_and_green_time(cars_to_pass, formulated_green_time):
    predicted_cars = 0

    # Define ranges and corresponding logic
    if 0 < cars_to_pass <= 30:
        predicted_cars = int(0.9 * cars_to_pass) + random.randint(-2, 2)
    elif 31 <= cars_to_pass <= 50:
        predicted_cars = int(0.85 * cars_to_pass) + random.randint(-5, 5)
    elif 51 <= cars_to_pass <= 100:
        predicted_cars = int(0.75 * cars_to_pass) + random.randint(-5, 5)
    elif 101 <= cars_to_pass < 120:
        predicted_cars = int(0.65 * cars_to_pass) + random.randint(-5, 5)
    elif cars_to_pass >= 120:
        predicted_cars = 60
    else:
        print("Invalid number of cars to pass.")
        return

    # Ensure predicted cars are non-negative
    predicted_cars = max(1, predicted_cars)
    predicted_green_time = formulated_green_time + random.choice([-1,-2, 0, 0, 0, 0, -5,2, 5,3,1,6, 0, 0, 0])
    
    return [predicted_cars, predicted_green_time]

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.25, batch_size=2000, memory_size=35000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

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
            return [random.randrange(self.action_size), [0 for _ in range(self.state_size)], [5 for _ in range(self.state_size)]]

        if np.random.rand() <= self.epsilon:
            # Exploration: random action
            action = random.choice(available_lanes)
            cars_to_pass = [random.randint(1, max(state[i], 1)) for i in range(self.state_size)]
            predicted_green_times = np.clip([random.randint(5, 120) for _ in range(self.state_size)], 5, 120)
            return [action, cars_to_pass, predicted_green_times]

        # Exploitation: action based on model prediction
        q_values = self.model.predict(state[np.newaxis])[0]
        available_lanes_q_values = [(q_values[i], i) for i in range(self.state_size) if state[i] > 0]
        best_action = max(available_lanes_q_values)[1]

        cars_to_pass = [min(int(q_values[i + self.state_size]), state[i]) if state[i] > 0 else 0 for i in range(self.state_size)]
        predicted_green_times = np.clip(np.round(q_values[self.action_size:]), 5, 120)

        # Add random noise to encourage exploration
        cars_to_pass = [int(max(1, c + np.random.normal(0, 10))) for c in cars_to_pass]
        cars_to_pass = [min(c, state[i]) for i, c in enumerate(cars_to_pass)]
        predicted_green_times += np.random.normal(0, 10, self.state_size)
        predicted_green_times = np.clip(predicted_green_times, 5, 120)

        return [best_action, cars_to_pass, predicted_green_times]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.array(states)
        next_states = np.array(next_states)
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

        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size, callbacks=[LearningRateScheduler(scheduler)])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name + ".weights.h5")

    def save(self, name):
        self.model.save_weights(name + ".weights.h5")

total_episodes = 1000
# Main training loop
if __name__ == "__main__":
    traffic_data = pd.read_csv("E:\\Codes\\SMART-TRAFFIC-AUTOMATION\\output_file2.csv", header=None)
    env = TrafficEnv(traffic_data)

    state_size = env.state.shape[0]
    action_size = state_size  # Number of lanes
    agent = DQNAgent(state_size=state_size, action_size=action_size, epsilon_min=0.1)

    # Resume training loop
    episode_to_resume = 716

    for e in range(episode_to_resume, total_episodes):
        # Load the saved model at the beginning of each episode
        if e % 4 == 0 and e > 0:
            agent.load(f"dqn_model_tf_{e-4}")

        # Reset epsilon to boost exploration every 100 episodes
        if (e % 100 == 0 and e<=1000) or e%716 ==0:
            agent.epsilon = 0.53  # Increase epsilon for more exploration
            agent.epsilon_decay = 0.997  # Adjust decay rate to slow down exploration reduction

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

        print(f"Episode {e+1}/{total_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        if e % 4 == 0:
            agent.save(f"dqn_model_tf_{e}")
