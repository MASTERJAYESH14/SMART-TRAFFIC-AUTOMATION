import numpy as np
import pandas as pd

class TrafficEnv:
    def __init__(self, traffic_data):
        self.initial_traffic_data = traffic_data[(traffic_data != 0).any(axis=1)].reset_index(drop=True)
        self.traffic_data = self.initial_traffic_data.copy()
        self.yellow_light_time = 3  # Yellow light time in seconds
        self.state_index = 0  # Start from the first state in the dataset
        self.state = self._get_next_valid_state()  # Set the state to the initial state
        self.initial_state = self.state.copy()  # Store the initial state
        self.current_lane = 0  # Start with the first lane
        self.current_time = 0  # Initialize time tracking variable
        self.training_step = 0  # Initialize the training step counter


    def reset(self):
        self.state_index = 0
        self.current_time = 0
        self.traffic_data = self.initial_traffic_data.copy()
        self.state = self.initial_state.copy()  # Reset to the stored initial state
        self.current_lane = np.random.randint(0, self.state.shape[0])
        self.training_step = 0
        return self.state

    def _get_next_valid_state(self):
        # Get the next valid state from traffic data without zeros rows
        if self.state_index < len(self.traffic_data):
            state = self.traffic_data.iloc[self.state_index].to_numpy()
            self.state_index += 1
            return state
        return np.zeros(self.traffic_data.shape[1])  # Return a dummy state if no valid states left

    def calculate_formula_green_time(self, cars_to_pass):
        return 6 + (cars_to_pass - 1) * 2

    def calculate_cycle_time(self, predicted_green_times):
        total_cycle_time = 0
        for green_time in predicted_green_times:
            total_cycle_time += green_time + self.yellow_light_time
        return total_cycle_time

    def calculate_reward(self, cars_to_pass, cars_waiting, predicted_green_time):
        reward = 0

        # Adaptive penalty based on the training phase (step count)
        penalty_factor = min(1, self.training_step / 1000)  # Gradually increase penalty with more training steps

        # Calculate the formula green time
        formula_green_time = self.calculate_formula_green_time(cars_to_pass)

        # Reward based on congestion levels and clearing lanes quickly
        if cars_waiting >= 100:
            if cars_to_pass >= int(cars_waiting * 0.6):
                reward += 500  # High reward for optimal decision
            else:
                reward += -750 * penalty_factor  # Apply adaptive penalty for wrong decision
        elif 50 <= cars_waiting < 100:
            if cars_to_pass >= int(cars_waiting * 0.7):
                reward += 500  # Medium reward for optimal decision
            else:
                reward += -700 * penalty_factor  # Apply adaptive penalty for wrong decision
        elif 30 <= cars_waiting < 50:
            if cars_to_pass >= int(cars_waiting * 0.8):
                reward += 500  # Small reward for optimal decision
            else:
                reward += -650 * penalty_factor  # Apply adaptive penalty for wrong decision
        elif cars_waiting < 30:
            if cars_to_pass == cars_waiting:
                reward += 500  # Small reward for allowing all cars to pass
            else:
                reward += -800 * penalty_factor  # Apply adaptive penalty for exceeding the number of cars

        # Reward based on the predicted green time
        if abs(predicted_green_time - formula_green_time) <= 5:
            reward += 100  # Reward for being close to the formula
        elif (predicted_green_time < formula_green_time):
            reward += -250  # Penalty for underestimating green time
        elif (predicted_green_time > formula_green_time + 5):
            reward += -450  # Penalty for overestimating green time

        # Apply additional penalty for predicting more cars than waiting
        if cars_to_pass > cars_waiting:
            reward += -300  # Penalty for overestimating the number of cars

        # Apply a time-decay factor to incentivize quicker congestion reduction
        time_decay_factor = max(0, 50 - self.current_time // 10)  # Reduce reward as time progresses
        reward += time_decay_factor

        # Print the formulated green time for reference
        print(f"Formulated Green Time: {formula_green_time} seconds")
        return reward

    def step(self, actions, predicted_green_times):
        original_state = self.state.copy()

        while self.state[self.current_lane] == 0:
            print(f"Skipping lane {self.current_lane + 1} due to 0 cars.")
            self.current_lane = (self.current_lane + 1) % len(self.state)
            
            if all(self.state[i] == 0 for i in range(len(self.state))):
                print("All lanes are empty. Moving to the next row in the dataset.")
                self.state_index += 1

                if self.state_index >= len(self.traffic_data):
                    print("End of dataset reached. Resetting to the first row.")
                    self.state_index = 0
                    self.state = self._get_next_valid_state()
                    done = False
                    return self.state, 0, done, {}

                self.state = self._get_next_valid_state()
                done = False
                return self.state, 0, done, {}

        cars_waiting = self.state[self.current_lane]
        cars_to_pass = max(1, min(actions[self.current_lane], cars_waiting))  # Ensure valid number of cars to pass
        predicted_green_time = max(1, predicted_green_times[self.current_lane])  # Ensure positive green time

        self.state[self.current_lane] -= cars_to_pass

        reward = self.calculate_reward(cars_to_pass, cars_waiting, predicted_green_time)

        self.current_time += predicted_green_time + self.yellow_light_time

        print(f"Current Lane: {self.current_lane + 1}")
        print(f"Original State: {original_state}")
        print(f"Cars Allowed to Pass: {cars_waiting}")
        print(f"Predicted Cars to Pass: {cars_to_pass}")
        print(f"Predicted Green Time: {predicted_green_time} seconds")
        print(f"New State: {self.state}")
        print(f"Reward: {reward}")

        self.current_lane = (self.current_lane + 1) % len(self.state)
        self.training_step += 1

        done = self.state_index >= len(self.traffic_data) - 1

        if not done:
            self.state_index += 1
            self.state = self._get_next_valid_state()

        if done:
            self.state_index = 0
            self.state = self.initial_state.copy()

        return self.state, reward, done, {}

# Example usage
traffic_data = pd.read_csv("output_file2.csv", header=None)
env = TrafficEnv(traffic_data)

# Reset environment
state = env.reset()

# Simulate steps
for _ in range(len(traffic_data)):
    actions = [state[i] for i in range(len(state))]  # Example actions
    predicted_green_times = [env.calculate_formula_green_time(state[i]) + np.random.randint(-2, 3) for i in range(len(state))]  # Predicted green time with some random noise
    state, reward, done, _ = env.step(actions, predicted_green_times)
    if done:
        print("Episode finished./-/-/-/-/-/-/-/-/--/-/-/-/--/-/-/-/-/-/--/-/-/-/-/-/--/-/-/-/-/-/--/-/-/-/-/--/-/-")
        break
