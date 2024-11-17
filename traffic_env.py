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

    def calculate_formula_green_time(self, predicted_cars_to_pass):
        return 6 + (predicted_cars_to_pass - 1) * 2

    def calculate_cycle_time(self, predicted_green_times):
        total_cycle_time = 0
        for green_time in predicted_green_times:
            total_cycle_time += green_time + self.yellow_light_time
        return total_cycle_time

    def calculate_reward(self, predicted_cars_to_pass, cars_allowed_to_pass, predicted_green_time):
        reward = 0

        # Calculate the formula green time
        formula_green_time = self.calculate_formula_green_time(predicted_cars_to_pass)

        # Calculate efficiency
        efficiency = predicted_cars_to_pass / predicted_green_time if predicted_green_time > 0 else 0

        # Normalize green time
        green_time_factor = predicted_green_time / formula_green_time if formula_green_time > 0 else 1

        # Reward based on congestion levels
        congestion_ratio = predicted_cars_to_pass / cars_allowed_to_pass if cars_allowed_to_pass > 0 else 0

        if cars_allowed_to_pass >= 100:
            if 0.6 <= congestion_ratio < 0.75:
                reward += 2000 * efficiency
            elif congestion_ratio >= 0.8:
                reward += 750
            elif (predicted_cars_to_pass == 1 or predicted_green_time<80):
                reward += -2000
            else:
                reward -= 1000
        elif 50 <= cars_allowed_to_pass < 100:
            if 0.7 <= congestion_ratio <= 0.87:
                reward += 2000 * efficiency
            elif congestion_ratio >= 0.9:
                reward += 300
            elif congestion_ratio <= 0.7 or predicted_green_time<45:
                reward -= 1500
            else:
                reward -= 1000
        elif 30 <= cars_allowed_to_pass < 50:
            if congestion_ratio >= 0.8:
                reward += 2000 * efficiency
            elif congestion_ratio < 0.8 or predicted_green_time<70:
                reward -= 1500
            else:
                reward -= 800
        elif cars_allowed_to_pass < 30:
            if predicted_cars_to_pass == cars_allowed_to_pass:
                reward += 2000 * efficiency
            elif abs(predicted_cars_to_pass-cars_allowed_to_pass)>3:
                reward+= -1800
            elif predicted_cars_to_pass < cars_allowed_to_pass:
                reward -= 1200
            elif predicted_green_time> 70:
                reward += -2000
            else:
                reward -= 1500
        
        if(formula_green_time>120):
            if(predicted_green_time==120):
                reward+= 950 *green_time_factor
            else:
                reward += -500
        else:

            if abs(predicted_green_time - formula_green_time) < 3:
                reward += 850 * green_time_factor
            elif predicted_green_time < formula_green_time:
                if abs(predicted_green_time - formula_green_time) > 5:
                    reward -= 450
                elif predicted_green_time * 3 < formula_green_time:
                    reward -= 600
            elif predicted_green_time > formula_green_time + 5:
                if predicted_green_time - formula_green_time > 5:
                    reward -= 450
                elif predicted_green_time > formula_green_time * 2:
                    reward -= 600

        # Penalties for extreme green times
        if predicted_green_time < 6:
            reward -= 2000

        # Penalty for overestimating cars
        if predicted_cars_to_pass > cars_allowed_to_pass:
            reward -= 2000

        # Time-decay factor
        time_decay_factor = max(0, 50 - self.current_time // 10)
        reward += time_decay_factor

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

        cars_allowed_to_pass = self.state[self.current_lane]  # Number of cars allowed to pass
        predicted_cars_to_pass = max(1, min(actions[self.current_lane], cars_allowed_to_pass))  # Ensure valid number of cars to pass
        predicted_green_time = max(1, predicted_green_times[self.current_lane])  # Ensure positive green time

        self.state[self.current_lane] -= predicted_cars_to_pass

        reward = self.calculate_reward(predicted_cars_to_pass, cars_allowed_to_pass, predicted_green_time)

        self.current_time += predicted_green_time + self.yellow_light_time

        print(f"Current Lane: {self.current_lane + 1}")
        print(f"Original State: {original_state}")
        print(f"Cars Allowed to Pass: {cars_allowed_to_pass}")
        print(f"Predicted Cars to Pass: {predicted_cars_to_pass}")
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