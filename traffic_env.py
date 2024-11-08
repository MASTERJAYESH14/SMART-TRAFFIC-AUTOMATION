import numpy as np
import pandas as pd

class TrafficEnv:
    def __init__(self, traffic_data):
        self.traffic_data = traffic_data
        self.yellow_light_time = 3  # Yellow light time in seconds
        self.state_index = 0  # Start from the first state in the dataset
        self.state = self.traffic_data.iloc[self.state_index].to_numpy()  # Initial state from data
        self.current_lane = 0  # Start with the first lane

    def reset(self):
        self.state_index = 0
        self.state = self.traffic_data.iloc[self.state_index].to_numpy()
        self.current_lane = 0  # Reset the current lane to the first lane
        return self.state

    def calculate_cycle_time(self, actions):
        total_cycle_time = 0
        for cars_to_pass in actions:
            green_time = 6 + (cars_to_pass - 1) * 2
            total_cycle_time += green_time + self.yellow_light_time
        return total_cycle_time

    def calculate_reward(self, cars_to_pass, actions):
        cars_waiting = self.state[self.current_lane]
        reward = 0

        # Calculate total cycle time based on actions
        total_cycle_time = self.calculate_cycle_time(actions)

        if total_cycle_time <= 120:
            reward += cars_to_pass  # Reward proportional to the number of cars passed
        else:
            if cars_waiting >= 100:
                if cars_to_pass <= int(cars_waiting * 0.5):
                    reward = 50  # High reward for optimal decision
                else:
                    reward = -100  # Heavy penalty for wrong decision
            elif 50 <= cars_waiting < 100:
                if cars_to_pass <= int(cars_waiting * 0.7):
                    reward = 40  # Medium reward for optimal decision
                else:
                    reward = -120  # Heavy penalty for wrong decision
            elif 30 <= cars_waiting < 50:
                if cars_to_pass <= int(cars_waiting * 0.8):
                    reward = 30  # Small reward for optimal decision
                else:
                    reward = -150  # Heavy penalty for wrong decision
            elif cars_waiting < 30:
                if cars_to_pass == cars_waiting:
                    reward = 20  # Small reward for allowing all cars to pass
                else:
                    reward = -200  # Penalty for exceeding the number of cars

        print(f"Calculating Reward: Cars Waiting: {cars_waiting}, Cars Allowed to Pass: {cars_to_pass}, Reward: {reward}")
        return reward

    def step(self, actions):
        original_state = self.state.copy()

        # Skip lanes with 0 cars and apply penalty if opened
        while self.state[self.current_lane] == 0:
            print(f"Skipping lane {self.current_lane + 1} due to 0 cars.")
            self.current_lane = (self.current_lane + 1) % len(self.state)
            # Check if we've cycled through all lanes
            if all(self.state[i] == 0 for i in range(len(self.state))):
                print("All lanes are empty. Ending simulation.")
                return self.state, -10, True, {}  # End the simulation with a penalty

        # Calculate cars waiting in the current lane
        cars_waiting = self.state[self.current_lane]

        # Determine cars to pass based on the action provided
        cars_to_pass = actions[self.current_lane]

        # Ensure we don't exceed the available cars in the lane
        if self.state[self.current_lane] >= cars_to_pass:
            self.state[self.current_lane] -= cars_to_pass
        else:
            cars_to_pass = self.state[self.current_lane]
            self.state[self.current_lane] = 0

        # Calculate reward based on the number of cars passed and cycle time
        reward = self.calculate_reward(cars_to_pass, actions)

        # Print details of the current state, original state, lane, green time, and reward
        print(f"Current Lane: {self.current_lane + 1}")
        print(f"Original State: {original_state}")
        print(f"Cars Allowed to Pass: {cars_to_pass}")
        print(f"New State: {self.state}")
        print(f"Reward: {reward}")

        self.current_lane = (self.current_lane + 1) % len(self.state)

        # Check if we've reached the last row in the traffic data
        done = self.state_index >= len(self.traffic_data) - 1
        if not done:
            self.state_index += 1
            self.state = self.traffic_data.iloc[self.state_index].to_numpy()
        return self.state, reward, done, {}

# Load the CSV file and create a traffic environment
traffic_data = pd.read_csv("output_file2.csv", header=None)
env = TrafficEnv(traffic_data)

# Reset environment
state = env.reset()

# Simulate steps
for _ in range(len(traffic_data)):
    # Allow all cars to pass based on the lane's total count
    actions = [state[i] for i in range(len(state))]
    state, reward, done, _ = env.step(actions)
    if done:
        break
