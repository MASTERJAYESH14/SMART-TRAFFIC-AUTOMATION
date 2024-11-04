import numpy as np
import pandas as pd

class TrafficEnv:
    def __init__(self, traffic_data):
        self.traffic_data = traffic_data
        self.yellow_light_time = 3  # Yellow light time in seconds
        self.state_index = 0  # Start from the first state in the dataset
        self.state = self.traffic_data.iloc[self.state_index].to_numpy()  # Initial state from data
        self.max_proportion_to_pass = 0.75  # Max proportion of cars that can pass
        self.min_cars_to_pass = 1  # Minimum cars threshold for full passing
        self.current_lane = 0  # Start with the first lane

    def reset(self):
        self.state_index = 0
        self.state = self.traffic_data.iloc[self.state_index].to_numpy()
        self.current_lane = 0  # Reset the current lane to the first lane
        return self.state

    def calculate_reward(self, lane_actions, cars_passed, original_state):
        cars_waiting_penalty = -sum(self.state) * 0.1  # Reduced penalty weight
        
        # Fairness penalty if a lane has over 50% of cars remaining after green light
        fairness_penalty = 0
        for i, cars in enumerate(self.state):
            if cars > original_state[i] * 0.5:
                fairness_penalty -= 100  # Keep this as is for strong disincentive

        # Base reward for passing cars and total cars passed
        reward = (cars_passed * 1.5) + cars_waiting_penalty + fairness_penalty
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

        # Calculate total cars across all lanes
        total_cars = sum(self.state)
        cars_waiting = self.state[self.current_lane]

        # Determine cars to pass based on the waiting threshold
        if total_cars <= 120:  # Allow all cars to pass if total time is within the threshold
            cars_to_pass = cars_waiting
        else:
            # Apply threshold logic for passing cars based on the waiting cars
            if cars_waiting >= 100:
                cars_to_pass = max(1, int(cars_waiting * 0.5))
            elif cars_waiting >= 50:
                cars_to_pass = max(1, int(cars_waiting * 0.7))
            elif cars_waiting >= 30:
                cars_to_pass = max(1, int(cars_waiting * 0.2))
            else:
                cars_to_pass = max(1, cars_waiting)  # Allow at least 1 car to pass

        # Ensure we don't exceed the available cars in the lane
        if self.state[self.current_lane] >= cars_to_pass:
            self.state[self.current_lane] -= cars_to_pass
        else:
            cars_to_pass = self.state[self.current_lane]
            self.state[self.current_lane] = 0

        # Calculate green time based on cars allowed to pass
        green_time = 6 + (cars_to_pass - 1) * 2

        # Calculate reward based on new and original state
        reward = self.calculate_reward(actions, cars_to_pass, original_state)

        # Print details of the current state, original state, lane, green time, and reward
        print(f"Current Lane: {self.current_lane + 1}")
        print(f"Original State: {original_state}")
        print(f"Cars Allowed to Pass: {cars_to_pass}")
        print(f"Green Time Duration: {green_time} seconds")
        print(f"New State: {self.state}")
        print(f"Reward: {reward}")

        # Move to the next lane in round-robin fashion
        self.current_lane = (self.current_lane + 1) % len(self.state)

        # Check if we've reached the last row in the traffic data
        done = self.state_index >= len(self.traffic_data) - 1
        if not done:
            self.state_index += 1
            self.state = self.traffic_data.iloc[self.state_index].to_numpy()
        return self.state, reward, done, {}

# Load the CSV file and create a traffic environment
traffic_data = pd.read_csv("combined_intersections.csv", header=None)
env = TrafficEnv(traffic_data)

# Reset environment
state = env.reset()

# Simulate steps
for _ in range(len(traffic_data)):
    # Calculate actions based on half the lane size as an example
    actions = [max(1, int(state[i] * 0.5)) for i in range(len(state))]
    state, reward, done, _ = env.step(actions)
    if done:
        break
