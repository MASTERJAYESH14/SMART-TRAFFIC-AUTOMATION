Smart Traffic Simulation:
Welcome to the Smart Traffic Simulation repository! This project leverages Deep Q-Learning to manage traffic lights efficiently, ensuring optimal traffic flow and reducing congestion.

Project Overview:
In this project, we implement a DQN (Deep Q-Network) agent to control traffic lights at an intersection. The agent's goal is to minimize the average waiting time of cars by dynamically adjusting the green time based on the traffic conditions.

Features:
Reinforcement Learning with DQN: Implemented using TensorFlow/Keras.

Traffic Environment: Simulates traffic at an intersection using realistic traffic data.

Dynamic Traffic Management: Adjusts green light timings based on current traffic conditions.

Reward System: Encourages the agent to optimize traffic flow while penalizing inefficiencies.

Folder Structure:
traffic_env.py: Contains the Traffic Environment class, simulating the intersection and calculating rewards.

DQN_Agent_TF.py: Implements the DQN agent, including training and evaluation.

output_file2.csv: Sample traffic data used for training the agent.
