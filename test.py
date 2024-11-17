import numpy as np
from DQN_Agent_TF import predict_cars_and_green_time

def calc_green_time(cars_at_intersection):
    return 6 + (cars_at_intersection - 1) * 2

def run_traffic_simulation(num_iterations=10):
    traffic_data = np.array([20, 15, 18,19])  
    lane = 0  

    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")
        
        print(f"Original Cars at Intersections: {traffic_data}")
        cars = traffic_data[lane]   
        green_time = calc_green_time(cars)
        ans = predict_cars_and_green_time(cars, green_time)
        print(f"Current lane : {lane + 1}")
        print(f"Actual cars to pass : {cars}")
        print(f"Predicted cars to pass : {ans[0]}")
        print(f"Predicted Green time : {ans[1]}")
        print(f"Actual green time: {green_time}")

        # Update traffic for the current lane
        traffic_data[lane] = traffic_data[lane] - ans[0]
        if traffic_data[lane] < 0: 
            traffic_data[lane] = 0 

        print("Traffic Now: ", traffic_data)

        # Add random cars to each lane
        random_addition = np.array([
            np.random.randint(25, 40),  
            np.random.randint(15, 30),  
            np.random.randint(30, 50),  
            np.random.randint(20, 25)   
        ])
                    
        traffic_data += random_addition
        print(f"Cars at Intersections after addition: {traffic_data}")

        lane = (lane + 1) % 4  

        print(f"Iteration {iteration + 1} completed.\n")

    print("Simulation complete!")

# Example usage:
run_traffic_simulation(num_iterations=10)
