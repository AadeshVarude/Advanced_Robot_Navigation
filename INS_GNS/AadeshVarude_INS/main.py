from propogation_model import * 
from visualize import  *
from ukf import * 


filename = 'C:/Users/dell/Desktop/Advanced_robot_navigation/INS/trajectory_data.csv'
data = load_data(filename)
# Extracting time and groundtruth data
time = data[:,0]
groundtruth_p = data[:,1:4]
groundtruth_q = data[:,4:7]

# # TASK 2 To run feedforward closed loop 
ukf =UKF("ff")
x_prop_hist=ukf.run()

# Plotting the lat lon and altitude and the angles
plot_position(time, groundtruth_p, x_prop_hist)

#Plotting Haversine distance 
avg_harversine_distance = calculate_and_plot_haversine_distances(groundtruth_p,x_prop_hist)

print()
print("Average Haversine Distance", avg_harversine_distance )
# TASK 3 To run feedback closed loop 
ukf =UKF("fb")
x_prop_hist=ukf.run()

# Plotting the lat lon and altitude and the angles
plot_position(time, groundtruth_p, x_prop_hist)

#Plotting Haversine distance 
avg_harversine_distance = calculate_and_plot_haversine_distances(groundtruth_p,x_prop_hist)
print()
print("Average Haversine Distance", avg_harversine_distance )
