import matplotlib.pyplot as plt
from haversine import haversine, Unit
import numpy as np

def plot_position(time, groundtruth_p, x_prop_hist):
    # Plot ground truth and predicted position for X axis
    plt.figure(figsize=(10, 5))
    plt.plot(time, groundtruth_p[:, 0], label='Ground Truth X position', color='blue')
    plt.plot(time, x_prop_hist[:, 0], linestyle='--', label='Predicted X position', color='red')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Ground Truth vs. Predicted X Position')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot ground truth and predicted position for Y axis
    plt.figure(figsize=(10, 5))
    plt.plot(time, groundtruth_p[:, 1], label='Ground Truth Y position', color='green')
    plt.plot(time, x_prop_hist[:, 1], linestyle='--', label='Predicted Y position', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Ground Truth vs. Predicted Y Position')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot ground truth and predicted position for Z axis
    plt.figure(figsize=(10, 5))
    plt.plot(time, groundtruth_p[:, 2], label='Ground Truth Z position', color='purple')
    plt.plot(time, x_prop_hist[:, 2], linestyle='--', label='Predicted Z position', color='brown')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Ground Truth vs. Predicted Z Position')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_angles(time, groundtruth_q, x_prop_hist):
    # Plot ground truth and predicted roll
    plt.figure(figsize=(10, 5))
    plt.plot(time, x_prop_hist[:, 3], linestyle='--', label='Predicted Roll', color='red')
    plt.plot(time, groundtruth_q[:, 0], label='Ground Truth Roll', color='blue')

    plt.xlabel('Time')
    plt.ylabel('Roll')
    plt.title('Ground Truth vs. Predicted Roll')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot ground truth and predicted pitch
    plt.figure(figsize=(10, 5))
    plt.plot(time, x_prop_hist[:, 4], linestyle='--', label='Predicted Pitch', color='orange')
    plt.plot(time, groundtruth_q[:, 1], label='Ground Truth Pitch', color='green')

    plt.xlabel('Time')
    plt.ylabel('Pitch')
    plt.title('Ground Truth vs. Predicted Pitch')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot ground truth and predicted yaw
    plt.figure(figsize=(10, 5))

    plt.plot(time, x_prop_hist[:, 5], linestyle='--', label='Predicted Yaw', color='brown')
    plt.plot(time, np.deg2rad(groundtruth_q[:, 2]), label='Ground Truth Yaw', color='purple')

    plt.xlabel('Time')
    plt.ylabel('Yaw')
    plt.title('Ground Truth vs. Predicted Yaw')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
# plot_position(time, groundtruth_p, x_prop_hist)
def calculate_and_plot_haversine_distances(groundtruth_p, x_prop_hist):
    """
    Calculate the Haversine distance between estimated positions and true GNSS-measured positions for each point,
    and plot these distances.

    Parameters:
    - groundtruth_p: A numpy array with true positions (latitude, longitude).
    - x_prop_hist: A numpy array with estimated positions (latitude, longitude).
    """

    # Ensure that the number of points in both datasets are the same
    if len(groundtruth_p) != len(x_prop_hist):
        raise ValueError("The number of estimated points must match the number of true points")

    distances = []  # List to store the distances

    for i in range(len(groundtruth_p)):
        true_pos = (groundtruth_p[i, 0], groundtruth_p[i, 1])  # True position (lat, lon)
        est_pos = (x_prop_hist[i, 0], x_prop_hist[i, 1])       # Estimated position (lat, lon)
        # Calculate Haversine distance in kilometers
        distance = haversine(true_pos, est_pos, unit='km')
        distances.append(distance)

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(distances, color='g')
    plt.title('Haversine Distance Between Estimated and True Positions')
    plt.xlabel('Data Point Index')
    plt.ylabel('Distance (km)')
    plt.grid(True)
    plt.show()

    return np.mean(distances)
