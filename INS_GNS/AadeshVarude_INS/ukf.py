import numpy as np
import scipy.io
import sympy as sp

from earth import *
from propogation_model import * 

import numpy as np
import scipy.io
import sympy as sp

class UKF:
  def __init__(self, model):
    self.model = model
    # Initialize the UKF class
    if self.model == "fb":
      self.n = 15  # Dimensionality of the state space
    else:
      self.n = 12  # Dimensionality of the state space

    self.alpha = 0.001  # Alpha parameter for tuning UKF weights
    self.k = 2  # K parameter for tuning UKF weights
    self.beta = 2  # Beta parameter for tuning UKF weights
    # Calculate lambda parameter for tuning UKF weights
    self.lamb = (self.alpha ** 2 ) * (self.n + self.k) - self.n
    # Calculate UKF weights
    self.wc , self.wm = self.get_weights()

  def get_weights(self):
    # Calculate the UKF weights
    wc = np.zeros(2*self.n + 1)  # Array to store covariance weights
    wm = np.zeros(2*self.n + 1)  # Array to store mean weights

    # Calculate covariance and mean weights for the first sigma point
    wc[0] = self.lamb / (self.n + self.lamb ) + (1 - self.alpha**2 + self.beta)
    wm[0] = self.lamb / (self.n + self.lamb)

    # Calculate covariance and mean weights for the remaining sigma points
    for i in range(1 , self.n+1):
      wm[i] = wm[ self.n + i] = wc[i] = wc[self.n + i] = 0.5 / (self.n + self.lamb)

    return wc, wm

  def get_sigma_points(self , x, n, P):
    # Calculate sigma points for the Unscented Kalman Filter (UKF)
    sigma_points = np.zeros((n, 2 * n + 1))  # Array to store sigma points
    residue =  np.linalg.cholesky((n + self.lamb) * P)  # Cholesky decomposition of the covariance matrix
    sigma_points[:,0 ] = x  # Set the mean as the first sigma point

    # Calculate remaining sigma points
    for i in range(1 , self.n + 1):
      sigma_points[:,i] = x + residue[:,i-1]
      sigma_points[:,n+i] = x - residue[:,i-1]

    return sigma_points

  def measurement_model_fb(self, x):
    # Measurement model for feedback (fb) mode
    C = np.zeros((6,x.shape[0]))
    C[0:3,0:3] = np.eye(3)
    C[3:,6:9] = np.eye(3)
    return C @ x

  def measurement_model_ff(self, x , z):
    # Measurement model for feedforward (ff) mode
    C = np.zeros((3,x.shape[0]))
    C[0:3,0:3] = np.eye(3)
    error_update = C @ x - z[:3]
    
    return error_update

  def run(self):
    # Run the Unscented Kalman Filter (UKF)
    data = load_data('C:/Users/dell/Desktop/Advanced_robot_navigation/INS/trajectory_data.csv')  # Load data from CSV file

    time = data[:,0]  # Extract time information from data

    groundtruth_p = data[:,1:4]  # Extract ground truth position data
    groundtruth_q = data[:,4:7]  # Extract ground truth orientation data
    u = data[:,7:13]  # Extract control inputs (gyroscope and accelerometer data)
    z = data[:,13:]  # Extract sensor measurements (position and velocity)

    if self.model =="fb":
      P = np.eye(15) * 0.1  # Initialize covariance matrix P
      Q = np.eye(15) * 1e2  # Process noise covariance matrix Q
      R_val = [2e-3, 2e-3, 2e-3, 2e-3, 2e-3, 2e-3]
      R = np.diag(R_val) # Measurement noise covariance matrix R
      x = np.zeros(15)  # Initialize previous state vector
    else:
      P = np.eye(12) * 0.1  # Initialize covariance matrix P
      Q = np.eye(12) * 1e2  # Process noise covariance matrix Q
      R_val = [5e-3, 5e-3, 5e-3]
      R = np.diag(R_val) # Measurement noise covariance matrix R
      x = np.zeros(12)  # Initialize previous state vector

    x_prop_hist = np.zeros((len(data), self.n))  # Array to store predicted states
    dt = 0  # Initialize time step

    
    x[:3] = groundtruth_p[0]  # Initialize position components of the previous state
    x[3:6] = groundtruth_q[0]  # Initialize orientation components of the previous state
    x[5] = np.deg2rad(195)  # Initialize yaw angle component of the previous state

    print("total length",len(time))

    for i in range(len(time)) :
      print("Doing :", i)

      # Prediction step for UKF
      sigma_points = self.get_sigma_points(x, self.n, P)  # Calculate sigma points
      X1 = np.zeros_like(sigma_points)  # Initialize array to store predicted sigma points

      for j in range(2 * self.n + 1):
        X = sigma_points[:,j]  # Extract sigma point
        X1[:,j] = propogation_model(X.reshape(self.n), u[i] , dt, model=self.model)  # Predict next state using propagation model

      x_mean = np.sum(X1 * self.wm, axis=1)  # Calculate mean predicted state
      d = X1 - x_mean.reshape(-1,1)  # Calculate residual

      P = d @ np.diag(self.wc) @ d.T + Q  # Update covariance matrix P

      # Update step

      sigma_points = X1
      if self.model =="fb":
        Z = np.zeros((6,self.n*2+1))

        for j in range(2 * self.n + 1):
          X = sigma_points[:,j]  # Extract sigma point
          Z[:,j] = self.measurement_model_fb(X.reshape(self.n))

        z_mean = np.sum(Z * self.wm, axis=1)  # Calculate mean predicted state
        d1 = Z - z_mean.reshape(-1,1)  # Calculate residual
        S = d1 @ np.diag(self.wc) @ d1.T + R  # Update covariance matrix S

        cross_cov = d @ np.diag(self.wc) @ d1.T

        kalman_gain = cross_cov @ np.linalg.inv(S)

        x = x_mean + kalman_gain @ (z[i]-z_mean)

        P = P - kalman_gain @ S @ kalman_gain.T

        x_prop_hist[i,:] = x
      else :

        Z_error = np.zeros((3,self.n*2+1))
        for j in range(2 * self.n + 1):
          X = sigma_points[:,j]  # Extract sigma point
          Z_error[:,j] = self.measurement_model_ff(X.reshape(self.n),z[i])

        error_mean = np.sum(Z_error * self.wm, axis=1)  # Calculate mean predicted state
        d1 = Z_error - error_mean.reshape(-1,1)  # Calculate residual
        S = d1 @ np.diag(self.wc) @ d1.T + R  # Update covariance matrix S

        cross_cov = d @ np.diag(self.wc) @ d1.T

        kalman_gain = cross_cov @ np.linalg.inv(S)

        P = P - kalman_gain @ S @ kalman_gain.T
        
        x_stored = x_mean.copy()
        x_stored[:3] = x_stored[:3] - error_mean
        
        x_prop_hist[i,:] = x_stored
        x_mean[9:] = error_mean
        x = x_mean

      dt=1

    return x_prop_hist
