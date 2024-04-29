import numpy as np
from scipy.spatial.transform import Rotation
from earth import *

def load_data(filepath):
  data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
  return data

def skew(x):
  x = x.reshape((3,1))

  return np.array([[0, -x[2,0], x[1,0]],
                    [x[2,0], 0, -x[0,0]],
                    [-x[1,0], x[0,0], 0]])

def propogation_model(x_prev, u , dt , model):

  # Earth's rate of rotation
  omega_E = 7.2921157e-5
  bg , ba = x_prev[9:12] , x_prev[12:15]

  if model == "ff" :
    omega_meas = u[:3]
    f_b = u[3:]
  else:
    omega_meas = u[:3] - bg
    f_b = u[3:] - ba
    
  # Calculate rotation matrix from previous state
  R_prev = Rotation.from_euler('xyz', x_prev[3:6]).as_matrix()

  # Calculate rotation matrix for Earth rate correction
  omega_e_i = np.array([[0, -omega_E, 0],
                              [omega_E, 0, 0],
                              [0, 0, 0]])

  omega_b_i = np.array([[0, -omega_meas[2], omega_meas[1]],
                                [omega_meas[2], 0, -omega_meas[0]],
                                [-omega_meas[1], omega_meas[0], 0]])

  # Extracting data from previsous state
  lat, lon, alt, vn, ve, vd = x_prev[0], x_prev[1], x_prev[2], x_prev[6], x_prev[7], x_prev[8]

  # Calculating the denominators for the wne calculation using the function provided in earth.py
  RN , RE, RE_cos = principal_radii(lat, alt)

  # Formualting wne
  w_n_e = np.array([ve/RE, -1*vn/RN , (-1*ve*np.tan(np.deg2rad(lat)))/RE])

  omega_n_e = np.array([[0, -w_n_e[2], w_n_e[1]],
                              [w_n_e[2], 0, -w_n_e[0]],
                              [-w_n_e[1], w_n_e[0], 0]])


  # Update the rotation matrix using the first-order approximation
  R_next = np.dot((np.eye(3) + dt * omega_b_i), R_prev) - (omega_e_i + omega_n_e) @ (R_prev *dt)

  # Velocity update
  v_n_t_prev = np.array([vn , ve , vd] )

  f_n_t = ( (R_prev + R_next) @ f_b ) / 2

  omega_e_i = skew(rate_n(lat))

  v_n_t = v_n_t_prev + dt * (f_n_t + gravity_n(lat, alt) - (omega_n_e + 2 * omega_e_i ) @ v_n_t_prev)


  # Position update
  alt_next = alt - dt*( vd + v_n_t[2] ) * 0.5

  RN_next , _ , _ = principal_radii(lat, alt_next)

  lat_next = np.deg2rad(lat) + dt * 0.5 * ((vn/RN) + (v_n_t[0] / RN_next) )

  _ , _ , RE_cos_next = principal_radii(np.rad2deg(lat_next), alt_next)

  lon_next = np.deg2rad(lon) + dt * 0.5 * ((ve/ RE_cos ) + (v_n_t[1] / RE_cos_next) )

  # Convert the updated rotation matrix to Euler angles
  euler_next = Rotation.from_matrix(R_next).as_euler('xyz')

  p_next = np.array([np.rad2deg(lat_next), np.rad2deg(lon_next), alt_next])

  x_next = np.concatenate((p_next , euler_next, v_n_t, x_prev[9:]))

  return x_next