#!/usr/bin/python
#
# Copyright 2023 MangDang- Adapted in 2025 by Reid Graves
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Description:
# The simplified single leg of mini pupper can be regarded as an RR robot arm on the same plane, 
# (The RR manipulator is commonly used as shorthand for a two revolute joint configuration in a single plane.)
# visualize the robot leg, and calculate the position of the end point from a given angle

import matplotlib.pyplot as plt 
import numpy as np  
from math import degrees, radians, sin, cos

def theta_2_position_rr(l1, l2, theta1, theta2):
    """
    Kinematics positive solution convert the input joint angle into the corresponding end point coordinates
    :param l1: up leg length
    :param l2: low leg length
    :param theta1: up leg joint angle
    :param theta2: low leg joint angle
    :return: endpoint coordinates of up leg and low leg
    """
    upLegEndpoint = [l1*cos(radians(theta1)), l1*sin(radians(theta1))]
    lowLegEndpoint = [l1*cos(radians(theta1))+l2*cos(radians(theta1+theta2)),
               l1*sin(radians(theta1))+l2*sin(radians(theta1+theta2))]

    return upLegEndpoint, lowLegEndpoint


def preprocess_drawing_data(points):
    """
    Process point coordinate data into a drawing format adapted to matplotlib
    :param points: point data
    :return: x-coordinate list and corresponding y-coordinate list of drawing data
    """
    xs = [0]
    ys = [0]
    xs.append(points[0][0])
    xs.append(points[1][0])
    ys.append(points[0][1])
    ys.append(points[1][1])
    return xs, ys

def annotate_angle(x0, y0, rad1, rad2, name, inverse=False):
    """
    draw angles for two lines
    :param x0: x coordinate of the center of the circle
    :param y0: y coordinate of the center of the circle
    :param rad1: starting angle
    :param rad2: end angle
    :param name: role name
    :param inverse: used to solve the overlapping problem of point 1
    :return: None
    """
    theta = np.linspace(rad1, rad2, 100)  # 0~rad
    r = 0.3  # circle radius
    x1 = r * np.cos(theta) + x0
    y1 = r * np.sin(theta) + y0
    plt.plot(x1, y1, color='red')
    plt.scatter(x0, y0, color='blue')
    degree = degrees((rad2 - rad1))
    if inverse:
        plt.annotate("%s=%.1f°" % (name, degree), [x0, y0], [x0 - r / 1.5, y0 - r / 1.5])
    else:
        plt.annotate("%s=%.1f°" % (name, degree), [x0, y0], [x0 + r / 1.5, y0 + r / 1.5])

# Joint information
# Up leg length: 5 cm and low leg length: 6 cm
link_length = [4.6, 5.2]  # in cm

# Joint angle initialization
joints_angle_origin = [-150, 90]
joints_angle = [0, 0]
print("Initial state of joint angle theta1=%d°, theta2=%d°", (joints_angle_origin[0], joints_angle_origin[1]))

# Input link parameters: each joint angle
for i in range(1, 3):
    joints_angle[i-1] = int(input("Please input the angle to be turned by the leg servo [%d]:" % i))
    print("The leg servo[{0}] will turn {1}°".format(i, joints_angle[i-1]))
    joints_angle[i-1] = joints_angle_origin[i-1]+joints_angle[i-1]


# Compute and preprocess plot data
points_origin = theta_2_position_rr(link_length[0], link_length[1], joints_angle_origin[0], joints_angle_origin[1])
print(f'The original upper leg and lower leg endpoints {points_origin}')
points_after = theta_2_position_rr( link_length[0], link_length[1], joints_angle[0], joints_angle[1])
print(f'The upper and lower leg endpoints after {points_after}')
data_origin = preprocess_drawing_data(points_origin)
print(f'original points shifted for plotting: {data_origin}')
data_after = preprocess_drawing_data(points_after)
print(f'later points shifted for plotting {data_after}')

# Draw
fig, ax = plt.subplots()  # build image
plt.plot(data_origin[0], data_origin[1], color='black', label='original')
plt.scatter(data_origin[0], data_origin[1], color='black')
plt.plot(data_after[0], data_after[1], color='red', label='after')
plt.scatter(data_after[0], data_after[1], color='blue')
ax.set(xlabel='X', ylabel='Y', title='Bittle leg kinematics')
ax. grid()
plt. axis("equal")
plt. legend(loc=2)

# Annotation
annotate_angle(data_origin[0][0], data_origin[1][0], 0, radians(joints_angle_origin[0]), "theta1_original", inverse=True)
annotate_angle(data_origin[0][1], data_origin[1][1], radians(joints_angle_origin[0]),radians(joints_angle_origin[0]+joints_angle_origin[1]), "theta2_original", inverse=True)
annotate_angle(data_after[0][0], data_after[1][0], 0, radians(joints_angle[0]), "theta1_after")
annotate_angle(data_after[0][1], data_after[1][1], radians(joints_angle[0]), radians(joints_angle[0]+joints_angle[1]), "theta2_after")
plt.show()