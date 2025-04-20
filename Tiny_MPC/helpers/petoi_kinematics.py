import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import beta
import casadi as cs
class PetoiKinematics:
    def __init__(self, render_mode='') -> None:
        
        """ Static parameters """
        self.body_length = self.a = .105 # m
        self.body_width = self.b = .1 # m
        self.leg_length = self.c = .046 # m
        self.foot_length = self.d = .046 # m
        self.upper_length = 0.01  # Shoulder to knee
        self.lower_length = 0.012  # Knee to foot
        self.shoulder_offset = 0.005  # Front shoulder x-offset

        """ Body and link mass """
        self.thigh_mass = 0.006 # kg
        self.foot_mass = 0.017 # kg
        self.thigh_coms = np.array([
            [0.0, -0.024575, 0.004372], # FL
            [0.0, -0.024575, 0.004372], # BL
            [0.0, 0.024575, 0.00777], # BR
            [0.0, 0.024575, 0.00777], # FR
        ])
        self.foot_coms = np.array([
            [-0.009232, 0.000238, -0.014614], # FL
            [-0.008998, 0.000238, -0.014638], # BL
            [0.008426, -0.000238, -0.01382], # BR
            [0.009497, -0.000238, -0.01382], # FR
        ])

        """ Dynamic parameters 
        Note that we make body angle and body height arrays to enable a smooth transition

        self.gamma is an interpolation between current angle and target angle
        """
        self.alphas = np.zeros([4]) # shoulder angles
        self.betas = np.zeros([4]) # knee angles
        self.gamma = np.array([0.]) # body angles
        self.gamma_granularity = np.deg2rad(2.) # increase/decrease body angle (roughly) 2deg each time
        self.h = np.array([0.]) # body vertial shift
        self.h_granularity = 0.1 # increase/decrease body height (roughly) 0.3cm each time
        self.T01_front = None
        self.T01_back = None
        self.T01_front_inv = None
        self.T01_back_inv = None

        """ Plotting initialization """
        self.render_mode = render_mode
        if self.render_mode == '3d':
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_box_aspect([1,1,0.5])
            self.ax.set_aspect('auto')
        elif self.render_mode == '2d':
            self.fig, self.ax = plt.subplots(1,1)
        else:
            self.fig, self.ax = None, None

        self.update_gamma_h(self.gamma[0], self.h[0])

    def update_gamma_h(self, gamma=None, h=None):
        """
        Update the target body angle and body height
        1. if gamma is given, replan the interpolation
        2. if gamma is not given, 
            2(a) if len(self.gamma)>1, select the next gamma in interpolation
            2(b) if len(self.gamma)=1, already at the target point, then don't update anything
        
        h is similar to gamma
        """

        if h is None:
            if len(self.h) == 1:
                pass
            else:
                self.h = self.h[1:]
        else:
            self.h = np.linspace(
                self.h[0], h, int(np.abs(h - self.h[0]) // self.h_granularity) + 1
            )
            if len(self.h) > 1: self.h = self.h[1:]

        if gamma is None:
            if len(self.gamma) == 1: 
                return
            else:
                self.gamma = self.gamma[1:]
                
        else:
            self.gamma = np.linspace(
                self.gamma[0], gamma, int(np.abs(gamma - self.gamma[0]) // self.gamma_granularity) + 1
            )
            if len(self.gamma) > 1: self.gamma = self.gamma[1:]


        # update the transformation matrix
        cg, sg = np.cos(self.gamma[0]), np.sin(self.gamma[0])
        self.T01_front = np.array([
            [cg, sg, -self.a/2*cg],
            [-sg, cg, self.a/2*sg],
            [0,  0,  1.]
        ])
        self.T01_back = np.array([
            [cg, sg,   self.a/2*cg],
            [-sg, cg, -self.a/2*sg],
            [0,  0,  1.]
        ])

        self.T01_front_inv = np.linalg.inv(self.T01_front)
        self.T01_back_inv = np.linalg.inv(self.T01_back)

    def render(self, alphas, betas):
        if self.render_mode != '3d' and self.render_mode != '2d':
            return
        plt.cla()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None]
        )

        """ plot legs """
        for i in range(4):
            ca, sa = np.cos(alphas[i]), np.sin(alphas[i])
            T01 = self.T01_front if i in [0,3] else self.T01_back
            y = -self.b/2 if i in [0,1] else self.b/2
            T02 = T01 @ np.array([
                [ca, -sa, self.c * sa],
                [sa,  ca, -self.c * ca],
                [0,    0, 1]
            ])
            foot = T02 @ np.array([
                -self.d * np.cos(betas[i]),
                self.d * np.sin(betas[i]),
                1
            ])

            if self.render_mode == '3d':
                self.ax.plot(
                    # shoulder  knee        foot
                    [T01[0, 2], T02[0, 2], foot[0]],    # x
                    [y,         y,          y],         # y
                    [T01[1, 2], T02[1, 2], foot[1]],    # z
                    linewidth=3
                )  
            elif self.render_mode == '2d':
                self.ax.plot(
                    # shoulder  knee        foot
                    [T01[0, 2], T02[0, 2], foot[0]], # x             
                    [T01[1, 2], T02[1, 2], foot[1]], # z
                    linewidth=3
                ) 
            else:
                pass
        # plot body
        """ plot body
        In the 1st version of the code, this comes before the leg plotting, 
        but then legs will cover the plot of body, so I put this after the leg plotting
        """
        if self.render_mode == '3d':
            self.ax.plot(
                [self.T01_front[0,2], self.T01_back[0,2]],
                [0, 0],
                [self.T01_front[1,2], self.T01_back[1,2]],
                'b-', linewidth=3
            )
            self.ax.plot(
                [self.T01_front[0,2], self.T01_front[0,2]],
                [-self.b/2, self.b/2],
                [self.T01_front[1,2], self.T01_front[1,2]],
                'b-', linewidth=3
            )

            self.ax.plot(
                [self.T01_back[0,2], self.T01_back[0,2]],
                [-self.b/2, self.b/2],
                [self.T01_back[1,2], self.T01_back[1,2]],
                'b-', linewidth=3
            )
            self.ax.scatter(0, 0, 0)

        elif self.render_mode == '2d':
            self.ax.plot(
                np.array([self.T01_front[0,2], self.T01_back[0,2]]),
                np.array([self.T01_front[1,2], self.T01_back[1,2]]),
                'b-', linewidth=3
            )
            self.ax.scatter(0,0) 
        
        else:
            pass

        if self.render_mode == '3d':
            self.ax.set_xlim3d([-10, 10])
            self.ax.set_ylim3d([-5, 5])
            self.ax.set_zlim3d([-9, 0])
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_aspect("auto")
            plt.pause(0.0001)
        elif self.render_mode == '2d':
            self.ax.set_xlim([-10, 10])
            self.ax.set_ylim([-9, 5])
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_aspect("equal")
            plt.pause(0.001)
    
    def leg_ik(self, xz:np.array, y_included=False):
        """
        compute the inverse kinematics for all 4 legs
        :param xz:
            np.array with shape [4,2]
            each row is the xz coordinate of the end effector wrt the body frame
        :return
            alphas: shoulder angles, np.array with shape [4,]
            betas: knee angles, np.array with shape [4,]
        """
        if not y_included:
            xz0 = np.concatenate([xz, np.ones([4, 1])], axis=1) # [4, 3]
        else:
            xz0 = xz.copy()
        xz0[:,1] -= self.h[0]
        alphas = []
        betas = []

        for i in range(4):
            T01_inv = self.T01_front_inv if i in [0,3] else self.T01_back_inv
            xz1 = T01_inv @ xz0[i]
            x1, z1 = xz1[0], xz1[1]

            L = np.sqrt(x1**2 + z1**2)
            alpha1_tilde = np.arccos((self.c**2 + L**2 - self.d**2) / (2 * self.c * L))
            alpha2_tilde = np.arcsin(x1/L)
            alphas.append(alpha1_tilde + alpha2_tilde)

            beta_tilde = np.arccos((self.c**2 + self.d**2 - L**2) / (2 * self.c * self.d))
            betas.append(np.pi/2 - beta_tilde)

        return np.array(alphas), np.array(betas)

    def forward_kinematics_front(self, leg_angles):
        """Returns foot position in body frame as 2D vector [x, z]"""
        alpha, beta = leg_angles[0], leg_angles[1]
        x = self.upper_length*cs.sin(alpha) + self.lower_length*cs.sin(alpha + beta)
        z = -self.upper_length*cs.cos(alpha) - self.lower_length*cs.cos(alpha + beta)
        return cs.vertcat(x + self.shoulder_offset, z)

    def forward_kinematics_back(self, leg_angles):
        # Similar implementation for back legs with appropriate offsets
        alpha, beta = leg_angles[0], leg_angles[1]
        x = self.upper_length*cs.sin(alpha) + self.lower_length*cs.sin(alpha + beta)
        z = -self.upper_length*cs.cos(alpha) - self.lower_length*cs.cos(alpha + beta)
        return cs.vertcat(x - self.shoulder_offset, z)
    
    def joints_to_com(self, alphas, betas):
        """
        compute center of mass based on joint angles
        :param alphas:
            np.array with shape [4,]
            each row is the angle of the shoulder joint
        :param betas:
            np.array with shape [4,]
            each row is the angle of the knee joint
        :return
            com_proj: np.array with shape [3,]
            each row is the xz coordinate of the center of mass projection on the ground
        """

        total_com = np.zeros(3)
        # [x,z,1]
        total_mass = 0.0

        for i in range(4):
            ca, sa = np.cos(alphas[i]), np.sin(alphas[i])
            cb, sb = np.cos(betas[i]), np.sin(betas[i])

            # Select shoulder-to-world transform
            T01 = self.T01_front if i in [0, 3] else self.T01_back

            # -------- Thigh COM --------
            # Position along thigh (first link)
            # self.thigh_coms[i] is assumed in local thigh frame
            thigh_com_local = self.thigh_coms[i]
            thigh_com_offset = np.array([
                ca * thigh_com_local[0] - sa * thigh_com_local[2],
                sa * thigh_com_local[0] + ca * thigh_com_local[2],
                1
            ])
            thigh_com_world = T01 @ thigh_com_offset

            # -------- Foot COM --------
            # Build T02 as in your FK code (shoulder → knee)
            T02 = T01 @ np.array([
                [ca, -sa, self.c * sa],
                [sa,  ca, -self.c * ca],
                [0,    0, 1]
            ])
            # self.foot_coms[i] assumed in local foot frame
            foot_com_local = self.foot_coms[i]
            # rotate by knee
            foot_com_offset = np.array([
                cb * foot_com_local[0] - sb * foot_com_local[2],
                sb * foot_com_local[0] + cb * foot_com_local[2],
                1
            ])
            foot_com_world = T02 @ foot_com_offset

            total_com += self.thigh_mass * thigh_com_world
            total_com += self.foot_mass * foot_com_world
            total_mass += self.thigh_mass + self.foot_mass

        # Final COM position
        com = total_com / total_mass
        com[1] *= -1
        return com[:2] # Return com x,z




if __name__ == '__main__':
    standing_gait = np.array([
        [-5., -5],
        [5, -5],
        [5, -5],
        [-5, -5]
    ])

    standing_gait = np.array(
        [[ 0.05024405 -0.04199575]
    [-0.05014449 -0.04199698]
    [ 0.08252691 -0.12852691],
    [-0.08242691 -0.12852691]]
    )
    dog = PetoiKinematics(render_mode='3d')
    try:
        while plt.fignum_exists(dog.fig.number):  # Check if the figure window is still open
            alphas, betas = dog.leg_ik(standing_gait)
            dog.render(alphas, betas)
            dog.update_gamma_h()
            print(np.rad2deg(alphas))
            print(np.rad2deg(betas))
            print()
    except KeyboardInterrupt:
        print("Exiting gracefully...")
    finally:
        plt.close(dog.fig)