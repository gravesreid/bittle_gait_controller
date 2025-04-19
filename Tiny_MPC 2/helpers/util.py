import numpy as np

class Util:

    def compute_pd_torque(u_angle_trajectory, kp, kd, ki, pid_dt=1e-3):
        """
        Compute joint torques using PD control law with aligned dimensions.
        
        Args:
            u_angle_trajectory: (N, 8) array of target joint angles
            kp: Proportional gain (float or 8-element array)
            kd: Derivative gain (float or 8-element array)
            ki: Integral gain (float or 8-element array)
            pid_dt: Time step between trajectory points
            
        Returns:
            torque: (N-2, 8) array of computed torques
        """
        trajectory = np.array(u_angle_trajectory)
        
        # Central differences for consistent dimensions (N-2, 8)
        # Position difference (Δθ between i+1 and i-1)
        del_theta = trajectory[2:] - trajectory[:-2]
        
        # Velocity (θ_dot = Δθ/(2Δt))
        theta_dot = del_theta / (2 * pid_dt)
        
        # Integral term (cumulative sum of position errors)
        int_theta = np.cumsum(del_theta * pid_dt, axis=0)
        
        # Torque calculation (all terms now have shape (N-2, 8))
        torque = (
            kp * del_theta +
            kd * theta_dot +
            ki * int_theta
        )
        
        return torque