import numpy as np
from spatialmath import SE3
from scipy.optimize import minimize



def find_joint_state(ur3e, initial_guess, desired_config, desired_pose : SE3) -> list:

    """Find the joint state that minimizes the difference from the desired config and satisfies the forward kinematics constraint."""
    def objective_function(x, desired_config):
        """Objective function to minimize. This should return the difference between the current joint state and the desired joint state."""
        return np.linalg.norm(x - desired_config)

    def constraint_function(x, desired_pose):
        """Constraint function for the optimization. This should return 0 when the forward kinematics of the current joint state match the desired pose."""
        current_pose = ur3e.fkine(x,tool=self.virtual_UR_tool).A
        pos_error = np.linalg.norm(current_pose[0:3,3] - desired_pose.A)
        ori_error = np.linalg.inv(current_pose[0:3,0:3]) @ desired_pose.A[0:3,0:3]
        return pos_error + ori_error
    
    # Define the constraint
    constraint = {'type': 'eq', 'fun': constraint_function, 'args': (desired_pose)}

    # Call the minimize function
    result = minimize(objective_function, initial_guess, args=(desired_config), constraints=constraint)

    # Return the optimized joint state
    return result.x