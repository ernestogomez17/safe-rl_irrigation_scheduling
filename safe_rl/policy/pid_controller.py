import torch
from torch import relu
import numpy as np


# class LagrangianPIDController:
#     def __init__(self, KP, KI, KD, thres, per_state=True):
#         self.KP = KP
#         self.KI = KI
#         self.KD = KD
#         self.thres = thres
#         self.error_old = 0
#         self.error_integral = 0

#     def update_threshold(self, new_threshold):
#         ''' Update the threshold while maintaining the controller state. '''
#         self.thres = new_threshold

#     def control(self, qc):
#         error_new = torch.mean(qc - self.thres)  # Compute new error
#         error_diff = relu(error_new - self.error_old)  # Difference from old error
#         self.error_integral = torch.mean(relu(self.error_integral + error_new))  # Update integral of error
#         self.error_old = error_new  # Update old error

#         # Calculate the control multiplier
#         multiplier = relu(self.KP * relu(error_new) + self.KI * self.error_integral +
#                           self.KD * error_diff)
#         return torch.mean(multiplier)


def projection(x):
    return np.maximum(0, x)

class LagrangianPIDController:
    '''
    A PID controller for updating a Lagrangian multiplier,
    correctly implementing the logic from Stooke et al., 2020.
    '''
    def __init__(self, KP, KI, KD):
        self.KP = KP
        self.KI = KI
        self.KD = KD

        # Initialize controller state variables
        self.error_old = 0.0
        self.error_integral = 0.0
        
        # The multiplier (λ) is the state of the controller
        self.multiplier = 0.0

    def update(self, error):
        '''
        This method is called once per epoch with the actual safety error.
        It updates the internal multiplier value based on the PID logic.
        
        The 'error' is defined as (actual_value - threshold), where a positive
        error means the constraint is being violated.
        '''
        error_new = float(error)
        
        # Allow negative derivative to decrease multiplier when safety improves
        error_diff = projection(error_new - self.error_old)

        # Only project the integral term, not individual errors
        self.error_integral = projection(self.error_integral + error_new)
        self.error_old = error_new

        # The new multiplier is calculated from scratch based on the PID terms
        p_term = self.KP * projection(error_new)
        i_term = self.KI * self.error_integral
        d_term = self.KD * error_diff

        # Only the final multiplier should be non-negative
        self.multiplier = projection(p_term + i_term + d_term)

    def get_multiplier(self):
        '''
        Gets the current lagrangian multiplier.
        '''
        return self.multiplier