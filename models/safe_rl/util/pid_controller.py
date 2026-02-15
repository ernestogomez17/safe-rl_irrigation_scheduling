import torch
from torch import relu
import numpy as np


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
        
        # D-term: only contributes when violations are GROWING (projection clips negative diffs)
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