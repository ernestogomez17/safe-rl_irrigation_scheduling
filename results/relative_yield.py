"""
relative_yield.py

Implementation of the Relative Yield (RY) calculation as per:
Roy, A., et al. (2021). Water Resources Research.
DOI: 10.1029/2020WR029004
"""

from typing import List

def calculate_relative_yield(
    ky_values: List[float],
    caet_values: List[float],
    cpet_values: List[float]
) -> float:
    """
    Calculates the Relative Yield (RY) based on cumulative actual and 
    potential evapotranspiration across different crop growth stages.

    Parameters:
    -----------
    ky_values : List[float]
        Crop yield response factor for each growth stage k (K_{y,k}).
    caet_values : List[float]
        Cumulative actual evapotranspiration up to growth stage k (CAET_k).
    cpet_values : List[float]
        Cumulative potential evapotranspiration up to growth stage k (CPET_k).

    Returns:
    --------
    float
        The overall Relative Yield (RY) for the season.
    """
    if not (len(ky_values) == len(caet_values) == len(cpet_values)):
        raise ValueError("All input lists must have the same length (representing the number of growth stages).")

    # 1. Calculate yield loss at each stage k (YL_k)
    # Equation 14: YL_k = K_{y,k} * (1 - CAET_k / CPET_k)
    yl_values = []
    for ky, caet, cpet in zip(ky_values, caet_values, cpet_values):
        if cpet <= 0.0:
            # Prevent division by zero if CPET is zero
            yl = 0.0
        else:
            # Deficit ratio
            deficit_ratio = 1.0 - (caet / cpet)
            # CAET can occasionally slightly exceed CPET due to numeric precision; clamp to 0
            deficit_ratio = max(0.0, deficit_ratio) 
            yl = ky * deficit_ratio
            
        yl_values.append(yl)

    # 2. Calculate the overall weighted Relative Yield (RY)
    # Equation 15: RY = 1 - (sum(K_{y,k} * YL_k) / sum(K_{y,k}))
    numerator = sum(ky * yl for ky, yl in zip(ky_values, yl_values))
    denominator = sum(ky_values)

    if denominator == 0.0:
        return 1.0  # Fallback if no yield response factors are provided

    overall_yield_loss = numerator / denominator
    ry = 1.0 - overall_yield_loss

    # Floor at 0.0 to ensure RY is physically meaningful (e.g., total crop failure)
    return max(0.0, ry)

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Example values for a hypothetical 4-stage grape season
    # (Initial, Development, Mid-Season, Late-Season)
    
    # 1. Yield response factors (K_y) for each stage
    K_y = [0.4, 0.6, 1.2, 0.8]  
    
    # 2. Cumulative Actual ET (CAET) at the end of each stage (in mm)
    CAET_schedule = [50.0, 110.0, 300.0, 420.0]
    
    # 3. Cumulative Potential ET (CPET) at the end of each stage (in mm)
    CPET_schedule = [52.0, 120.0, 330.0, 450.0]

    # Calculate RY
    final_ry = calculate_relative_yield(K_y, CAET_schedule, CPET_schedule)
    
    print(f"Calculated Relative Yield (RY): {final_ry:.4f} ({final_ry * 100:.1f}%)")