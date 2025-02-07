from constants import CONSTANTS
import numpy as np

class RelativityHandler:
    """Handle relativistic effects in stellar structure"""
    
    def __init__(self, mass: float, radius: float, active: bool = True):
        self.mass = mass
        self.radius = radius
        self.active = active
        self.compactness = CONSTANTS['G'] * mass / (radius * CONSTANTS['c']**2)
        
    def correct_pressure(self, P_classical: float, density: float) -> float:
        """Apply TOV corrections to pressure"""
        if not self.active:
            return P_classical
            
        # Implement TOV equation corrections
        compactness = 2 * CONSTANTS['G'] * self.mass / (CONSTANTS['c']**2 * self.radius)
        
        # Add post-Newtonian corrections
        P_corrected = P_classical * (1 + first_post_newtonian_term(compactness))
        
        return P_corrected
        
    def correct_temperature(self, T: float, r: float) -> float:
        """Apply relativistic temperature corrections"""
        if not self.active:
            return T
            
        # Redshift correction
        redshift = 1.0 / np.sqrt(1 - 2*self.compactness*r/self.radius)
        
        # Add frame dragging effects for rapidly rotating stars
        frame_dragging = 1 + 0.1 * self.compactness
        
        return T * redshift * frame_dragging
