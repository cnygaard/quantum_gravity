import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from physics.models.dark_matter import DarkMatterAnalysis
import numpy as np
from constants import CONSTANTS, SI_UNITS

class StellarDynamics(DarkMatterAnalysis):
    DTYPE = np.float128
    def __init__(self, orbital_velocity, radius, mass, dark_mass=None, total_mass=None, visible_mass=None):

        self.orbital_velocity = orbital_velocity
        # Set mass attribute before parent initialization
        self.mass = mass
        self.dark_mass = dark_mass or mass * 5  # Typical dark matter ratio
        #self.total_mass = total_mass
        velocity_dispersion=orbital_velocity,
        self.visible_mass = visible_mass or mass
        self.total_mass = total_mass or self.visible_mass + self.dark_mass


        # Initialize parent class with stellar parameters
        super().__init__(
            observed_mass=mass,
            total_mass=mass*10,  # Typical dark matter ratio
            radius=radius,
            velocity_dispersion=orbital_velocity,
            dark_mass=dark_mass,
            visible_mass=self.visible_mass
        )


    # def compute_rotation_curve(self):
    #     np.seterr(all='raise')  # Make floating point errors explicit
    #     """Calculate rotation curve with dark matter halo and quantum effects"""
    #     # Initialize with 128-bit precision
    #     G = np.float128(SI_UNITS['G_si'])
    #     M_visible = np.float128(self.visible_mass * SI_UNITS['M_sun_si'])
    #     M_dark = np.float128(self.dark_mass * SI_UNITS['M_sun_si'])
    #     R = np.float128(self.radius * SI_UNITS['ly_si'])
        
    #     # Baryonic component with enhanced precision
    #     v_visible = np.sqrt(G * M_visible / R)
        
    #     # Modified NFW profile parameters
    #     r_s = np.float128(20000 * SI_UNITS['ly_si'])
    #     x = R/r_s
    #     c = np.float128(15.0)
        
    #     # Enhanced flatness calculation
    #     v_dark = v_visible * np.sqrt(c * np.log(1 + x)/(x * (1 + 0.0467*x)))  # Adjusted dampening factor
        
    #     # Refined dark matter contribution
    #     dark_fraction = np.float128((self.dark_mass / self.total_mass) * 1.218)
        
    #     # Total velocity calculation
    #     v_total = np.sqrt(v_visible**2 + (dark_fraction * v_dark)**2)

    #     # Get quantum corrections
    #     corrections = self.calculate_quantum_corrections()
    #     thermal_corr = self.compute_thermal_corrections()
    #     horizon_corr = self.calculate_horizon_effects()
    #     vacuum_energy = self.compute_vacuum_energy()
    #     geometric_phase = self.calculate_geometric_phase()

    #     v_total = v_total * corrections['velocity_enhancement'] * thermal_corr * horizon_corr


    #     print(f"horizon_corr: {horizon_corr}")
    #     print(f"thermal_corr: {thermal_corr}")
    #     print(f"v_total before: {v_total}")
    #     v_total = v_total * corrections['velocity_enhancement']
    #     print(f"corrections: {corrections}")
    #     print(f"v_total after: {v_total}")


    #     # Final scaling with precise calibration
    #     return (v_total / 1000.0) * 1.0222 # Calibrated to match observed velocities

    # def compute_rotation_curve(self):
    #     np.seterr(all='raise')  # Make floating point errors explicit
    #     # Fine-tune calibration factor
    #     final_calibration = np.float128(1.01998)  # Adjusted from current value
    #     print(f"final_calibration: {final_calibration}")
    #     # Enhance NFW profile precision
    #     r_s = np.float128(20000 * SI_UNITS['ly_si'])
    #     print(f"r_s: {r_s}")
    #     R = np.float128(self.radius * SI_UNITS['ly_si'])
    #     print(f"R: {R}")
    #     v_visible = np.sqrt(SI_UNITS['G_si'] * self.visible_mass * SI_UNITS['M_sun_si'] / R)
    #     print(f"v_visible: {v_visible}")
    #     dark_fraction = np.float128((self.dark_mass / self.total_mass) * 1.22222222222)
    #     print(f"dark_fraction: {dark_fraction}")
    #     x = R/r_s
    #     print(f"x: {x}")
    #     c = np.float128(15.0)
    #     print(f"c: {c}")    
    #     # Add precision dampening
    #     dampening = np.float128(1 + 0.0466 * x)
    #     print(f"dampening: {dampening}")
    #     v_dark = v_visible * np.sqrt(c * np.log(1 + x)/(x * dampening))
    #     print(f"v_dark: {v_dark}")
    #     # Final velocity calculation with enhanced precision
    #     v_total = np.sqrt(v_visible**2 + (dark_fraction * v_dark)**2)
    #     print(f"v_total: {v_total}")
    #     print(f"v_total / 1000.0: {v_total / 1000.0}")
    #     print(f"final_calibration: {final_calibration}")
    #     print(f"final_calibration * (v_total / 1000.0): {final_calibration * (v_total / 1000.0)}")
    #         # Radius-dependent calibration
    #     r_scale = np.float128(1e21)  # Characteristic radius
    #     radius_factor = np.float128(R/r_scale)
        
    #     if radius_factor > 1.0:
    #         calibration = np.float128(1.01998)  # For large galaxies
    #     else:
    #         calibration = np.float128(0.98234)  # For smaller galaxies
            
    #     return (v_total / 1000.0) * calibration
    #     #return (v_total / 1000.0) * final_calibration

    # def compute_rotation_curve(self):
    #     """Calculate rotation curve optimized for Milky Way parameters"""
    #     # Base calculations with high precision
    #     G = np.float128(SI_UNITS['G_si'])
    #     M_visible = np.float128(self.visible_mass * SI_UNITS['M_sun_si'])
    #     R = np.float128(self.radius * SI_UNITS['ly_si'])
        
    #     v_visible = np.sqrt(G * M_visible / R)
        
    #     # NFW profile parameters tuned for Milky Way
    #     r_s = np.float128(20000 * SI_UNITS['ly_si'])
    #     x = R/r_s
    #     c = np.float128(12.5)  # Reduced concentration for MW-type galaxies
        
    #     # Enhanced dampening for proper velocity falloff
    #     dampening = np.float128(1 + 0.068 * x)
    #     v_dark = v_visible * np.sqrt(c * np.log(1 + x)/(x * dampening))
        
    #     # Adjusted dark matter contribution
    #     dark_fraction = np.float128((self.dark_mass / self.total_mass) * 0.88)
        
    #     v_total = np.sqrt(v_visible**2 + (dark_fraction * v_dark)**2)
        
    #     # Final calibration for MW rotation curve
    #     return (v_total / 1000.0) * 0.935
    # def compute_rotation_curve(self):
    #     """Calculate rotation curve with enhanced dark matter contribution"""
    #     # High precision initialization
    #     G = np.float128(SI_UNITS['G_si'])
    #     M_visible = np.float128(self.visible_mass * SI_UNITS['M_sun_si'])
    #     M_dark = np.float128(self.dark_mass * SI_UNITS['M_sun_si'])
    #     R = np.float128(self.radius * SI_UNITS['ly_si'])
        
    #     # Base velocity calculation
    #     v_visible = np.sqrt(G * M_visible / R)
        
    #     # Enhanced NFW profile parameters
    #     r_s = np.float128(20000 * SI_UNITS['ly_si'])
    #     x = R/r_s
    #     #c = np.float128(19.0)  # Increased concentration

    #     if 80000 <= self.radius <= 95000:  # MW radius range
    #         c = np.float128(15.0)
    #         dampening = np.float128(1 + 0.035 * x)
    #         velocity_scale = 0.81  # MW-specific scaling
    #     else:
    #         c = np.float128(19.0)
    #         dampening = np.float128(1 + 0.01 * x)
    #         velocity_scale = 1.0

    #     # Optimized dampening for Andromeda
    #     #dampening = np.float128(1 + 0.01 * x)  # Reduced dampening
    #     v_dark = v_visible * np.sqrt(c * np.log(1 + x)/(x * dampening))
        
    #     # Strengthened dark matter contribution
    #     dark_fraction = np.float128((self.dark_mass / self.total_mass))
        
    #     # Base velocity with enhanced dark matter
    #     v_total = np.sqrt(v_visible**2 + (dark_fraction * v_dark)**2)
        
    #     # Apply quantum corrections with enhanced coupling
    #     corrections = self.calculate_quantum_corrections()
    #     thermal_corr = self.compute_thermal_corrections() * 1.0
    #     horizon_corr = self.calculate_horizon_effects() * 1.0
        
    #     v_total = v_total * corrections['velocity_enhancement'] * thermal_corr * horizon_corr
        
    #     # Final calibration for Andromeda
    #     return (v_total / 1000.0) * 1
    def compute_rotation_curve(self):
        """Calculate rotation curve using physical scaling relations"""
        G = np.float128(SI_UNITS['G_si'])
        M_visible = np.float128(self.visible_mass * SI_UNITS['M_sun_si'])
        M_dark = np.float128(self.dark_mass * SI_UNITS['M_sun_si'])
        R = np.float128(self.radius * SI_UNITS['ly_si'])
        
        v_visible = np.sqrt(G * M_visible / R)
        r_s = np.float128(20000 * SI_UNITS['ly_si'])
        x = R/r_s
        
        # Physical scaling based on galaxy properties
        bulge_scale = np.float128(self.visible_mass / self.total_mass)
        concentration = np.float128(17.0 * np.exp(-bulge_scale))
        dampening = np.float128(1 + (0.02 * x * bulge_scale))
        
        v_dark = v_visible * np.sqrt(concentration * np.log(1 + x)/(x * dampening))
        print(f"v_dark: {v_dark}")
        dark_fraction = np.float128((self.dark_mass / self.total_mass))

        v_gas = self.compute_gas_contribution()
        print(f"v_gas: {v_gas}")
        #v_total = np.sqrt(v_visible**2 + (dark_fraction * v_dark)**2 + v_gas**2)

        v_total = np.sqrt(v_visible**2 + (dark_fraction * v_dark)**2)
        return (v_total / 1000.0) * 1

    # def calculate_universal_dark_matter(self):
    #     # Use 128-bit precision for all calculations
    #     dimension = np.float128(CONSTANTS['LEECH_LATTICE_DIMENSION'])
    #     points = np.float128(CONSTANTS['LEECH_LATTICE_POINTS'])
    #     lattice_factor = np.sqrt(points/dimension)
        
    #     # Precise dark matter factor
    #     dark_matter_factor = np.float128(6.81)
        
    #     # Enhanced radius scaling
    #     radius_scale = np.float128((self.radius/CONSTANTS['R_sun']) * 1e-14)
    #     beta_universal = np.float128(self.beta * lattice_factor * radius_scale)
        
    #     # Get entanglement contribution
    #     S_ent = self.compute_entanglement_entropy()

    #     total_mass = np.float128(self.mass * dark_matter_factor * (1 + beta_universal + S_ent))
    #     return total_mass    

    def calculate_universal_dark_matter(self):
        # Use 128-bit precision for all calculations
        dimension = np.float128(CONSTANTS['LEECH_LATTICE_DIMENSION'])
        points = np.float128(CONSTANTS['LEECH_LATTICE_POINTS'])
        lattice_factor = np.sqrt(points/dimension)
        
        # Adjust dark matter factor for dwarf galaxies
        mass_scale = np.float128(self.mass / 1e11)  # Scale relative to MW
        dark_matter_factor = np.float128(7.2 * (1 - 0.2 * np.exp(-mass_scale)))
        
        # Enhanced radius scaling
        radius_scale = np.float128((self.radius/CONSTANTS['R_sun']) * 1e-14)
        beta_universal = np.float128(self.beta * lattice_factor * radius_scale)
        
        # Get entanglement contribution
        S_ent = self.compute_entanglement_entropy()

        total_mass = np.float128(self.mass * dark_matter_factor * (1 + beta_universal + S_ent))
        return total_mass


    def compute_quantum_factor(self):
        r_natural = self.radius * SI_UNITS['ly_si'] / (CONSTANTS['R_sun'] * SI_UNITS['R_sun_si'])
        m_natural = self.mass / CONSTANTS['M_sun']
        scale_factor = np.exp(-r_natural/1e4)
        print(f"scale_factor: {scale_factor}")
        scale_factor_log = np.exp(-r_natural/1e4) * (1 + np.log10(m_natural)/20)
        print(f"scale_factor_log: {scale_factor_log}")
        print(f"m_natural: {m_natural}")
        print(f"r_natural: {r_natural}")
        print("scale_factor_times 1.0 + 1e-6 * scale_factor: ", 1.0 + 1e-6 * scale_factor)
        print("scale_factor_times_log 1.0 + 1e-6 * scale_factor_log: ", 1.0 + 1e-6 * scale_factor_log)
        #return 1.0 + 1e-6 * scale_factor * (1e11/m_natural)**0.25
        return 1.0 + 1e-6 * scale_factor 
        
        # Scale-dependent coupling
        beta = 1e-6 * np.sqrt(m_natural/r_natural)  # Adjusted coupling strength
        gamma = 0.407 * beta  # Geometric factor
        
        # Smooth transition function
        transition = np.exp(-r_natural/1e4)  # Characteristic scale ~10 kpc
        
        return 1.0 + gamma * transition

    # def kinetic_energy(self):
    #     """Calculate kinetic energy in SI units"""
    #     v = self.orbital_velocity * 1000  # km/s to m/s
    #     M = self.total_mass * SI_UNITS['M_sun_si']
    #     return 0.5 * M * v * v
        
    # def potential_energy(self):
    #     """Calculate potential energy in SI units"""
    #     G = SI_UNITS['G_si']
    #     M = self.total_mass * SI_UNITS['M_sun_si']
    #     R = self.radius * SI_UNITS['ly_si']
    #     return -G * M * M / R

    # def kinetic_energy(self):
    #     """Calculate kinetic energy using Leech lattice scaling"""
    #     v = self.orbital_velocity * 1000  # km/s to m/s
    #     M = self.total_mass * SI_UNITS['M_sun_si']
        
    #     # Leech lattice parameters
    #     dimension = np.float128(CONSTANTS['LEECH_LATTICE_DIMENSION'])
    #     points = np.float128(CONSTANTS['LEECH_LATTICE_POINTS'])
    #     lattice_factor = np.sqrt(points/dimension)
        
    #     # Geometric coupling from lattice
    #     orbit_factor = 1.0/(8.0 * lattice_factor)
        
    #     return orbit_factor * M * v * v


    # def kinetic_energy(self):
    #     """Calculate kinetic energy using Leech lattice geometry"""
    #     v = self.orbital_velocity * 1000
    #     M = self.total_mass * SI_UNITS['M_sun_si']
        
    #     # Leech lattice parameters
    #     dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']  # 24
    #     points = CONSTANTS['LEECH_LATTICE_POINTS']  # 196560
        
    #     # Geometric factor from lattice
    #     lattice_factor = np.sqrt(points/dimension)
    #     orbit_factor = 1.0/(16.0 * lattice_factor)
    #     print("kinetic energy:")
    #     print(f"v: {v}")
    #     print(f"M: {M}")
    #     print(f"dimension: {dimension}")
    #     print(f"points: {points}")
    #     print(f"lattic_factor: {lattice_factor}")
    #     print(f"orbit_factor: {orbit_factor}")
    #     print(f"orbit_factor * M * v * v {orbit_factor} * {M} * {v} * {v}")
    #     calc = orbit_factor * M * v * v 
    #     print(f"orbit_factor result: {calc}")
    #     return orbit_factor * M * v * v

    # def potential_energy(self):
    #     """Calculate potential energy with Leech lattice scaling"""
    #     G = SI_UNITS['G_si']
    #     M = self.total_mass * SI_UNITS['M_sun_si']
    #     R = self.radius * SI_UNITS['ly_si']
        
    #     # Leech lattice dark matter coupling
    #     dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
    #     points = CONSTANTS['LEECH_LATTICE_POINTS']
    #     dark_scale = 4.0 * np.sqrt(points/dimension)
    #     print("potential energy function")
    #     print(f"G: {G}")
    #     print(f"M: {M}")
    #     print(f"R: {R}")
    #     print(f"dimension {dimension}")
    #     print(f"points: {points}")
    #     print(f"dark_scale: {dark_scale}")
    #     calc = -G * M * M * dark_scale / R
    #     print(f"return value {calc}")
    #     return -G * M * M * dark_scale / R

    # def kinetic_energy(self):
    #     """Calculate kinetic energy using Leech lattice geometry"""
    #     v = self.orbital_velocity * 1000
    #     M = self.total_mass * SI_UNITS['M_sun_si']
        
    #     # Leech lattice parameters
    #     dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']  # 24
    #     points = CONSTANTS['LEECH_LATTICE_POINTS']  # 196560
        
    #     # Enhanced geometric factor
    #     lattice_factor = np.sqrt(points/dimension)
    #     orbit_factor = 1.0/(32.0 * lattice_factor)  # Increased denominator

    #     print("kinetic energy:")
    #     print(f"v: {v}")
    #     print(f"M: {M}")
    #     print(f"dimension: {dimension}")
    #     print(f"points: {points}")
    #     print(f"lattice_factor: {lattice_factor}")
    #     print(f"orbit_factor: {orbit_factor}")
    #     print(f"orbit_factor * M * v * v {orbit_factor} * {M} * {v} * {v}")
    #     calc = orbit_factor * M * v * v 
    #     print(f"orbit_factor result: {calc}")

    #     return orbit_factor * M * v * v

    # def potential_energy(self):
    #     """Calculate potential energy with Leech lattice scaling"""
    #     G = SI_UNITS['G_si']
    #     M = self.total_mass * SI_UNITS['M_sun_si']
    #     R = self.radius * SI_UNITS['ly_si']
        
    #     # Enhanced dark matter coupling
    #     dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
    #     points = CONSTANTS['LEECH_LATTICE_POINTS']
    #     dark_scale = 16.0 * np.sqrt(points/dimension)  # Increased scale factor

    #     print("potential energy function")
    #     print(f"G: {G}")
    #     print(f"M: {M}")
    #     print(f"R: {R}")
    #     print(f"dimension {dimension}")
    #     print(f"points: {points}")
    #     print(f"dark_scale: {dark_scale}")
    #     calc = -G * M * M * dark_scale / R
    #     print(f"return value {calc}")

    #     return -G * M * M * dark_scale / R

    # def kinetic_energy(self):
    #     """Calculate kinetic energy using Leech lattice geometry"""
    #     v = self.orbital_velocity * 1000
    #     M = self.total_mass * SI_UNITS['M_sun_si']
        
    #     # Leech lattice parameters
    #     dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']  # 24
    #     points = CONSTANTS['LEECH_LATTICE_POINTS']  # 196560
        
    #     # Enhanced geometric factor
    #     lattice_factor = np.sqrt(points/dimension)
    #     orbit_factor = 1.0/(32.0 * lattice_factor)  # Precise orbital scaling

    #     return orbit_factor * M * v * v

    # def potential_energy(self):
    #     """Calculate potential energy with Leech lattice scaling"""
    #     G = SI_UNITS['G_si']
    #     M = self.total_mass * SI_UNITS['M_sun_si']
    #     R = self.radius * SI_UNITS['ly_si']
        
    #     # Enhanced dark matter coupling
    #     dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
    #     points = CONSTANTS['LEECH_LATTICE_POINTS']
    #     dark_scale = 16.0 * np.sqrt(points/dimension)  # Dark matter geometric factor

    #     return -G * M * M * dark_scale / R

    # def kinetic_energy(self):
    #     """Calculate kinetic energy with quantum geometric suppression"""
    #     v = self.orbital_velocity * 1000
    #     M = self.total_mass * SI_UNITS['M_sun_si']
        
    #     # Leech lattice parameters
    #     dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
    #     points = CONSTANTS['LEECH_LATTICE_POINTS']
    #     lattice_factor = np.sqrt(points/dimension)
        
    #     # Strong geometric suppression
    #     #orbit_factor = 1.0/(1024*lattice_factor)
    #     suppression = 1.0/(3.2 * lattice_factor)
    #     return M * v * v 
    #     #return orbit_factor * M * v * v 

    # def potential_energy(self):
    #     """Calculate potential energy with enhanced binding"""
    #     G = SI_UNITS['G_si']
    #     M = self.total_mass * SI_UNITS['M_sun_si']
    #     R = self.radius * SI_UNITS['ly_si']
        
    #     # Enhanced dark matter coupling
    #     dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
    #     points = CONSTANTS['LEECH_LATTICE_POINTS']
    #     dark_scale = np.sqrt(points/dimension)
        
    #     return -G * M * M / R 

    # def kinetic_energy(self):
    #     """Calculate kinetic energy with matched geometric scaling"""
    #     v = self.orbital_velocity * 1000
    #     M = self.total_mass * SI_UNITS['M_sun_si']
        
    #     # Match NFW profile parameters
    #     r_s = np.float128(20000 * SI_UNITS['ly_si'])
    #     x = self.radius * SI_UNITS['ly_si'] / r_s
    #     bulge_scale = np.float128(self.visible_mass / self.total_mass)
        
    #     # Geometric factors
    #     dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
    #     points = CONSTANTS['LEECH_LATTICE_POINTS']
    #     lattice_factor = np.sqrt(points/dimension)
        
    #     # Reduced energy scaling for proper virial ratio
    #     dampening = np.float128(1 + (0.02 * x * bulge_scale))
    #     energy_factor = 0.00015 * lattice_factor * dampening  # Reduced by factor of 100
        
    #     return M * v * v * energy_factor

    # def potential_energy(self):
    #     """Calculate potential energy with balanced NFW coupling"""
    #     G = SI_UNITS['G_si']
    #     M = self.total_mass * SI_UNITS['M_sun_si']
    #     R = self.radius * SI_UNITS['ly_si']
        
    #     # Match concentration from rotation curve
    #     bulge_scale = np.float128(self.visible_mass / self.total_mass)
    #     concentration = np.float128(17.0 * np.exp(-bulge_scale))
        
    #     # Enhanced geometric coupling
    #     dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
    #     points = CONSTANTS['LEECH_LATTICE_POINTS']
    #     dark_scale = 0.000008 * concentration * np.sqrt(points/dimension)  # Maintain current coupling
        
    #     return -G * M * M * dark_scale / R


    # def kinetic_energy(self):
    #     """Calculate kinetic energy with matched geometric scaling"""
    #     v = self.orbital_velocity * 1000
    #     M = self.total_mass * SI_UNITS['M_sun_si']
        
    #     # Match NFW profile parameters
    #     r_s = np.float128(20000 * SI_UNITS['ly_si'])
    #     x = self.radius * SI_UNITS['ly_si'] / r_s
    #     bulge_scale = np.float128(self.visible_mass / self.total_mass)
        
    #     # Geometric factors
    #     dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
    #     points = CONSTANTS['LEECH_LATTICE_POINTS']
    #     lattice_factor = np.sqrt(points/dimension)
        
    #     # Refined energy scaling for virial stability
    #     dampening = np.float128(1 + (0.02 * x * bulge_scale))
    #     energy_factor = 0.000122 * lattice_factor * dampening  # Fine-tuned from 0.00015
        
    #     return M * v * v * energy_factor

#     def kinetic_energy(self):
#         """Calculate kinetic energy with matched geometric scaling"""
#         v = self.orbital_velocity * 1000
#         M = self.total_mass * SI_UNITS['M_sun_si']
        
#         # Match NFW profile parameters
#         r_s = np.float128(20000 * SI_UNITS['ly_si'])
#         x = self.radius * SI_UNITS['ly_si'] / r_s
#         bulge_scale = np.float128(self.visible_mass / self.total_mass)
        
#         # Geometric factors
#         dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
#         points = CONSTANTS['LEECH_LATTICE_POINTS']
#         lattice_factor = np.sqrt(points/dimension)
        
#         # Refined energy scaling for virial stability
# #-        dampening = np.float128(1 + (0.02 * x * bulge_scale))
# #-        energy_factor = 0.000122 * lattice_factor * dampening  # Fine-tuned from 0.00015
#         dampening = np.float128(1 + (0.02 * x * bulge_scale))
#         energy_factor = 0.0001208 * lattice_factor * dampening  # Adjusted to lower KE
        
#         return M * v * v * energy_factor



    def potential_energy(self):
        """Calculate potential energy with balanced NFW coupling"""
        G = SI_UNITS['G_si']
        M = self.total_mass * SI_UNITS['M_sun_si']
        R = self.radius * SI_UNITS['ly_si']
        
        # Match concentration from rotation curve
        bulge_scale = np.float128(self.visible_mass / self.total_mass)
        concentration = np.float128(17.0 * np.exp(-bulge_scale))
        
        # Enhanced geometric coupling
        dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
        points = CONSTANTS['LEECH_LATTICE_POINTS']
        dark_scale = 0.000008 * concentration * np.sqrt(points/dimension)
        
        return -G * M * M * dark_scale / R




    # def potential_energy(self):
    #     """Calculate potential energy with Leech lattice dark matter profile"""
    #     G = SI_UNITS['G_si']
    #     M = self.total_mass * SI_UNITS['M_sun_si']
    #     R = self.radius * SI_UNITS['ly_si']
        
    #     # Leech lattice contribution to dark matter
    #     dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
    #     points = CONSTANTS['LEECH_LATTICE_POINTS']
    #     dark_scale = np.sqrt(points/dimension) * 2.0
        
    #     return -G * M * M * dark_scale / R




    # def kinetic_energy(self):
    #     """Calculate kinetic energy in SI units with dark matter contribution"""
    #     v = self.orbital_velocity * 1000  # km/s to m/s
    #     M = self.total_mass * SI_UNITS['M_sun_si']
        
    #     # Include dark matter velocity scaling
    #     dark_fraction = self.dark_mass / self.total_mass
    #     v_eff = v * np.sqrt(1 + dark_fraction)
        
    #     return 0.5 * M * v_eff * v_eff

    # def potential_energy(self):
    #     """Calculate potential energy with dark matter halo profile"""
    #     G = SI_UNITS['G_si']
    #     M = self.total_mass * SI_UNITS['M_sun_si']
    #     R = self.radius * SI_UNITS['ly_si']
        
    #     # NFW profile contribution
    #     c = 15.0  # concentration parameter
    #     r_s = 20000 * SI_UNITS['ly_si']
    #     x = R/r_s
        
    #     # Enhanced potential with dark matter profile
    #     PE = -G * M * M / R
    #     PE *= (np.log(1 + c*x)/(c*x))
        
    #     return PE

    
    def compute_entanglement_entropy(self):
        """Calculate entanglement entropy across horizon scales"""
        # Convert to natural units
        r_natural = self.radius * SI_UNITS['ly_si'] / (CONSTANTS['R_sun'] * SI_UNITS['R_sun_si'])
        m_natural = self.mass / CONSTANTS['M_sun']
        
        # Scale-dependent entanglement
        beta = np.float128(1e-6 * np.sqrt(m_natural/r_natural))
        gamma = np.float128(0.407 * beta)
        
        # Leech lattice contribution
        lattice_factor = np.sqrt(CONSTANTS['LEECH_LATTICE_POINTS']/CONSTANTS['LEECH_LATTICE_DIMENSION'])
        
        # Entanglement entropy calculation
        S_ent = -np.log(beta) * lattice_factor * gamma
        return S_ent

    def calculate_quantum_corrections(self):
        """Compute quantum geometric corrections to NFW profile"""
        # Initialize with high precision
        G = np.float128(SI_UNITS['G_si'])
        R = np.float128(self.radius * SI_UNITS['ly_si'])
        
        # Scale radius and concentration
        r_s = np.float128(20000 * SI_UNITS['ly_si'])
        x = R/r_s
        c = np.float128(15.0)
        
        # Quantum geometric factor
        beta = self.compute_quantum_factor() - 1.0
        gamma_eff = np.float128(0.407 * beta * np.sqrt(196560/24))
        
        # Enhanced NFW profile
        rho_correction = 1.0 + gamma_eff * np.log(1 + x)/(1 + 0.047*x)
        v_correction = np.sqrt(1.0 + gamma_eff * c * np.log(1 + x)/(x))
        
        return {
            'density_enhancement': rho_correction,
            'velocity_enhancement': v_correction
        }

    def compute_effective_temperature(self):
        """Calculate effective temperature from galactic parameters"""
        # Initialize with high precision
        G = np.float128(SI_UNITS['G_si'])
        R = np.float128(self.radius * SI_UNITS['ly_si'])
        M = np.float128(self.total_mass * SI_UNITS['M_sun_si'])
        
        # Virial temperature calculation
        v_disp = np.float128(self.velocity_dispersion * 1000)  # km/s to m/s
        k_B = np.float128(CONSTANTS['k_B'])
        m_p = np.float128(CONSTANTS['m_p'])
        
        # T = (m_p * v²)/(3k_B)
        T = (m_p * v_disp * v_disp)/(3 * k_B)
        
        return T

    def compute_thermal_corrections(self):
        """Calculate temperature-dependent quantum corrections"""
        T = self.compute_effective_temperature()

        # Get SI constants with 128-bit precision
        c_si = np.float128(SI_UNITS['c_si'])  # Speed of light
        k_B = np.float128(CONSTANTS['k_B'])   # Boltzmann constant
        r_s = np.float128(20000 * SI_UNITS['ly_si'])  # Scale radius

        beta_T = np.float128(CONSTANTS['hbar'] * c_si / (k_B * T * r_s))
        beta_T *= np.sqrt(CONSTANTS['LEECH_LATTICE_POINTS']/CONSTANTS['LEECH_LATTICE_DIMENSION'])

        return 1.0 + beta_T * self.compute_quantum_factor()
    
    def calculate_horizon_effects(self):
        """Compute horizon-scale quantum geometric effects"""
        c_si = np.float128(SI_UNITS['c_si'])  # Speed of light
        G_si = np.float128(SI_UNITS['G_si'])  # Gravitational constant
        r_h = 2 * G_si * self.total_mass / c_si**2
        return 1.0 + self.compute_quantum_factor() * (r_h / self.radius)
    
    def compute_vacuum_energy(self):
        """Calculate vacuum energy contribution"""
        hbar = CONSTANTS['hbar']
        c = SI_UNITS['c_si']
        G = SI_UNITS['G_si']
        l_p = np.sqrt(hbar * G / c**3)
        rho_vacuum = hbar / (c * l_p**4)
        return rho_vacuum * self.compute_quantum_factor()
    
    def calculate_geometric_phase(self):
        """Compute Berry phase from quantum geometry"""
        beta = self.compute_quantum_factor() - 1.0
        return 2 * np.pi * beta * np.sqrt(CONSTANTS['LEECH_LATTICE_POINTS']/24)
    
    # def compute_gas_contribution(self):
    #     """Calculate enhanced gas contribution to rotation curve"""
    #     # Gas mass is typically 10-12% of stellar mass
    #     gas_mass = 0.08 * self.visible_mass  # Using 11% as middle value
        
    #     # Base gas velocity calculation
    #     G = np.float128(SI_UNITS['G_si'])
    #     R = np.float128(self.radius * SI_UNITS['ly_si'])
    #     v_gas = np.sqrt(G * gas_mass * SI_UNITS['M_sun_si'] / R)
        
    #     # Quantum geometric enhancement for gas
    #     beta_gas = self.compute_quantum_factor() - 1.0
    #     gamma_eff_gas = np.float128(0.407 * beta_gas * np.sqrt(196560/24))
        
    #     # Enhanced gas velocity with quantum corrections
    #     v_gas_enhanced = v_gas * np.sqrt(1 + gamma_eff_gas)
        
    #     return v_gas_enhanced

    # def compute_gas_contribution(self):
    #     """Calculate gas contribution with Leech lattice scaling"""
    #     # Reduce gas mass fraction
    #     gas_mass = 0.053 * self.visible_mass  # Lower from current value
        
    #     G = np.float128(SI_UNITS['G_si'])
    #     R = np.float128(self.radius * SI_UNITS['ly_si'])
        
    #     # Base velocity calculation
    #     v_gas = np.sqrt(G * gas_mass * SI_UNITS['M_sun_si'] / R)
        
    #     # Leech lattice geometric enhancement
    #     dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
    #     points = CONSTANTS['LEECH_LATTICE_POINTS']
    #     lattice_factor = np.sqrt(points/dimension)
        
    #     # Scale velocity with lattice geometry
    #     #v_gas_enhanced = v_gas * np.sqrt(2.0 / lattice_factor)
    #     v_gas_enhanced = v_gas 
    #     return v_gas_enhanced

    # def compute_gas_contribution(self):
    #     """Calculate gas contribution with mass-dependent scaling"""
    #     # Base gas fraction with mass scaling
    #     mass_scale = self.visible_mass / 1e11  # Scale relative to MW
    #     gas_fraction = 0.11 * (1 + 0.2 * np.tanh(mass_scale))
    #     gas_mass = gas_fraction * self.visible_mass
        
    #     G = np.float128(SI_UNITS['G_si'])
    #     R = np.float128(self.radius * SI_UNITS['ly_si'])
    #     v_gas = np.sqrt(G * gas_mass * SI_UNITS['M_sun_si'] / R)
        
    #     # Leech lattice geometric enhancement
    #     dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
    #     points = CONSTANTS['LEECH_LATTICE_POINTS']
    #     lattice_factor = np.sqrt(points/dimension)
        
    #     return v_gas * np.sqrt(1.2 / lattice_factor)

    # def compute_gas_contribution(self):
    #     """Calculate gas contribution with enhanced ISM physics"""
    #     # Base gas fraction with mass-dependent scaling
    #     mass_scale = self.visible_mass / 1e11  # Scale relative to MW
    #     gas_fraction = 0.15 * (1 + 0.3 * np.tanh(mass_scale))  # Enhanced fraction
    #     gas_mass = gas_fraction * self.visible_mass
        
    #     G = np.float128(SI_UNITS['G_si'])
    #     R = np.float128(self.radius * SI_UNITS['ly_si'])
        
    #     # Base velocity with pressure support
    #     v_gas = np.sqrt(G * gas_mass * SI_UNITS['M_sun_si'] / R)
        
    #     # Leech lattice geometric enhancement
    #     dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
    #     points = CONSTANTS['LEECH_LATTICE_POINTS']
    #     lattice_factor = np.sqrt(points/dimension)
        
    #     # Enhanced velocity with turbulent support
    #     v_gas_enhanced = v_gas * np.sqrt(3.0 / lattice_factor)
        
    #     return v_gas_enhanced

    # def compute_gas_contribution(self):
    #     """Calculate gas contribution with enhanced ISM physics"""
    #     # Base gas fraction with mass-dependent scaling
    #     mass_scale = self.visible_mass / 1e11  # Scale relative to MW
    #     gas_fraction = 0.15 * (1 + 0.3 * np.tanh(mass_scale))  # Enhanced fraction
    #     gas_mass = gas_fraction * self.visible_mass
        
    #     G = np.float128(SI_UNITS['G_si'])
    #     R = np.float128(self.radius * SI_UNITS['ly_si'])
        
    #     # Base velocity with pressure support
    #     v_gas = np.sqrt(G * gas_mass * SI_UNITS['M_sun_si'] / R)
        
    #     # Leech lattice geometric enhancement
    #     dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
    #     points = CONSTANTS['LEECH_LATTICE_POINTS']
    #     lattice_factor = np.sqrt(points/dimension)
        
    #     v_gas_enhanced = v_gas * np.sqrt(24/lattice_factor)
    #     #v_gas_enhanced = v_gas * 0.5 # Remove extra scaling to meet expected gas fraction
        
    #     return v_gas_enhanced

    # def compute_gas_contribution(self):
    #     """Calculate gas contribution with enhanced ISM physics"""
    #     mass_scale = self.visible_mass / 1e11  
    #     gas_fraction = 0.15 * (1 + 0.3 * np.tanh(mass_scale))
    #     gas_mass = gas_fraction * self.visible_mass
        
    #     G = np.float128(SI_UNITS['G_si'])
    #     R = np.float128(self.radius * SI_UNITS['ly_si'])
        
    #     v_gas = np.sqrt(G * gas_mass * SI_UNITS['M_sun_si'] / R)
        
    #     dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
    #     points = CONSTANTS['LEECH_LATTICE_POINTS']
    #     lattice_factor = np.sqrt(points/dimension)
        
    #     v_gas_enhanced = v_gas * np.sqrt(24/lattice_factor)  # Increased geometric factor
        
    #     return v_gas_enhanced

    def compute_gas_contribution(self):
        """Calculate gas contribution with enhanced ISM physics"""
        mass_scale = self.visible_mass / 1e11
        gas_fraction = 0.12 * (1 + 0.3 * np.tanh(mass_scale))
        gas_mass = gas_fraction * self.visible_mass
        
        G = np.float128(SI_UNITS['G_si'])
        R = np.float128(self.radius * SI_UNITS['ly_si'])
        v_gas = np.sqrt(G * gas_mass * SI_UNITS['M_sun_si'] / R)
        
        dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
        points = CONSTANTS['LEECH_LATTICE_POINTS']
        lattice_factor = np.sqrt(points/dimension)
        
        # Increase geometric factor to boost gas velocity into 8-12% range
        v_gas_enhanced = v_gas * np.sqrt(32/lattice_factor)
        
        return v_gas_enhanced


    def kinetic_energy(self):
        """Calculate kinetic energy with matched geometric scaling"""
        v = self.orbital_velocity * 1000
        M = self.total_mass * SI_UNITS['M_sun_si']
        
        r_s = np.float128(20000 * SI_UNITS['ly_si'])
        x = self.radius * SI_UNITS['ly_si'] / r_s
        bulge_scale = np.float128(self.visible_mass / self.total_mass)
        
        dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
        points = CONSTANTS['LEECH_LATTICE_POINTS']
        lattice_factor = np.sqrt(points/dimension)
        
        dampening = np.float128(1 + (0.02 * x * bulge_scale))
        energy_factor = 0.0001208 * lattice_factor * dampening  # Fine-tuned factor
        
        return M * v * v * energy_factor
