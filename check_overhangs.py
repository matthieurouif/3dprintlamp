import numpy as np
import matplotlib.pyplot as plt

# Re-implement the profile logic from lamp.py
def bulbous_base_profile(z_norm):
    COLLAR_START_Z = 0.90
    z_body = np.clip(z_norm / COLLAR_START_Z, 0, 1)
    
    # Invert
    z_inverted = 1.0 - z_body
    k = 0.5 
    
    # Base Curve (Scaled by 0.8)
    r_body_original = 35 + 40 * np.sin(np.power(z_inverted, k) * np.pi)
    r_body_base = r_body_original * 0.8
    
    # Blend Ledge
    LEDGE_RADIUS = 38.0
    blend_top = np.power(z_body, 8) 
    r_body = r_body_base * (1 - blend_top) + LEDGE_RADIUS * blend_top
    
    mask_neck = z_norm > COLLAR_START_Z
    COLLAR_RADIUS = 29.0
    r = np.where(mask_neck, COLLAR_RADIUS, r_body)
    return r

def analyze():
    BASE_HEIGHT = 120.0
    z_mm = np.linspace(0, BASE_HEIGHT, 1000)
    z_norm = z_mm / BASE_HEIGHT
    
    r_mm = bulbous_base_profile(z_norm)
    
    # Calculate gradients (dr/dz)
    dr = np.diff(r_mm)
    dz = np.diff(z_mm)
    slope = dr / dz
    
    # Angle from Horizontal (Overhang Angle)
    # 0 deg = Horizontal (Extreme Overhang)
    # 90 deg = Vertical (Safe)
    # Standard safe printing is > 45 deg.
    # Bambu can do > 25-30 deg.
    
    # Calculate angle from horizontal
    angles_deg = np.degrees(np.arctan2(dz, np.abs(dr)))
    
    # Find worst offenders (excluding the flat collar top which is vertical/horizontal explicitly)
    # We care about the "widening" parts (dr > 0).
    # If dr < 0 (narrowing), it's self-supporting.
    
    widening_mask = dr > 0.05 # filtering noise
    
    if np.any(widening_mask):
        worst_angle = np.min(angles_deg[widening_mask])
        worst_z_idx = np.argmin(angles_deg[widening_mask])
        # Mapping back to original z is tricky with boolean masking
        # Let's just iterate
        pass
    
    print(f"--- Overhang Analysis ---")
    min_angle = 90.0
    critical_z = 0
    
    for i in range(len(slope)):
        if dr[i] > 0.001: # Expanding outwards
            # Angle from horizontal
            angle = np.degrees(np.arctan2(dz[i], dr[i]))
            if angle < min_angle:
                min_angle = angle
                critical_z = z_mm[i]
                
    print(f"Minimum Overhang Angle (from horizontal): {min_angle:.2f} degrees")
    print(f"At Height Z = {critical_z:.2f} mm")
    
    if min_angle < 30:
        print("WARNING: Steep overhangs detected. Supports likely needed.")
    elif min_angle < 45:
        print("CAUTION: Overhangs between 30-45 deg. Bambu should handle it, but check layers.")
    else:
        print("SAFE: All angles > 45 degrees.")

if __name__ == "__main__":
    analyze()
