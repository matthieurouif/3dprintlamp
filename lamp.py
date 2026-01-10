import numpy as np
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool

from stl import mesh
import trimesh
from text_generator import text_to_3d_mesh, warp_text_to_cylinder_ledge

# --- HELPER FUNCTIONS ---

def to_cartesian(r, theta, z):
    """Converts cylindrical coordinates to cartesian."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y, z))

def create_faces_from_grid(num_layers, num_radial, start_idx=0, flip_normals=False):
    """Generates triangular faces from a grid of vertices."""
    faces = []
    for i in range(num_layers - 1):
        for j in range(num_radial):
            # Indices for the four corners of a quad
            p1 = start_idx + i * num_radial + j
            p2 = start_idx + i * num_radial + (j + 1) % num_radial
            p3 = start_idx + (i + 1) * num_radial + (j + 1) % num_radial
            p4 = start_idx + (i + 1) * num_radial + j
            
            if flip_normals:
                # Clockwise winding for inner walls
                faces.append([p1, p3, p2])
                faces.append([p1, p4, p3])
            else:
                # Counter-clockwise winding for outer walls
                faces.append([p1, p2, p3])
                faces.append([p1, p3, p4])
    return np.array(faces)

def generate_brim_mesh(inner_radius, outer_radius, thickness=0.4, num_radial=180, z_offset=0):
    """
    Generates a thin disk (brim) to help with bed adhesion.
    """
    theta_vals = np.linspace(0, 2 * np.pi, num_radial, endpoint=False)
    
    verts = []
    # Bottom vertices
    for t in theta_vals:
        verts.append([inner_radius * np.cos(t), inner_radius * np.sin(t), z_offset])
    for t in theta_vals:
        verts.append([outer_radius * np.cos(t), outer_radius * np.sin(t), z_offset])
    # Top vertices
    for t in theta_vals:
        verts.append([inner_radius * np.cos(t), inner_radius * np.sin(t), z_offset + thickness])
    for t in theta_vals:
        verts.append([outer_radius * np.cos(t), outer_radius * np.sin(t), z_offset + thickness])
        
    verts = np.array(verts)
    faces = []
    
    # helper for quads
    def add_quad(v1, v2, v3, v4):
        faces.append([v1, v2, v3])
        faces.append([v1, v3, v4])

    N = num_radial
    for j in range(N):
        j_next = (j + 1) % N
        
        # Bottom face (normal down)
        add_quad(j, j + N, j_next + N, j_next)
        # Top face (normal up)
        add_quad(j + 2*N, j_next + 2*N, j_next + 3*N, j + 3*N)
        # Inner wall (normal inward)
        add_quad(j, j_next, j_next + 2*N, j + 2*N)
        # Outer wall (normal outward)
        add_quad(j + N, j + 3*N, j_next + 3*N, j_next + N)
        
    data = np.zeros(len(faces), dtype=mesh.Mesh.dtype)
    for i, f in enumerate(faces):
        for k in range(3):
            data['vectors'][i][k] = verts[f[k]]
            
    return mesh.Mesh(data)

def generate_revolved_mesh(
    height, layers, num_radial, profile_func, 
    num_ribs=0, rib_depth=0, wall_thickness=0, smooth_inner=True,
    inner_profile_func=None,
    hole_mask_func=None  # New: Function (theta, z) -> bool
):
    """
    Generates a revolved mesh. 
    Supports watertight hole generation if hole_mask_func provided.
    """
    z_vals = np.linspace(0, height, layers)
    theta_vals = np.linspace(0, 2 * np.pi, num_radial, endpoint=False) # looped
    
    # Create grids
    theta_grid, z_grid = np.meshgrid(theta_vals, z_vals)
    z_norm_grid = z_grid / height
    
    # 1. Calculate Radii (Outer and Inner)
    base_r = profile_func(z_norm_grid)
    taper = np.sin(z_norm_grid * np.pi) ** 0.5
    r_outer = base_r + (rib_depth * taper * np.cos(num_ribs * theta_grid))
    
    if inner_profile_func:
        r_inner = inner_profile_func(z_norm_grid)
    elif wall_thickness > 0:
        r_inner_base = base_r - wall_thickness
        if smooth_inner:
            r_inner = r_inner_base
        else:
            r_inner = r_inner_base + (rib_depth * taper * np.cos(num_ribs * theta_grid))
        r_inner = np.maximum(r_inner, 0.1)
    else:
        r_inner = None # Single surface if no thickness

    # 2. Compute Vertices
    # Shape: (layers, num_radial, 3)
    # We flatten them later, but keep grid structure for indexing
    
    def get_verts(r, t, z):
        x = r * np.cos(t)
        y = r * np.sin(t)
        return np.stack([x, y, z], axis=-1)

    verts_outer = get_verts(r_outer, theta_grid, z_grid)
    verts_inner = get_verts(r_inner, theta_grid, z_grid) if r_inner is not None else None
    
    # 3. Hole Mask Logic
    # mask[i, j] = True if inside hole (SKIP geometry)
    mask = np.zeros(theta_grid.shape, dtype=bool)
    if hole_mask_func:
        # Evaluate mask for each point
        # A face is a hole if ALL its vertices are in the hole? 
        # Or if the centroid is? 
        # Better: mask vertices. If a face has ANY masked vertex, it's affected.
        # Strict: If all 4 corners of a quad are in hole, skip face.
        # Boundary: If some are in, some out -> we need to bridge?
        # Simplest grid approach:
        # Evaluated at vertices.
        for i in range(layers):
            for j in range(num_radial):
                mask[i, j] = hole_mask_func(theta_vals[j], z_vals[i])

    faces = []

    # Helper to add quad (split to 2 tris)
    # n_rad is num_radial
    def add_quad(v1, v2, v3, v4):
        faces.append([v1, v2, v3])
        faces.append([v1, v3, v4])

    N_rad = num_radial
    N_lay = layers
    
    # Offsets for vertex indices
    # Outer vertices: 0 to N_lay*N_rad - 1
    # Inner vertices: N_lay*N_rad to 2*N_lay*N_rad - 1
    offset_inner = N_lay * N_rad

    # 4. Generate Body Faces (Outer and Inner)
    for i in range(N_lay - 1):
        for j in range(N_rad):
            j_next = (j + 1) % N_rad
            
            # Indices in the grid
            idx_curr = i * N_rad + j
            idx_right = i * N_rad + j_next
            idx_up = (i + 1) * N_rad + j
            idx_up_right = (i + 1) * N_rad + j_next
            
            # Check mask
            # If ANY vertex of the quad is in the hole, we skip the SURFACE face?
            # Or if ALL?
            # User wants a hole. 
            # If we skip if ALL are in hole, we get ragged edges.
            # Let's say: If the CENTROID is in hole.
            # Or simple: if mask[i,j] is True, current vertex is deleted.
            # If a quad references a deleted vertex, it can't exist?
            # Standard approach:
            # cell (i,j) is 'active' if !mask[i,j].
            # Actually, standard is mask is on cells.
            # Let's assess mask at cell center?
            
            z_c = (z_vals[i] + z_vals[i+1]) / 2
            # Angle center. Handle wrap
            t1 = theta_vals[j]
            t2 = theta_vals[j_next]
            if t2 < t1: t2 += 2*np.pi
            t_c = (t1 + t2) / 2
            
            is_hole = False
            if hole_mask_func:
                is_hole = hole_mask_func(t_c, z_c) # Using mask on FACES (Cells)
            
            if not is_hole:
                # Add Outer Face
                # CCW winding
                add_quad(idx_curr, idx_right, idx_up_right, idx_up)
                
                # Add Inner Face (if exists)
                if verts_inner is not None:
                    # Invert winding for inner surface (CW)
                    # Vertices are shifted by offset_inner
                    i1 = idx_curr + offset_inner
                    i2 = idx_right + offset_inner
                    i3 = idx_up_right + offset_inner
                    i4 = idx_up + offset_inner
                    pass
                    # Quad: 1-4-3-2 (CW)
                    add_quad(i1, i4, i3, i2)

    # 5. Generate Bridge Walls (Hole Tunneling)
    # We iterate boundaries of cells.
    # Vertical boundaries: between (i, j) and (i, j+1)
    # Horizontal boundaries: between (i, j) and (i+1, j)
    
    if verts_inner is not None and hole_mask_func:
        # Loop over all potential edges
        
        # Horizontal Edges (between Layer i and i+1)
        for i in range(N_lay - 1):
            for j in range(N_rad):
                # Cell Below: (i, j)
                # Cell Above: (i+1, j) -- wait, grid logic
                # Face (i,j) is bounded by ring i and ring i+1.
                # So "Horizontal" neighbors are (i,j) and (i-1,j).
                pass
        
        # Simpler: Iterate all cells. 
        # If Cell(i,j) is VOID and Neighbor is SOLID -> Build Wall.
        
        def is_cell_hole(i, j):
            if i < 0 or i >= N_lay - 1: return True # Outside bounds treated as void/open? No, top/bottom closed separately.
            # Actually bounds are rims.
            # Return True if hole mask.
            # Recalculate center
            z_c = (z_vals[i] + z_vals[i+1]) / 2
            t1 = theta_vals[j]
            t2 = theta_vals[(j+1)%N_rad]
            if t2 < t1: t2 += 2*np.pi
            t_c = (t1 + t2) / 2
            return hole_mask_func(t_c, z_c)

        for i in range(N_lay - 1):
            for j in range(N_rad):
                curr_hole = is_cell_hole(i, j)
                
                # Check Right Neighbor (Mod N_rad)
                j_next = (j + 1) % N_rad
                next_hole = is_cell_hole(i, j_next)
                
                # If one is hole and other is solid -> Vertical Wall (along grid vertical line)
                if curr_hole != next_hole:
                    # Edge vertices are at j_next (shared edge)
                    # Vertices: bottom=(i, j_next), top=(i+1, j_next)
                    
                    idx_bott = i * N_rad + j_next
                    idx_top = (i + 1) * N_rad + j_next
                    
                    idx_bott_in = idx_bott + offset_inner
                    idx_top_in = idx_top + offset_inner
                    
                    # Determine winding
                    # If Curr is Hole, Next is Solid: We are "entering" solid from left.
                    # Normal points towards Hole (into void).
                    # Actually normal points OUT of solid.
                    # Solid is on Right.
                    # Face normal should point LEFT (negative theta).
                    
                    if curr_hole and not next_hole:
                        # Solid on right. Normal points -Theta.
                        # Outer(bott) -> Outer(top) -> Inner(top) -> Inner(bott) ??
                        # Let's visualize.
                        # Wall connecting Outer to Inner.
                        add_quad(idx_bott, idx_bott_in, idx_top_in, idx_top)
                    else:
                        # Solid on left. Normal points +Theta.
                        add_quad(idx_bott, idx_top, idx_top_in, idx_bott_in)

                # Check Top Neighbor
                if i < N_lay - 2: # Don't check top boundary of mesh yet
                    idx_above_i = i + 1
                    above_hole = is_cell_hole(idx_above_i, j)
                    
                    if curr_hole != above_hole:
                        # Edge vertices are at ring i+1
                        # Vertices: left=(i+1, j), right=(i+1, j+1)
                        j_next = (j + 1) % N_rad
                        
                        idx_left = (i + 1) * N_rad + j
                        idx_right = (i + 1) * N_rad + j_next
                        
                        idx_left_in = idx_left + offset_inner
                        idx_right_in = idx_right + offset_inner
                        
                        if curr_hole and not above_hole:
                            # Solid above. Normal points Down (-Z).
                            add_quad(idx_left, idx_right, idx_right_in, idx_left_in)
                        else:
                            # Solid below. Normal points Up (+Z).
                            add_quad(idx_left, idx_left_in, idx_right_in, idx_right)

    # 6. Rims (Top and Bottom) - Must respect holes
    # If a cell touching top/bottom is a hole, we shouldn't cap it?
    # Or assuming holes don't touch top/bottom.
    # Our hole is z=10. Base 0..150. Safe.
    
    if verts_inner is not None:
        # Bottom Rim (Layer 0)
        # Connect Outer(0, j) to Inner(0, j)
        for j in range(N_rad):
            # Check if cell (0, j) is hole? No, rim is boundary.
            j_next = (j + 1) % N_rad
            # Normal down
            add_quad(j, j_next, j_next + offset_inner, j + offset_inner)
            
        # Top Rim (Layer N-1)
        # Connect Outer(N-1, j) to Inner
        base_idx = (N_lay - 1) * N_rad
        for j in range(N_rad):
            j_next = (j + 1) % N_rad
            i1 = base_idx + j
            i2 = base_idx + j_next
            i3 = i2 + offset_inner
            i4 = i1 + offset_inner
            # Normal up
            add_quad(i1, i4, i3, i2)

    # Convert numpy-stl mesh to trimesh
    # Faces list of lists -> numpy array
    faces = np.array(faces)
    
    verts_flat = verts_outer.reshape(-1, 3)
    if verts_inner is not None:
        verts_inner_flat = verts_inner.reshape(-1, 3)
        all_verts = np.concatenate([verts_flat, verts_inner_flat])
    else:
        all_verts = verts_flat
        
    # Create trimesh object
    tmesh = trimesh.Trimesh(vertices=all_verts, faces=faces)
    
    return tmesh

# --- CONSTANTS ---
BASE_HEIGHT = 120
SHADE_HEIGHT = 120
COLLAR_RADIUS = 29.0
LEDGE_RADIUS = 32.0

def base_hole_mask(theta, z):
    """
    Defines the cable hole at the back bottom.
    Hole at z=10, Back (Theta ~ Pi).
    Diameter 8mm (Radius 4mm).
    """
    z_center = 10.0
    r_hole = 4.0 # 8mm Diameter
    
    dz = z - z_center
    if abs(dz) > r_hole + 1.0: return False
    
    d_theta = theta - np.pi
    # Normalize
    d_theta = (d_theta + np.pi) % (2*np.pi) - np.pi
    
    # We estimate local radius at z=10. Bulbous part is roughly 35-75. 
    # At z=10 it's around 40-50.
    r_local = 45.0 
    
    # Arc length
    arc = r_local * d_theta
    dist_sq = dz**2 + arc**2
    return dist_sq < (r_hole**2)

def ribbed_shade_profile(z_norm):
    """The lantern shape for the shade (Outer Surface)."""
    # Starts at 31, Peak 85, Ends 31
    r_main = 31.0 + 54.0 * np.sin(z_norm * np.pi)
    
    # Blend bottom
    mask_bottom = z_norm < 0.05
    blend = z_norm / 0.05
    r_bottom = 31.0 * (1 - blend) + r_main * blend
    r = np.where(mask_bottom, r_bottom, r_main)
    return r

def cylinder_shade_inner_profile(z_norm):
    """A straight cylinder for the inner wall."""
    return np.full_like(z_norm, 29.5)

# --- REFINED BASE GENERATION ---

def generate_lamp_base(socket_radius, output_filename, deboss_text=None):
    """
    Generates a lamp base with specified socket radius and optional debossed text.
    """
    print(f"Generating {output_filename} (Socket R={socket_radius})...")
    
    # 1. Generate the Main Shell (Outer Body)
    # We use a slightly modified profile that stops exactly at the ledge height
    LEDGE_HEIGHT = BASE_HEIGHT - 12.0 # 108mm
    
    def shell_profile(z_norm):
        # z_norm is relative to total height (120)
        # But we only want the bulbous part up to 108
        z = z_norm * BASE_HEIGHT
        
        # Original logic shifted to fit 108
        z_rel = np.clip(z / LEDGE_HEIGHT, 0, 1)
        z_inverted = 1.0 - z_rel
        k = 0.5
        r_body_original = 35 + 40 * np.sin(np.power(z_inverted, k) * np.pi)
        r_body_base = r_body_original * 0.82857
        
        # Transition to LEDGE_RADIUS at the top
        blend_top = np.power(z_rel, 8)
        r_bulbous = r_body_base * (1 - blend_top) + LEDGE_RADIUS * blend_top
        
        return np.where(z > LEDGE_HEIGHT, COLLAR_RADIUS, r_bulbous)

    def shell_inner_profile(z_norm):
        r_outer = shell_profile(z_norm)
        return np.maximum(r_outer - 3.0, 1.0)

    # Base Body Mesh
    base_shell = generate_revolved_mesh(
        height=BASE_HEIGHT,
        layers=150,
        num_radial=360,
        profile_func=shell_profile,
        num_ribs=45,
        rib_depth=1.0,
        inner_profile_func=shell_inner_profile,
        hole_mask_func=base_hole_mask
    )
    
    # 2. Generate the Socket Holder (Ring)
    # Height: 22mm (from 98 to 120)
    # Top 12mm is the Collar (R=29)
    # Bottom 10mm is the extra extension (Thickness 10mm)
    HOLDER_HEIGHT = 22.0
    
    def holder_outer_profile_truncated(z_norm):
        z = z_norm * HOLDER_HEIGHT
        # z in [0, 22]. 0 is bottom (Z=98), 22 is top (Z=120)
        # z > 10 (which is Z > 108) -> Collar
        return np.where(z > 10, COLLAR_RADIUS, socket_radius + 10.0)

    def holder_inner_profile_truncated(z_norm):
        return np.full_like(z_norm, socket_radius)

    socket_holder = generate_revolved_mesh(
        height=HOLDER_HEIGHT,
        layers=40, # Adjusted layers for shorter height
        num_radial=180,
        profile_func=holder_outer_profile_truncated,
        inner_profile_func=holder_inner_profile_truncated
    )
    # Translate to top
    socket_holder.vertices[:, 2] += (BASE_HEIGHT - HOLDER_HEIGHT)
    
    # 3. Combine and Deboss
    # Use concatenate instead of union to avoid boolean errors with non-manifold shell
    
    # Deboss text on the top disk (at Z=120)
    if deboss_text:
        print(f"Applying deboss: '{deboss_text}'")
        # Ring at top is from socket_radius to COLLAR_RADIUS (29.0)
        # Center the text in this ring
        font_size = 5.0
        text_mesh = text_to_3d_mesh(deboss_text, font_size=font_size, extrusion_height=1.5)
        
        # Calculate radius to center the text
        # If invert_radial=True, R = radius - Y_radial. 
        # Range is [radius - font_size, radius]
        # Center is radius - font_size/2
        # We want radius - font_size/2 = (COLLAR_RADIUS + socket_radius) / 2
        target_radius = (COLLAR_RADIUS + socket_radius) / 2.0 + (font_size / 2.0)
        
        text_mesh = warp_text_to_cylinder_ledge(
            text_mesh, 
            radius=target_radius,
            base_height=119.5, # Slightly below surface to ensure intersection/cut
            invert_radial=True,
            scale_x=1.0
        )
        
        try:
            # Note: Subtract from socket_holder as it forms the top surface
            socket_holder_processed = socket_holder.process(validate=True)
            text_mesh_processed = text_mesh.process(validate=True)
            socket_holder_with_text = socket_holder_processed.difference(text_mesh_processed)
            final_base = trimesh.util.concatenate([base_shell, socket_holder_with_text])
        except Exception as e:
            print(f"Deboss failed, using union to visualize position: {e}")
            final_base = trimesh.util.concatenate([base_shell, socket_holder, text_mesh])
    else:
        final_base = trimesh.util.concatenate([base_shell, socket_holder])

    # Save
    final_base.export(output_filename)
    print(f"Saved {output_filename}")

# --- MAIN EXECUTION ---

# 1. Generate G9 Base (18mm hole -> 9mm radius)
generate_lamp_base(9.0, 'functional_base.stl', deboss_text="Made with ❤️ in Paris")

# 2. Generate E14 Base (25mm hole -> 12.5mm radius)
generate_lamp_base(12.5, 'functional_base_e14.stl', deboss_text="Made with ❤️ in Paris")

# 3. Generate the Shade (Update to return trimesh and save)
print("\nGenerating Shade...")
shade_mesh = generate_revolved_mesh(
    height=SHADE_HEIGHT,
    layers=120,
    num_radial=180,
    profile_func=ribbed_shade_profile,
    num_ribs=45,
    rib_depth=3.5,
    wall_thickness=0,
    inner_profile_func=cylinder_shade_inner_profile
)

# Add a base disk to the shade for stability
# This disk extends from center to the outer edge at z=0
print("Adding stability disk to shade base...")
base_disk_outer_radius = 31.5  # Slightly larger than the shade bottom (31mm)
base_disk_thickness = 2.0  # 2mm thick disk

base_disk = generate_brim_mesh(
    inner_radius=0.1,  # Nearly solid (small center hole for manifold mesh)
    outer_radius=base_disk_outer_radius,
    thickness=base_disk_thickness,
    num_radial=180,
    z_offset=0
)

# Convert numpy-stl mesh to trimesh
base_disk_trimesh = trimesh.Trimesh(
    vertices=base_disk.vectors.reshape(-1, 3),
    faces=np.arange(len(base_disk.vectors) * 3).reshape(-1, 3)
)

# Combine shade and base disk
shade_with_base = trimesh.util.concatenate([shade_mesh, base_disk_trimesh])

shade_with_base.export('diffusion_shade.stl')
shade_with_base.export('lamp_shade.stl')
print("Saved shades with stability disk.")

print("\nDone!")