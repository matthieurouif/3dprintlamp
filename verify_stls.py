import numpy as np
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool

import trimesh

def verify_stl(filename, expected_socket_r):
    print(f"\nVerifying {filename}...")
    mesh = trimesh.load(filename)
    
    # 1. Height
    z_min, z_max = mesh.bounds[0][2], mesh.bounds[1][2]
    height = z_max - z_min
    print(f"  Height: {height:.2f}mm (Expected: 120.00)")
    
    # 2. Socket Hole Diameter (Checking at the top z=120)
    # We slice at z=115 to check the hole
    slice_z = 115.0
    plane_origin = [0, 0, slice_z]
    plane_normal = [0, 0, 1]
    slice_mesh = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    
    if slice_mesh:
        # The slice will have multiple loops. The innermost one should be the socket.
        # We can look for the smallest radius
        r_inner = np.min(np.linalg.norm(slice_mesh.vertices[:, :2], axis=1))
        print(f"  Inner Socket Radius: {r_inner:.2f}mm (Expected: ~{expected_socket_r:.2f})")
    else:
        print("  Could not slice at z=115")

    # 3. Socket Holder Depth
    # Slice at z=90 to see if the holder is still there
    slice_z_deep = 90.0
    slice_deep = mesh.section(plane_origin=[0, 0, slice_z_deep], plane_normal=[0, 0, 1])
    if slice_deep:
         r_deep = np.min(np.linalg.norm(slice_deep.vertices[:, :2], axis=1))
         print(f"  Inner Radius at z=90: {r_deep:.2f}mm (Expected: ~{expected_socket_r:.2f})")
    else:
         print(f"  No geometry at z=90 (Holder depth issue?)")

    # 4. Truncation Verification
    # Slice at z=80 to see if the holder is GONE
    slice_z_truncated = 80.0
    slice_trunc = mesh.section(plane_origin=[0, 0, slice_z_truncated], plane_normal=[0, 0, 1])
    if slice_trunc:
         r_trunc = np.min(np.linalg.norm(slice_trunc.vertices[:, :2], axis=1))
         if r_trunc < 15.0: # If we see small radius, it means holder is still there
             print(f"  FAILURE: Inner radius at z=80 is {r_trunc:.2f}mm (Should be empty or >30mm)")
         else:
             print(f"  SUCCESS: No internal holder at z=80 (Inner R: {r_trunc:.2f}mm)")
    else:
         print(f"  SUCCESS: No geometry at z=80 (Empty base shell is expected to be single layer or absent in section if not closed)")

verify_stl('functional_base.stl', 9.0)
verify_stl('functional_base_e14.stl', 12.5)
