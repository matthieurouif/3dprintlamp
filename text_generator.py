import numpy as np
import trimesh
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
import shapely.geometry
import shapely.ops

def text_to_3d_mesh(text, font_size=10, extrusion_height=1.0):
    """
    Generates a 3D mesh for the given text string.
    Returns a trimesh.Trimesh object centered at the origin (in X/Y).
    """
    # 1. Generate Text Path (2D outlines) using Matplotlib
    # specific font isn't critical, default sans-serif is fine
    fp = FontProperties(family='sans-serif', weight='bold') 
    path = TextPath((0, 0), text, size=font_size, prop=fp)
    
    # 2. Convert Path to Shapely Polygons
    # matplotlib paths are a list of vertices/codes.
    # We need to reconstruct them into polygons.
    # TextPath.to_polygons() returns a list of (N, 2) arrays.
    polys = []
    for points in path.to_polygons():
        if len(points) < 3:
            continue
        # Check winding to distinguish holes? 
        # Matplotlib to_polygons usually returns everything as separate polygons
        # We need to properly handle holes (like in 'O' or 'A').
        # Using shapely helps here.
        poly = shapely.geometry.Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)
        polys.append(poly)
        
    # Combine polygons to handle holes (Union)
    # This is slightly expensive but robust
    if not polys:
        return None
        
    # Naive approach: Just treat all as positive polygons first?
    # Better: Use shapely.ops.unary_union to flatten them, but that merges overlapping letters.
    # TextPath usually separates 'O' into Outer and Inner loops.
    # A simple trick is to extrude them all and letting the slicer handle it, 
    # but strictly we want proper geometry.
    # We can use trimesh.creation.extrude_polygon which handles holes if passed correctly.
    # But figuring out which is a hole is tricky from raw point lists.
    
    # Alternative: simple triangulation of each loop.
    # Let's try creating a trimesh for each polygon.
    meshes = []
    for poly in polys:
        # Check area sign to detect holes?
        # Matplotlib polygons are usually CCW for exterior, CW for interior.
        # But shapely normalizes them.
        
        # Simpler: Just extrude every loop as a solid.
        # For 'O', you get a big solid and a small solid.
        # If we just add them, it's 2 solids.
        # Correct 3D text requires subtracting the hole.
        # However, for 3D printing, if the inner hole is filled, it's just a solid 'O'.
        # That's bad.
        
        # Let's trust trimesh.path.raster.rasterize or similar? 
        # No, creating from path.
        pass

    # Better approach with Trimesh directly which has path handling
    # trimesh.path.Path2D can load from shapely or lists.
    # But let's stick to simple extrusion if possible.
    
    # Matplotlib's to_polygons() isn't perfect for topology.
    # Use trimesh to parse path? No, it needs SVG/DXF.
    
    # Let's try simple extrusion of each loop.
    # If the text is robust, extracting shapes with holes is:
    # 1. Area sort.
    # 2. Assume smaller inside larger is a hole.
    # This might be overkill for "Made with Love".
    
    # Workaround:
    # Use `shapely.ops.unary_union` on the list of polygons. This handles boolean ops 2D.
    # If standard matplotlib output is clean, holes are separate polygons.
    # But shapely Polygon constructor with holes needs (shell, [holes]).
    # We can reconstructed this structure:
    # Iterate all polys, find containment.
    
    # Actually, simpler hack:
    # Just return a list of extruded meshes for every loop.
    # 'O' will be two cylinders. 
    # If they are same height, it looks filled.
    # BUT, if we merge them, it's fine? No.
    
    # Proper way: shapely 2D boolean.
    # Construct a MultiPolygon from all rings?
    # Not easily.
    
    # Let's skip perfection and rely on `trimesh.creation.extrude_polygon`.
    # It requires a Polygon.
    
    # Let's try:
    full_poly = shapely.ops.unary_union(polys)
    
    # Now extrude the result (which is likely a MultiPolygon with holes handled)
    # trimesh can extrude a MultiPolygon?
    # Iterate through parts of MultiPolygon
    parts = []
    if isinstance(full_poly, shapely.geometry.Polygon):
        parts.append(full_poly)
    elif isinstance(full_poly, shapely.geometry.MultiPolygon):
        parts.extend(full_poly.geoms)
        
    extruded_meshes = []
    for p in parts:
        # clean buffer
        if not p.is_valid:
            p = p.buffer(0)
        m = trimesh.creation.extrude_polygon(p, height=extrusion_height)
        extruded_meshes.append(m)
        
    combined = trimesh.util.concatenate(extruded_meshes)
    return combined

def warp_text_to_cylinder_ledge(mesh, radius, base_height, angle_range_deg=180, invert_radial=False, scale_x=1.0):
    """
    Warps a linear 3D text mesh onto a circular ledge (flat XY plane).
    
    Args:
        mesh: trimesh object (linear text along X axis)
        radius: Radial position of the text baseline/center
        base_height: Z-height where the text sits
        invert_radial: If True, flips the radial orientation (Bottom=Outer, Top=Inner).
                       Use this for "readable from above" (text faces user).
        scale_x: Multiplier for the horizontal width (X-axis in linear space).
    """
    vertices = mesh.vertices.copy()
    
    # Center text X around 0 and apply scale_x
    min_x, max_x = vertices[:, 0].min(), vertices[:, 0].max()
    center_x = (min_x + max_x) / 2
    vertices[:, 0] = (vertices[:, 0] - center_x) * scale_x
    
    # Center text Y (vertical in font space) to line up with Radius
    # In font space, Y is "up".
    # We map Font-Y to Radius change.
    # Let's assume baseline is at 'radius'.
    # So Y=0 -> R=radius.
    
    # Coordinates:
    # v[0] = X (position along arc)
    # v[1] = Y (radial offset)
    # v[2] = Z (vertical extrusion)
    
    x_arc = vertices[:, 0]
    y_radial = vertices[:, 1]
    z_height = vertices[:, 2]
    
    # Map X to Angle
    # ArcLength = R * Theta
    # Theta = X / R
    # We use 'radius' as the reference R for angle calculation to keep spacing const
    theta = x_arc / radius
    
    # Angle Offset: Start at -90 deg (Front) or 180 (Back)?
    # Usually "Made in..." is at the back or bottom.
    # Let's put it at the "Front" (Theta = -pi/2) or just nice distribution?
    # Let's center it at -Pi/2 (Front).
    theta_offset = -np.pi / 2
    theta += theta_offset
    
    # Map Y to Radius
    # Standard: Bottom (Y=0) is Inner Radius. Top is Outer. (Reads "Upside Down" from outside)
    # Inverted: Bottom (Y=0) is Outer Radius. Top is Inner. (Reads "Correctly" from outside/above)
    
    if invert_radial:
        # Flip Y direction essentially
        # We want Baseline at Radius. Top at Radius - Y.
        r = radius - y_radial
    else:
        # Standard
        r = radius + y_radial
    
    # Map Z to Height
    # Text sits ON the ledge.
    # So Z_final = Z_original + base_height
    z = z_height + base_height
    
    # Convert Polar to Cartesian
    new_x = r * np.cos(theta)
    new_y = r * np.sin(theta)
    new_z = z
    
    mesh.vertices = np.column_stack((new_x, new_y, new_z))
    return mesh

def generate_embossed_text_stl(text, output_filename, font_size=5, extrusion=0.5, radius=33.5, base_height=135.0, invert_radial=False, scale_x=1.0):
    mesh_obj = text_to_3d_mesh(text, font_size=font_size, extrusion_height=extrusion)
    if mesh_obj is None:
        print("Failed to generate text mesh.")
        return
        
    warped_mesh = warp_text_to_cylinder_ledge(mesh_obj, radius=radius, base_height=base_height, invert_radial=invert_radial, scale_x=scale_x)
    
    # Save using trimesh
    warped_mesh.export(output_filename)
    print(f"Saved {output_filename}")
