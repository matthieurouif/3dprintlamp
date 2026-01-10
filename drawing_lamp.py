import cadquery as cq
import math

# --- Configuration ---
LAMP_HEIGHT = 150       # Height of the lamp
LAMP_WIDTH = 100        # Width/Depth of the square profile
WALL_THICKNESS = 2      # Thickness of the pattern lines
TEXT_STRING = "Photoroom"
FONT_SIZE = 14
LED_DIAMETER = 60.0     # Fit for Bambu LED Kit 001 (Puck)
BASE_HEIGHT = 18.0      # Height of the LED base block

def make_heart(size=10):
    """
    Creates a helper heart shape using a 2D spline/path.
    Returns a Workplane with a Face.
    """
    # Parametric curve close?
    # t from 0 to 2pi
    wp = cq.Workplane("XY").parametricCurve(lambda t: (
        size * 16 * math.sin(t)**3, 
        size * (13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
    ), start=0, stop=2*math.pi, makeWire=True)
    
    # Convert wire to face explicitly
    wire = wp.val()
    face = cq.Face.makeFromWires(wire)
    return cq.Workplane("XY").add(face)

def make_led_base():
    """
    Creates a square mounting base for the LED Kit 001.
    """
    # 1. Main Square Block (Matches Lamp Width)
    base = cq.Workplane("XY").rect(LAMP_WIDTH, LAMP_WIDTH).extrude(BASE_HEIGHT)
    
    # 2. LED Puck Cutout (Through hole or pocket?)
    # We want a pocket coming from the bottom, with a lip at the top to hold it.
    
    # Pocket D=60mm from bottom up to (Height - 3mm)
    # Creating a Lip at the top (Remaining 3mm) with smaller hole.
    
    pocket_depth = BASE_HEIGHT - 3.0 # Leave 3mm ceiling
    
    # Large Hole (Holder)
    base = base.faces("<Z").workplane().circle(LED_DIAMETER / 2).cutBlind(-pocket_depth)
    
    # Small Hole (Light passage, Retaining Lip)
    # Diameter 50mm (smaller than 60) to hold the puck
    base = base.faces(">Z").workplane().circle(50.0 / 2).cutBlind(-BASE_HEIGHT)
    
    # 3. Cable Slot
    # A slot from the center out to the back edge
    cable_width = 8.0
    # Cutting from bottom face through
    base = base.faces("<Z").workplane().center(0, LAMP_WIDTH/4).rect(cable_width, LAMP_WIDTH/2).cutBlind(-BASE_HEIGHT)
    
    return base

# --- 1. Create the 2D "Drawing" (The Top Profile) ---

# Create the outer frame
frame = cq.Workplane("XY").rect(LAMP_WIDTH, LAMP_WIDTH).rect(LAMP_WIDTH - WALL_THICKNESS*2, LAMP_WIDTH - WALL_THICKNESS*2)

# Create the Text
text_geo = (
    cq.Workplane("XY")
    .center(0, 15) 
    .text(TEXT_STRING, FONT_SIZE, 5, font="Arial", kind="regular", halign="center")
)

# Create Hearts
heart_1 = make_heart(0.8).translate((-25, -20)) 
heart_2 = make_heart(0.8).translate((25, -20))  

# Combine the elements into one 2D profile
# Use .add() instead of .union() because union expects solids, and these are non-overlapping faces.
profile_elements = text_geo.add(heart_1.val()).add(heart_2.val())

# --- "Keith Haring" Style Processing ---
thick_profile = profile_elements.faces(">Z").edges().toPending().offset2D(1.5, kind="arc")

# Connector lines (Squiggles/Bars)
c1 = cq.Workplane("XY").rect(LAMP_WIDTH-5, 2).translate((0, 15))
c2 = cq.Workplane("XY").rect(2, LAMP_WIDTH-5).translate((-25, 0))
c3 = cq.Workplane("XY").rect(2, LAMP_WIDTH-5).translate((25, 0))

connectors = c1.add(c2).add(c3)

# --- 2. Vertical Extrusion ---
# Robust approach: Extrude components separately, then Union the solids.
# This avoids issues with 2D boolean operations on Wires.

body_frame = frame.extrude(LAMP_HEIGHT)
body_profile = thick_profile.extrude(LAMP_HEIGHT)
body_connectors = connectors.extrude(LAMP_HEIGHT)

# Union the solids
lamp_body = body_frame.union(body_profile).union(body_connectors)

# Move it up so it sits ON TOP of the base
lamp_body = lamp_body.translate((0, 0, BASE_HEIGHT))

# --- 3. Add the Base ---
base_mount = make_led_base()

# Union
final_assembly = base_mount.union(lamp_body)

# --- Export ---
print("Exporting STL...")
cq.exporters.export(final_assembly, "haring_photoroom_lamp.stl")
print("STL generated: haring_photoroom_lamp.stl")