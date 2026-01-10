import numpy as np
import trimesh
from text_generator import generate_embossed_text_stl

try:
    generate_embossed_text_stl("Made with ❤️ in Paris", "test_text.stl", font_size=5, extrusion=1.0)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
