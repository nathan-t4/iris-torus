import numpy as np
import time
import os

import meshcat
import meshcat.geometry as g

from pinocchio.visualizer import MeshcatVisualizer

def MeshcatSTLVisualizer():
    vis = meshcat.Visualizer().open()

    vis["navy_door"].set_object(
        g.StlMeshGeometry.from_file(
            os.path.join(os.getcwd(), "./models/navy_door_765x1450.stl")
        )
    )

    time.sleep(10)

if __name__ == '__main__':
    MeshcatSTLVisualizer()