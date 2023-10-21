import time 
import sys

import numpy as np
import pinocchio as pin

from os.path import join, dirname, abspath
from pinocchio.visualize import MeshcatVisualizer

def MeshcatURDFVisualizer():
    MODEL_DIR = join(dirname(dirname(str(abspath(__file__)))), "models")

    URDF_FILENAME = "navy_door.urdf"

    mesh_dir = MODEL_DIR
    urdf_path = join(MODEL_DIR, URDF_FILENAME)

    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_path, mesh_dir, pin.JointModelFreeFlyer()
    )
    
    viz = MeshcatVisualizer(model, collision_model, visual_model)

    try:
        viz.initViewer(open=True)
    except ImportError as err:
        print("Error while initializing the viewer. It seems you should install Python meshcat")
        print(err)
        sys.exit(0)

    viz.loadViewerModel()

    time.sleep(10)

if __name__ == '__main__':
    MeshcatURDFVisualizer()