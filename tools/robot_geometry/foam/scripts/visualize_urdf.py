import pybullet as p
import pybullet_data
import time
import argparse

def load_urdf(urdf_path, use_gui=True):
    # Start PyBullet simulation
    if use_gui:
        physics_client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    else:
        physics_client = p.connect(p.DIRECT)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Set search path
    p.setGravity(0, 0, -9.81)  # Set gravity

    # Load plane and robot URDF
    plane_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
    
    # Run the simulation loop
    try:
        while use_gui:
            p.stepSimulation()
            
            # Get the current position of the robot's base
            pos, orn = p.getBasePositionAndOrientation(robot_id)
            # If the robot's base goes below the plane (z < 0), reset its z-position to 0.
            if pos[2] < 3:
                p.resetBasePositionAndOrientation(robot_id, [pos[0], pos[1], 0], orn)
            
            time.sleep(1 / 240.0)  # Step simulation at ~240Hz
    except KeyboardInterrupt:
        pass
    
    # Disconnect from PyBullet
    p.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a URDF file into PyBullet simulation.")
    parser.add_argument("urdf_file", type=str, help="Path to the URDF file.")
    parser.add_argument("--nogui", action="store_true", help="Run without GUI.")
    args = parser.parse_args()
    
    load_urdf(args.urdf_file, use_gui=not args.nogui)