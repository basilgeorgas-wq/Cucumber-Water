import numpy as np
import threading
from itertools import combinations
import swift
import time
from spatialmath.base import *
from spatialgeometry import Cuboid, Sphere
from typing import List
from roboticstoolbox import DHRobot, DHLink, jtraj, RevoluteDH, trapezoidal
from ir_support import RectangularPrism, line_plane_intersection, CylindricalDHRobotPlot
from ir_support.robots.DHRobot3D import DHRobot3D
import keyboard
from math import pi, radians, sin, cos, atan2, sqrt
import matplotlib.pyplot as plt
import os

# Import the OMRON_TM5_700 class
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from DOBOT_CR5 import DOBOT_CR5
from MOTOMAN_GP4 import MOTOMAN_GP4
from OMRON_TM5 import OMRON_TM5_700
def create_simple_dh_robot():
    """Create a simple DH robot for comparison (without 3D models)"""
    links = []
    d = [0.147, 0,     0,      0.116,  0.116, 0.105]
    a = [0,     -0.427, -0.357, 0, 0, 0]
    alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]
    offset = [0,0,0,0,0,0]
    qlim =[ [-2*pi,2*pi],  
            [-2*pi,2*pi],
            [-8*pi/9 ,8*pi/9], 
            [-2*pi,2*pi],
            [-2*pi,2*pi], 
            [-2*pi,2*pi]]
    

    for i in range(6):
        link = RevoluteDH(
            d=d[i], 
            a=a[i], 
            alpha=alpha[i], 
            qlim=qlim[i], 
            offset=offset[i]
        )
        links.append(link)
    
    robot = DHRobot(links, name='TM5_DH_Simple')
    return robot

def add_cylinder_geometry(robot, radius=0.03, multicolor=True):
    """Attach cylinder geometry to robot links"""
    cyl_viz = CylindricalDHRobotPlot(robot, cylinder_radius=radius, multicolor=multicolor)
    cyl_viz.create_cylinders()
    return robot

def compare_transforms(robot_3d, robot_dh, q):
    """Compare forward kinematics between 3D model and DH model"""
    print("\n=== Forward Kinematics Comparison ===")
    print(f"Joint configuration: {np.round(np.degrees(q), 2)} degrees")
    
    # Set both robots to same configuration
    robot_3d.q = q
    robot_dh.q = q
    
    # Get end effector transforms
    T_3d = robot_3d.fkine(q)
    T_dh = robot_dh.fkine(q)
    
    print("\n3D Model End Effector:")
    print(f"  Position: {np.round(T_3d.t, 4)}")
    print(f"  Distance from base: {np.linalg.norm(T_3d.t):.4f}m")
    
    print("\nDH Model End Effector:")
    print(f"  Position: {np.round(T_dh.t, 4)}")
    print(f"  Distance from base: {np.linalg.norm(T_dh.t):.4f}m")
    
    # Calculate difference
    pos_diff = np.linalg.norm(T_3d.t - T_dh.t)
    print(f"\nPosition difference: {pos_diff*1000:.2f}mm")
    
    if pos_diff < 0.01:  # Less than 10mm
        print("✓ DH parameters match 3D model well!")
    else:
        print("✗ Warning: Significant difference detected - check DH parameters")
    
    return pos_diff

def animate_comparison(env, robot_3d, robot_dh, cycles=2):
    """Animate both robots simultaneously to show alignment"""
    print("\n=== Animating Robot Comparison ===")
    
    steps_per_joint = 60
    
    for cycle in range(cycles):
        print(f"\nCycle {cycle + 1}/{cycles}")
        
        for joint_idx in range(len(robot_3d.links)):
            q_min = robot_3d.qlim[0, joint_idx]
            q_max = robot_3d.qlim[1, joint_idx]
            
            print(f"  Joint {joint_idx}: [{np.degrees(q_min):.1f}°, {np.degrees(q_max):.1f}°]")
            
            q_current = robot_3d.q.copy()
            
            # Move to minimum
            q_target = q_current.copy()
            q_target[joint_idx] = q_min
            
            for step in range(steps_per_joint):
                alpha = step / (steps_per_joint - 1)
                q = q_current + alpha * (q_target - q_current)
                robot_3d.q = q
                robot_dh.q = q
                env.step(0.05)
                time.sleep(0.02)
            
            q_current = robot_3d.q.copy()
            
            # Move to maximum
            q_target = q_current.copy()
            q_target[joint_idx] = q_max
            
            for step in range(steps_per_joint):
                alpha = step / (steps_per_joint - 1)
                q = q_current + alpha * (q_target - q_current)
                robot_3d.q = q
                robot_dh.q = q
                env.step(0.05)
                time.sleep(0.02)
            
            # Return to middle
            q_current = robot_3d.q.copy()
            q_target = q_current.copy()
            q_target[joint_idx] = (q_min + q_max) / 2
            
            for step in range(steps_per_joint // 2):
                alpha = step / (steps_per_joint // 2 - 1)
                q = q_current + alpha * (q_target - q_current)
                robot_3d.q = q
                robot_dh.q = q
                env.step(0.05)
                time.sleep(0.02)

def test_multiple_configurations(robot_3d, robot_dh):
    """Test DH parameters at multiple joint configurations"""
    print("\n=== Testing Multiple Configurations ===")
    
    test_configs = [
        ("Zero position", [0, 0, 0, 0, 0, 0]),
        ("Home position", [0, -pi/2, 0, -pi/2, 0, 0]),
        ("Extended", [0, -pi/4, -pi/4, 0, pi/2, 0]),
        ("Random 1", [pi/6, -pi/3, pi/4, -pi/6, pi/3, pi/4]),
        ("Random 2", [-pi/4, -2*pi/3, pi/6, pi/4, -pi/4, -pi/3]),
    ]
    
    max_error = 0
    worst_config = None
    
    for name, q in test_configs:
        print(f"\n{name}:")
        error = compare_transforms(robot_3d, robot_dh, q)
        
        if error > max_error:
            max_error = error
            worst_config = name
    
    print(f"\n=== Summary ===")
    print(f"Maximum error: {max_error*1000:.2f}mm at {worst_config}")
    
    if max_error < 0.005:
        print("Status: EXCELLENT - DH parameters are very accurate")
    elif max_error < 0.02:
        print("Status: GOOD - DH parameters are acceptable")
    elif max_error < 0.05:
        print("Status: FAIR - Consider refining DH parameters")
    else:
        print("Status: POOR - DH parameters need correction")



    
def main():
    """Main execution function"""
    print("=== DH Parameter Verification Tool ===")
    print("This tool overlays cylinder geometry with the 3D robot model")
    print("to help verify DH parameter accuracy.\n")
    
    # Create both robots
    print("Loading 3D robot model...")
    robot_3d = DOBOT_CR5()
    
    print("Creating DH cylinder model...")
    robot_dh = create_simple_dh_robot()
    if hasattr(robot_3d, 'base') and robot_3d.base is not None:
        robot_dh.base = robot_3d.base
    add_cylinder_geometry(robot_dh, radius=0.03, multicolor=True)
    
    # Set initial configuration
    q_init = [0, -pi/2, 0, -pi/2, 0, 0]
    robot_3d.q = q_init
    robot_dh.q = q_init
    
    # Create Swift environment
    env = swift.Swift()
    env.launch(realtime=True)
    
    # Add both robots
    robot_3d.add_to_env(env)
    env.add(robot_dh)
    
    print("\n=== Visualization Ready ===")
    print("- Solid model: 3D CAD model")
    print("- Cylinders: DH parameter representation")
    print("- If they align well, your DH parameters are correct!")
    
    # Initial comparison
    compare_transforms(robot_3d, robot_dh, q_init)
    
    print("\nOptions:")
    print("1. Run full configuration test (static)")
    print("2. Animate both robots (watch alignment)")
    print("3. Manual control (use teach pendant)")
    print("4. Just observe current position")
    
    choice = input("\nEnter choice (1/2/3/4): ").strip()
    
    if choice == "1":
        test_multiple_configurations(robot_3d, robot_dh)
    elif choice == "2":
        animate_comparison(env, robot_3d, robot_dh, cycles=2)
    elif choice == "3":
        print("\nStarting teach pendant...")
        print("Use sliders to move joints. Watch how cylinders align with 3D model!")
        fig = robot_3d.teach(robot_3d.q, block=False)
        
        try:
            while True:
                if keyboard.is_pressed('enter'):
                    break
                # Sync both robots
                robot_dh.q = robot_3d.q
                print(f"\rq = {np.round(np.degrees(robot_3d.q), 1)}°", end='', flush=True)
                fig.step(0.05)
                env.step(0.05)
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nInterrupted")
        
        compare_transforms(robot_3d, robot_dh, robot_3d.q)
        keyboard.unhook_all()
        plt.close("all")
    
    # Keep Swift window open
    print("\n\nPress Ctrl+C to exit")
    try:
        while True:
            env.step(0.05)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()