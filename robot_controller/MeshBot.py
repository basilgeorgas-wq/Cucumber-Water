import numpy as np
import swift
import time
from roboticstoolbox import DHRobot, RevoluteDH
from ir_support import CylindricalDHRobotPlot
from math import pi
import matplotlib.pyplot as plt
import keyboard

def create_dh():
    "Create DH"
    links = []
    
  
    d = [0.330, 0.0, 0.0, 0.290, 0.0, 0.1] 
    a = [0.0, -0.260, -0.015, 0.0, 0.0, 0.0] 
    alpha = [pi/2, 0, pi/2, -pi/2, -pi/2, 0.0]
    offset = [0,0,0,pi/2,0,0]
     
    qlim = [
        [-pi, pi], 
        [-pi/2, pi/2], 
        [-pi/2, pi/2], 
        [-pi, pi], 
        [-pi, pi], 
        [-(270 * pi/180), (270 * pi/180)]
    ]
    
    for i in range(6):
        link = RevoluteDH(
            d=d[i], 
            a=a[i], 
            alpha=alpha[i], 
            qlim=qlim[i], 
            offset=offset[i]
        )
        links.append(link)
    
    robot = DHRobot(links, name='DH')
    
    # Add cylinder geometry
    cyl_viz = CylindricalDHRobotPlot(robot, cylinder_radius=0.03, multicolor=True)
    cyl_viz.create_cylinders()
    
    return robot

def animate_robot(env, robot, cycles=2):
    """Animate the robot through its joint ranges"""
    print("\n=== Animating Robot ===")
    
    steps_per_joint = 30
    
    for cycle in range(cycles):
        print(f"\nCycle {cycle + 1}/{cycles}")
        
        for joint_idx in range(len(robot.links)):
            q_min = robot.qlim[0, joint_idx]
            q_max = robot.qlim[1, joint_idx]
            
            print(f"  Joint {joint_idx}: [{np.degrees(q_min):.1f}°, {np.degrees(q_max):.1f}°]")
            
            q_current = robot.q.copy()
            
            # Move to minimum
            q_target = q_current.copy()
            q_target[joint_idx] = q_min
            
            for step in range(steps_per_joint):
                alpha = step / (steps_per_joint - 1)
                q = q_current + alpha * (q_target - q_current)
                robot.q = q
                env.step(0.05)
                time.sleep(0.02)
            
            q_current = robot.q.copy()
            
            # Move to maximum
            q_target = q_current.copy()
            q_target[joint_idx] = q_max
            
            for step in range(steps_per_joint):
                alpha = step / (steps_per_joint - 1)
                q = q_current + alpha * (q_target - q_current)
                robot.q = q
                env.step(0.05)
                time.sleep(0.02)
            
            # Return to middle
            q_current = robot.q.copy()
            q_target = q_current.copy()
            q_target[joint_idx] = (q_min + q_max) / 2
            
            for step in range(steps_per_joint // 2):
                alpha = step / (steps_per_joint // 2 - 1)
                q = q_current + alpha * (q_target - q_current)
                robot.q = q
                env.step(0.05)
                time.sleep(0.02)

def main():
    """Main execution function"""
    print("=== DH Cylindrical Robot Visualizer ===")
    print("Displaying MOTOMAN GP4 as cylinder geometry only\n")
    
    # Create robot
    print("Creating DH cylinder model...")
    robot = create_dh()
    
    # Set initial configuration
    q_init = [0, -pi/2, 0, -pi/2, 0, 0]
    robot.q = q_init
    
    # Create Swift environment
    env = swift.Swift()
    env.launch(realtime=True)
    
    # Add robot
    env.add(robot)
    
    print("\n=== Visualization Ready ===")
    print(f"Initial joint configuration: {np.round(np.degrees(q_init), 1)}°")
    
    T = robot.fkine(q_init)
    print(f"End effector position: {np.round(T.t, 4)}")
    print(f"Distance from base: {np.linalg.norm(T.t):.4f}m")
    
    print("\nOptions:")
    print("1. Animate robot through joint ranges")
    print("2. Manual control (teach pendant)")
    print("3. Just observe current position")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        animate_robot(env, robot, cycles=2)
    elif choice == "2":
        print("\nStarting teach pendant...")
        print("Use sliders to move joints. Press Enter when done.")
        fig = robot.teach(robot.q, block=False)
        
        try:
            while True:
                if keyboard.is_pressed('enter'):
                    break
                print(f"\rq = {np.round(np.degrees(robot.q), 1)}°", end='', flush=True)
                fig.step(0.05)
                env.step(0.05)
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nInterrupted")
        
        T = robot.fkine(robot.q)
        print(f"\n\nFinal end effector position: {np.round(T.t, 4)}")
        
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