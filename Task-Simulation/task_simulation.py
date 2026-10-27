import os
import swift
import sys
from pathlib import Path
from spatialmath import SE3
import keyboard
from math import pi
import numpy as np
import threading
import time
import matplotlib as plt

# Add the Robot Controller folder to Python's search path
environment_path = Path(__file__).parent.parent / "environment_controller"
robot_controller_path = Path(__file__).parent.parent / "robot_controller"
sys.path.append(str(robot_controller_path))
sys.path.append(str(environment_path))

# Now that the path is added, imports will work
from OMRON_TM5 import OMRON_TM5_700
from DOBOT_CR5 import DOBOT_CR5
from MOTOMAN_GP4 import MOTOMAN_GP4

from environment_controller import Environment_Manager
from movement_handler import ikine_move, rmrc_move, linear_move, start_joystick_control, create_linear_path

class TaskSimulation():
    def __init__(self):
        self.env = None
        self.OMRON = None
        self.DOBOT = None
        self.Motoman = None
        self.DoBotCR7 = None
        self.environment = None
        
        # Emergency stop control flags
        self.emergency_stop_active = False
        self.simulation_running = True
        self.pause_event = threading.Event()
        self.pause_event.set()  # Initially not paused
        
        # Lock for thread-safe flag access
        self.estop_lock = threading.Lock()
        
        # Start emergency stop listener thread
        self.estop_thread = threading.Thread(target=self._emergency_stop_listener, daemon=True)
        self.estop_thread.start()
    
        # Define workspace limits (x, y, z boundaries in meters)
        self.WORKSPACE_LIMITS = {
            'x': [-0.8, 0.8],      # x min/max
            'y': [-0.2, 1.0],      # y min/max
            'z': [0.0, 1.5]        # z min/max
        }
    
    def _emergency_stop_listener(self):
        """Background thread that listens for emergency stop commands"""
        print("\n" + "="*50)
        print("EMERGENCY STOP CONTROLS:")
        print("  Press 'e': Trigger Emergency Stop")
        print("  Press 'q': Quit Simulation")
        print("="*50 + "\n")
        
        # Set up keyboard hotkeys
        keyboard.add_hotkey('e', self.trigger_emergency_stop, suppress=False)
        keyboard.add_hotkey('q', self.quit_simulation, suppress=False)
        
        # Keep thread alive
        while self.simulation_running:
            time.sleep(0.1)
        
        # Clean up hotkeys when done
        try:
            keyboard.unhook_all()
        except:
            pass

    def trigger_emergency_stop(self):
        """Activate emergency stop and present user with options"""
        with self.estop_lock:
            if not self.emergency_stop_active:
                self.emergency_stop_active = True
                self.pause_event.clear()  # Pause the simulation
                
                print("\n" + "!"*50)
                print("!!! EMERGENCY STOP ACTIVATED !!!")
                print("!"*50)
                print("\nRobot motion has been halted.")
                print("\nOptions:")
                print("  Press 'r': Resume simulation")
                print("  Press 'q': End simulation")
                print("="*50 + "\n")
                
                # Set up resume hotkey (remove first if it exists)
                try:
                    keyboard.remove_hotkey('r')
                except:
                    pass
                keyboard.add_hotkey('r', self.resume_simulation, suppress=False)

    def resume_simulation(self):
        """Resume simulation after emergency stop"""
        with self.estop_lock:
            if self.emergency_stop_active:
                print("\n" + "="*50)
                print("RESUMING SIMULATION...")
                print("="*50 + "\n")
                
                self.emergency_stop_active = False
                self.pause_event.set()  # Resume execution
                
                # Remove the resume hotkey
                try:
                    keyboard.remove_hotkey('r')
                except:
                    pass
   
    def quit_simulation(self):
        """End the simulation"""
        print("\n" + "="*50)
        print("ENDING SIMULATION...")
        print("="*50 + "\n")
        
        with self.estop_lock:
            self.simulation_running = False
            self.emergency_stop_active = False
            self.pause_event.set()  # Unblock any waiting threads
        
        # Clean up keyboard hooks
        try:
            keyboard.unhook_all()
        except:
            pass
        
        # Clean up environment
        if self.env is not None:
            try:
                self.env.close()
            except:
                pass
        
        sys.exit(0)
    
    def check_pause(self):
        """Check if simulation should pause (called in main loop)"""
        if self.emergency_stop_active:
            self.pause_event.wait()  # Block until resumed
        
        return self.simulation_running
    
    def initiate_joystick_control(self, robot, duration=300):
        start_joystick_control(
        robot=robot,
        env=self.env,
        duration=duration,  # Max duration in seconds (5 minutes)
        dt=0.1,             # Update rate (10 Hz)
        Kv=0.3,             # Linear velocity gain (m/s)
        Kw=0.8,             # Angular velocity gain (rad/s)
        show_status=True    # Print status updates
    )

    #TODO fix Collision(Non-functional), link collision also not integrated, 
    #     Add hardware e-stop(Figure out how to have joystick permanently running during program?)
    def launch_simulation(self):
        """Create the Swift environment and run the simulation"""
        # Create the Swift environment
        self.env = swift.Swift()
        self.env.launch(realtime=True)

        # Initialise environment manager + Collision objects
        self.environment = Environment_Manager(self.env)
        faces, vertices, face_normals = self.environment.get_collision_data()
        collisions_list = [] 
 
        # Initialise Motoman
        self.Motoman = MOTOMAN_GP4()
        self.Motoman.setbase(SE3(-1.1, -0.2, 0.92)* SE3.Rz(np.pi))
        self.Motoman.add_to_env(self.env)

        # Initialise Omron
        self.OMRON = OMRON_TM5_700()
        self.OMRON.setbase(SE3(-0.1, 0, 0.92))
        self.OMRON.add_to_env(self.env)

        # Initialise Dobot
        self.DOBOT = DOBOT_CR5()
        self.DOBOT.setbase(SE3(-0.1, -1.5, 0.92))
        self.DOBOT.add_to_env(self.env)

        #Teach code, set to also update in swift.env. TODO Figure out why it creates a 2nd figure in the while loop. 
        fig = self.OMRON.teach(self.OMRON.q, block=False)  # Matplotlib teach pendant

        while not keyboard.is_pressed('enter'):
            fig.step(0.05)
            self.OMRON.q = self.OMRON.q  # Sync DH robot       
            self.env.step(0.05)           # Update Swift visualization

        fig.close()

        path = create_linear_path(start_pos=self.Motoman.fkine(self.Motoman.q).t,
                                  end_pos=SE3(-1.1, -0.5, 1.2).t,
                                  steps=100)
        rmrc_move(self.Motoman,
                  q0=self.Motoman.q,
                  env=self.env,
                  cartesian_path = path)


        ikine_move(
            robot=self.Motoman,
            target_pose=SE3(-0.8, -0.2, 1.2) @ SE3.RPY([0, pi/2, 0]),
            env=self.env,
            q0=self.Motoman.q,
            faces = faces, 
            vertices = vertices,
            face_normals = face_normals,
            show_path_dots=False
        )

        if not self.check_pause():
            return
        
        print("Motoman performing slicing")
        x = 0
        for i in range(5):
            linear_move(
                robot = self.Motoman, target_pose= SE3(-0.8, x, 1.1) @ SE3.RPY([0, pi/2, 0]), env=self.env,
                steps=10,  
                position_tolerance=0.005,  
                check_reachability=True,
                faces = faces, 
                vertices = vertices,
                face_normals = face_normals,
                show_path_dots=False
            )
            if not self.check_pause():
                return
            linear_move(
                robot = self.Motoman, target_pose= SE3(-0.8, x, 0.95) @ SE3.RPY([0, 95* pi/180, 0]), env=self.env,
                steps=10,  
                position_tolerance=0.005,  
                check_reachability=True,
                faces = faces, 
                vertices = vertices,
                face_normals = face_normals,
                show_path_dots=False
            )
            if not self.check_pause():
                return
            x += -0.05
            self.env.step(0.05)

        ikine_move(
            robot=self.Motoman,
            target_pose=SE3(-0.9, -0.2, 1.8) @ SE3.RPY([0, pi/2, 0]),
            env=self.env,
            steps = 30,
            q0=self.Motoman.q,
            faces = faces, 
            vertices = vertices,
            face_normals = face_normals,
            show_path_dots=False
        )
        if not self.check_pause():
            return
        
        ikine_move(
            robot=self.DOBOT,
            target_pose=SE3(-0.65, -1.4, 1.05)@ SE3.RPY([0, 0, 0]),
            env=self.env,
            q0=self.DOBOT.q,
            faces=faces, 
            vertices=vertices,
            face_normals=face_normals,
            collisions_list=collisions_list,
            show_path_dots=False
        )
        if not self.check_pause():
            return

        ikine_move(
            robot=self.DOBOT,
            target_pose=SE3(0, -0.7, 1.2),
            env=self.env,
            q0=self.DOBOT.q,
            faces=faces, 
            vertices=vertices,
            face_normals=face_normals,
            collisions_list=collisions_list,
            show_path_dots=False
        )
        if not self.check_pause():
            return

        print("Omron moving sliced pieces")
        ikine_move(
            robot=self.OMRON,
            target_pose=SE3(-0.77, -0.1, 0.95) @ SE3.RPY([0, pi/2, 0]),
            env=self.env,
            q0=self.OMRON.q,
            faces=faces, 
            vertices=vertices,
            face_normals=face_normals,
            collisions_list=collisions_list,
            show_path_dots=False
        )
        if not self.check_pause():
            return

        ikine_move(
            robot=self.OMRON,
            target_pose=SE3(0, -0.3, 1.2)@ SE3.RPY([0, pi/2, 0]),
            env=self.env,
            q0=self.OMRON.q,
            faces=faces, 
            vertices=vertices,
            face_normals=face_normals,
            collisions_list=collisions_list,
            show_path_dots=False
        )
        if not self.check_pause():
            return

        ikine_move(
            robot=self.OMRON,
            target_pose=SE3(0, -0.7, 1.4)@ SE3.RPY([0, pi, 0]),
            env=self.env,
            q0=self.OMRON.q,
            faces=faces, 
            vertices=vertices,
            face_normals=face_normals,
            collisions_list=collisions_list,
            show_path_dots=False
        )
        if not self.check_pause():
            return

        # self.initiate_joystick_control(robot=self.DOBOT, duration=300)
        
        if not self.check_pause():
            return

        self.env.hold()


if __name__ == "__main__":
    sim = TaskSimulation()
    sim.launch_simulation()