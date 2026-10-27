import os
import swift
import swift_windows_fix  # noqa: F401  - fixes Swift's "404 File not found" for meshes on Windows
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

        # Every robot is checked against the others (each move skips itself)
        robots = [self.Motoman, self.OMRON, self.DOBOT]

        #Teach code, set to also update in swift.env. TODO Figure out why it creates a 2nd figure in the while loop. 
        fig = self.OMRON.teach(self.OMRON.q, block=False)  # Matplotlib teach pendant

        while not keyboard.is_pressed('enter'):
            fig.step(0.05)
            self.OMRON.q = self.OMRON.q  # Sync DH robot       
            self.env.step(0.05)           # Update Swift visualization

        fig.close()

        # Settings shared by every move below
        self._move_kwargs = dict(env=self.env, faces=faces, vertices=vertices,
                                 face_normals=face_normals, other_robots=robots,
                                 collisions_list=collisions_list, show_path_dots=False)
        self.step_log = []
        self.run_task()

        # self.initiate_joystick_control(robot=self.DOBOT, duration=300)

        if not self.check_pause():
            return

        self.env.hold()

    # ======================================================================
    # Task settings (metres / radians). Change these when real grippers are added.
    # ======================================================================
    GRIPPER_LENGTH_OMRON = 0.0     # flange -> fingertip of the Omron gripper
    GRIPPER_LENGTH_DOBOT = 0.0     # flange -> fingertip of the Dobot gripper
    FLANGE_TO_GLASS = 0.05         # flange -> glass centre when the Dobot holds it (+ gripper)

    SLICE_POSITION = (-0.8, 0.05, 0.94)   # where the cut slice is placed after slicing
    SINK_GLASS_POSITION = (-0.65, -1.4, 1.05)  # glass centre under the tap
    HANDOVER_GLASS_POSITION = (0.0, -0.6, 1.1) # glass centre when held out for the Omron
    APPROACH_HEIGHT = 0.10         # Omron goes this far above a pick/drop point first
    DROP_CLEARANCE = 0.08          # Omron fingertip height above the glass rim when dropping
                                   # (wrist link + glass collision padding need ~6.6 cm)

    TOOL_DOWN = SE3.Rx(pi)         # tool z-axis pointing straight down

    @staticmethod
    def grip_facing(yaw):
        """Dobot side-grip on an upright glass: tool z horizontal, at heading `yaw`
        (0 = facing +y, pi/2 = facing -x); flange y points down so the glass stays upright."""
        return SE3.Rz(yaw) * SE3.Rx(-pi / 2)

    # ======================================================================
    def _move(self, label, robot, target, linear=False, **kwargs):
        """Run one move with the shared settings. Returns True if the robot got there."""
        if not self.check_pause():
            return False
        args = dict(self._move_kwargs)
        args.update(kwargs)
        if linear:
            args.pop('collisions_list', None)
            q_matrix, ok, _ = linear_move(robot=robot, target_pose=target, **args)
            success = q_matrix is not None and ok
        else:
            q_matrix, _, ok = ikine_move(robot=robot, target_pose=target, q0=robot.q, **args)
            success = q_matrix is not None and ok
        self.step_log.append((label, success))
        return success

    def _dobot_flange_for_glass(self, glass_position, yaw):
        """Flange pose that puts the glass centre at glass_position."""
        T_grip = self.grip_facing(yaw)
        reach = self.FLANGE_TO_GLASS + self.GRIPPER_LENGTH_DOBOT
        offset = T_grip.R[:, 2] * reach            # flange -> glass, along tool z
        return SE3(*(np.array(glass_position) - offset)) * T_grip

    def run_task(self):
        env_mgr = self.environment
        H, R = env_mgr.GLASS_HEIGHT, env_mgr.GLASS_RADIUS

        # ---- Props: the Dobot starts holding an empty glass ----------------
        # Glass frame: origin at its base, z up. In the flange frame the glass
        # centre sits FLANGE_TO_GLASS along tool z, upright (= -flange y).
        reach = self.FLANGE_TO_GLASS + self.GRIPPER_LENGTH_DOBOT
        T_flange_glass = (SE3(0, 0, reach) * SE3.Rx(pi / 2) * SE3(0, 0, -H / 2)).A
        glass_axis = [(T_flange_glass @ np.array([0, 0, z, 1.0]))[:3] for z in (0.0, H)]
        glass = env_mgr.add_glass()
        self.DOBOT.attach(glass, T_flange_glass,
                          collision_segments=[tuple(glass_axis)], collision_radius=R)
        water = None
        cuc_slice = None

        # ---- 1. Motoman slices the cucumber --------------------------------
        # Straight line (RMRC) from home to above the cucumber, knife held level.
        # (The old RMRC end point (-1.1, -0.5, 1.2) is over the Motoman's own base:
        #  joint 2 runs out of travel unless the knife is allowed to flop over.)
        print("Motoman moving above cucumber (RMRC straight line)")
        if self.check_pause():
            rmrc_steps = 100
            path = create_linear_path(start_pos=self.Motoman.fkine(self.Motoman.q).t,
                                      end_pos=SE3(-0.8, -0.2, 1.2).t,
                                      steps=rmrc_steps)
            kw = self._move_kwargs
            q_traj, _, _ = rmrc_move(self.Motoman, q0=self.Motoman.q, env=self.env,
                                     cartesian_path=path,
                                     faces=kw['faces'], vertices=kw['vertices'],
                                     face_normals=kw['face_normals'],
                                     other_robots=kw['other_robots'])
            self.step_log.append(("Motoman above cucumber (RMRC)", len(q_traj) == rmrc_steps))

        print("Motoman performing slicing")
        x = 0
        for i in range(5):
            self._move(f"Motoman slice {i+1} up", self.Motoman,
                       SE3(-0.8, x, 1.1) @ SE3.RPY([0, pi/2, 0]), linear=True,
                       steps=10, position_tolerance=0.005, check_reachability=True)
            self._move(f"Motoman slice {i+1} down", self.Motoman,
                       SE3(-0.8, x, 0.95) @ SE3.RPY([0, 95*pi/180, 0]), linear=True,
                       steps=10, position_tolerance=0.005, check_reachability=True)
            x += -0.05
            self.env.step(0.05)

        # The cut slice appears next to the cucumber (stand-in until the
        # cucumber_cut_*.stl pieces are swapped in during slicing)
        cuc_slice = env_mgr.add_slice(SE3(*self.SLICE_POSITION))

        # Retract, tilted 15 deg: the level orientation isn't reachable up there
        print("Motoman retracting")
        self._move("Motoman retract", self.Motoman,
                   SE3(-0.9, -0.2, 1.8) @ SE3.RPY([0, 75*pi/180, 0]), steps=30)

        # ---- 2. Dobot fills the glass at the sink ---------------------------
        print("Dobot filling the glass at the sink")
        filled = self._move("Dobot to sink", self.DOBOT,
                            self._dobot_flange_for_glass(self.SINK_GLASS_POSITION, pi/2))
        if filled:
            water = env_mgr.add_water()
            # water cylinder is centred on its origin: put it in the lower part of the glass
            self.DOBOT.attach(water, T_flange_glass @ SE3(0, 0, 0.6 * H / 2 + 0.005).A)
            for _ in range(10):
                self.env.step(0.05)

        # ---- 3. Dobot holds the glass out for the Omron -------------------
        print("Dobot holding out the glass")
        held = self._move("Dobot hold out glass", self.DOBOT,
                          self._dobot_flange_for_glass(self.HANDOVER_GLASS_POSITION, 0.0))

        # ---- 4. Omron picks up the slice ----------------------------------
        print("Omron picking up the slice")
        sx, sy, sz = self.SLICE_POSITION
        pick_z = sz + env_mgr.SLICE_RADIUS + 0.005 + self.GRIPPER_LENGTH_OMRON
        above_ok = self._move("Omron above slice", self.OMRON,
                              SE3(sx, sy, pick_z + self.APPROACH_HEIGHT) * self.TOOL_DOWN)
        picked = above_ok and self._move("Omron pick slice", self.OMRON,
                                         SE3(sx, sy, pick_z) * self.TOOL_DOWN, linear=True,
                                         steps=10, position_tolerance=0.005)
        if picked:
            self.OMRON.attach(cuc_slice)          # slice now moves with the Omron
            self._move("Omron lift slice", self.OMRON,
                       SE3(sx, sy, pick_z + self.APPROACH_HEIGHT) * self.TOOL_DOWN, linear=True,
                       steps=10, position_tolerance=0.005)

        # ---- 5. Omron drops the slice into the glass ------------------------
        if picked and held:
            print("Omron putting the slice in the glass")
            gx, gy, gz = self.HANDOVER_GLASS_POSITION
            drop_z = gz + H / 2 + self.DROP_CLEARANCE + self.GRIPPER_LENGTH_OMRON
            over = self._move("Omron above glass", self.OMRON,
                              SE3(gx, gy, drop_z + self.APPROACH_HEIGHT) * self.TOOL_DOWN)
            dropped = over and self._move("Omron lower to glass", self.OMRON,
                                          SE3(gx, gy, drop_z) * self.TOOL_DOWN, linear=True,
                                          steps=10, position_tolerance=0.005)
            if dropped:
                # Release: slice drops to the bottom of the glass and moves with it
                self.OMRON.detach(cuc_slice)
                cuc_slice.T = (SE3(gx, gy, gz - H / 2 + env_mgr.SLICE_RADIUS + 0.005)).A
                self.DOBOT.attach(cuc_slice)
                self.env.step(0.05)
                self._move("Omron move away", self.OMRON,
                           SE3(gx, gy, drop_z + self.APPROACH_HEIGHT) * self.TOOL_DOWN, linear=True,
                           steps=10, position_tolerance=0.005)
        else:
            print("Skipping the drop: " +
                  ("slice was not picked up" if not picked else "glass is not held out"))

        # ---- Summary --------------------------------------------------------
        print("\n" + "=" * 50 + "\nTASK SUMMARY\n" + "=" * 50)
        for label, ok in self.step_log:
            print(f"  {'OK    ' if ok else 'FAILED'}  {label}")
        print("=" * 50)


if __name__ == "__main__":
    sim = TaskSimulation()
    sim.launch_simulation()