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
from gripper import Gripper
from movement_handler import (ikine_move, rmrc_move, linear_move, start_joystick_control,
                              create_linear_path, find_approach_configuration,
                              generate_ik_initial_guesses, solve_ik_with_strategies,
                              wrap_to_reference, get_joint_limits)

class TaskSimulation():
    def __init__(self, use_teach_pendant=False):
        # True: open the Omron teach pendant first; press Enter to start the task
        self.use_teach_pendant = use_teach_pendant
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

        self._mount_grippers()

        # The Dobot holds the (empty) glass upright from the very start
        self._setup_dobot_with_glass(faces, vertices, face_normals, robots)

        # Optional teach pendant (run with --teach). Press Enter to start the task.
        if self.use_teach_pendant:
            print("Teach pendant open - move the Omron's sliders, press Enter to start the task")
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
    # Grippers: distances along the tool axis from the robot's last DH frame
    OMRON_FLANGE_FACE = 0.006      # DH frame -> Omron flange face (gripper mounts here)
    DOBOT_FLANGE_FACE = 0.015      # DH frame -> Dobot flange face
    GRIPPER_LENGTH_OMRON = OMRON_FLANGE_FACE + 0.150   # -> Robotiq 2F-85 grip point
                                                       #    (finger pads, 1.3 cm above the tips)
    GRIPPER_LENGTH_DOBOT = DOBOT_FLANGE_FACE + 0.189   # -> AG-95 grip point (finger pad centre)
    FLANGE_TO_GLASS = 0.0          # extra offset from the grip point to the glass centre

    SLICE_POSITION = (-0.8, 0.05, 0.94)   # where the cut slice is placed after slicing
    GLASS_COUNTER_POSITION = (-0.38, -1.9, 0.913) # glass base on the counter at the start
    GLASS_PLACE_POSITION = (-0.38, -0.95, 0.913)  # ... and where it is served at the end
                                                  #   (a straight move from the handover)
    GLASS_COUNTER_GRIP_YAW = pi / 2               # Dobot grips it facing -x (from the robot's side)
    GLASS_APPROACH = 0.10                         # Dobot's fingers slide in level from this far
    GLASS_PLACE_APPROACH = 0.06                   # glass is lowered onto the counter from this high
    SINK_GLASS_POSITION = (-0.685, -1.345, 1.04)  # glass centre under the tap spout
    SINK_SLIDE = 0.12              # the glass slides in/out horizontally this far under the spout
    HANDOVER_GLASS_POSITION = (0.0, -0.5, 1.05) # glass centre when held out for the Omron
                                                # (middle of the region both arms + grippers can use)
    APPROACH_HEIGHT = 0.10         # Omron goes this far above a pick/drop point first
    PICK_GAP = 0.0                 # grip point height above the slice centre when picking
    DROP_CLEARANCE = 0.06          # Omron grip point height above the glass rim when dropping
    DROP_APPROACH_HEIGHT = 0.08    # Omron comes down onto the drop point from this far above
                                   # (the slice hangs 1.9 cm below it, fingertips 1.3 cm below)

    # Omron tool pointing straight down, wrist turned 90 deg so the Robotiq's
    # fingers close along world y (across the slice's flat faces) and stay
    # clear of the Omron's wrist camera
    TOOL_DOWN = SE3.Rz(pi / 2) * SE3.Rx(pi)

    CUT_PITCH = np.deg2rad(105)    # knife angle while slicing (90 = level, 105 = 15 deg nose-down)
    CUT_FLANGE_X = -0.91           # Motoman hand position while slicing (blade reaches x = -0.8)
    CUT_UP_Z = 1.12                # hand height above the cucumber
    CUT_DOWN_Z = 0.97              # hand height at the bottom of the cut (blade edge ~0.916)

    # Cucumber: the Omron grips its front end (sticking out of the fridge) at an
    # angle, and later lays it down on the bench in front of the Motoman
    CUCUMBER_GRIP_FROM_END = 0.03          # grip point distance from the cucumber's end
    CUCUMBER_GRIP_TILT = np.deg2rad(60)    # hand tilt from the cucumber's axis (90 = square on)
    FRIDGE_APPROACH = 0.10                 # Omron's fingers slide in level from this far out
    FRIDGE_PULL_OUT = 0.02                 # pulled straight out this far, then a path is planned
    CUCUMBER_BENCH_ORIGIN = (-0.8, -0.2, 0.94)   # cucumber end on the bench (it lies along +y)
    CUCUMBER_PLACE_APPROACH = 0.08         # Omron comes down onto the bench from this far above

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

    def _move_above(self, label, robot, T_above, T_down):
        """Go to T_above in an arm shape that can then move straight down to T_down."""
        if not self.check_pause():
            return False
        kw = self._move_kwargs
        q_hint = find_approach_configuration(
            robot, T_above, T_down, robot.q,
            faces=kw['faces'], vertices=kw['vertices'], face_normals=kw['face_normals'],
            other_robots=kw['other_robots'])
        if q_hint is None:
            print(f"{label}: no arm configuration allows a clear straight descent")
            self.step_log.append((label, False))
            return False
        return self._move(label, robot, T_above, preferred_q=q_hint)

    # ---- Keeping the glass upright -----------------------------------------
    GLASS_TILT_LIMIT = np.deg2rad(20)        # max tilt of the glass once it has water in it
    GLASS_TILT_LIMIT_EMPTY = np.deg2rad(45)  # ... and while it's still empty
    glass_tilt_limit = GLASS_TILT_LIMIT      # the limit in force right now

    def _glass_tilt(self, q):
        """Angle between the glass's axis (= -flange y) and straight up."""
        up = -self.DOBOT.flange_pose(q)[:3, 1]
        return np.arccos(np.clip(up[2], -1.0, 1.0))

    def _make_upright_rule(self):
        def keep_glass_upright(q):
            return self._glass_tilt(q) <= self.glass_tilt_limit
        return keep_glass_upright

    def _project_upright(self, q, max_jump=0.6):
        """Nearby joints with the same hand position but the glass upright (None if none).

        Applies the smallest rotation that turns the glass axis back to vertical,
        then solves IK starting from q, and rejects answers that jump to a
        different arm shape (they'd tilt the glass on the way)."""
        D = self.DOBOT
        if self._glass_tilt(q) <= self.glass_tilt_limit:
            return np.array(q, dtype=float)              # already within the allowed tilt
        T = D.flange_pose(q)
        up = -T[:3, 1]                                   # current glass axis
        target_up = np.array([0.0, 0.0, 1.0])
        axis = np.cross(up, target_up)
        s, c = np.linalg.norm(axis), float(np.dot(up, target_up))
        if s < 1e-9:
            if c > 0:
                return np.array(q, dtype=float)          # already upright
            axis, s = np.array([1.0, 0.0, 0.0]), 0.0     # upside down: flip about x
        else:
            axis = axis / s
        angle = np.arctan2(s, c)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R_fix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)   # Rodrigues
        target = SE3.Rt(R_fix @ T[:3, :3], T[:3, 3], check=False)
        sol = D.ikine_LM(target, q0=q, joint_limits=True, slimit=1)
        if not sol.success:
            return None
        # IK returns angles in -pi..pi; use the equivalent nearest q (joints turn > 360 deg)
        q_new = wrap_to_reference(sol.q, q, get_joint_limits(D))
        if np.linalg.norm(q_new - q) > max_jump:
            return None
        return q_new

    def _mount_grippers(self):
        """Robotiq 2F-85 on the Omron, DH Robotics AG-95 on the Dobot."""
        self.omron_gripper = Gripper(self.OMRON, "robotiq_2f85", self.env,
                                     mount_offset=self.OMRON_FLANGE_FACE)
        self.dobot_gripper = Gripper(self.DOBOT, "dh_ag95", self.env,
                                     mount_offset=self.DOBOT_FLANGE_FACE)

    def _setup_dobot_with_glass(self, faces, vertices, face_normals, robots):
        """Put the glass on the kitchen counter, ready for the Dobot to pick up."""
        env_mgr = self.environment
        H, R = env_mgr.GLASS_HEIGHT, env_mgr.GLASS_RADIUS

        # Glass frame: origin at its base, z up. When held, the glass centre sits
        # at the AG-95's grip point (FLANGE_TO_GLASS further along tool z), upright
        # (= -flange y).
        reach = self.FLANGE_TO_GLASS + self.GRIPPER_LENGTH_DOBOT
        self.T_flange_glass = (SE3(0, 0, reach) * SE3.Rx(pi / 2) * SE3(0, 0, -H / 2)).A
        # Collision capsule that fits inside the glass's height (axis from R to H-R),
        # so a glass standing on the counter doesn't count as hitting it
        self.glass_axis = [tuple((self.T_flange_glass @ np.array([0, 0, z, 1.0]))[:3])
                           for z in (R, H - R)]
        self.glass = env_mgr.add_glass(SE3(*self.GLASS_COUNTER_POSITION))
        if getattr(self, 'dobot_gripper', None) is not None:
            self.dobot_gripper.open()

    def _glass_centre_on_counter(self, base_position, lift=0.0):
        x, y, z = base_position
        return (x, y, z + self.environment.GLASS_HEIGHT / 2 + lift)

    def pick_glass(self):
        """Dobot picks the glass up off the counter (side grip, coming in level)."""
        D, glass = self.DOBOT, self.glass
        R = self.environment.GLASS_RADIUS
        yaw = self.GLASS_COUNTER_GRIP_YAW
        T_grasp = self._dobot_flange_for_glass(self._glass_centre_on_counter(self.GLASS_COUNTER_POSITION), yaw)
        T_pre = SE3(-self.GLASS_APPROACH * T_grasp.R[:, 2]) * T_grasp     # back along the tool axis

        print("Dobot picking up the glass from the counter")
        self.dobot_gripper.open()
        ok = (self._move_above("Dobot to glass", D, T_pre, T_grasp)
              and self._move("Dobot reach glass", D, T_grasp, linear=True, steps=10,
                             position_tolerance=0.005))
        if not ok:
            return False
        self.dobot_gripper.grip()
        self.env.step(0.3)
        D.attach(glass, self.T_flange_glass)                 # carried, glass collision after lift
        if not self._move("Dobot lift glass", D, SE3(0, 0, 0.03) * T_grasp, linear=True, steps=5,
                          position_tolerance=0.005):
            return False
        D.detach(glass)
        D.attach(glass, self.T_flange_glass, collision_segments=[self.glass_axis], collision_radius=R)
        # From now on every Dobot move (IK, path checks, RRT) keeps the glass upright
        # (within 45 deg while empty, 20 deg once it has water in it)
        self.glass_tilt_limit = self.GLASS_TILT_LIMIT_EMPTY
        D.path_constraint = self._make_upright_rule()
        D.project_to_constraint = self._project_upright
        return True

    def place_glass(self, extras=()):
        """Dobot sets the glass (and whatever is in it) down on the counter and lets go."""
        D, glass = self.DOBOT, self.glass
        yaw = self.GLASS_COUNTER_GRIP_YAW
        T_place = self._dobot_flange_for_glass(
            self._glass_centre_on_counter(self.GLASS_PLACE_POSITION, lift=0.005), yaw)
        T_above = SE3(0, 0, self.GLASS_PLACE_APPROACH) * T_place

        print("Dobot putting the glass down on the counter")
        ok = (self._move_glass("Dobot glass above counter", T_above)
              and self._move_glass("Dobot lower glass onto counter", T_place, steps=8))
        if not ok:
            return False
        # Let go: the glass, water and slice stay on the counter
        self.dobot_gripper.open()
        for shape in (glass,) + tuple(extras):
            if shape is not None:
                D.detach(shape)
        glass.T = SE3(*self.GLASS_PLACE_POSITION).A
        D.path_constraint = None
        D.project_to_constraint = None
        self.env.step(0.3)
        T_back = SE3(-self.GLASS_APPROACH * T_place.R[:, 2]) * T_place
        self._move("Dobot move away from glass", D, T_back, linear=True, steps=8,
                   position_tolerance=0.005)
        return True

    def _sink_front_position(self):
        """Glass position in front of the tap: same height, slid out along +x."""
        x, y, z = self.SINK_GLASS_POSITION
        return (x + self.SINK_SLIDE, y, z)

    def _move_glass(self, label, target, steps=40):
        """Dobot move while carrying the glass: a straight line first (an upright
        glass stays upright), path planning (RRT) only if the line is blocked."""
        if not self.check_pause():
            return False
        args = {k: v for k, v in self._move_kwargs.items() if k != 'collisions_list'}
        buf_ok = False
        q_matrix, ok, _ = linear_move(robot=self.DOBOT, target_pose=target, steps=steps,
                                      position_tolerance=0.005, **args)
        if q_matrix is not None and ok:
            buf_ok = True
        else:
            print(f"{label}: straight line blocked - planning a path around")
            q_matrix, _, ok = ikine_move(robot=self.DOBOT, target_pose=target, q0=self.DOBOT.q,
                                         **self._move_kwargs)
            buf_ok = q_matrix is not None and ok
        self.step_log.append((label, buf_ok))
        return buf_ok

    def _dobot_flange_for_glass(self, glass_position, yaw):
        """Flange pose that puts the glass centre at glass_position."""
        T_grip = self.grip_facing(yaw)
        reach = self.FLANGE_TO_GLASS + self.GRIPPER_LENGTH_DOBOT
        offset = T_grip.R[:, 2] * reach            # flange -> glass, along tool z
        return SE3(*(np.array(glass_position) - offset)) * T_grip

    # ---- Cucumber handling ------------------------------------------------
    def _cucumber_grasp(self):
        """Omron flange pose in the cucumber's own frame (mesh y = its axis, origin at
        the gripped end). Fingers close across the cucumber (mesh x); the hand points
        along the cucumber, tilted CUCUMBER_GRIP_TILT downwards."""
        psi = self.CUCUMBER_GRIP_TILT
        z = np.array([0.0, np.cos(psi), -np.sin(psi)])     # tool axis: into the cucumber, downwards
        x = np.array([1.0, 0.0, 0.0])
        y = np.cross(z, x)
        grip = SE3.Rt(np.column_stack([x, y, z]), [0, self.CUCUMBER_GRIP_FROM_END, 0])
        return grip * SE3(0, 0, -self.GRIPPER_LENGTH_OMRON)

    def _cucumber_axis_in_flange(self, T_cucumber, q):
        """Cucumber's axis (both ends) in the Omron's flange frame, for its collision model."""
        T_inv = np.linalg.inv(self.OMRON.flange_pose(q))
        L = self.environment.CUCUMBER_LENGTH
        return [tuple((T_inv @ (T_cucumber.A @ np.array([0, y, 0, 1.0])))[:3] for y in (0.0, L))]

    def _choose_cucumber_place(self):
        """Pick how the cucumber lies on the bench (end first either way, any roll
        about its own axis) so the Omron can lower it straight down to there."""
        kw = self._move_kwargs
        G = self._cucumber_grasp()
        x, y0, z = self.CUCUMBER_BENCH_ORIGIN
        L = self.environment.CUCUMBER_LENGTH
        for yaw, origin in ((0.0, (x, y0, z)), (pi, (x, y0 + L, z))):
            for roll_deg in (0, 30, -30, 60, -60, 90, -90, 120, -120, 180):
                T_cuc = SE3(*origin) * SE3.Rz(yaw) * SE3.Ry(np.deg2rad(roll_deg))
                T_place = SE3(0, 0, 0.005) * T_cuc * G               # 5 mm above the bench
                T_above = SE3(0, 0, self.CUCUMBER_PLACE_APPROACH) * T_place
                q = find_approach_configuration(
                    self.OMRON, T_above, T_place, self.OMRON.q, faces=kw['faces'],
                    vertices=kw['vertices'], face_normals=kw['face_normals'],
                    other_robots=kw['other_robots'], steps=8, max_attempts=20)
                if q is not None:
                    return T_cuc, T_place, T_above
        return None

    def fetch_cucumber(self):
        """Omron takes the cucumber off the fridge's top shelf and lays it on the
        bench in front of the Motoman. Returns True on success."""
        O, gripper, cucumber = self.OMRON, self.omron_gripper, self.environment.cucumber
        R = self.environment.CUCUMBER_RADIUS
        T_cuc = SE3(np.asarray(cucumber.T))
        T_grip = T_cuc * self._cucumber_grasp()
        T_pre = SE3(self.FRIDGE_APPROACH, 0, 0.01) * T_grip        # level, in front of the fridge

        print("Omron fetching the cucumber from the fridge")
        gripper.open()
        ok = (self._move_above("Omron to fridge", O, T_pre, T_grip)
              and self._move("Omron reach into fridge", O, T_grip, linear=True,
                             steps=10, position_tolerance=0.005))
        if not ok:
            return False
        gripper.grip("grip_cucumber")
        self.env.step(0.3)
        O.attach(cucumber)                                         # carried, no collision yet

        # Lift 1.5 cm off the shelf (the cucumber is touching the shelf, so its
        # collision check starts after this), then pull it straight out
        T_lift = SE3(0, 0, 0.015) * T_grip
        if not self._move("Omron lift cucumber", O, T_lift, linear=True, steps=4,
                          position_tolerance=0.005):
            return False
        O.detach(cucumber)
        O.attach(cucumber, collision_segments=self._cucumber_axis_in_flange(T_cuc * SE3(0, 0, 0.015), O.q),
                 collision_radius=R)
        if not self._move("Omron pull cucumber out", O, SE3(self.FRIDGE_PULL_OUT, 0, 0) * T_lift,
                          linear=True, steps=10, position_tolerance=0.005):
            return False

        # Carry it to the bench and lay it down
        choice = self._choose_cucumber_place()
        if choice is None:
            print("No way found to lay the cucumber on the bench")
            self.step_log.append(("Omron place cucumber", False))
            return False
        T_cuc_bench, T_place, T_above = choice
        ok = (self._move_above("Omron carry cucumber to bench", O, T_above, T_place)
              and self._move("Omron lower cucumber", O, T_place, linear=True, steps=8,
                             position_tolerance=0.005))
        if not ok:
            return False
        gripper.open()
        O.detach(cucumber)
        cucumber.T = T_cuc_bench.A                                  # settles onto the bench
        self.env.step(0.3)
        self._move("Omron move away from cucumber", O, T_above, linear=True, steps=8,
                   position_tolerance=0.005)
        self._park(O, "Omron back to start (out of the Motoman's way)")
        return True

    def _park(self, robot, label):
        """Send a robot back to the joint angles it started the task with."""
        q_home = self._home_q[robot.name]
        return self._move(label, robot, SE3(robot.fkine(q_home).A), preferred_q=q_home)

    def run_task(self):
        env_mgr = self.environment
        # Where each robot started: it can return here to get out of the others' way
        self._home_q = {r.name: np.array(r.q, dtype=float)
                        for r in (self.Motoman, self.OMRON, self.DOBOT)}
        H, R = env_mgr.GLASS_HEIGHT, env_mgr.GLASS_RADIUS

        water = None
        cuc_slice = None

        # ---- 0. Omron takes the cucumber out of the fridge ----------------------
        if not self.fetch_cucumber():
            print("Cucumber not delivered to the bench - the Motoman will cut thin air")

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

        # Knife tilted 15 deg nose-down, hand just behind the bench edge: the blade
        # reaches the board over the cucumber while the wrist housing stays clear
        # of the bench (at the old level pose the wrist was lower than the blade).
        print("Motoman performing slicing")
        cut_pose = SE3.RPY([0, self.CUT_PITCH, 0])
        y = 0
        for i in range(5):
            self._move(f"Motoman slice {i+1} up", self.Motoman,
                       SE3(self.CUT_FLANGE_X, y, self.CUT_UP_Z) @ cut_pose, linear=True,
                       steps=10, position_tolerance=0.005, check_reachability=True)
            self._move(f"Motoman slice {i+1} down", self.Motoman,
                       SE3(self.CUT_FLANGE_X, y, self.CUT_DOWN_Z) @ cut_pose, linear=True,
                       steps=10, position_tolerance=0.005, check_reachability=True)
            y += -0.05
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
        T_front = self._dobot_flange_for_glass(self._sink_front_position(), pi/2)
        have_glass = self.pick_glass()
        filled = (have_glass
                  and self._move_glass("Dobot to front of tap", T_front)
                  and self._move_glass("Dobot slide glass under tap",
                                       self._dobot_flange_for_glass(self.SINK_GLASS_POSITION, pi/2),
                                       steps=15))
        if filled:
            self.glass_tilt_limit = self.GLASS_TILT_LIMIT      # full glass: keep it level
            water = env_mgr.add_water()
            # water cylinder is centred on its origin: put it in the lower part of the glass
            self.DOBOT.attach(water, self.T_flange_glass @ SE3(0, 0, 0.6 * H / 2 + 0.005).A)
            for _ in range(10):
                self.env.step(0.05)

        # ---- 3. Dobot holds the glass out for the Omron -------------------
        if filled:
            self._move_glass("Dobot slide glass out from tap", T_front, steps=15)
        print("Dobot holding out the glass")
        held = have_glass and self._move_glass("Dobot hold out glass",
                                self._dobot_flange_for_glass(self.HANDOVER_GLASS_POSITION, 0.0))

        # ---- 4. Omron picks up the slice ----------------------------------
        print("Omron picking up the slice")
        sx, sy, sz = self.SLICE_POSITION
        pick_z = sz + self.PICK_GAP + self.GRIPPER_LENGTH_OMRON   # flange height: grip at slice centre
        above_ok = self._move_above("Omron above slice", self.OMRON,
                                    SE3(sx, sy, pick_z + self.APPROACH_HEIGHT) * self.TOOL_DOWN,
                                    SE3(sx, sy, pick_z) * self.TOOL_DOWN)
        picked = above_ok and self._move("Omron pick slice", self.OMRON,
                                         SE3(sx, sy, pick_z) * self.TOOL_DOWN, linear=True,
                                         steps=10, position_tolerance=0.005)
        if picked:
            self.omron_gripper.grip()
            self.env.step(0.3)
            self.OMRON.attach(cuc_slice)          # slice now moves with the Omron
            self._move("Omron lift slice", self.OMRON,
                       SE3(sx, sy, pick_z + self.APPROACH_HEIGHT) * self.TOOL_DOWN, linear=True,
                       steps=10, position_tolerance=0.005)

        # ---- 5. Omron drops the slice into the glass ------------------------
        if picked and held:
            print("Omron putting the slice in the glass")
            gx, gy, gz = self.HANDOVER_GLASS_POSITION
            drop_z = gz + H / 2 + self.DROP_CLEARANCE + self.GRIPPER_LENGTH_OMRON
            over = self._move_above("Omron above glass", self.OMRON,
                                    SE3(gx, gy, drop_z + self.DROP_APPROACH_HEIGHT) * self.TOOL_DOWN,
                                    SE3(gx, gy, drop_z) * self.TOOL_DOWN)
            dropped = over and self._move("Omron lower to glass", self.OMRON,
                                          SE3(gx, gy, drop_z) * self.TOOL_DOWN, linear=True,
                                          steps=10, position_tolerance=0.005)
            if dropped:
                # Release: slice drops to the bottom of the glass and moves with it
                self.omron_gripper.open()
                self.OMRON.detach(cuc_slice)
                cuc_slice.T = (SE3(gx, gy, gz - H / 2 + env_mgr.SLICE_RADIUS + 0.005)).A
                self.DOBOT.attach(cuc_slice)
                self.env.step(0.05)
                self._move("Omron move away", self.OMRON,
                           SE3(gx, gy, drop_z + self.DROP_APPROACH_HEIGHT) * self.TOOL_DOWN, linear=True,
                           steps=10, position_tolerance=0.005)
        else:
            print("Skipping the drop: " +
                  ("slice was not picked up" if not picked else "glass is not held out"))

        # ---- 6. Glass with the slice goes back on the counter -------------------
        self._park(self.OMRON, "Omron back to start")
        if have_glass:
            if self.place_glass(extras=(water, cuc_slice if cuc_slice in [a['shape'] for a in
                                                                          self.DOBOT.__dict__.get('_attachments', [])]
                                        else None)):
                self._park(self.DOBOT, "Dobot back to start")

        # ---- Summary --------------------------------------------------------
        print("\n" + "=" * 50 + "\nTASK SUMMARY\n" + "=" * 50)
        for label, ok in self.step_log:
            print(f"  {'OK    ' if ok else 'FAILED'}  {label}")
        print("=" * 50)


if __name__ == "__main__":
    # py -3.11 Task-Simulation/task_simulation.py           -> runs the task straight away
    # py -3.11 Task-Simulation/task_simulation.py --teach   -> teach pendant first
    sim = TaskSimulation(use_teach_pendant="--teach" in sys.argv)
    sim.launch_simulation()