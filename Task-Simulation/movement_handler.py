import numpy as np
from scipy import linalg
from spatialmath.base import *
from spatialmath import SE3
from roboticstoolbox import jtraj, trapezoidal
import pygame
import keyboard
import time

from collision_handler import (
    check_single_config_collision, 
    is_collision,
    find_collision_free_path_rrt
)

def ikine_move(robot, target_pose, env=None, q0=None, 
               steps=50, interpolation=2, point_freq=1, 
               visualize=True, show_path_dots=True,
               max_attempts=50, tolerance=0.01,
               # Collision parameters
               check_collision=True, faces=None, vertices=None, 
               face_normals=None, collisions_list=None,
               collision_free_search=True, stop_on_collision=True,
               other_robots=None):
   
    # Validate collision parameters
    check_collision = validate_collision_params(
        check_collision, faces, vertices, face_normals
    )
    
    # Setup
    q_start = robot.q.copy() if q0 is None else q0
    T_target = SE3(target_pose) if isinstance(target_pose, np.ndarray) else target_pose
    
    print_section_header("IKINE MOVE", check_collision)
    
    # Create collision checking function if enabled
    collision_check_fn = None

    if check_collision:
        def collision_check_fn(q):
            result, _ = check_single_config_collision(
                robot, q, faces, vertices, face_normals, return_once_found=True,
                other_robots=other_robots
            )
            return result
    
    # ========== PHASE 1: Solve IK for target ==========
    #Create varied initial guesses - current, zero, random
    initial_guesses = generate_ik_initial_guesses(robot, q_start, max_attempts)
    
    #solve IK with different masks - position only, position bias, position+orientation
    ik_result = solve_ik_with_strategies(
        robot, T_target, initial_guesses,
        tolerance=tolerance,
        check_collision_fn=collision_check_fn
    )
    
    #Logging
    if not ik_result['success']:
        print(f"IK failed after {ik_result['attempts']} attempts (error: {ik_result['error']*1000:.2f}mm)")
        print_section_footer()
        return None, None, False
    
    if not ik_result['collision_free'] and stop_on_collision:
        print(f"IK solution in collision (error: {ik_result['error']*1000:.2f}mm) - ABORTED")
        print_section_footer()
        return None, None, False
    
    q_target = ik_result['q']
    status = "collision-free" if ik_result['collision_free'] else "IN COLLISION"
    print(f"IK solved (attempts: {ik_result['attempts']}, error: {ik_result['error']*1000:.2f}mm, {status})")
    
    # ========== PHASE 2: Generate trajectory ==========
    #Set interpolation - quintic or trapezoidal
    if interpolation == 1:
        q_matrix = jtraj(q_start, q_target, steps).q
    elif interpolation == 2:
        s = trapezoidal(0, 1, steps).q
        q_matrix = np.empty((steps, robot.n))
        for i in range(steps):
            q_matrix[i, :] = (1 - s[i]) * q_start + s[i] * q_target
    else:
        raise ValueError("interpolation must be 1 (Quintic) or 2 (Trapezoidal)")
    
    # ========== PHASE 3: Check trajectory for collisions ==========
    #Attempt RRT if path still has collisions
    trajectory_success = True

    if check_collision:
        
        print("\nChecking trajectory for collisions...")
        traj_collision = is_collision(
            robot, list(q_matrix), faces, vertices, face_normals,
            collisions_list=collisions_list, env=env, return_once_found=True,
            other_robots=other_robots
        )
        
        if traj_collision:
            print("Trajectory has collision")
            trajectory_success = False
            
            if collision_free_search:
                print("Searching for collision-free path with RRT...")
                
                q_path, rrt_success = find_collision_free_path_rrt(
                    robot, q_start, q_target,
                    faces, vertices, face_normals,
                    env=env, collisions_list=collisions_list,
                    max_iterations=1000,
                    visualize=False,
                    other_robots=other_robots
                )
                
                if rrt_success:
                    print(f"RRT found collision-free path with {len(q_path)} waypoints")
                    q_matrix = np.array(q_path)
                    trajectory_success = True
                else:
                    print("RRT could not find collision-free path")
                    
                    if stop_on_collision:
                        print("Motion ABORTED due to collision")
                        print_section_footer()
                        return None, None, False
                    else:
                        print("Proceeding with colliding path (stop_on_collision=False)")
            else:
                if stop_on_collision:
                    print("Motion ABORTED due to collision (collision_free_search=False)")
                    print_section_footer()
                    return None, None, False
                else:
                    print("Proceeding with colliding path (stop_on_collision=False)")
        else:
            print("Trajectory is collision-free")
    
    # ========== PHASE 4: Visualization ==========
    if visualize and env is not None:
        if not trajectory_success:
            print("\n WARNING: Visualizing trajectory that contains collisions!")
        
        animate_trajectory_unified(
            robot, q_matrix, T_target, env,
            point_freq=point_freq,
            collision_free=trajectory_success,
            show_dots=show_path_dots
        )
    
    print_section_footer()
    return q_matrix, q_target, trajectory_success

def linear_move(robot, target_pose, env=None, q0=None, steps=50, 
                point_freq=1, visualize=True, show_path_dots=True, 
                max_ik_attempts=10, 
                position_tolerance=0.005, check_reachability=True, 
                smooth_failures=True,
                # Collision parameters
                check_collision=True, faces=None, vertices=None, 
                face_normals=None, stop_on_collision=True,
                other_robots=None):
    #Increase waypoints to create smoother path

    # Validate collision parameters
    check_collision = validate_collision_params(
        check_collision, faces, vertices, face_normals
    )
    
    # Initialize
    q_start = robot.q.copy() if q0 is None else q0
    T_target = SE3(target_pose) if isinstance(target_pose, np.ndarray) else target_pose
    T_start = robot.fkine(q_start)
    
    linear_dist = np.linalg.norm(T_target.t - T_start.t)
    step_size = linear_dist / (steps - 1) if steps > 1 else 0
    
    print_section_header("LINEAR MOVE", check_collision)
    print(f"Distance: {linear_dist:.4f}m | Waypoints: {steps} | Step: {step_size*1000:.2f}mm")
    
    # Create collision checking function if enabled
    collision_check_fn = None
    if check_collision:
        def collision_check_fn(q):
            result, _ = check_single_config_collision(
                robot, q, faces, vertices, face_normals, return_once_found=True,
                other_robots=other_robots
            )
            return result
    
    # ========== PHASE 1: Check target reachability ==========
    if check_reachability:
        print(f"\nChecking target reachability (tolerance: {position_tolerance*1000:.1f}mm)...")
        target_reachable = check_target_reachable(
            robot, T_target, q_start, 
            tolerance=position_tolerance,
            max_attempts=max_ik_attempts,
            check_collision_fn=collision_check_fn
        )
        
        if not target_reachable:
            print("Target not reachable")
            print_section_footer()
            return None, False, {'error': 'target_unreachable'}
        else:
            print("Target is reachable")
    
    # Initialize trajectory storage
    q_matrix = np.zeros((steps, robot.n))
    q_matrix[0, :] = q_start
    success_flags = np.zeros(steps, dtype=bool)
    success_flags[0] = True
    
    diagnostics = {
        'ik_attempts': np.zeros(steps),
        'position_errors': np.zeros(steps),
        'max_joint_velocities': np.zeros(steps),
        'failed_waypoints': [],
        'collision_waypoints': []
    }
    
    q_prev = q_start.copy()
    all_success = True
    
    print(f"\nSolving IK for {steps} waypoints...")
    
    # ========== PHASE 2: Solve IK for all waypoints ==========
    for i in range(steps):
        if i % max(1, steps // 20) == 0:
            print(f"{int(100*i/steps)}% ", end='', flush=True)
        
        # Interpolate pose
        alpha = i / (steps - 1) if steps > 1 else 1.0
        T_interp = T_start.interp(T_target, alpha)
        
        # Generate initial guesses for this waypoint
        initial_guesses = [q_prev, q_start, np.zeros(robot.n)]
        
        # Solve IK
        ik_result = solve_ik_with_strategies(
            robot, T_interp, initial_guesses,
            tolerance=position_tolerance,
            check_collision_fn=None 
            
        )
        
        if ik_result['success']:
            q_matrix[i, :] = ik_result['q']
            success_flags[i] = True
            diagnostics['ik_attempts'][i] = ik_result['attempts']
            diagnostics['position_errors'][i] = ik_result['error']
            
            # Check collision
            if check_collision and collision_check_fn is not None:
                collision = collision_check_fn(q_matrix[i, :])
                
                if collision:
                    diagnostics['collision_waypoints'].append(i)
                    print(f"\n Waypoint {i} in collision!")
                    success_flags[i] = False
                    all_success = False
            
            # Track joint velocities
            if i > 0:
                dq = q_matrix[i, :] - q_matrix[i-1, :]
                diagnostics['max_joint_velocities'][i] = np.max(np.abs(dq))
            
            q_prev = q_matrix[i, :].copy()
            
        else:
            print(f"\n Waypoint {i} IK failed (error: {ik_result['error']*1000:.1f}mm)")
            success_flags[i] = False
            diagnostics['failed_waypoints'].append(i)
            all_success = False
    
    print("100% ✓")
    
    # ========== PHASE 3: Handle failures ==========
    has_failures = len(diagnostics['failed_waypoints']) > 0 or len(diagnostics['collision_waypoints']) > 0
    
    if has_failures:
        if stop_on_collision and len(diagnostics['collision_waypoints']) > 0:
            print(f"\n✗ Motion ABORTED: {len(diagnostics['collision_waypoints'])} waypoints in collision")
            print_section_footer()
            return None, False, diagnostics
        
        if smooth_failures:
            failed_set = set(diagnostics['failed_waypoints'] + diagnostics['collision_waypoints'])
            print(f"\n Smoothing {len(failed_set)} problematic waypoints...")
            print(" WARNING: Smoothing may not guarantee collision-free path!")
            
            # Interpolate through failures
            i = 0
            while i < steps:
                if not success_flags[i]:
                    start_fail = i
                    while i < steps and not success_flags[i]:
                        i += 1
                    end_fail = i - 1
                    
                    start_valid = max(0, start_fail - 1)
                    end_valid = min(steps - 1, end_fail + 1)
                    
                    if success_flags[start_valid] and success_flags[end_valid]:
                        q_start_interp = q_matrix[start_valid, :]
                        q_end_interp = q_matrix[end_valid, :]
                        num_failed = end_fail - start_fail + 1
                        
                        for j, idx in enumerate(range(start_fail, end_fail + 1)):
                            alpha = (j + 1) / (num_failed + 1)
                            q_matrix[idx, :] = (1 - alpha) * q_start_interp + alpha * q_end_interp
                            success_flags[idx] = True
                        
                        print(f"   Interpolated waypoints {start_fail} to {end_fail}")
                else:
                    i += 1
    
    # ========== PHASE 4: Summary ==========
    print(f"\n{'='*70}")
    print(f"TRAJECTORY SUMMARY")
    print(f"{'='*70}")
    
    if diagnostics['failed_waypoints']:
        print(f" IK failures: {len(diagnostics['failed_waypoints'])}/{steps}")
    if diagnostics['collision_waypoints']:
        print(f" Collisions: {len(diagnostics['collision_waypoints'])}/{steps}")
    if not diagnostics['failed_waypoints'] and not diagnostics['collision_waypoints']:
        print(f" All waypoints solved successfully")
    
    # ========== PHASE 5: Visualization ==========
    if visualize and env is not None:
        if not all_success:
            print(f"\n WARNING: Visualizing trajectory with known issues")
        
        total_error = animate_trajectory_unified(
            robot, q_matrix, T_target, env,
            point_freq=point_freq,
            collision_free=all_success,
            show_dots=show_path_dots
        )
        
        if total_error > position_tolerance:
            all_success = False
    
    print_section_footer()
    
    return q_matrix, all_success, diagnostics

#Position only currently
def rmrc_move(robot, q0, cartesian_path, env, delta_t=0.05, 
              min_manip=0.01, damping_factor=0.01
              ):
    
    # Get dimensions of path
    dimension = cartesian_path.shape[0]
    steps = cartesian_path.shape[1]
    
    # Initialize storage
    q_trajectory = []
    manipulability = np.zeros(steps)
    velocity_error = np.zeros((dimension, steps))
    
    q_trajectory.append(q0)
    robot.q = q0
    
    # RMRC loop
    for i in range(steps - 1):
        # Calculate desired Cartesian velocity
        x_dot = (cartesian_path[:, i+1] - cartesian_path[:, i]) / delta_t
        
        # Get Jacobian at CURRENT configuration in trajectory
        # (This is the key - we use q_trajectory[i], which may differ from robot.q)
        q_current = q_trajectory[i]
        J = robot.jacob0(q_current)
        J = J[:dimension, :]  # Take only position rows
        
        # Calculate manipulability
        manipulability[i] = np.sqrt(max(0, linalg.det(J @ J.T)))
        
        # Solve for joint velocities
        if manipulability[i] < min_manip:
            # Use Damped Least Squares for low manipulability
            q_dot = linalg.inv(J.T @ J + damping_factor * np.eye(robot.n)) @ J.T @ x_dot
            if i % 10 == 0:
                print(f"  Step {i}: Low manip ({manipulability[i]:.3f}), using DLS")
        else:
            # Use pseudoinverse for good manipulability
            q_dot = linalg.pinv(J) @ x_dot
        
        # Calculate velocity error
        velocity_error[:, i] = x_dot - J @ q_dot
        
        # Update joint configuration
        q_next = q_current + delta_t * q_dot
        q_trajectory.append(q_next)
        
        robot.q = q_next
        env.step(delta_t)

    final_pos = robot.fkine(robot.q).t
    target_pos = cartesian_path[:, -1]
    error_mm = np.linalg.norm(final_pos - target_pos) * 1000
    print(f"RMRC: Complete. Final error: {error_mm:.2f}mm")
    
    return q_trajectory, manipulability, velocity_error

def jog_velocity(robot, velocity_cmd, dt=0.1, lambda_dls=0.1, max_joint_vel=1.0, 
                 enforce_limits=True, manipulability_threshold=0.05):

    # Get Jacobian in base frame
    J = robot.jacob0(robot.q)
    
    # Calculate manipulability
    manipulability = np.sqrt(max(0, linalg.det(J @ J.T)))
    
    # Adaptive DLS damping near singularities
    if manipulability < manipulability_threshold:
        lambda_adaptive = lambda_dls * (1 + (manipulability_threshold - manipulability) / manipulability_threshold)
    else:
        lambda_adaptive = lambda_dls
    
    # Damped Least Squares inverse
    JTJ = J.T @ J
    lambda_I = (lambda_adaptive ** 2) * np.eye(J.shape[1])
    Jinv_dls = np.linalg.inv(JTJ + lambda_I) @ J.T
    
    # Convert to joint velocities
    dq = Jinv_dls @ velocity_cmd
    
    # Safety: Clamp joint velocities
    dq_clamped = np.clip(dq, -max_joint_vel, max_joint_vel)
    velocity_limited = not np.allclose(dq, dq_clamped)
    
    # Integrate to get new position
    q_new = robot.q + dq_clamped * dt
    
    # Check joint limits
    within_limits = True
    if enforce_limits:
        for i in range(len(q_new)):
            if hasattr(robot.links[i], 'qlim') and robot.links[i].qlim is not None:
                q_min, q_max = robot.links[i].qlim
                if q_new[i] < q_min or q_new[i] > q_max:
                    q_new[i] = np.clip(q_new[i], q_min, q_max)
                    within_limits = False

    # Status information
    status = {
        'manipulability': manipulability,
        'velocity_limited': velocity_limited,
        'within_limits': within_limits,
        'joint_velocities': dq_clamped,
        'near_singularity': manipulability < manipulability_threshold,
    }
    
    return q_new, status

def start_joystick_control(robot, env, duration=300, dt=0.1, 
                           Kv=0.3, Kw=0.8, lambda_dls=0.1, 
                           max_joint_vel=1.0, show_status=True):
    # Initialize pygame and joystick
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No joystick detected.")
        return None

    joy = pygame.joystick.Joystick(0)
    joy.init()

    joystick_deadzone = 0.1
    
    # Trajectory storage
    trajectory = []
    trajectory.append(robot.q.copy())
    
    # Timing
    start_time = time.time()
    iteration = 0

    try:
        while (time.time() - start_time) < duration:
            pygame.event.pump()
            iteration += 1
            current_time = time.time() - start_time
            
            # Read joystick inputs
            axes_raw = [joy.get_axis(i) for i in range(min(4, joy.get_numaxes()))]
            axes = [0 if abs(a) < joystick_deadzone else a for a in axes_raw]
            while len(axes) < 4:
                axes.append(0)
            
            # Read buttons
            num_buttons = joy.get_numbuttons()
            buttons = [joy.get_button(i) for i in range(num_buttons)]
            
            def get_button(idx, default=0):
                return buttons[idx] if idx < len(buttons) else default
            
            # Map joystick to 6D velocity (only translation and rotation)
            vx = Kv * axes[0]
            vy = -Kv * axes[1]
            vz = 0  # No Z translation control
            wx = Kw * axes[2]
            wy = -Kw * axes[3]
            wz = 0  # No Z rotation control
            
            velocity_cmd = np.array([vx, vy, vz, wx, wy, wz])
            
            # Calculate new joint configuration
            q_new, status = jog_velocity(
                robot, velocity_cmd, dt, lambda_dls, max_joint_vel
            )
            
            # Apply motion
            robot.q = q_new
            trajectory.append(robot.q.copy())
            
            if env is not None:
                env.step(dt)
            
            # Status display
            if show_status and iteration % 10 == 0:
                T = robot.fkine(robot.q)
                vel_norm = np.linalg.norm(velocity_cmd[:3])
                
                status_str = f"[{iteration:4d}] Pos: [{T.t[0]:6.3f}, {T.t[1]:6.3f}, {T.t[2]:6.3f}]"
                status_str += f" | Vel: {vel_norm:5.3f} m/s"
                status_str += f" | Man: {status['manipulability']:5.3f}"
                
                if status['near_singularity']:
                    status_str += " ⚠️ SING"
                if status['velocity_limited']:
                    status_str += " 🔴 VLIM"
                if not status['within_limits']:
                    status_str += " 🔴 JLIM"
                
                print(status_str, end='\r')
            
            # Exit command (Start button or keyboard)
            if get_button(9) or keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
                print("\n\n✋ Exiting joystick control...")
                break

    except KeyboardInterrupt:
        print("\n\n⚠️  [INTERRUPT] Control stopped by user...")
    
    finally:
        pygame.quit()
        print("\n" + "=" * 70)
        print(f"📊 SESSION SUMMARY")
        print("=" * 70)
        print(f"Duration: {time.time() - start_time:.1f} seconds")
        print(f"Iterations: {iteration}")
        print(f"Trajectory points: {len(trajectory)}")
        print(f"Final position: {robot.fkine(robot.q).t}")
        print("=" * 70)
        print("🎮 Joystick control ended. Resuming code execution...")
        
        return np.array(trajectory)

#Helper Functions
def validate_collision_params(check_collision, faces, vertices, face_normals):
    """Validate Obstacle Data is passed."""
    if check_collision and (faces is None or vertices is None or face_normals is None):
        print("Obstacle data not provided. Collision checking disabled.")
        return False
    return check_collision

def print_section_header(title, collision_enabled=False):
    """Print consistent section headers."""
    suffix = " WITH COLLISION AVOIDANCE" if collision_enabled else ""
    print("\n" + "="*70)
    print(f"{title}{suffix}")
    print("="*70)

def print_section_footer():
    """Print section footer."""
    print("="*70 + "\n")

def generate_ik_initial_guesses(robot, q_start, max_attempts=10):
    """Generate varied initial guesses."""

    guesses = [
        q_start,                           # Current position
        np.zeros(robot.n),                 # Zero configuration
        robot.qz if hasattr(robot, 'qz') else np.zeros(robot.n)  # Default pose
    ]
    
    # Add random configurations
    for _ in range(max_attempts - len(guesses)):
        q_rand = np.zeros(robot.n)
        for j in range(robot.n):
            if hasattr(robot.links[j], 'qlim') and robot.links[j].qlim is not None:
                q_min, q_max = robot.links[j].qlim
                q_rand[j] = np.random.uniform(q_min, q_max)
            else:
                q_rand[j] = np.random.uniform(-np.pi, np.pi)
        guesses.append(q_rand)
    
    return guesses

def solve_ik_with_strategies(robot, T_target, initial_guesses, 
                            tolerance=0.01, check_collision_fn=None):

    best_solution = None
    best_error = float('inf')
    
    # Define IK masks to attempt with
    strategies = [
        {'mask': [1,1,1,1,1,1], 'name': 'full'},
        {'mask': [1,1,1,0,0,0], 'name': 'position_only'},
        {'mask': [1,1,1,0.1,0.1,0.1], 'name': 'position_weighted'},
    ]
    
    attempt_count = 0
    
    for q_init in initial_guesses:
        #run through masks
        for strategy in strategies:
            attempt_count += 1
            
            try:
                ik_solution = robot.ikine_LM(T_target, q0=q_init, mask=strategy['mask'])
                
                if ik_solution is not None and hasattr(ik_solution, 'q'):
                    T_achieved = robot.fkine(ik_solution.q)
                    pos_error = np.linalg.norm(T_target.t - T_achieved.t)
                    
                    # Check if this solution is better
                    if pos_error < best_error:
                        best_error = pos_error
                        best_solution = ik_solution.q
                    
                    # Check if solution meets tolerance
                    if ik_solution.success and pos_error < tolerance:
                        # Check collision
                        is_collision_free = True
                        if check_collision_fn is not None:
                            is_collision_free = not check_collision_fn(ik_solution.q)
                            
                            if not is_collision_free:
                                continue  # Try next strategy
                        
                        # Found valid solution
                        return {
                            'success': True,
                            'q': ik_solution.q,
                            'error': pos_error,
                            'attempts': attempt_count,
                            'collision_free': is_collision_free
                        }
            
            except Exception as e:
                continue
    
    # No perfect solution found, return best effort
    if best_solution is not None:
        is_collision_free = True
        if check_collision_fn is not None:
            is_collision_free = not check_collision_fn(best_solution)
        
        return {
            'success': best_error < tolerance,  
            'q': best_solution,
            'error': best_error,
            'attempts': attempt_count,
            'collision_free': is_collision_free
        }
    
    # Complete failure
    return {
        'success': False,
        'q': None,
        'error': best_error,
        'attempts': attempt_count,
        'collision_free': False
    }

def check_target_reachable(robot, T_target, q_start, tolerance=0.01, 
                          max_attempts=10, check_collision_fn=None):
    """Check if target pose is reachable using IK."""
    initial_guesses = generate_ik_initial_guesses(robot, q_start, max_attempts)
    
    result = solve_ik_with_strategies(
        robot, T_target, initial_guesses, 
        tolerance=tolerance,
        check_collision_fn=check_collision_fn
    )
    
    return result['success'] and result['error'] < tolerance

def animate_trajectory_unified(robot, q_matrix, T_target, env, 
                               point_freq=1, collision_free=True, 
                               show_dots=True):
    """Animate trajectory in Swift environment."""
    if env is None:
        return 0.0
    
    points = []
    steps = len(q_matrix)
    
    print("\n Animating trajectory")
    
    for i, q in enumerate(q_matrix):
        robot.q = q
        env.step(0.05)
        
        # Add visual dots
        if show_dots and i % point_freq == 0:
            from spatialgeometry import Sphere
            color = [0.0, 1.0, 0.0, 1.0] if collision_free else [1.0, 0.5, 0.0, 1.0]
            sphere = Sphere(radius=0.03, color=color)
            sphere.T = robot.fkine(q).A
            env.add(sphere)
            points.append(sphere)
    
    # Calculate final error
    T_final = robot.fkine(robot.q)
    pos_error = np.linalg.norm(T_target.t - T_final.t)
    
    print(f" Animation complete")
    print(f" Final error: {pos_error*1000:.2f}mm")
    
    return pos_error

def create_linear_path(start_pos, end_pos, steps=50):
    """
    Create a straight line path between two positions.
    
    Args:
        start_pos: Starting position [x, y, z] (array or list)
        end_pos: Ending position [x, y, z]
        steps: Number of waypoints
    
    Returns:
        cartesian_path: Shape (3, steps) for RMRC
    """
    start_pos = np.array(start_pos)
    end_pos = np.array(end_pos)
    
    # Create path: each column is a waypoint
    cartesian_path = np.zeros((3, steps))
    
    for i in range(steps):
        alpha = i / (steps - 1)  # 0 to 1
        cartesian_path[:, i] = (1 - alpha) * start_pos + alpha * end_pos
    
    return cartesian_path