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

# A move only counts as successful if the tool ends within this angle of the
# requested orientation (unless the move is called with position_only=True)
DEFAULT_ORIENTATION_TOLERANCE = np.deg2rad(1.0)   # radians

def ikine_move(robot, target_pose, env=None, q0=None, 
               steps=50, interpolation=2, point_freq=1, 
               visualize=True, show_path_dots=True,
               max_attempts=50, tolerance=0.01,
               # Collision parameters
               check_collision=True, faces=None, vertices=None, 
               face_normals=None, collisions_list=None,
               collision_free_search=True, stop_on_collision=True,
               other_robots=None,
               # Orientation parameters
               orientation_tolerance=DEFAULT_ORIENTATION_TOLERANCE,
               position_only=False):
    """Joint-space move to target_pose.

    orientation_tolerance : max orientation error in radians (default 1 deg)
    position_only         : True to ignore orientation entirely (only if the
                            tool direction really doesn't matter for this move)
    """
   
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
    
    # Solve IK: full pose (unless position_only), within joint limits,
    # collision-free, and closest to the current joints
    ik_result = solve_ik_with_strategies(
        robot, T_target, initial_guesses,
        tolerance=tolerance,
        check_collision_fn=collision_check_fn if stop_on_collision else None,
        orientation_tolerance=orientation_tolerance,
        position_only=position_only,
        q_reference=q_start
    )
    
    #Logging
    if not ik_result['success']:
        if ik_result['reason'] == 'collision':
            print(f"IK solution in collision (error: {ik_result['error']*1000:.2f}mm) - ABORTED")
        else:
            print(f"IK failed after {ik_result['attempts']} attempts: "
                  f"{describe_ik_failure(ik_result, tolerance, orientation_tolerance)}")
        print_section_footer()
        return None, None, False
    
    q_target = ik_result['q']
    if not stop_on_collision and collision_check_fn is not None and collision_check_fn(q_target):
        ik_result['collision_free'] = False
    status = "collision-free" if ik_result['collision_free'] else "IN COLLISION"
    ori_txt = "ignored" if position_only else f"{np.degrees(ik_result['ori_error']):.2f}deg"
    print(f"IK solved (attempts: {ik_result['attempts']}, error: {ik_result['error']*1000:.2f}mm, "
          f"orientation: {ori_txt}, {status})")
    
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
                other_robots=None,
                # Orientation / smoothness parameters
                orientation_tolerance=DEFAULT_ORIENTATION_TOLERANCE,
                position_only=False,
                max_joint_step=np.deg2rad(20)):
    """Straight-line Cartesian move, solved as IK at each waypoint.

    orientation_tolerance : max orientation error per waypoint, radians
    position_only         : True to ignore orientation entirely
    max_joint_step        : largest allowed joint change between waypoints;
                            bigger means the arm flipped configuration
    """

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
        reach = check_target_reachable(
            robot, T_target, q_start, 
            tolerance=position_tolerance,
            max_attempts=max_ik_attempts,
            check_collision_fn=collision_check_fn,
            orientation_tolerance=orientation_tolerance,
            position_only=position_only
        )
        
        if not reach['success']:
            print("Target not reachable: " +
                  describe_ik_failure(reach, position_tolerance, orientation_tolerance))
            print_section_footer()
            return None, False, {'error': 'target_unreachable', 'reason': reach['reason']}
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
        'collision_waypoints': [],
        'jump_waypoints': []
    }
    
    q_prev = q_start.copy()
    all_success = True
    
    print(f"\nSolving IK for {steps} waypoints...")
    
    # ========== PHASE 2: Solve IK for all waypoints ==========
    # Waypoint 0 is the robot's current joints (no IK, so no jump at the start)
    for i in range(1, steps):
        if i % max(1, steps // 20) == 0:
            print(f"{int(100*i/steps)}% ", end='', flush=True)
        
        # Interpolate pose
        alpha = i / (steps - 1) if steps > 1 else 1.0
        T_interp = T_start.interp(T_target, alpha)
        
        # Previous waypoint first, so IK stays on the same arm configuration
        initial_guesses = [q_prev, q_start, np.zeros(robot.n)]
        
        ik_result = solve_ik_with_strategies(
            robot, T_interp, initial_guesses,
            tolerance=position_tolerance,
            check_collision_fn=None,           # collisions checked below, per waypoint
            orientation_tolerance=orientation_tolerance,
            position_only=position_only,
            q_reference=q_prev,
            max_solutions=len(initial_guesses)
        )
        
        if ik_result['success']:
            q_i = ik_result['q']
            dq = q_i - q_prev
            diagnostics['max_joint_velocities'][i] = np.max(np.abs(dq))
            
            # A big joint change between neighbouring waypoints = configuration flip
            if np.max(np.abs(dq)) > max_joint_step:
                print(f"\n Waypoint {i}: joint jump of {np.degrees(np.max(np.abs(dq))):.1f}deg "
                      f"(limit {np.degrees(max_joint_step):.0f}deg)")
                diagnostics['jump_waypoints'].append(i)
                q_matrix[i, :] = q_prev         # placeholder, never zeros
                success_flags[i] = False
                all_success = False
                continue
            
            q_matrix[i, :] = q_i
            success_flags[i] = True
            diagnostics['ik_attempts'][i] = ik_result['attempts']
            diagnostics['position_errors'][i] = ik_result['error']
            
            # Check collision
            if check_collision and collision_check_fn is not None:
                if collision_check_fn(q_i):
                    diagnostics['collision_waypoints'].append(i)
                    print(f"\n Waypoint {i} in collision!")
                    success_flags[i] = False
                    all_success = False
            
            q_prev = q_i.copy()
            
        else:
            print(f"\n Waypoint {i} IK failed: "
                  f"{describe_ik_failure(ik_result, position_tolerance, orientation_tolerance)}")
            q_matrix[i, :] = q_prev             # placeholder, never zeros
            success_flags[i] = False
            diagnostics['failed_waypoints'].append(i)
            all_success = False
    
    print("100% ✓")
    
    # ========== PHASE 3: Handle failures ==========
    has_failures = (len(diagnostics['failed_waypoints']) > 0 or len(diagnostics['collision_waypoints']) > 0
                    or len(diagnostics['jump_waypoints']) > 0)
    
    if has_failures:
        if stop_on_collision and len(diagnostics['collision_waypoints']) > 0:
            print(f"\n✗ Motion ABORTED: {len(diagnostics['collision_waypoints'])} waypoints in collision")
            print_section_footer()
            return None, False, diagnostics
        
        if smooth_failures:
            failed_set = set(diagnostics['failed_waypoints'] + diagnostics['collision_waypoints']
                             + diagnostics['jump_waypoints'])
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
    
    # Anything still unresolved (e.g. the final waypoint failed) -> don't move
    if not np.all(success_flags):
        bad = np.where(~success_flags)[0].tolist()
        print(f"\n✗ Motion ABORTED: waypoints {bad} could not be solved or smoothed")
        print_section_footer()
        return None, False, diagnostics
    
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

def rmrc_move(robot, q0, cartesian_path, env, delta_t=0.05,
              min_manip=0.01, damping_factor=0.01,
              # Orientation
              target_orientation=None, position_only=False,
              # Safety
              max_joint_vel=1.5,
              check_collision=True, faces=None, vertices=None, face_normals=None,
              other_robots=None, stop_on_collision=True):
    """Resolved-rate motion control along a straight Cartesian path.

    cartesian_path     : 3 x N array of positions (e.g. from create_linear_path)
    target_orientation : None  -> hold the starting tool orientation the whole way
                         SE3 / SO3 / 3x3 -> blend smoothly from start to this orientation
    position_only      : True = old behaviour (orientation free to drift)
    max_joint_vel      : rad/s; if exceeded the whole step is slowed, not distorted
    Collision checking works like ikine_move (pass faces/vertices/face_normals,
    and other_robots). The move stops before entering a collision or a joint limit.

    Returns (q_trajectory, manipulability, velocity_error) as before.
    velocity_error is 6 x N (3 x N if position_only).
    """
    from spatialmath import UnitQuaternion
    from spatialmath.base import trlog, vex

    check_collision = validate_collision_params(check_collision, faces, vertices, face_normals)
    print_section_header("RMRC MOVE", check_collision)

    steps = cartesian_path.shape[1]
    dim = 3 if position_only else 6
    q_trajectory = [np.array(q0, dtype=float)]
    manipulability = np.zeros(steps)
    velocity_error = np.zeros((dim, steps))
    qlim = get_joint_limits(robot)

    # Orientation schedule: hold, or slerp from start to target
    R_start = robot.fkine(q0).R
    if target_orientation is None:
        R_end = R_start
    else:
        R_end = target_orientation.R if hasattr(target_orientation, 'R') else np.asarray(target_orientation)
    quat_start, quat_end = UnitQuaternion(R_start), UnitQuaternion(R_end)
    same_orientation = np.allclose(R_start, R_end, atol=1e-9)
    def R_desired(i):
        # (spatialmath's interp divides by zero when start == end, so skip it then)
        if same_orientation or steps <= 1:
            return R_end
        return quat_start.interp(quat_end, i / (steps - 1)).R

    robot.q = q_trajectory[0]
    stopped = None
    slowed_steps = 0
    max_pos_dev, max_ori_dev = 0.0, 0.0

    for i in range(steps - 1):
        q = q_trajectory[i]
        T = robot.fkine(q)

        # Closed loop: aim for the NEXT path point from where the tool ACTUALLY is,
        # so any drift is corrected on the following step
        v = (cartesian_path[:, i + 1] - T.t) / delta_t
        if position_only:
            x_dot = v
        else:
            R_err = R_desired(i + 1) @ T.R.T              # rotation still needed
            omega = vex(trlog(R_err, check=False)) / delta_t if not np.allclose(R_err, np.eye(3)) \
                else np.zeros(3)
            x_dot = np.r_[v, omega]

        J = robot.jacob0(q)[:dim, :]
        manipulability[i] = np.sqrt(max(0.0, np.linalg.det(J @ J.T)))

        if manipulability[i] < min_manip:
            # Damped least squares near singularities
            q_dot = J.T @ np.linalg.solve(J @ J.T + damping_factor * np.eye(dim), x_dot)
        else:
            q_dot = np.linalg.pinv(J) @ x_dot

        # Joint speed limit: slow the whole step down (keeps the direction = stays on the line)
        peak = np.max(np.abs(q_dot))
        if peak > max_joint_vel:
            q_dot *= max_joint_vel / peak
            slowed_steps += 1

        velocity_error[:, i] = x_dot - J @ q_dot
        q_next = q + delta_t * q_dot

        # Stop BEFORE leaving the joint limits
        if np.any(q_next < qlim[0]) or np.any(q_next > qlim[1]):
            j = int(np.argmax((q_next < qlim[0]) | (q_next > qlim[1])))
            stopped = f"joint {j+1} would pass its limit at step {i+1}/{steps}"
            break

        # Stop BEFORE entering a collision
        if check_collision and stop_on_collision:
            hit, _ = check_single_config_collision(robot, q_next, faces, vertices, face_normals,
                                                   return_once_found=True, other_robots=other_robots)
            if hit:
                stopped = f"collision ahead at step {i+1}/{steps}"
                break

        q_trajectory.append(q_next)
        robot.q = q_next
        if env is not None:
            env.step(delta_t)

        # Track how far the tool strays from the path
        T_now = robot.fkine(q_next)
        max_pos_dev = max(max_pos_dev, np.linalg.norm(T_now.t - cartesian_path[:, i + 1]))
        if not position_only:
            R_d = R_desired(i + 1)
            max_ori_dev = max(max_ori_dev, np.arccos(np.clip((np.trace(R_d.T @ T_now.R) - 1) / 2, -1, 1)))

    # Summary
    T_final = robot.fkine(robot.q)
    pos_err = np.linalg.norm(T_final.t - cartesian_path[:, len(q_trajectory) - 1]) * 1000
    if stopped:
        print(f"RMRC STOPPED: {stopped}")
    print(f"RMRC: {len(q_trajectory)}/{steps} steps. Final error: {pos_err:.2f}mm. "
          f"Max deviation from line: {max_pos_dev*1000:.2f}mm"
          + ("" if position_only else f", from orientation: {np.degrees(max_ori_dev):.2f}deg"))
    if slowed_steps:
        print(f"  {slowed_steps} steps slowed to respect max_joint_vel")
    if np.any(manipulability[:len(q_trajectory) - 1] < min_manip):
        print(f"  Passed near a singularity (min manipulability "
              f"{manipulability[:len(q_trajectory) - 1].min():.4f}) - damped least squares used")
    print_section_footer()

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
    """Initial guesses for IK: current joints first, then defaults, then random."""
    guesses = [
        np.array(q_start, dtype=float),    # Current position (gives the nearest solution)
        np.zeros(robot.n),                 # Zero configuration
        np.array(robot.qz) if hasattr(robot, 'qz') else np.zeros(robot.n)  # Default pose
    ]
    qlim = get_joint_limits(robot)
    for _ in range(max(0, max_attempts - len(guesses))):
        guesses.append(np.random.uniform(qlim[0], qlim[1]))
    return guesses


def get_joint_limits(robot):
    """2 x n array of joint limits (falls back to +/-pi where none is set)."""
    qlim = np.zeros((2, robot.n))
    for j in range(robot.n):
        lim = getattr(robot.links[j], 'qlim', None)
        if lim is None or np.any(np.isnan(lim)):
            qlim[:, j] = [-np.pi, np.pi]
        else:
            qlim[:, j] = lim
    return qlim


def pose_errors(robot, q, T_target):
    """(position error in metres, orientation error in radians) of q against T_target."""
    T = robot.fkine(q)
    pos_err = np.linalg.norm(T.t - T_target.t)
    R_err = T.R.T @ T_target.R
    ori_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
    return pos_err, ori_err


def wrap_to_reference(q, q_ref, qlim):
    """For each joint pick q + 2*pi*k (still inside its limits) closest to q_ref.

    Joints with more than 360 deg of travel (e.g. Omron/Dobot/Motoman wrists)
    have several equivalent angles; this stops the robot spinning a full turn
    when a nearby equivalent exists.
    """
    q = np.array(q, dtype=float)
    for j in range(len(q)):
        best = q[j]
        for k in range(-3, 4):
            cand = q[j] + 2 * np.pi * k
            if qlim[0, j] - 1e-9 <= cand <= qlim[1, j] + 1e-9:
                if not (qlim[0, j] - 1e-9 <= best <= qlim[1, j] + 1e-9) or \
                        abs(cand - q_ref[j]) < abs(best - q_ref[j]):
                    best = cand
        q[j] = best
    return q


def solve_ik_with_strategies(robot, T_target, initial_guesses,
                             tolerance=0.01, check_collision_fn=None,
                             orientation_tolerance=DEFAULT_ORIENTATION_TOLERANCE,
                             position_only=False, q_reference=None,
                             max_solutions=5):
    """Solve IK and return the valid solution closest to q_reference.

    A solution is valid only if ALL of these hold:
      - position error  < tolerance (m)
      - orientation error < orientation_tolerance (rad), unless position_only=True
      - every joint inside its limits
      - collision-free (if check_collision_fn is given)

    Returns a dict: success, q, error, ori_error, attempts, collision_free, reason.
    On failure q is None and `reason` says what blocked it.
    """
    qlim = get_joint_limits(robot)
    q_ref = np.array(initial_guesses[0] if q_reference is None else q_reference, dtype=float)
    mask = [1, 1, 1, 0, 0, 0] if position_only else None

    valid = []
    rejected = {'not_converged': 0, 'orientation': 0, 'limits': 0, 'collision': 0}
    best_pos, best_ori = float('inf'), float('inf')
    attempts = 0

    for q_init in initial_guesses:
        attempts += 1
        try:
            sol = robot.ikine_LM(T_target, q0=q_init, joint_limits=True, slimit=1, mask=mask)
        except Exception:
            rejected['not_converged'] += 1
            continue

        q = wrap_to_reference(sol.q, q_ref, qlim)
        pos_err, ori_err = pose_errors(robot, q, T_target)
        best_pos = min(best_pos, pos_err)
        if pos_err < tolerance:
            best_ori = min(best_ori, ori_err)

        if pos_err >= tolerance:
            rejected['not_converged'] += 1
            continue
        if not position_only and ori_err >= orientation_tolerance:
            rejected['orientation'] += 1
            continue
        if np.any(q < qlim[0] - 1e-9) or np.any(q > qlim[1] + 1e-9):
            rejected['limits'] += 1
            continue
        if check_collision_fn is not None and check_collision_fn(q):
            rejected['collision'] += 1
            continue

        valid.append((np.linalg.norm(q - q_ref), q, pos_err, ori_err))
        if len(valid) >= max_solutions:
            break

    if valid:
        _, q, pos_err, ori_err = min(valid, key=lambda v: v[0])
        return {'success': True, 'q': q, 'error': pos_err, 'ori_error': ori_err,
                'attempts': attempts, 'collision_free': True, 'reason': None}

    # Nothing valid: report the most useful reason
    if rejected['collision'] > 0:
        reason = 'collision'
    elif rejected['limits'] > 0:
        reason = 'joint_limits'
    elif rejected['orientation'] > 0:
        reason = 'orientation'
    else:
        reason = 'unreachable'

    # If the full pose failed, check whether the POSITION alone is reachable.
    # If it is, the real problem is the requested tool orientation.
    if not position_only and reason in ('unreachable', 'joint_limits'):
        for q_init in initial_guesses[:10]:
            try:
                sol = robot.ikine_LM(T_target, q0=q_init, joint_limits=True, slimit=1,
                                     mask=[1, 1, 1, 0, 0, 0])
            except Exception:
                continue
            q = sol.q
            if (pose_errors(robot, q, T_target)[0] < tolerance
                    and np.all(q >= qlim[0] - 1e-9) and np.all(q <= qlim[1] + 1e-9)):
                reason = 'orientation'
                break
    return {'success': False, 'q': None, 'error': best_pos, 'ori_error': best_ori,
            'attempts': attempts, 'collision_free': reason != 'collision', 'reason': reason}


def describe_ik_failure(result, tolerance, orientation_tolerance):
    """Plain-language explanation of why IK failed."""
    reason = result['reason']
    if reason == 'collision':
        return "every solution found is in collision"
    if reason == 'joint_limits':
        return "solutions found are outside the joint limits"
    if reason == 'orientation':
        return ("the position is reachable, but not with this tool orientation - "
                "change the target orientation, or pass position_only=True "
                "if the tool direction doesn't matter for this move")
    return (f"target out of reach (closest {result['error']*1000:.1f} mm, "
            f"limit {tolerance*1000:.1f} mm)")


def check_target_reachable(robot, T_target, q_start, tolerance=0.01,
                           max_attempts=10, check_collision_fn=None,
                           orientation_tolerance=DEFAULT_ORIENTATION_TOLERANCE,
                           position_only=False):
    """Check if target pose is reachable using IK. Returns the solve result dict."""
    initial_guesses = generate_ik_initial_guesses(robot, q_start, max_attempts)
    return solve_ik_with_strategies(
        robot, T_target, initial_guesses,
        tolerance=tolerance,
        check_collision_fn=check_collision_fn,
        orientation_tolerance=orientation_tolerance,
        position_only=position_only,
        q_reference=q_start
    )

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