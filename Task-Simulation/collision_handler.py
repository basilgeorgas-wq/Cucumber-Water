import numpy as np
from spatialgeometry import Sphere
from spatialmath.base import transl
import time
from itertools import combinations
from ir_support import line_plane_intersection
from roboticstoolbox import jtraj

# ---------------------------------------------------------------------------
# Collision model
#   - Each robot link = straight segment between consecutive joint frames
#     (from fkine_all), inflated by `margin` metres to give the link thickness.
#   - Each obstacle = oriented box, rebuilt from the 8 vertices that
#     RectangularPrism produces for it (so the faces/vertices/normals interface
#     used everywhere else stays the same).
#   - Segment vs box is tested in the box's own frame with slab clipping, which
#     catches a link crossing a face AND a link sitting completely inside a box.
# ---------------------------------------------------------------------------

DEFAULT_LINK_MARGIN = 0.03   # metres of "padding" around each link centreline
DEFAULT_IGNORE_LINKS = (0,)  # segment 0 = base column, bolted to the bench, never moves


def boxes_from_vertices(vertices):
    """Rebuild oriented boxes from RectangularPrism vertices (8 per box).

    RectangularPrism vertex order: v0=(-,-,-), v1=(+,-,-), v3=(-,+,-), v4=(-,-,+).
    Returns a list of dicts: centre (3,), R (3x3, columns = box axes), half (3,).
    """
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) % 8 != 0:
        raise ValueError("Collision vertices must come in groups of 8 (one RectangularPrism per obstacle)")

    boxes = []
    for k in range(0, len(vertices), 8):
        v = vertices[k:k + 8]
        ex, ey, ez = v[1] - v[0], v[3] - v[0], v[4] - v[0]
        lengths = np.array([np.linalg.norm(ex), np.linalg.norm(ey), np.linalg.norm(ez)])
        R = np.column_stack([ex / lengths[0], ey / lengths[1], ez / lengths[2]])
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
            raise ValueError(f"Obstacle {k // 8} vertices are not a rectangular box")
        boxes.append({'centre': v.mean(axis=0), 'R': R, 'half': lengths / 2})
    return boxes


def segment_box_intersection(p0, p1, box, margin=0.0):
    """Does segment p0->p1 touch the box (grown by `margin`)?

    Returns (hit, point) where point is the first contact point in world frame.
    """
    R, c, h = box['R'], box['centre'], box['half'] + margin
    a = R.T @ (np.asarray(p0, float) - c)   # segment start in box frame
    d = R.T @ (np.asarray(p1, float) - c) - a
    t_enter, t_exit = 0.0, 1.0

    for k in range(3):
        if abs(d[k]) < 1e-12:               # parallel to this slab
            if abs(a[k]) > h[k]:
                return False, None
        else:
            t1 = (-h[k] - a[k]) / d[k]
            t2 = (h[k] - a[k]) / d[k]
            if t1 > t2:
                t1, t2 = t2, t1
            t_enter = max(t_enter, t1)
            t_exit = min(t_exit, t2)
            if t_enter > t_exit:
                return False, None

    point = c + R @ (a + t_enter * d)
    return True, point


def get_link_segments(robot, q, ignore_links=DEFAULT_IGNORE_LINKS):
    """List of (link_index, start_point, end_point) for every non-zero-length link."""
    poses = get_link_poses(robot, q)          # list of n+1 4x4 arrays
    segments = []
    for i in range(len(poses) - 1):
        if i in ignore_links:
            continue
        p0, p1 = poses[i][:3, 3], poses[i + 1][:3, 3]
        if np.linalg.norm(p1 - p0) < 1e-6:    # e.g. coincident wrist frames
            continue
        segments.append((i, p0, p1))
    return segments


def _mark_collision(point, env, collisions_list):
    """Drop a red sphere in Swift at a collision point."""
    if env is not None and collisions_list is not None:
        new_collision = Sphere(radius=0.05, color=[1.0, 0.0, 0.0, 1.0])
        new_collision.T = transl(point[0], point[1], point[2])
        env.add(new_collision)
        collisions_list.append(new_collision)


def check_single_config_collision(robot, q, faces, vertices, face_normals,
                                  env=None, collisions_list=None,
                                  return_once_found=True,
                                  margin=DEFAULT_LINK_MARGIN,
                                  ignore_links=DEFAULT_IGNORE_LINKS):
    """Check one joint configuration. Returns (collision_detected, collision_points).

    `faces` and `face_normals` are accepted for backwards compatibility; the
    box test only needs the vertices.
    """
    boxes = boxes_from_vertices(vertices)
    collision_points = []

    for _, p0, p1 in get_link_segments(robot, q, ignore_links):
        for box in boxes:
            hit, point = segment_box_intersection(p0, p1, box, margin)
            if hit:
                collision_points.append(point)
                _mark_collision(point, env, collisions_list)
                if return_once_found:
                    return True, collision_points

    return len(collision_points) > 0, collision_points


def is_collision(robot, q_list, faces, vertices, face_normals,
                 collisions_list=None, env=None, return_once_found=True,
                 margin=DEFAULT_LINK_MARGIN, ignore_links=DEFAULT_IGNORE_LINKS):
    """Check a list of configurations (a trajectory). True if any collides."""
    found = False
    for q in q_list:
        hit, _ = check_single_config_collision(
            robot, q, faces, vertices, face_normals,
            env=env, collisions_list=collisions_list,
            return_once_found=return_once_found,
            margin=margin, ignore_links=ignore_links
        )
        if hit:
            found = True
            if return_once_found:
                return True
    return found


def collision_report(robot, q, vertices, obstacle_names=None,
                     margin=DEFAULT_LINK_MARGIN, ignore_links=DEFAULT_IGNORE_LINKS):
    """Debug helper: list of (link_index, obstacle_name) pairs that collide."""
    boxes = boxes_from_vertices(vertices)
    names = obstacle_names or [f"obstacle_{k}" for k in range(len(boxes))]
    hits = []
    for i, p0, p1 in get_link_segments(robot, q, ignore_links):
        for name, box in zip(names, boxes):
            if segment_box_intersection(p0, p1, box, margin)[0]:
                hits.append((i, name))
    return hits

#Barycentric check intersections
def is_intersection_point_inside_triangle(intersect_p, triangle_verts):
    # Using barycentric coordinates to check if point is inside triangle
    u = triangle_verts[1, :] - triangle_verts[0, :]
    v = triangle_verts[2, :] - triangle_verts[0, :]

    uu = np.dot(u, u)
    uv = np.dot(u, v)
    vv = np.dot(v, v)

    w = intersect_p - triangle_verts[0, :]
    wu = np.dot(w, u)
    wv = np.dot(w, v)

    D = uv * uv - uu * vv

    # Get and test parametric coords (s and t)
    s = (uv * wv - vv * wu) / D
    if s < 0.0 or s > 1.0:  # intersect_p is outside Triangle
        return False

    t = (uv * wu - uu * wv) / D
    if t < 0.0 or (s + t) > 1.0:  # intersect_p is outside Triangle
        return False

    return True

#Link poses
def get_link_poses(robot, q=None):
    if q is None:
        return robot.fkine_all().A
    return robot.fkine_all(q).A

#RRT Pathfinding 
def find_collision_free_path_rrt(robot, q_start, q_goal, faces, vertices, face_normals,
                                 env=None, collisions_list=None,
                                 max_iterations=1000, step_size=0.2, 
                                 goal_sample_rate=0.1, goal_threshold=0.3,
                                 visualize=False):
    
    print(f"Starting RRT search (max iterations: {max_iterations})...")
    
    # Check if start and goal are collision-free
    start_collision, _ = check_single_config_collision(
        robot, q_start, faces, vertices, face_normals, return_once_found=True
    )
    
    goal_collision, _ = check_single_config_collision(
        robot, q_goal, faces, vertices, face_normals, return_once_found=True
    )
    
    if start_collision:
        print("Start configuration is in collision!")
        return [], False
    
    if goal_collision:
        print("Goal configuration is in collision!")
        return [], False
    
    # Initialize tree
    tree = [{'q': np.array(q_start), 'parent': None, 'cost': 0}]
    
    # Get joint limits
    q_min = np.array([robot.links[i].qlim[0] if hasattr(robot.links[i], 'qlim') and robot.links[i].qlim is not None else -np.pi 
                     for i in range(robot.n)])
    q_max = np.array([robot.links[i].qlim[1] if hasattr(robot.links[i], 'qlim') and robot.links[i].qlim is not None else np.pi 
                     for i in range(robot.n)])
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        # Sample random configuration (bias toward goal)
        if np.random.random() < goal_sample_rate:
            q_rand = q_goal
        else:
            q_rand = np.random.uniform(q_min, q_max)
        
        # Find nearest node in tree
        nearest_idx = 0
        min_dist = np.linalg.norm(tree[0]['q'] - q_rand)
        
        for i, node in enumerate(tree):
            dist = np.linalg.norm(node['q'] - q_rand)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        q_near = tree[nearest_idx]['q']
        
        # Extend toward sample
        direction = q_rand - q_near
        dist = np.linalg.norm(direction)
        
        if dist > step_size:
            direction = direction / dist * step_size
        
        q_new = q_near + direction
        
        # Clip to joint limits
        q_new = np.clip(q_new, q_min, q_max)
        
        # Check if extension is collision-free
        collision = check_line_collision(
            robot, q_near, q_new, faces, vertices, face_normals, steps=5
        )
        
        if not collision:
            # Add to tree
            new_cost = tree[nearest_idx]['cost'] + np.linalg.norm(q_new - q_near)
            tree.append({'q': q_new, 'parent': nearest_idx, 'cost': new_cost})
            
            # Visualize if requested
            if visualize and env is not None and iteration % 50 == 0:
                sphere = Sphere(radius=0.02, color=[0.0, 0.5, 1.0, 0.3])
                T = robot.fkine(q_new).A
                sphere.T = T
                env.add(sphere)
                env.step(0.01)
            
            # Check if goal reached
            dist_to_goal = np.linalg.norm(q_new - q_goal)
            if dist_to_goal < goal_threshold:
                # Try direct connection to goal
                collision = check_line_collision(
                    robot, q_new, q_goal, faces, vertices, face_normals, steps=10
                )
                
                if not collision:
                    # Add goal to tree
                    goal_cost = tree[-1]['cost'] + dist_to_goal
                    tree.append({'q': np.array(q_goal), 'parent': len(tree) - 1, 'cost': goal_cost})
                    
                    # Extract path
                    path = extract_path(tree)
                    
                    elapsed = time.time() - start_time
                    print(f"✓ RRT succeeded in {iteration + 1} iterations ({elapsed:.2f}s)")
                    print(f"  Path length: {len(path)} waypoints")
                    
                    # Smooth path
                    path = smooth_path(robot, path, faces, vertices, face_normals)
                    print(f"  Smoothed to: {len(path)} waypoints")
                    
                    return path, True
        
        # Progress update
        if iteration % 100 == 0 and iteration > 0:
            print(f"  Iteration {iteration}/{max_iterations} (tree size: {len(tree)})")
    
    elapsed = time.time() - start_time
    print(f"✗ RRT failed after {max_iterations} iterations ({elapsed:.2f}s)")
    print(f"  Final tree size: {len(tree)}")
    
    return [], False

def check_line_collision(robot, q1, q2, faces, vertices, face_normals, steps=10):
    for i in range(steps + 1):
        alpha = i / steps
        q_interp = (1 - alpha) * np.array(q1) + alpha * np.array(q2)
        
        collision, _ = check_single_config_collision(
            robot, q_interp, faces, vertices, face_normals, return_once_found=True
        )
        
        if collision:
            return True
    
    return False

def extract_path(tree):
    path = []
    current_idx = len(tree) - 1  # Goal node
    
    while current_idx is not None:
        path.append(tree[current_idx]['q'])
        current_idx = tree[current_idx]['parent']
    
    path.reverse()
    return path

#Removes unnecessary waypoints
def smooth_path(robot, path, faces, vertices, face_normals, max_attempts=100):
    if len(path) <= 2:
        return path
    
    smoothed = path.copy()
    
    for _ in range(max_attempts):
        if len(smoothed) <= 2:
            break
        
        # Pick two random waypoints
        i = np.random.randint(0, len(smoothed) - 2)
        j = np.random.randint(i + 2, len(smoothed))
        
        # Try direct connection
        collision = check_line_collision(
            robot, smoothed[i], smoothed[j], 
            faces, vertices, face_normals, steps=20
        )
        
        if not collision:
            # Remove intermediate waypoints
            smoothed = smoothed[:i+1] + smoothed[j:]
    
    return smoothed

#Interpolates between waypoints
def fine_interpolation(q1, q2, max_step_radians=np.deg2rad(1)):    
    steps = 2
    while np.any(max_step_radians < np.abs(np.diff(jtraj(q1, q2, steps).q, axis=0))):
        steps += 1
    return jtraj(q1, q2, steps).q
def interpolate_waypoints(waypoint_list, max_step_radians=np.deg2rad(1)):
    q_matrix = []
    for i in range(len(waypoint_list) - 1):
        for q in fine_interpolation(waypoint_list[i], waypoint_list[i+1], max_step_radians):
            q_matrix.append(q)
    return q_matrix

def visualize_collision_spheres(robot, q, env, radius=0.03, color=None):
    if color is None:
        color = [0.0, 1.0, 0.0, 0.5]
    
    robot.q = q
    spheres = []
    
    tr = get_link_poses(robot, q)
    for i in range(len(tr)):
        sphere = Sphere(radius=radius, color=color)
        sphere.T = tr[i]
        env.add(sphere)
        spheres.append(sphere)
    
    return spheres