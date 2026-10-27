import numpy as np
from spatialgeometry import Sphere
from spatialmath.base import transl
import time
from roboticstoolbox import jtraj

# ---------------------------------------------------------------------------
# Collision model
#   - Each robot link = 1-4 capsules fitted around its 3D mesh (loaded from
#     <robot mesh folder>/collision_capsules.json). Without that file, links fall
#     back to straight lines between joint frames padded by `margin` metres.
#   - Each obstacle = oriented box, rebuilt from the 8 vertices that
#     RectangularPrism produces for it (so the faces/vertices/normals interface
#     used everywhere else stays the same).
#   - A tool (e.g. the knife) can be added as extra capsules in the flange frame.
#   - Robots are checked against each other as capsule-vs-capsule distances.
#   - Segment vs box is tested in the box's own frame with slab clipping, which
#     catches a link crossing a face AND a link sitting completely inside a box.
# ---------------------------------------------------------------------------

DEFAULT_LINK_MARGIN = 0.03   # metres of padding around each link centreline (fallback model only)
FITTED_CLEARANCE = 0.01      # extra metres added to the mesh-fitted capsules
HULL_CLEARANCE = 0.002       # extra metres kept between a link's convex hull and an obstacle
DEFAULT_IGNORE_LINKS = (0,)  # link 0 = robot base, bolted down, never moves (skipped vs obstacles)


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


def get_collision_capsules(robot, q, margin=DEFAULT_LINK_MARGIN,
                           ignore_links=DEFAULT_IGNORE_LINKS):
    """Everything on the robot that can collide, as capsules (line + radius).

    Returns a list of (label, p0, p1, radius). Labels are link indices (int)
    or 'tool'. Tool capsules come from the optional robot attributes:
        robot.collision_tool_segments : list of (p0, p1) in the FLANGE frame
                                        (last DH frame, before robot.tool)
        robot.collision_tool_radius   : radius in metres for those segments
    """
    poses = get_link_poses(robot, q)
    fitted = getattr(robot, 'collision_link_capsules', None)
    if fitted:
        # Capsules fitted to the 3D meshes (robot_controller/make_collision_capsules.py),
        # stored in each link's DH frame. Label k = mesh link k (0 = base).
        capsules = []
        for k, link_caps in enumerate(fitted):
            if k in ignore_links:
                continue
            T = poses[k]
            R, t = T[:3, :3], T[:3, 3]
            for a, b, r in link_caps:
                capsules.append((k, R @ a + t, R @ b + t, r + FITTED_CLEARANCE))
    else:
        # Fallback: lines between joint centres, padded by `margin`
        capsules = [(i, p0, p1, margin) for i, p0, p1 in _segments_from_poses(poses, ignore_links)]

    tool_segments = getattr(robot, 'collision_tool_segments', None)
    if tool_segments:
        T_flange = poses[-1]
        r_tool = getattr(robot, 'collision_tool_radius', margin)
        for a, b in tool_segments:
            p0 = (T_flange @ np.r_[a, 1.0])[:3]
            p1 = (T_flange @ np.r_[b, 1.0])[:3]
            capsules.append(('tool', p0, p1, r_tool))
    return capsules


def _segments_from_poses(poses, ignore_links):
    out = []
    for i in range(len(poses) - 1):
        if i in ignore_links:
            continue
        p0, p1 = poses[i][:3, 3], poses[i + 1][:3, 3]
        if np.linalg.norm(p1 - p0) >= 1e-6:
            out.append((i, p0, p1))
    return out


def segment_segment_distance(p1, q1, p2, q2):
    """Closest distance between segments p1-q1 and p2-q2.

    Returns (distance, closest_point_on_1, closest_point_on_2).
    (Ericson, Real-Time Collision Detection, 5.1.9)
    """
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = np.dot(d1, d1), np.dot(d2, d2), np.dot(d2, r)
    eps = 1e-12
    if a <= eps and e <= eps:
        s = t = 0.0
    elif a <= eps:
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e <= eps:
            t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    c1, c2 = p1 + d1 * s, p2 + d2 * t
    return np.linalg.norm(c1 - c2), c1, c2


def _mark_collision(point, env, collisions_list):
    """Drop a red sphere in Swift at a collision point."""
    if env is not None and collisions_list is not None:
        new_collision = Sphere(radius=0.05, color=[1.0, 0.0, 0.0, 1.0])
        new_collision.T = transl(point[0], point[1], point[2])
        env.add(new_collision)
        collisions_list.append(new_collision)


_capsule_cache = {}      # other robots don't move while one robot plans: reuse their capsules
_box_cache = {}


def _cached_boxes(vertices):
    key = (id(vertices), np.shape(vertices))
    boxes = _box_cache.get(key)
    if boxes is None:
        boxes = boxes_from_vertices(vertices)
        _box_cache.clear()
        _box_cache[key] = boxes
    return boxes


def _other_robot_capsules(other, margin):
    """Capsules of a robot at its current pose, cached until it moves."""
    tool = getattr(other, 'collision_tool_segments', None)
    key = (id(other), np.asarray(other.q, float).tobytes(), np.asarray(other.base).tobytes(),
           id(tool), len(tool) if tool else 0, margin)
    caps = _capsule_cache.get(key)
    if caps is None:
        caps = get_collision_capsules(other, other.q, margin, ignore_links=())
        if len(_capsule_cache) > 32:
            _capsule_cache.clear()
        _capsule_cache[key] = caps
    return caps


def hull_box_overlap(V_world, N_world, box, clearance=0.0):
    """Does a convex hull touch the box (grown by `clearance`)?

    Separating-axis test using the box's 3 axes and the hull's face normals.
    (Edge-edge axes are skipped, which can only report extra overlaps, never
    miss a real one - so it errs on the safe side.)
    """
    axes = np.vstack([box['R'].T, N_world])                 # (k,3)
    proj = V_world @ axes.T                                 # hull projected on each axis
    h_min, h_max = proj.min(axis=0), proj.max(axis=0)
    c = axes @ box['centre']
    r = np.abs(axes @ box['R']) @ (box['half'] + clearance)  # box half-extent along each axis
    separated = (h_min > c + r) | (h_max < c - r)
    return not np.any(separated)


def _link_hull_hits_box(robot, poses, k, box):
    """Second-stage check of link k against a box, using its convex hull(s)
    (the last link can have a second hull for a mounted gripper)."""
    T = poses[k]
    R, t = T[:3, :3], T[:3, 3]
    for V, N, pad in robot.collision_link_hulls[k]:
        if hull_box_overlap(V @ R.T + t, N @ R.T, box, HULL_CLEARANCE + pad):
            return True
    return False


def convex_shapes_overlap(A, B, clearance=0.0):
    """Separating-axis test between two convex shapes given as (points, face normals).

    Uses both shapes' face normals (edge-edge axes skipped: may report an extra
    overlap, never misses one). A shape with no normals (a line segment) only
    contributes its points. `clearance` is the total gap required between them.
    """
    VA, NA = A
    VB, NB = B
    axes = [n for n in (NA, NB) if n is not None and len(n)]
    if not axes:
        return True
    axes = np.vstack(axes)
    pa, pb = VA @ axes.T, VB @ axes.T
    separated = (pa.min(0) > pb.max(0) + clearance) | (pb.min(0) > pa.max(0) + clearance)
    return not np.any(separated)


def _world_shapes(robot, poses, label, capsule):
    """Precise convex shapes for one capsule's part: the link's hull(s) if it has
    them, otherwise the capsule's own segment. Returns [(points, normals, pad)]."""
    hulls = getattr(robot, 'collision_link_hulls', None)
    if isinstance(label, (int, np.integer)) and hulls and label < len(hulls) and poses is not None:
        T = poses[label]
        R, t = T[:3, :3], T[:3, 3]
        return [(V @ R.T + t, N @ R.T, pad) for V, N, pad in hulls[label]]
    _, p0, p1, r = capsule
    return [(np.vstack([p0, p1]), None, r)]       # segment, padded by its radius


def _parts_touch(shapes_a, shapes_b):
    for VA, NA, pa in shapes_a:
        for VB, NB, pb in shapes_b:
            if NA is None and NB is None:
                return True                        # two plain capsules: trust the capsule test
            if convex_shapes_overlap((VA, NA), (VB, NB), HULL_CLEARANCE + pa + pb):
                return True
    return False


_pose_cache = {}


def _other_robot_poses(other):
    """Link poses of a (non-moving) robot, cached until it moves."""
    key = (id(other), np.asarray(other.q, float).tobytes(), np.asarray(other.base).tobytes())
    poses = _pose_cache.get(key)
    if poses is None:
        poses = get_link_poses(other, other.q)
        if len(_pose_cache) > 32:
            _pose_cache.clear()
        _pose_cache[key] = poses
    return poses


def _capsule_aabbs(capsules):
    """(N,3) min and max corners of each capsule's bounding box."""
    p0 = np.array([c[1] for c in capsules])
    p1 = np.array([c[2] for c in capsules])
    r = np.array([c[3] for c in capsules])[:, None]
    return np.minimum(p0, p1) - r, np.maximum(p0, p1) + r


def _collision_pairs(robot, q, vertices, other_robots, margin, ignore_links,
                     obstacle_names=None, stop_at_first=True):
    """Core check. Returns a list of (my_label, what_was_hit, contact_point)."""
    hits = []

    # 0. Optional rule the moving robot must obey (e.g. "keep the glass upright").
    #    Set robot.path_constraint = f(q) -> True if allowed. A pose that breaks it
    #    is treated like a collision, so IK, path checks and RRT all avoid it.
    constraint = getattr(robot, 'path_constraint', None)
    if constraint is not None and not constraint(q):
        name = getattr(constraint, '__name__', 'path constraint')
        hits.append(('constraint', name, get_link_poses(robot, q)[-1][:3, 3]))
        return hits
    all_capsules = get_collision_capsules(robot, q, margin, ignore_links=())   # one fkine
    my_capsules = [c for c in all_capsules if c[0] not in ignore_links]
    hulls = getattr(robot, 'collision_link_hulls', None)
    poses = get_link_poses(robot, q) if hulls else None
    hull_verdict = {}                 # (link, box index) -> precise answer, computed once

    # 1. Robot (links + tool) vs static obstacle boxes
    #    Stage 1: capsules (fast, a bit fat). Stage 2: if a link's capsule touches
    #    a box, its convex hull decides (precise to a few mm).
    if vertices is not None and len(vertices) > 0:
        boxes = _cached_boxes(vertices)
        names = obstacle_names or [f"obstacle_{k}" for k in range(len(boxes))]
        for label, p0, p1, rad in my_capsules:
            for bi, (name, box) in enumerate(zip(names, boxes)):
                hit, point = segment_box_intersection(p0, p1, box, rad)
                if not hit:
                    continue
                if hulls and isinstance(label, (int, np.integer)) and label < len(hulls):
                    key = (label, bi)
                    if key not in hull_verdict:
                        hull_verdict[key] = _link_hull_hits_box(robot, poses, label, box)
                    if not hull_verdict[key]:
                        continue          # capsule touched, the real link shape doesn't
                hits.append((label, name, point))
                if stop_at_first:
                    return hits

    # 2. Robot vs other robots, at the other robots' CURRENT joint angles.
    #    Base columns are included here: another arm can hit them.
    others = [o for o in (other_robots or []) if o is not robot]
    if others and all_capsules:
        if poses is None:
            poses = get_link_poses(robot, q)
        a_lo, a_hi = _capsule_aabbs(all_capsules)
        for other in others:
            theirs = _other_robot_capsules(other, margin)
            if not theirs:
                continue
            b_lo, b_hi = _capsule_aabbs(theirs)
            other_poses = _other_robot_poses(other)
            pair_verdict = {}
            # Broad phase: only pairs whose bounding boxes overlap can touch
            overlap = np.all((a_lo[:, None, :] <= b_hi[None, :, :]) &
                             (b_lo[None, :, :] <= a_hi[:, None, :]), axis=2)
            for i, j in zip(*np.nonzero(overlap)):
                label, a0, a1, ra = all_capsules[i]
                olabel, b0, b1, rb = theirs[j]
                dist, c1, c2 = segment_segment_distance(a0, a1, b0, b1)
                if dist >= ra + rb:
                    continue
                # Capsules touch: let the precise shapes (hulls) decide
                key = (label, olabel) if (label != 'tool' and olabel != 'tool') else (label, olabel, i, j)
                if key not in pair_verdict:
                    pair_verdict[key] = _parts_touch(
                        _world_shapes(robot, poses, label, all_capsules[i]),
                        _world_shapes(other, other_poses, olabel, theirs[j]))
                if not pair_verdict[key]:
                    continue
                hits.append((label, f"{other.name}:{olabel}", (c1 + c2) / 2))
                if stop_at_first:
                    return hits
    return hits


def check_single_config_collision(robot, q, faces, vertices, face_normals,
                                  env=None, collisions_list=None,
                                  return_once_found=True,
                                  margin=DEFAULT_LINK_MARGIN,
                                  ignore_links=DEFAULT_IGNORE_LINKS,
                                  other_robots=None):
    """Check one joint configuration. Returns (collision_detected, collision_points).

    Checks the robot's links (and tool, if it has one) against the obstacle
    boxes, and against `other_robots` at their current poses.
    `faces` and `face_normals` are accepted for backwards compatibility; the
    box test only needs the vertices.
    """
    hits = _collision_pairs(robot, q, vertices, other_robots, margin, ignore_links,
                            stop_at_first=return_once_found)
    points = [p for _, _, p in hits]
    for p in points:
        _mark_collision(p, env, collisions_list)
    return len(points) > 0, points


def is_collision(robot, q_list, faces, vertices, face_normals,
                 collisions_list=None, env=None, return_once_found=True,
                 margin=DEFAULT_LINK_MARGIN, ignore_links=DEFAULT_IGNORE_LINKS,
                 other_robots=None):
    """Check a list of configurations (a trajectory). True if any collides."""
    found = False
    for q in q_list:
        hit, _ = check_single_config_collision(
            robot, q, faces, vertices, face_normals,
            env=env, collisions_list=collisions_list,
            return_once_found=return_once_found,
            margin=margin, ignore_links=ignore_links,
            other_robots=other_robots
        )
        if hit:
            found = True
            if return_once_found:
                return True
    return found


def collision_report(robot, q, vertices, obstacle_names=None,
                     margin=DEFAULT_LINK_MARGIN, ignore_links=DEFAULT_IGNORE_LINKS,
                     other_robots=None):
    """Debug helper: list of (link_index or 'tool', what_it_hit) pairs."""
    hits = _collision_pairs(robot, q, vertices, other_robots, margin, ignore_links,
                            obstacle_names=obstacle_names, stop_at_first=False)
    return [(label, what) for label, what, _ in hits]

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

# ---------------------------------------------------------------------------
# Path planning: RRT-Connect in joint space
# ---------------------------------------------------------------------------
DEFAULT_EDGE_RESOLUTION = np.deg2rad(2.0)   # max joint change between collision checks


def _joint_limits(robot):
    qlim = np.zeros((2, robot.n))
    for j in range(robot.n):
        lim = getattr(robot.links[j], 'qlim', None)
        qlim[:, j] = [-np.pi, np.pi] if lim is None or np.any(np.isnan(lim)) else lim
    return qlim


def check_line_collision(robot, q1, q2, faces, vertices, face_normals, steps=None,
                         other_robots=None, resolution=DEFAULT_EDGE_RESOLUTION):
    """True if the straight joint-space line q1 -> q2 hits anything.

    Checked every `resolution` radians of the largest joint change (or at a
    fixed number of `steps` if given), so long lines are never under-sampled.
    """
    q1, q2 = np.asarray(q1, float), np.asarray(q2, float)
    if steps is None:
        steps = max(1, int(np.ceil(np.max(np.abs(q2 - q1)) / resolution)))
    for i in _bisection_order(steps):
        alpha = i / steps
        hit, _ = check_single_config_collision(
            robot, (1 - alpha) * q1 + alpha * q2, faces, vertices, face_normals,
            return_once_found=True, other_robots=other_robots)
        if hit:
            return True
    return False


def _bisection_order(steps):
    """0..steps in the order end, start, middle, quarters, eighths ... A blocked line
    is usually caught after a few checks instead of walking it end to end."""
    order, seen = [steps, 0], {0, steps}
    queue = [(0, steps)]
    while queue:
        nxt = []
        for a, b in queue:
            if b - a > 1:
                m = (a + b) // 2
                if m not in seen:
                    seen.add(m)
                    order.append(m)
                nxt += [(a, m), (m, b)]
        queue = nxt
    return order


class _Tree:
    """RRT tree with vectorised nearest-neighbour lookup."""
    def __init__(self, q_root):
        self.Q = [np.asarray(q_root, float)]
        self.parent = [-1]

    def nearest(self, q):
        d = np.linalg.norm(np.asarray(self.Q) - q, axis=1)
        return int(np.argmin(d))

    def add(self, q, parent):
        self.Q.append(np.asarray(q, float))
        self.parent.append(parent)
        return len(self.Q) - 1

    def path_to_root(self, idx):
        path = []
        while idx != -1:
            path.append(self.Q[idx])
            idx = self.parent[idx]
        return path


def find_collision_free_path_rrt(robot, q_start, q_goal, faces, vertices, face_normals,
                                 env=None, collisions_list=None,
                                 max_iterations=8000, step_size=0.3,
                                 goal_sample_rate=0.1, goal_threshold=0.3,
                                 visualize=False, other_robots=None,
                                 resolution=DEFAULT_EDGE_RESOLUTION,
                                 max_time=60.0, sample_margin=np.pi / 2, seed=None):
    """Plan a collision-free joint-space path with RRT-Connect.

    Two trees grow, one from the start and one from the goal, and try to join
    each other every iteration - far better than a single tree at getting
    through narrow gaps (e.g. under the cupboard).

    Sampling first stays inside a box around start and goal (+/- sample_margin
    per joint, within joint limits); if that fails, it widens to the full limits.

    Returns (waypoints, success). Waypoints run start -> goal; every straight
    segment between them has been collision-checked at `resolution`.
    (goal_sample_rate / goal_threshold / visualize are kept for compatibility.)
    """
    rng = np.random.default_rng(seed)
    q_start = np.asarray(q_start, float)
    q_goal = np.asarray(q_goal, float)
    qlim = _joint_limits(robot)

    def collides(q):
        return check_single_config_collision(robot, q, faces, vertices, face_normals,
                                             return_once_found=True,
                                             other_robots=other_robots)[0]

    def edge_free(a, b):
        return not check_line_collision(robot, a, b, faces, vertices, face_normals,
                                        other_robots=other_robots, resolution=resolution)

    # If the robot has a path constraint, it can supply robot.project_to_constraint(q)
    # -> nearby q that obeys it (or None). Random samples almost never obey a
    # constraint like "glass upright", so every new node is pulled onto it.
    project = getattr(robot, 'project_to_constraint', None)
    if getattr(robot, 'path_constraint', None) is None:
        project = None

    print(f"Starting RRT-Connect search (max {max_iterations} iterations, {max_time:.0f}s)...")
    if collides(q_start):
        print("Start configuration is in collision!")
        return [], False
    if collides(q_goal):
        print("Goal configuration is in collision!")
        return [], False
    if edge_free(q_start, q_goal):
        print("Direct path is already collision-free")
        return [q_start, q_goal], True

    # Sampling regions: near start/goal first, then everything
    lo_near = np.maximum(qlim[0], np.minimum(q_start, q_goal) - sample_margin)
    hi_near = np.minimum(qlim[1], np.maximum(q_start, q_goal) + sample_margin)
    regions = [(lo_near, hi_near), (qlim[0], qlim[1])]

    t0 = time.time()
    for phase, (lo, hi) in enumerate(regions):
        tree_a, tree_b = _Tree(q_start), _Tree(q_goal)
        a_is_start = True
        budget = max_iterations // len(regions)

        for it in range(budget):
            if time.time() - t0 > max_time:
                break
            q_rand = rng.uniform(lo, hi)        # only sets a direction to grow in

            # EXTEND tree_a one step towards the sample
            i_near = tree_a.nearest(q_rand)
            q_near = tree_a.Q[i_near]
            d = q_rand - q_near
            dist = np.linalg.norm(d)
            q_new = q_rand if dist <= step_size else q_near + d / dist * step_size
            if project is not None and dist > step_size:
                q_new = project(q_new)
            if q_new is None or not edge_free(q_near, q_new):
                tree_a, tree_b, a_is_start = tree_b, tree_a, not a_is_start
                continue
            i_new = tree_a.add(q_new, i_near)

            # CONNECT: tree_b keeps stepping towards q_new until blocked or joined.
            # Each step must get closer (a constraint projection can push sideways),
            # and the number of steps is capped so this can never loop forever.
            j = tree_b.nearest(q_new)
            last_dist = np.inf
            for _ in range(200):
                q_b = tree_b.Q[j]
                d = q_new - q_b
                dist = np.linalg.norm(d)
                if dist >= last_dist - 1e-3 or time.time() - t0 > max_time:
                    break                      # not making progress
                last_dist = dist
                q_step = q_new if dist <= step_size else q_b + d / dist * step_size
                if project is not None and dist > step_size:
                    q_step = project(q_step)
                if q_step is None or not edge_free(q_b, q_step):
                    break
                j = tree_b.add(q_step, j)
                if dist <= step_size:          # trees joined
                    side_a = tree_a.path_to_root(i_new)[::-1]   # root_a ... q_new
                    side_b = tree_b.path_to_root(j)             # q_new ... root_b
                    path = side_a + side_b[1:]
                    if not a_is_start:
                        path = path[::-1]
                    elapsed = time.time() - t0
                    print(f"✓ RRT-Connect succeeded in {it + 1} iterations ({elapsed:.1f}s)"
                          + (" after widening the search" if phase else ""))
                    print(f"  Raw path: {len(path)} waypoints")
                    path = smooth_path(robot, path, faces, vertices, face_normals,
                                       other_robots=other_robots, resolution=resolution)
                    print(f"  Shortened to: {len(path)} waypoints")
                    return path, True

            tree_a, tree_b, a_is_start = tree_b, tree_a, not a_is_start

        if phase == 0:
            print("  Not found near start/goal - widening search to full joint range")

    print(f"✗ RRT-Connect failed ({time.time() - t0:.1f}s)")
    return [], False


#Removes unnecessary waypoints
def smooth_path(robot, path, faces, vertices, face_normals, max_attempts=40,
                other_robots=None, resolution=DEFAULT_EDGE_RESOLUTION, seed=None):
    """Shorten a path: greedily skip to the furthest waypoint reachable in a
    straight (collision-free) line, then try some random shortcuts."""
    path = [np.asarray(q, float) for q in path]
    if len(path) <= 2:
        return path

    def free(a, b):
        return not check_line_collision(robot, a, b, faces, vertices, face_normals,
                                        other_robots=other_robots, resolution=resolution)

    # Greedy pass
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not free(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j

    # Random shortcuts between points along segments
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        if len(out) <= 2:
            break
        i = rng.integers(0, len(out) - 2)
        j = rng.integers(i + 2, len(out))
        if free(out[i], out[j]):
            out = out[:i + 1] + out[j:]
    return out


#Interpolates between waypoints
def fine_interpolation(q1, q2, max_step_radians=np.deg2rad(1)):
    # Quintic (jtraj) peaks at 1.875x the average speed, so start from that estimate
    q1, q2 = np.asarray(q1, float), np.asarray(q2, float)
    steps = max(2, int(np.ceil(1.875 * np.max(np.abs(q2 - q1)) / max_step_radians)) + 1)
    while np.any(max_step_radians < np.abs(np.diff(jtraj(q1, q2, steps).q, axis=0))):
        steps += 1
    return jtraj(q1, q2, steps).q
def interpolate_waypoints(waypoint_list, max_step_radians=np.deg2rad(1)):
    q_matrix = []
    for i in range(len(waypoint_list) - 1):
        segment = fine_interpolation(waypoint_list[i], waypoint_list[i+1], max_step_radians)
        # skip each segment's first point after the first (it repeats the previous end)
        for q in (segment if i == 0 else segment[1:]):
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