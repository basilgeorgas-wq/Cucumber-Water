# Cucumber-Water

A three-arm robotic system that prepares and serves cucumber water, simulated in
Python with the Robotics Toolbox and the Swift 3D viewer.

Five empty glasses start on the kitchen counter. The robots take a cucumber out
of the fridge, slice it, fill each glass at the sink, drop a slice into each one
and serve them in a line on the counter. Everything is driven from a control
panel with an emergency stop, a teach pendant and safety demos.

## The robots

| Robot | Tool | Job |
|---|---|---|
| Omron TM5-700 | Robotiq 2F-85 gripper | Takes the cucumber out of the fridge, puts it on the bench, then drops a slice into each glass |
| Motoman GP4 | Kitchen knife | Slices the cucumber (one slice per glass) |
| Dobot CR5 | DH Robotics AG-95 gripper | Picks up each glass, fills it at the tap, holds it out for the slice, then serves it |

## The task, step by step

1. **Fetch the cucumber:** the Omron grips the cucumber's end, which sticks out of the fridge's top shelf, pulls it out and lays it on the bench in front of the Motoman.
2. **Slice:** the Motoman moves above the cucumber in a straight line (resolved-rate motion control) and cuts it. A slice appears after each cut.
3. **For each glass:**
   1. The Dobot picks the glass up off the counter and lifts it clear of the others.
   2. It slides the glass under the tap to fill it, keeping it upright (within 45° while empty, 20° once full).
   3. It holds the glass out in the space shared by the two arms.
   4. The Omron picks up a slice and drops it into the glass.
   5. The Dobot puts the glass down in the serving line and returns to its start pose.
4. **Result:** five filled glasses, each with a slice, in a straight line 10 cm apart.

The empty glasses start in two staggered rows, because the AG-95's open fingers are 16 cm wide. Five glasses in one row with room for the fingers won't fit within the Dobot's reach.

## Setup (Windows, Python 3.11)

1. Install **Python 3.11** from python.org (tkinter, used by the control panel, comes with it).
2. In a terminal in the project folder:

   ```
   py -3.11 -m pip install --only-binary :all: -r requirements.txt
   py -3.11 -m pip install --no-deps ir-support==1.4.3
   ```

   `ir-support` is installed with `--no-deps` because its own dependency list pulls in packages that fail to build on Windows. The package versions are pinned on purpose. Newer `spatialgeometry` (1.4.x) crashes the Swift viewer.

## Running it

```
py -3.11 Task-Simulation/task_simulation.py
```

Swift opens in your browser and the control panel opens in its own window. Wait until the panel says **READY**, then press **START**.

| Command | What it does |
|---|---|
| `py -3.11 Task-Simulation/task_simulation.py` | Control panel (normal use) |
| `... --speed 2` | Start at 2x playback speed (also adjustable on the Settings tab) |
| `... --no-gui` | Old mode: runs the task straight away; keyboard e-stop `e`, resume `r`, quit `q` |
| `... --no-gui --teach` | Old mode with the matplotlib teach pendant first (press Enter to start) |

Closing the control panel shuts everything down.

## Control panel

### Run controls

| Control | What it does |
|---|---|
| **START** | Runs the task. It always starts from the start position (the scene is reset first). |
| **PAUSE / RESUME** | A normal stop and carry on. |
| **EMERGENCY STOP** (or **Esc** / **Space**) | Every robot freezes at once, even part-way through a move. The stop latches. |
| **RESET E-STOP** | Acknowledges the e-stop. Only possible once the light curtain is clear. The robots stay frozen until **RESUME**. |
| **Step mode + NEXT STEP** | The task waits before every move until NEXT STEP is pressed. |
| **RESET SCENE** | Abandons the run (even mid-move) and puts robots, glasses and cucumber back at the start. |

Releasing the e-stop takes two separate actions, reset and then resume, so the robots can never restart just because the stop was cleared. This follows the e-stop requirements of ISO 13850.

The status area shows the current step and progress (e.g. "Glass 2 of 5").

### Tabs

- **Step log:** every move as it happens, `OK` in green or `FAILED` in red, labelled with its glass number.
- **Teach pendant** (works when no task is running):
  - Choose a robot, then drag the **joint sliders**. The robot follows.
  - **±X / ±Y / ±Z** move the hand in a straight line (1–10 cm steps).
  - **Open / close gripper** and **Go home**.
  - A pose that would hit something is refused with the reason (e.g. "would hit bench").
  - Live **manipulability** readout, with a warning near a singularity that says which joints to bend.
- **Settings:** number of glasses (1–5), playback speed, show/hide the obstacle hitboxes and the collision markers.
- **Safety:**
  - **Light curtain:** a beam across the open side of the work area. Press *Person walks in* and the person breaks it, triggering the e-stop. It can't be reset until they walk away.
  - **Pot in the sink:** drops an obstacle onto the Dobot's route from the counter to the tap. A robot whose move would hit it stops before touching it and plans a new path round it. Later moves avoid it too.

### Good to know

Each arm's start pose is a **singularity**: the arm points straight up (elbow singularity) and J5 is at 0° (wrist singularity). Most straight-line jogs are impossible from there. Bend J3 and J5 with the sliders first (just J5 on the Motoman). The teach tab tells you this.

## How it works

### Collision checking

- **Obstacles** (bench, cupboard, tap, fridge walls and shelves, the open fridge door, glasses standing on the counter) are boxes.
- **Robot links** are modelled in two stages:
  1. **Capsules** fitted to each link's 3D model: fast, and slightly oversized.
  2. **Convex hulls** of each link: precise to a few millimetres. They decide whenever a capsule touches something.
- **Grippers and carried objects** (glass, cucumber, knife) are included in every check.
- **Robots are checked against each other**, at the other robots' current poses.
- **Validation:** against the real 3D meshes, the checker missed no collisions, both against obstacles and robot-to-robot.

### Motion planning

- **Inverse kinematics:** must meet both position and orientation tolerance and respect joint limits. Where several solutions exist, it picks the one closest to the current pose, so the arm doesn't flip.
- **Planned moves:** **RRT-Connect** finds a collision-free path when a straight line is blocked. The path is then shortened and smoothed.
- **Straight-line moves:**
  - used for approaches, grasps and the slicing cuts
  - checked step by step for collisions and sudden joint jumps
- **Approach planning:** chooses an arm pose above a target that allows a clear straight descent.
- **Resolved-rate motion control (RMRC):** straight Cartesian paths, using damped least squares near singularities.
- **Keeping the glass upright:** a planning constraint that every IK solution and path has to satisfy.
- **Replanning:** if an obstacle appears mid-move, the robot stops at the next frame and plans a new path from where it is.

### Tested

The five-glass task was run over multiple random seeds with:
- every step OK
- every glass ending with water and one slice, in a line
- the full glass never tilted
- no contact between any robot's 3D model and an obstacle

The control panel has an automated test of 42 checks covering every button and tab.

## Project layout

```
Task-Simulation/
  task_simulation.py      the task: robots, glasses, every step; run this file
  control_panel.py        the control panel window (tkinter) and its e-stop state machine
  movement_handler.py     IK, straight-line moves, planned moves, RMRC, animation
  collision_handler.py    collision checking (capsules, convex hulls, robot vs robot), RRT-Connect
  swift_windows_fix.py    fixes Swift's "404 File not found" for 3D models on Windows
environment_controller/
  environment_controller.py   kitchen, obstacles (hitboxes), cucumber, glasses, water, slices
  visual_files/               kitchen, cucumber, glass and slice models
robot_controller/
  DOBOT_CR5.py, OMRON_TM5.py, MOTOMAN_GP4.py   the three robots (DH models + 3D meshes)
  gripper.py                  mounts a gripper on a robot, with open/closed models
  attachments.py              lets a robot carry objects (and include them in collision checks)
  collision_model.py          loads each robot's collision capsules and hulls
  make_collision_capsules.py  developer tool: rebuilds the collision shapes from the 3D models
                              (needs trimesh + pycollada; only rerun if a model changes)
  Gripper_stl/                gripper models, collision shapes and licences
```

## Changing the setup

Positions and distances are named settings near the top of the task section of `Task-Simulation/task_simulation.py`. These are the main ones:

| Setting | What it controls |
|---|---|
| `GLASS_START_POSITIONS` | Where the empty glasses start |
| `GLASS_SERVE_POSITIONS` | Where the filled glasses are served |
| `SLICE_POSITIONS` | Where the slices appear |
| `SINK_GLASS_POSITION`, `HANDOVER_GLASS_POSITION` | Filling and handover points |
| `GLASS_TILT_LIMIT`, `GLASS_TILT_LIMIT_EMPTY` | How upright the glass must stay |
| `CURTAIN_X`, `DEMO_OBSTACLE_CENTRE` | Light curtain position, and where the pot goes |

New spots need checking for reach: the Dobot's reach and its gripper width limit where glasses can go.

## Known limitations

- **Self-collision is not checked yet.** Robots are checked against obstacles and each other, not against their own links. A check is written in `collision_handler.py` but switched off (`SELF_COLLISION_CHECK = False`) while it's made less cautious.
- **The cucumber stays whole after slicing.** The slices appear beside it.
- **Planning is randomised (RRT).** A failed plan is retried automatically, but a very slow computer could still occasionally run out of planning time.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No suitable Python runtime found` | Python 3.11 isn't installed, or isn't registered with the `py` launcher. Install it from python.org. |
| Browser shows "Application error: a client-side exception" | `spatialgeometry` is too new. Reinstall from `requirements.txt` (1.1.0). |
| Swift shows the scene but no robots (404 errors in the browser console) | Fixed by `swift_windows_fix.py`. Make sure it's in `Task-Simulation/`. |
| `Slider.__init__() got an unexpected keyword argument` (teach pendant) | `matplotlib` is too new. Reinstall from `requirements.txt` (3.8.4). |

## Credits

| Model | Source | Licence |
|---|---|---|
| Robotiq 2F-85 gripper | [PickNik Robotics, ros2_robotiq_gripper](https://github.com/PickNikRobotics/ros2_robotiq_gripper) | BSD-3-Clause |
| DH Robotics AG-95 gripper | [Ian Chuang, dh_ag95_gripper_ros2](https://github.com/ian-chuang/dh_ag95_gripper_ros2) | Apache-2.0 |

Licence files and the changes made are in `robot_controller/Gripper_stl/`.
