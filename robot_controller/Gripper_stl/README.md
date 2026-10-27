# Gripper models

Assembled from open-source ROS description packages into single STL files,
with the mounting face at z = 0 and the fingers pointing along +z.
Used by robot_controller/gripper.py.

| File | What it is |
|---|---|
| robotiq_2f85_open.stl / _grip.stl / _grip_cucumber.stl | Robotiq 2F-85: fully open (8.5 cm) / closed on the 1.1 cm cucumber slice / closed on the 3.8 cm cucumber |
| dh_ag95_open.stl / _grip.stl | DH Robotics AG-95: fully open (9.5 cm) / closed on the 7.2 cm glass |
| *_collision.json | Collision capsules + convex hull covering every finger position (made by make_collision_capsules.py) |

## Sources and licences

| Gripper | Source | Licence |
|---|---|---|
| Robotiq 2F-85 | PickNik Robotics, ros2_robotiq_gripper (robotiq_description) - https://github.com/PickNikRobotics/ros2_robotiq_gripper | BSD-3-Clause (LICENSE_robotiq_2f85_BSD-3.txt) |
| DH Robotics AG-95 | Ian Chuang, dh_ag95_gripper_ros2 (dh_ag95_description) - https://github.com/ian-chuang/dh_ag95_gripper_ros2 | Apache-2.0 (LICENSE_dh_ag95_Apache-2.0.txt) |

Changes from the originals: parts merged at the stated finger positions,
simplified to <= 20k faces, AG-95 shifted 9.7 cm so its mounting face is at z = 0.
