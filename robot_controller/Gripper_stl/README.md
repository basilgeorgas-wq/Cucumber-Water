# Gripper models

Assembled from open-source ROS description packages into single STL files
(one fully open, one fully closed), with the mounting face at z = 0 and the
fingers pointing along +z.

| File | Gripper | Source | Licence |
|---|---|---|---|
| robotiq_2f85_open/closed.stl | Robotiq 2F-85 | PickNik Robotics, ros2_robotiq_gripper (robotiq_description) - https://github.com/PickNikRobotics/ros2_robotiq_gripper | BSD-3-Clause (LICENSE_robotiq_2f85_BSD-3.txt) |
| dh_ag95_open/closed.stl | DH Robotics AG-95 | Ian Chuang, dh_ag95_gripper_ros2 (dh_ag95_description) - https://github.com/ian-chuang/dh_ag95_gripper_ros2 | Apache-2.0 (LICENSE_dh_ag95_Apache-2.0.txt) |

Changes: parts merged at the gripper's open/closed joint limits, simplified to
<= 20k faces, AG-95 shifted 9.7 cm so its mounting face is at z = 0.
