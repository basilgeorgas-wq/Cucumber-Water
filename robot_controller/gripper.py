"""
Mount a two-finger gripper on a robot's flange.

    g = Gripper(robot, "robotiq_2f85", env, mount_offset=0.006)
    g.open()                  # fingers fully open
    g.grip()                  # fingers closed: <name>_grip.stl
    g.grip("grip_cucumber")   # another closed position: <name>_grip_cucumber.stl

Model files live in robot_controller/Gripper_stl/:
    <name>_open.stl, <name>_grip*.stl  - finger positions (mount face z = 0,
                                          fingers along +z)
    <name>_collision.json              - collision shapes covering all positions
                                          (made by make_collision_capsules.py)

The gripper's collision shapes are added to the robot's last link, so every
collision check (obstacles, other robots, path planning) includes it.
See Gripper_stl/README.md for where the models come from and their licences.
"""
import glob
import json
import os

import numpy as np
import spatialgeometry as geometry
from spatialmath import SE3

GRIPPER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Gripper_stl")
_HIDDEN = SE3(0, 0, -50).A       # the unused finger state is parked out of sight


class Gripper:

    def __init__(self, robot, name, env, mount_offset=0.0, mount_rotation=0.0,
                 color=(0.25, 0.25, 0.28, 1.0)):
        """
        robot          : robot with AttachmentMixin (DOBOT_CR5, OMRON_TM5_700, ...)
        name           : 'robotiq_2f85' or 'dh_ag95'
        mount_offset   : metres from the last DH frame to the flange face (along tool z)
        mount_rotation : radians about tool z (which way the fingers open)
        """
        self.robot = robot
        self.name = name
        self.T_mount = (SE3(0, 0, mount_offset) * SE3.Rz(mount_rotation)).A

        self.meshes = {}
        for path in sorted(glob.glob(os.path.join(GRIPPER_DIR, f"{name}_*.stl"))):
            state = os.path.basename(path)[len(name) + 1:-4]      # e.g. 'open', 'grip_cucumber'
            mesh = geometry.Mesh(path, color=color)
            env.add(mesh)
            self.meshes[state] = mesh
        self.state = None
        self._add_collision_shapes()
        self.open()

    # ---- finger states ---------------------------------------------------
    def open(self):
        self._show("open")

    def grip(self, state="grip"):
        self._show(state)

    def _show(self, state):
        if state == self.state:
            return
        if state not in self.meshes:
            raise ValueError(f"{self.name} has no '{state}' model; available: {sorted(self.meshes)}")
        for s, mesh in self.meshes.items():
            self.robot.detach(mesh)
            self.robot.attach(mesh, self.T_mount if s == state else self.T_mount @ _HIDDEN)
        self.state = state

    # ---- collision -------------------------------------------------------
    def _add_collision_shapes(self):
        """Append the gripper's capsules and hull to the robot's last link."""
        path = os.path.join(GRIPPER_DIR, f"{self.name}_collision.json")
        with open(path) as f:
            data = json.load(f)
        R, t = self.T_mount[:3, :3], self.T_mount[:3, 3]
        last = self.robot.n                    # DH frame of the flange

        caps = getattr(self.robot, 'collision_link_capsules', None)
        if caps:
            caps[last] = list(caps[last]) + [
                (R @ np.array(a) + t, R @ np.array(b) + t, float(r))
                for a, b, r in data['capsules']]

        hulls = getattr(self.robot, 'collision_link_hulls', None)
        if hulls:
            h = data['hull']
            V = np.array(h['vertices']) @ R.T + t
            N = np.array(h['normals']) @ R.T
            hulls[last] = list(hulls[last]) + [(V, N, float(h.get('pad', 0.0)))]
