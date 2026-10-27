"""
Let a robot carry objects (glass, water, cucumber slice ...) that move rigidly
with its end-effector flange.

Use by listing AttachmentMixin BEFORE DHRobot3D in the class definition:

    class DOBOT_CR5(AttachmentMixin, DHRobot3D):
        ...

    robot.attach(glass_mesh)            # keeps the object's current offset from the flange
    robot.attach(glass_mesh, T_rel)     # or give the offset (4x4, flange frame) yourself
    robot.detach(glass_mesh)

Also fixes a lag in DHRobot3D: it redraws the 3D links *before* storing the
new joint angles, so the picture was always one step behind robot.q.
"""
import numpy as np


class AttachmentMixin:

    def attach(self, shape, T_rel=None, collision_segments=None, collision_radius=None):
        """Carry `shape` with the flange.

        T_rel              : 4x4 pose of the shape in the flange frame; if None,
                             the shape's current world pose is kept.
        collision_segments : optional list of (p0, p1) in the flange frame to add
                             to the robot's collision model while attached
        collision_radius   : radius for those segments (metres)
        """
        T_flange = self.flange_pose()
        if T_rel is None:
            T_rel = np.linalg.inv(T_flange) @ np.asarray(shape.T)
        attachments = self.__dict__.setdefault('_attachments', [])
        attachments.append({'shape': shape, 'T_rel': np.asarray(T_rel),
                            'segments': collision_segments, 'radius': collision_radius})
        shape.T = T_flange @ np.asarray(T_rel)
        self._refresh_attachment_collision()

    def detach(self, shape):
        """Stop carrying `shape` (it stays where it is)."""
        attachments = self.__dict__.get('_attachments', [])
        self.__dict__['_attachments'] = [a for a in attachments if a['shape'] is not shape]
        self._refresh_attachment_collision()

    def flange_pose(self, q=None):
        """World pose (4x4) of the last DH frame, ignoring robot.tool."""
        return self._get_transforms(self.q if q is None else q)[-1]

    # ---- internals -------------------------------------------------------
    def _refresh_attachment_collision(self):
        """Rebuild collision_tool_segments: the robot's own tool + attached objects."""
        if not hasattr(self, '_base_tool_segments'):
            self.__dict__['_base_tool_segments'] = list(getattr(self, 'collision_tool_segments', None) or [])
            self.__dict__['_base_tool_radius'] = getattr(self, 'collision_tool_radius', None)
        segments = list(self._base_tool_segments)
        radius = self._base_tool_radius
        for a in self.__dict__.get('_attachments', []):
            if a['segments']:
                segments += a['segments']
                # one radius per robot: use the largest so nothing is under-sized
                radius = a['radius'] if radius is None else max(radius, a['radius'])
        self.__dict__['collision_tool_segments'] = segments or None
        self.__dict__['collision_tool_radius'] = radius

    def _update_attachments(self):
        attachments = self.__dict__.get('_attachments')
        if not attachments:
            return
        T_flange = self.flange_pose()
        for a in attachments:
            a['shape'].T = T_flange @ a['T_rel']

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        # Redraw AFTER the new joints/base are stored (DHRobot3D draws before)
        if name in ('q', 'base') and hasattr(self, 'links_3d'):
            self._update_3dmodel()
            self._update_attachments()
