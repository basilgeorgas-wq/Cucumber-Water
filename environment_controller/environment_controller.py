import swift
import spatialgeometry as geometry
import os
from spatialmath import SE3
from math import pi
from ir_support import RectangularPrism
from spatialgeometry import Cuboid
import numpy as np

class Environment_Manager():
    def __init__(self, env):
        print("Initializing Environment Manager")
        self.env = env
        self.obstacles = []  # Store obstacle data
        
        self.add_cucumber(-0.8, -0.2, 0.94)
        self.add_enclosure(0, 0, 0)

        # Bench hitbox
        bench, bench_vertices, bench_faces, bench_normals = self.create_obstacle(
            x=-0.6, y=-1.235, z=0.46,
            x_length=0.6, y_length=3.2, z_length=0.9,
            env=self.env,
            name="bench"
        )
        self.obstacles.append({
            'name': 'bench',
            'vertices': bench_vertices,
            'faces': bench_faces,
            'face_normals': bench_normals
        })

        # Cupboard - sized to the kitchen model (x -0.93..-0.49 incl. handles, y -2.09..-0.55)
        cupboard, cupboard_vertices, cupboard_faces, cupboard_normals = self.create_obstacle(
            x=-0.71, y=-1.32, z=1.88,
            x_length=0.44, y_length=1.54, z_length=0.77,
            env=self.env,
            name="cupboard"
        )
        self.obstacles.append({
            'name': 'cupboard',
            'vertices': cupboard_vertices,
            'faces': cupboard_faces,
            'face_normals': cupboard_normals
        })

        # Sink tap (in the kitchen model, previously had no hitbox): upright neck + spout
        for name, (cx, cy, cz), (lx, ly, lz) in [
            ("tap_neck",  (-0.83, -1.41, 1.09), (0.09, 0.11, 0.36)),   # x -0.875..-0.785, z 0.91..1.27
            ("tap_spout", (-0.755, -1.385, 1.2075), (0.22, 0.14, 0.135)), # out to x -0.645, z 1.14..1.275
        ]:
            _, v, f, n = self.create_obstacle(x=cx, y=cy, z=cz, x_length=lx, y_length=ly,
                                              z_length=lz, env=self.env, name=name)
            self.obstacles.append({'name': name, 'vertices': v, 'faces': f, 'face_normals': n})

        # Fridge body hitbox
        fridge, fridge_vertices, fridge_faces, fridge_normals = self.create_obstacle(
            x=-0.6, y=0.75, z=0.8,
            x_length=0.7, y_length=0.7, z_length=1.7,
            env=self.env,
            name="fridge"
        )
        self.obstacles.append({
            'name': 'fridge',
            'vertices': fridge_vertices,
            'faces': fridge_faces,
            'face_normals': fridge_normals
        })

        # Fridge door hitbox
        door, door_vertices, door_faces, door_normals = self.create_obstacle(
            x=0.1, y=0.9, z=0.62,
            rotx=0, roty=0, rotz=-28*pi/180,
            x_length=0.7, y_length=0.1, z_length=1.1,
            env=self.env,
            name="fridge_door"
        )
        self.obstacles.append({
            'name': 'fridge_door',
            'vertices': door_vertices,
            'faces': door_faces,
            'face_normals': door_normals
        })

    def add_cucumber(self, x, y, z, rotx=0.0, roty=0.0, rotz=0.0):
        current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visual_files")
        stl_path = os.path.join(current_dir, "cucumber_base.stl")
        cucumber = geometry.Mesh(stl_path, pose=SE3(float(x), float(y), float(z))*SE3.Rx(rotx)*SE3.Ry(roty)*SE3.Rz(rotz), 
        scale=[1.0, 1.0, 1.0], color=(0.0, 1.0, 0.0, 1.0))
        self.env.add(cucumber)
        print("Cucumber Added")

    # ---- Props the robots carry -------------------------------------------
    # glass-cup.stl is modelled 1.2 m wide x 1.5 m tall with its base at z=0,
    # so it is scaled down to a real glass size.
    GLASS_SCALE = 0.06                      # -> 7.2 cm wide, 9 cm tall
    GLASS_RADIUS = 0.6 * GLASS_SCALE
    GLASS_HEIGHT = 1.5 * GLASS_SCALE
    SLICE_RADIUS = 0.019                    # cuc_slice.stl: 3.8 cm disc, 1.1 cm thick

    def add_glass(self, pose=None):
        """Glass (base at the mesh origin). Returns the Swift mesh."""
        current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visual_files")
        s = self.GLASS_SCALE
        glass = geometry.Mesh(os.path.join(current_dir, "glass-cup.stl"),
                              pose=pose if pose is not None else SE3(),
                              scale=[s, s, s], color=(0.8, 0.9, 1.0, 0.35))
        self.env.add(glass)
        return glass

    def add_water(self, pose=None):
        """Water inside the glass: translucent blue cylinder centred on its own origin."""
        water = geometry.Cylinder(radius=self.GLASS_RADIUS * 0.85, length=self.GLASS_HEIGHT * 0.6,
                                  pose=pose if pose is not None else SE3(),
                                  color=(0.2, 0.5, 1.0, 0.5))
        self.env.add(water)
        return water

    def add_slice(self, pose):
        """One cucumber slice (disc standing on edge, centred on its own origin)."""
        current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visual_files")
        cuc_slice = geometry.Mesh(os.path.join(current_dir, "cuc_slice.stl"),
                                  pose=pose, color=(0.3, 0.8, 0.3, 1.0))
        self.env.add(cuc_slice)
        return cuc_slice

    def add_enclosure(self, x=0.0, y=0.0, z=0.0, rotx=0.0, roty=0.0, rotz=0.0):
        current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visual_files")
        stl_path = os.path.join(current_dir, "Ethan_Kitchen.stl")
        enclosure = geometry.Mesh(
            stl_path,
            pose=SE3(float(x), float(y), float(z)) * SE3.Rx(rotx) * SE3.Ry(roty) * SE3.Rz(rotz),
            scale=[1, 1, 1],       
        )
        enclosure.collide = False        
        self.env.add(enclosure)
        print("Environment Added")

    def create_obstacle(self, x, y, z, rotx=0, roty=0, rotz=0, x_length=0, y_length=0, z_length=0, env=None, name="obstacle"):
        """
        Create a cuboid obstacle in the Swift environment, prepared for collision checking.
        """
        lwh = [x_length, y_length, z_length]
        centre = [x, y, z]
        pose = SE3(float(x), float(y), float(z)) * SE3.Rx(rotx) * SE3.Ry(roty) * SE3.Rz(rotz)
        
        obstacle = Cuboid(scale=lwh, color=[1.0, 0.0, 0.0, 0.2])
        obstacle.T = pose
        obstacle.name = name
        env.add(obstacle)
        
        # Get base geometry centered at origin
        vertices, faces, face_normals = RectangularPrism(lwh[0], lwh[1], lwh[2], center=[0, 0, 0]).get_data()
        
        # Transform vertices by the pose
        vertices = np.array(vertices)
        vertices_homogeneous = np.hstack([vertices, np.ones((len(vertices), 1))])
        vertices_transformed = (pose.A @ vertices_homogeneous.T).T[:, :3]

        # Normals are directions, so they must be rotated with the box too
        # (previously left in the box's local frame, which broke rotated boxes like the fridge door)
        face_normals_rotated = (pose.R @ np.array(face_normals, dtype=float).T).T

        return obstacle, vertices_transformed, faces, face_normals_rotated
    
    def get_collision_data(self):
        """
        Returns aggregated collision data for all obstacles.
        Handles variable-length faces (quads, triangles, etc).
        
        Returns:
        --------
        faces : list of lists
            Combined face indices from all obstacles (each face is a list of vertex indices)
        vertices : np.ndarray
            Combined vertices from all obstacles (N x 3)
        face_normals : np.ndarray
            Combined face normals from all obstacles (M x 3)
        """
        all_faces = []
        all_vertices = []
        all_face_normals = []
        vertex_offset = 0
        
        for obstacle_data in self.obstacles:
            vertices = np.array(obstacle_data['vertices'])
            faces = obstacle_data['faces']  # Keep as list of lists
            face_normals = np.array(obstacle_data['face_normals'])
            
            # Offset face indices to account for accumulated vertices
            faces_offset = []
            for face in faces:
                # Convert to list if numpy array, then offset each index
                face_list = [int(idx) + vertex_offset for idx in face]
                faces_offset.append(face_list)
            
            all_vertices.append(vertices)
            all_faces.extend(faces_offset)  # Extend, not append, to flatten the list
            all_face_normals.append(face_normals)
            
            # Update offset for next obstacle
            vertex_offset += len(vertices)
        
        # Combine into single arrays
        vertices_combined = np.vstack(all_vertices) if all_vertices else np.array([])
        face_normals_combined = np.vstack(all_face_normals) if all_face_normals else np.array([])
        
        return all_faces, vertices_combined, face_normals_combined

if __name__ == "__main__":
    env = swift.Swift()
    env.launch(realtime=True)
    manager = Environment_Manager(env)
    
    faces, vertices, face_normals = manager.get_collision_data()
    
    env.hold()
