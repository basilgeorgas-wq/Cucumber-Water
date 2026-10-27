import swift
import spatialgeometry as geometry
import numpy as np
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
from attachments import AttachmentMixin
from collision_model import load_collision_capsules
import time
import os


# Useful variables
from math import pi

# -----------------------------------------------------------------------------------#
class MOTOMAN_GP4(AttachmentMixin, DHRobot3D): 
    def __init__(self):
        self.has_knife = False
        self.knife_mesh = None
        
        links = self._create_DH()     
        link3D_names = dict(link0 = 'Ethan_Base_GP4', 
                            link1 = 'Ethan_J1_GP4', 
                            link2 = 'Ethan_J2_GP4',
                            link3 = 'Ethan_J3_GP4',
                            link4 = 'Ethan_J4_GP4',
                            link5 = 'Ethan_J5_GP4', 
                            link6 = 'Ethan_EE_GP4',
                            )
        qtest = [0,
                 -pi/2,
                 0,
                 pi/2,
                 0,
                 0
                 ]
        
        qtest_transforms = [spb.transl(0,0,0) ,                                                    
                            spb.transl(0,0,0) , 
                            spb.transl(0,0,0) ,
                            spb.transl(0,0,0) , 
                            spb.transl(0,0.0045,0.005) ,
                            spb.transl(0,0.0045,0.005) ,
                            spb.transl(0,0.0045,0.005) ]
        
        current_path = os.path.join(os.path.dirname(__file__), 'Motoman_GP4_stl')
        super().__init__(links, link3D_names, name = 'Motoman_GP4', link3d_dir = current_path, qtest = qtest, qtest_transforms = qtest_transforms)
        # Collision capsules fitted to the 3D meshes (see make_collision_capsules.py)
        load_collision_capsules(self, current_path,
                                [link3D_names[f'link{k}'] for k in range(self.n + 1)])
        self.q = qtest
        
        # File is 'Knife.dae' (capital K) - case matters on Linux/macOS
        knife_file = os.path.join(current_path, 'Knife.dae')
        if os.path.exists(knife_file):
            self.knife_mesh = geometry.Mesh(knife_file, color=(0.7, 0.7, 0.7))
            self.has_knife = True

            # Knife collision model, in the flange frame (last DH frame).
            # The blade is flat (~2.4 mm thick) and lies in the flange x-z plane.
            # Outline = convex hull of Knife.dae as drawn, simplified to 10 points
            # (x, z) in metres. Heel near the flange, tip at z ~ 0.25 m.
            outline_xz = [(-0.0097, -0.0356), (0.0305, -0.0314), (0.0278, 0.0454),
                          (0.0213, 0.1119), (0.0129, 0.1649), (-0.0048, 0.2241),
                          (-0.0148, 0.2416), (-0.0230, 0.2493), (-0.0335, 0.1585),
                          (-0.0289, -0.0211)]
            pts = [[x, 0.0, z] for x, z in outline_xz]
            # Capsules round the outline, plus one down the middle of the blade
            segments = [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
            segments.append(([0.0004, 0.0, -0.033], [-0.0230, 0.0, 0.2493]))
            self.collision_tool_segments = segments
            self.collision_tool_radius = 0.002   # half blade thickness + a little slack
            print(f"Knife loaded from: {knife_file}")
            # Set initial knife position
            T_ee = self.fkine(self.q)
            self.knife_mesh.T = T_ee.A @ self._qtest_transforms[6]
        else:
            print(f"Warning: Knife file not found at {knife_file}")

    # -----------------------------------------------------------------------------------#
    def _create_DH(self):
        links = []
        
        # Length in Z
        d = [0.330, 0.0, 0.0, 0.290, 0.0, 0.1] 
        # Length in X
        a = [0.0, -0.260, -0.015, 0.0, 0.0, 0.0] 
        # Twist in X
        alpha = [pi/2, 0, pi/2, -pi/2, pi/2, 0.0]
        
        offset = [0,0,0,-pi/2,0,0]

        #Pulled from MOTOMAN GP4 manual. Unsure why some are more than 360 degrees. 
        qlim = [
            [-170*pi/180, 170*pi/180],
            [-110*pi/180, 130*pi/180],
            [-65*pi/180, 200*pi/180],
            [-200*pi/180 - offset[3], 200*pi/180 - offset[3]],
            [-123*pi/180, 123*pi/180],
            [-455*pi/180, 455*pi/180]
        ]
        
        n_links = 6

        for i in range(n_links):
            link = rtb.RevoluteDH(d= d[i], a=a[i], alpha= alpha[i], qlim= qlim[i], offset=offset[i])
            links.append(link)
        
        return links
    
    def setbase(self, T):
        self.base = T
    
    # Override to add knife to environment
    def add_to_env(self, env):
        super().add_to_env(env)  # Add robot
        if self.has_knife:
            env.add(self.knife_mesh)  # Add knife
    
    # Override to update knife position with end-effector
    def _update_3dmodel(self):
        super()._update_3dmodel()  # Update robot
        if self.has_knife:
            # Get the actual link transforms used for the 3D models
            link_transforms = self._get_transforms(self.q)
            # Use the same transform as the end-effector link (link 6)
            self.knife_mesh.T = link_transforms[6] @ self._relation_matrices[6]
                 
    # -----------------------------------------------------------------------------------#
    def test(self):
        env = swift.Swift()
        env.launch(realtime= True)      
        self.q = self._qtest  
        self.add_to_env(env)
        q_goal = [self.q[i]-pi/3 for i in range(self.n)]
        qtraj = rtb.jtraj(self.q, q_goal, 50).q
        t = self.fkine(self.q).t
        x1, y1, z1 = t[0], t[1], t[2]
        max_reach_q = self._qtest
       
        T = self.fkine(max_reach_q)
        ee_pos_max = T.t
        max_reach = np.linalg.norm(ee_pos_max)

        # Create sphere to visualise reach
        reach = geometry.Sphere(radius=max_reach, pose = SE3(0,0,0), color = (0,0.5,0,0.5))
        print("max reach= ", max_reach)
        env.add(reach)
        
        for q in qtraj:
            self.q = q
            env.step(0.02)
            t = self.fkine(self.q).t
            x1, y1, z1 = t[0], t[1], t[2]
            # Create sphere at end-effector position in swift
            point1 = geometry.Sphere(0.01, pose = SE3(x1, y1, z1), color = (0.5,0,0,1))
            env.add(point1)
    
        env.hold()
        time.sleep(3)

# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":  
    r = MOTOMAN_GP4()
    r.test()