2
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
from roboticstoolbox import DHLink, DHRobot, jtraj


# Useful variables
from math import pi

# -----------------------------------------------------------------------------------#
class DOBOT_CR5(AttachmentMixin, DHRobot3D):
    """ 
    Example usage:
    >>> from ir-support import OMRON_TM5
    >>> import swift

    >>> r = OMRON_TM5()
    >>> q = [0,-pi/2,pi/4,0,0,0]
    >>> r.q = q
    >>> q_goal = [r.q[i]-pi/4 for i in range(r.n)]
    >>> env = swift.Swift()
    >>> env.launch(realtime= True)
    >>> r.add_to_env(env)
    >>> qtraj = rtb.jtraj(r.q, q_goal, 50).q
    >>> for q in qtraj:
    >>>    r.q = q
    >>>    env.step(0.02)

    """     
    def __init__(self):         
        # DH links
        links = self._create_DH()     
        # Names of the robot link files in the directory
        link3D_names = dict(link0 = 'Base_CR5',  
                            link1 = 'Joint1_CR5',  
                            link2 = 'Joint2_CR5',  
                            link3 = 'Joint3_CR5',  
                            link4 = 'Joint4_CR5',  
                            link5 = 'Joint5_CR5',  
                            link6 = 'EE_CR5', 
                            )

        # A joint config and the 3D object transforms to match that config
        qtest = [pi,
                 -pi/2,
                 0,
                 -pi/2,
                 0,
                 0
                 ]
        
        qtest_transforms = [spb.transl(0,0,0) ,                                                    
                            spb.transl(0,0,0) @ spb.trotz(pi/2), 
                            spb.transl(0,0,0) @ spb.trotz(pi/2),
                            spb.transl(0,0,0) @ spb.trotz(pi/2), 
                            spb.transl(0,0,0) @ spb.trotz(pi/2), #spb.rpy2tr(0,0,0, order ='xyz'),
                            spb.transl(0,0,0) @ spb.trotz(pi/2), 
                            spb.transl(0,0,0) @ spb.trotz(pi/2)]
        
        current_path = os.path.join(os.path.dirname(__file__), 'DoBot_stl')
        super().__init__(links, link3D_names, name = 'DOBOT', link3d_dir = current_path, qtest = qtest, qtest_transforms = qtest_transforms)
        # Collision capsules fitted to the 3D meshes (see make_collision_capsules.py)
        load_collision_capsules(self, current_path,
                                [link3D_names[f'link{k}'] for k in range(self.n + 1)])
        self.q = qtest

    # -----------------------------------------------------------------------------------#
    def _create_DH(self):
        """
        Create robot's standard DH model
        """
        # 4-6 Incorrect
        links = []
        d = [0.147, 0,     0,      0.116,  0.116, 0.105]
        a = [0,     -0.427, -0.357, 0, 0, 0]
        alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]
        offset = [0,0,0,0,0,0]
        qlim = [ [-2*pi,2*pi],  
                 [-2*pi,2*pi],
                 [-8*pi/9 ,8*pi/9], 
                 [-2*pi,2*pi],
                 [-2*pi,2*pi], 
                 [-2*pi,2*pi]]
        
        n_links = 6

        for i in range(n_links):
            link = rtb.RevoluteDH(d= d[i], a=a[i], alpha= alpha[i], qlim= qlim[i], offset=offset[i])
            links.append(link)
                
        # l0 = DHLink(d=0.147, a=0.147, alpha=pi/2, qlim=[-pi, pi])
        # l1 = DHLink(d= 0.427, a=0, alpha=0, qlim=[-pi, pi])
        # l2 = DHLink(d= 0, a= 0.122, alpha= -pi/2, offset= 0, qlim= [-(155 * pi/180), (155 * pi/180)])
        # # Dodgy Links
        # l3 = DHLink(d= 0.357, a= 0, alpha= 0, offset= 0, qlim= [-pi, pi])
        # l4 = DHLink(d= 0, a= 0.116, alpha= pi/2, offset= 0, qlim= [-pi, pi])
        # l5 = DHLink(d= 0.116, a= 0, alpha= 0, offset= 0, qlim= [-(270 * pi/180), (270 * pi/180)])
        return links
    
    def setbase(self, T):
        """
        Set the base transform of the robot
        """
        self.base = T
                 
    # -----------------------------------------------------------------------------------#
    def test(self):
        """
        Test the class by adding 3d objects into a new Swift window and do a simple movement,
        while plotting end effector positions
        """
        env = swift.Swift()
        env.launch(realtime= True)      
        self.q = self._qtest  
        # self.base = SE3(0.5,0.5,0)
        self.add_to_env(env)
        q_goal = [self.q[i]-pi/3 for i in range(self.n)]
        qtraj = rtb.jtraj(self.q, q_goal, 50).q
        t = self.fkine(self.q).t
        x1, y1, z1 = t[0], t[1], t[2]
        # fig = self.plot(self.q)
        max_reach_q = self._qtest
       
        T = self.fkine(max_reach_q)
        ee_pos_max = T.t
        max_reach = np.linalg.norm(ee_pos_max)

        #create sphere to visualise reach
        reach = geometry.Sphere(radius=max_reach, pose = SE3(0,0,0), color = (0,0.5,0,0.5))
        print("max reach= ", max_reach)
        env.add(reach)
        for q in qtraj:
            #self.q = max_reach_q
            self.q = q
            env.step(0.02)
            t = self.fkine(self.q).t
           
            x1, y1, z1 = t[0], t[1], t[2]
            # create sphere at end-effector position in swift
            point1 = geometry.Sphere(0.01, pose = SE3(x1, y1, z1), color = (0.5,0,0,1))
            env.add(point1)
    
      
        env.hold() # Uncomment to keep Swift from closing
        time.sleep(3)


    

# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":  
    r = DOBOT_CR5()
    r.test()


    
