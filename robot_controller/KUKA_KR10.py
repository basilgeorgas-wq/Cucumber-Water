2
import swift
import spatialgeometry as geometry
import numpy as np
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
import time
import os
from roboticstoolbox import DHLink, DHRobot, jtraj


# Useful variables
from math import pi

# -----------------------------------------------------------------------------------#
class KUKA_KR10(DHRobot3D):
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
        link3D_names = dict(link0 = 'Base_k', 
                            link1 = 'Axis1', 
                            link2 = 'Axis2', 
                            link3 = 'Axis3', 
                            link4 = 'Axis4', 
                            link5 = 'Axis5',
                            link6 = 'Axis6'
                            )

        # A joint config and the 3D object transforms to match that config
        qtest = [0,
                 0,
                 0,
                 0,
                 0,
                 0
                 ]
        
        qtest_transforms = [spb.transl(0,0,0) ,                                                    
                            spb.transl(0,0,0) , 
                            spb.transl(0,0,0) ,
                            spb.transl(0,0,0) , 
                            spb.transl(0,0,0) , #spb.rpy2tr(0,0,0, order ='xyz'),
                            spb.transl(0,0,0) , 
                            spb.transl(0,0,0) ]
        
        current_path = os.path.join(os.path.dirname(__file__), 'Kuka_stl')
        super().__init__(links, link3D_names, name = 'KUKA_KR10', link3d_dir = current_path, qtest = qtest, qtest_transforms = qtest_transforms)
        self.q = qtest

    # -----------------------------------------------------------------------------------#
    def _create_DH(self):
        """
        Create robot's standard DH model
        """
        
        links = []
        d = [0.4, 0,     0,      0.515,  0, 0.087]
        a = [0,     0.56, -0.035, 0,     0, 0]
        alpha = [-pi/2, 0, pi/2, -pi/2, pi/2, 0]
        offset = [0,-pi/2,pi/2,0,0,0]
        qlim = [ [-(170 * pi/180), (170 * pi/180)],  [-pi/2, 135*pi/180],[-(80 * pi/180), (165 * pi/180)], 
                [-pi, pi],[-120*pi/180, 120*pi/180], [-2*pi, 2*pi]]
        
        n_links = 6

        for i in range(n_links):
            link = rtb.RevoluteDH(d= d[i], a=a[i], alpha= alpha[i], qlim= qlim[i], offset=offset[i])
            links.append(link)
        
        # link(1)=Link('alpha',-pi/2,'a',0, 'd',0.4, 'offset',0, 'qlim',[deg2rad(-170), deg2rad(170)]);
        # link(2)=Link('alpha',0,'a',0.56, 'd',0, 'offset',-pi/2, 'qlim',[deg2rad(-90), deg2rad(135)]);
        # link(3)=Link('alpha',pi/2,'a',-0.035, 'd',0, 'offset',pi/2, 'qlim',[deg2rad(-80), deg2rad(165)]);
        # link(4)=Link('alpha',-pi/2,'a',0, 'd',0.515, 'offset',0, 'qlim',[deg2rad(-185), deg2rad(185)]);
        # link(5)=Link('alpha',pi/2,'a',0, 'd',0, 'offset',0, 'qlim',[deg2rad(-120), deg2rad(120)]);
        # link(6)=Link('alpha',0,'a',0, 'd',0.087, 'offset',0, 'qlim',[deg2rad(-360), deg2rad(360)]);
        
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

    def attach_knife(self):
        """
        Attach the knife as the current tool on the robot
        """
        # Adjust the offset so the knife aligns with the end effector flange
        knife_tool = SE3(0, 0, 0.13) * SE3.Rx(pi/2) * SE3.Rz(pi/2) 
        self.tool = knife_tool
        print("Knife attached to end effector.")

    

# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":  
    r = KUKA_KR10()
    r.test()


    
