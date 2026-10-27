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
class Rebel(DHRobot3D):
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
        link3D_names = dict(link0 = 'Base_Rebel', 
                            link1 = 'Joint1_R', 
                            link2 = 'Joint2_R', 
                            link3 = 'Joint3_R', 
                            link4 = 'Joint4_R', 
                            link5 = 'Joint5_R'
                            )

        # A joint config and the 3D object transforms to match that config
        qtest = [0,
                -pi/2,
                 0,
                 -pi/2,
                 0,
                 ]
        
        qtest_transforms = [spb.transl(0,0,0) ,                                                    
                            spb.transl(0,0,0) @ spb.trotz(pi/2), 
                            spb.transl(0,0,0) @ spb.trotz(pi/2),
                            spb.transl(0,0,0) @ spb.trotz(pi/2), 
                            spb.transl(0,0,0) @ spb.trotz(pi/2), #spb.rpy2tr(0,0,0, order ='xyz'),
                            spb.transl(0,0,0) @ spb.trotz(pi/2)]
        
        current_path = os.path.join(os.path.dirname(__file__), 'Rebel_stl')
        super().__init__(links, link3D_names, name = 'Rebel', link3d_dir = current_path, qtest = qtest, qtest_transforms = qtest_transforms)
        self.q = qtest

    # -----------------------------------------------------------------------------------#
    def _create_DH(self):
        """
        Create robot's standard DH model
        """
        
        links = []
        d = [0.1452, 0,     0,      0.123,  0.112,0.1132]
        a = [0,     -0.334, -0.3167, 0,     0, 0]
        alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]
        offset = [0,0,0,0,0,0]
        qlim = [ [-(270 * pi/180), (270 * pi/180)],  [-pi, pi],[-(155 * pi/180), (155 * pi/180)], 
                [-pi, pi],[-pi, pi], [-(270 * pi/180), (270 * pi/180)]]
        
        n_links = 5

        for i in range(n_links):
            link = rtb.RevoluteDH(d= d[i], a=a[i], alpha= alpha[i], qlim= qlim[i], offset=offset[i])
            links.append(link)
        
        # links[0] = DHLink(d= 0.1452, a= 0, alpha= pi/2, offset= 0, qlim= [-(270 * pi/180), (270 * pi/180)])
        # links[1]= DHLink(d= 0, a= -0.329, alpha= 0, offset= 0, qlim= [-pi, pi])
        # links[2] = DHLink(d= 0, a= -0.3115, alpha= 0, offset= 0, qlim= [-(155 * pi/180), (155 * pi/180)])
        # links[3] = DHLink(d= 0.123, a= 0, alpha= pi/2, offset= 0, qlim= [-pi, pi])
        # links[4] = DHLink(d= 0.122, a= 0, alpha= -pi/2, offset= 0, qlim= [-pi, pi])
        # links[5] = DHLink(d= 0.1132, a= 0, alpha= 0, offset= 0, qlim= [-(270 * pi/180), (270 * pi/180)])
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
    r = Rebel()
    r.test()


    
