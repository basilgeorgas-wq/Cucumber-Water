import swift
import spatialgeometry as geometry
import numpy as np
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
import time
import os
from roboticstoolbox import jtraj
from math import pi


# -----------------------------------------------------------------------------------#
class DOBOT_CR7(DHRobot3D):
    """ 
    DOBOT CR7 - Collaborative Robot Arm
    Approximate DH model and 3D mesh integration.
    
    Example usage:
    >>> from robot_controller.DOBOT_CR7 import DOBOT_CR7
    >>> import swift
    >>> r = DOBOT_CR7()
    >>> env = swift.Swift()
    >>> env.launch(realtime=True)
    >>> r.add_to_env(env)
    >>> q_goal = [r.q[i] - pi/4 for i in range(r.n)]
    >>> qtraj = rtb.jtraj(r.q, q_goal, 50).q
    >>> for q in qtraj:
    >>>     r.q = q
    >>>     env.step(0.02)
    """

    def __init__(self):
        # DH links
        links = self._create_DH()

        # Names of the robot link files in the directory
        link3D_names = dict(
            link0='BaseCR7',
            link1='Joint1CR7',
            link2='Joint2CR7',
            link3='Joint3CR7',
            link4='Joint4CR7',
            link5='Joint5CR7',
            link6='EECR7'
        )

        # Test joint configuration and corresponding transforms
        qtest = [0, -pi/2, 0, -pi/2, 0, 0]
        qtest_transforms = [
            spb.transl(0, 0, 0),
            spb.transl(0, 0, 0) @ spb.trotz(pi/2),
            spb.transl(0, 0, 0) @ spb.trotz(pi/2),
            spb.transl(0, 0, 0) @ spb.trotz(pi/2),
            spb.transl(0, 0, 0) @ spb.trotz(pi/2),
            spb.transl(0, 0, 0) @ spb.trotz(pi/2),
            spb.transl(0, 0, 0) @ spb.trotz(pi/2)
        ]

        # Path to mesh folder
        current_path = os.path.join(os.path.dirname(__file__), 'DoBotCR7_stl')

        super().__init__(
            links,
            link3D_names,
            name='DOBOT_CR7',
            link3d_dir=current_path,
            qtest=qtest,
            qtest_transforms=qtest_transforms
        )
        self.q = qtest

        try:
            for link in self.links_3d:
                link.color = (0.1, 0.1, 0.1, 1.0)  # dark grey like the real Dobot
            print("Applied custom colors to DoBot CR7 links.")
        except Exception as e:
            print(f"Color assignment skipped: {e}")

        




    # -----------------------------------------------------------------------------------#
    def _create_DH(self):
        """
        Create robot's standard DH model
        (values approximate for the DOBOT CR7)
        """

        d = [0.110, 0, 0, 0.060, 0.050, 0.080]
        a = [0, 0.135, 0.147, 0, 0, 0]
        alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]
        offset = [0, 0, 0, 0, 0, 0]
        qlim = [
            [-(270 * pi / 180), (270 * pi / 180)],
            [-pi, pi],
            [-(155 * pi / 180), (155 * pi / 180)],
            [-pi, pi],
            [-pi, pi],
            [-(270 * pi / 180), (270 * pi / 180)]
        ]

        links = []
        for i in range(6):
            link = rtb.RevoluteDH(
                d=d[i], a=a[i], alpha=alpha[i], qlim=qlim[i], offset=offset[i]
            )
            links.append(link)

        return links

    # -----------------------------------------------------------------------------------#
    def setbase(self, T):
        """
        Set the base transform of the robot
        """
        self.base = T

    # -----------------------------------------------------------------------------------#
    def test(self):
        """
        Test the DOBOT_CR7 model by visualizing in Swift
        and performing a simple joint movement.
        """
        env = swift.Swift()
        env.launch(realtime=True)
        self.q = self._qtest
        self.add_to_env(env)

        q_goal = [self.q[i] - pi / 4 for i in range(self.n)]
        qtraj = rtb.jtraj(self.q, q_goal, 50).q

        t = self.fkine(self.q).t
        x1, y1, z1 = t[0], t[1], t[2]

        # Display reach sphere
        max_reach_q = self._qtest
        T = self.fkine(max_reach_q)
        ee_pos_max = T.t
        max_reach = np.linalg.norm(ee_pos_max)
        reach = geometry.Sphere(
            radius=max_reach, pose=SE3(0, 0, 0), color=(0, 0.5, 0, 0.3)
        )
        env.add(reach)
        print("Max reach =", round(max_reach, 3), "m")

        for q in qtraj:
            self.q = q
            env.step(0.02)
            t = self.fkine(self.q).t
            point = geometry.Sphere(0.01, pose=SE3(t[0], t[1], t[2]), color=(0.5, 0, 0, 1))
            env.add(point)

        env.hold()
        time.sleep(2)


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    r = DOBOT_CR7()
    r.test()



    
