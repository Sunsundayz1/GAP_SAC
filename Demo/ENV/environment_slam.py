#! usr/bin/env python
# Authors: deitieslulces #
# create path planning

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from scipy.spatial.transform import Rotation as R
#from respawnGoal import Respawn
from respawnGoal_1 import Respawn
from copy import copy


class Env(object):
    def __init__(self, action_dim=2):  
        self.initGoal = True   
        self.get_goalbox = False  
        self.position = Pose()   

        self.action_dim = action_dim
        self.past_distance = 0
        self.stopped = 0
        self.num_goal = 0

        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        

    def reset(self):

        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

