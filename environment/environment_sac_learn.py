#! usr/bin/env python
# Authors: deitieslulces #
# get path point of SAC_new_rnd

import rospy
import numpy as np
import math
import os
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from scipy.spatial.transform import Rotation as R
from respawnGoal import Respawn
from copy import copy
dirPath = os.path.dirname(os.path.realpath(__file__))


class Env(object):
    def __init__(self, action_dim=2):
        self.goal_x = 0.5
        self.goal_y = 1
        self.heading = 0
        self.initGoal = True
        self.get_goalbox = False

        self.action_dim = action_dim
        self.past_distance = 0
        self.stopped = 0
        self.num_goal = 0
        # self.po = self.position
        self.coordinate = []

        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()

        # Keys CTRL + c will stop script
        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        # you can stop turtlebot by publishing an empty Twist
        # message
        rospy.loginfo("Stopping TurtleBot")
        self.pub_cmd_vel.publish(Twist())
        rospy.sleep(1)

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        self.past_distance = goal_distance

        return goal_distance

    def getOdometry(self, odom):

        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        # print (orientation_list)
        _, _, yaw = euler_from_quaternion(orientation_list)
        # r = R.from_quat(orientation_list)
        # _, _, yaw = r.as_euler('xyz', degrees=True)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        # print 'yaw', yaw
        # print 'gA', goal_angle

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)
        # rospy.loginfo('heading: %.2f', self.heading)

    def getState(self, scan, past_action):
        scan_range = []
        heading = self.heading
        min_range = 0.136
        done = False

        for i in range(len(scan.ranges)):

            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)

            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)

        if min_range > min(scan_range) > 0:
            done = True

        # for pa in past_action:
        #     scan_range.append(pa)

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        self.coordinate.append((self.position.x, self.position.y))
        np.savetxt(dirPath + '/coordinate.txt', self.coordinate)
        # current_distance = self.getGoalDistace()

        if current_distance < 0.3:
            self.get_goalbox = True

        # print ("scan_range: ", scan_range)
        return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], done

    def setReward(self, state, action, done):
        obstacle_min_range = state[-2]
        current_distance = state[-3]
        heading = state[-4]
        # print('current_distance:',current_distance, self.past_distance)

        i = 2 + action[1] / 0.75
        angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
        tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
        yaw_reward = tr
        # print("yaw_reward: ",yaw_reward,"action: ", action)

        distance_rate = 2 ** (current_distance / self.goal_distance)
        # distance_rate = (self.past_distance - current_distance)
        '''
        if distance_rate > 0:
            reward = 2

        else:
            reward = -2
        '''

        if obstacle_min_range < 0.2:
            obs_reward = -5
        else:         # if distance_rate <=0, reward = 0
            obs_reward = 0

        reward = ((round(yaw_reward * 5, 2)) * distance_rate) + obs_reward

        self.past_distance = current_distance

        # reward = yaw_reward + obs_reward
        # print(reward)

        if done:
            rospy.loginfo("Collision!!")
            # reward = -500
            reward = -500
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            # reward = 1000
            reward = 1000
            self.pub_cmd_vel.publish(Twist())
            self.num_goal += 1
            if self.num_goal <= 10:
                print(self.num_goal)
            if self.num_goal <= 100:
                self.goal_x, self.goal_y = 0.5,1
                self.reset()
            else:
                self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)

            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False
            done = True

        return reward, done

    def step(self, action, past_action):

        vel_cmd = Twist()

        #linear_vel = 0.17
        linear_vel = action[0]
        ang_vel = action[1]
        # vel_cmd.linear.x = 0.15
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = ang_vel

        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data, past_action)
        reward, done = self.setReward(state, action, done)

        return np.asarray(state), reward, done

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
        # else:
        # self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)

        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(data, [0] * self.action_dim)

        return np.asarray(state)
