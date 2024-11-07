#! usr/bin/env python
# Authors: deitieslulces #

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
# from scipy.spatial.transform import Rotation as R
from respawnGoal import Respawn
from copy import copy


class Env(object):
    def __init__(self, action_dim=2):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()

        self.action_dim = action_dim
        self.past_distance = 0
        self.stopped = 0
        self.num_goal = 0

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

    def getGoalDistace(self):  #获得欧式直线距离
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        self.past_distance = goal_distance

        return goal_distance

    def getOdometry(self, odom):

        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
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

    # 实参 data, [0] * self.action_dim
    def getState(self, scan, past_action):
        #scanheader: 
        # seq: 0
        # stamp: 
        #     secs: 0
        #     nsecs: 200000000
        # frame_id: "base_scan"
        # angle_min: 0.0
        # angle_max: 6.28318977355957  360度
        # angle_increment: 0.2731821835041046
        # time_increment: 0.0
        # scan_time: 0.0
        # range_min: 0.11999999731779099
        # range_max: 3.5
        # ranges: [3.3566431999206543, inf, inf, 2.658447504043579, 2.678169012069702, inf, inf, 1.963521957397461, 2.378981113433838, 2.939527988433838, inf, inf, inf, 2.242290735244751, 0.9218885898590088, 0.8314639329910278, 3.390415906906128, 3.2125446796417236, 3.2618539333343506, 2.118931531906128, inf, inf, 1.8623141050338745, 3.3532679080963135]
        # intensities: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        #  ranges 有24个维度

        # 激光扫描数据类型，包含的过去动作列表
        scan_range = []
        # 机器人的朝向
        heading = self.heading
        # 与障碍物的最小阈值
        min_range = 0.125
        # 是否发生碰撞
        done = False
        #print("len{}".format(len(scan.ranges)))
        for i in range(len(scan.ranges)):
            # 如果是无穷大 则添加3.5
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            # 如果激光扫描的不是一个数字 则append 0
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        #len(scan_range)  = 24

        # 返回列表的最小值，保留两位小数,min 取最小值，round保留两位小数
        obstacle_min_range = round(min(scan_range), 4)
        # 获取最小距离 ，障碍物的角度，获得其索引
        obstacle_angle = np.argmin(scan_range)
        # 
        if min_range > min(scan_range) > 0:  #即小于最小阈值则判定为碰撞
            done = True

        # for pa in past_action:
        #     scan_range.append(pa)

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        # current_distance = self.getGoalDistace()
        #print(current_distance)

        if current_distance < 0.2:
            self.get_goalbox = True  #则成功到达

        # print ("scan_range: ", scan_range)

        #返回一个包含激光扫描数据、机器人朝向、与目标的距离、最小距离障碍物的距离和角度的状态数组，和是否碰撞
        return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], done ,obstacle_min_range

    def setReward(self, state, action, done):
        goal_flag=False
        obstacle_min_range = state[-2] #最小障碍物的距离
        current_distance = state[-3]    #当前位置与目标位置的距离
        heading = state[-4] #获得偏航角
        # print("state的长度{}".format(len(state))) #364
        # print('current_distance:',current_distance, self.past_distance)

        i = 2 + action[1] / 0.75
        angle = pi / 4 + heading + (pi / 8 * i) 
        tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
        yaw_reward = tr # 偏航的奖励值
        # print("yaw_reward: ",yaw_reward,"action: ", action)

        #distance_rate = 2 ** ((current_distance / self.goal_distance)-1)  #计算距离
        distance_rate = 2 **((self.past_distance / current_distance)-1)
        
        '''
        if distance_rate > 0:
            reward = 2

        else:
            reward = -2
        '''

        if obstacle_min_range < 0.25:
            obs_reward = -10
        else:         # if distance_rate <=0, reward = 0
            obs_reward = 0
        
        reward = ((round(yaw_reward * 5, 2)) * distance_rate) + obs_reward+distance_rate

        self.past_distance = current_distance

        # reward = yaw_reward + obs_reward
        #print(reward)

        if done:
            goal_flag=False
            rospy.loginfo("Collision!!")
            reward = -650
            #reward = -800
            self.pub_cmd_vel.publish(Twist())#  发布空消息使其停止

        if self.get_goalbox:
            goal_flag = True
            rospy.loginfo("Goal!!")
            reward = 1000
            #reward = 800
            self.pub_cmd_vel.publish(Twist()) #发布速度指令，发布空消息使其停止
            self.num_goal += 1
            #print("第%d次goal"%(self.num_goal))


            #  更换目标点
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)

            self.goal_distance = self.getGoalDistace()   #获得当前位置到目标点的直线距离
            self.get_goalbox = False

        return reward, done ,goal_flag

    def step(self, action, past_action):

        vel_cmd = Twist()

        # linear_vel = 0.15
        linear_vel = action[0]
        # [角速度]
        ang_vel = action[1]
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)
        data = None

        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done , obstacle_min_range = self.getState(data, past_action)  # state 是列表 长度为28
        # print("*****obstacle_min_range:{}".format(obstacle_min_range))

        reward, done ,goal_flag= self.setReward(state, action, done)  # reward=4.02

        return np.asarray(state), reward, done ,goal_flag,obstacle_min_range # 状态、奖励、是否碰撞

    def reset(self):

        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("gazebo/reset_simulation service call failed")

        # 存储一个激光雷达数据
        data = None
        while data is None:
            try:
                # 主题、数据类型
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass
        
        # initGoal初始化默认为true
        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        
        # 获取到目标点的距离
        self.goal_distance = self.getGoalDistace()
        state, done ,obstacle_min_range= self.getState(data, [0] * self.action_dim) # 传入data 和 一个列表推倒式
        #len(state)=28
        return np.asarray(state)
