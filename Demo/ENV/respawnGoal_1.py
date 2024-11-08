
import numpy as np
import rospy
import random
import time
import os
import sys
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
import math
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
dirPath = os.path.dirname(os.path.realpath(__file__))

class Respawn():
    def __init__(self):
        
        self.start_time = time.time()
        print('开始时间{}'.format(time.ctime(self.start_time)))
        self.modelPath = 'Your ROS workspace/turtlebot_ws/src/turtlebot3_simulations-master/turtlebot3_gazebo/models/turtlebot3_square/goal_box/model.sdf'
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()


        self.modelPath_1 = 'Your ROS workspace/turtlebot_ws/src/turtlebot3_simulations-master/turtlebot3_gazebo/models/turtlebot3_square/goal_point/model.sdf'
        self.f = open(self.modelPath_1, 'r')
        self.model_1 = self.f.read()

        self.modelPath_2 = 'Your ROS workspace/src/turtlebot_ws/src/turtlebot3_simulations-master/turtlebot3_gazebo/models/turtlebot3_square/goal_point_1/model.sdf'
        self.f = open(self.modelPath_2, 'r')
        self.model_2 = self.f.read()

        self.modelPath_3 = 'Your ROS workspace/turtlebot_ws/src/turtlebot3_simulations-master/turtlebot3_gazebo/models/turtlebot3_square/goal_point_2/model.sdf'
        self.f = open(self.modelPath_3, 'r')
        self.model_3 = self.f.read()


        self.stage = 4
        self.goal_position = Pose()

        self.modelName = 'goal'

        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)

        self.check_model = False
        self.index = 0

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            # model.name =['ground_plane', 'Untitled', 'obstacle_1', 'obstacle_2', 'obstacle_3', 'obstacle_4', 'turtlebot3_burger']
            if model.name[i] == "goal":
                self.check_model = True

    def respawnModel(self, i=None, x=-1, y=1, t=0):
        while True:
            if not self.check_model:
                self.goal_position.position.x = x
                self.goal_position.position.y = y
                modelName = self.modelName + "{}".format(i)+"{}".format(t)
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                if i is None and t == 0:
                    spawn_model_prox(modelName, self.model, 'robotos_name_space', self.goal_position, "world")
                elif i and t == 0:
                    spawn_model_prox(modelName, self.model_1, 'robotos_name_space', self.goal_position, "world")
                elif i and t == 1:
                    spawn_model_prox(modelName, self.model_2, 'robotos_name_space', self.goal_position, "world")
                elif i and t == 2:
                    spawn_model_prox(modelName, self.model_3, 'robotos_name_space', self.goal_position, "world")
                elif i and t == 3:
                    spawn_model_prox(modelName, self.model_4, 'robotos_name_space', self.goal_position, "world")
                elif i and t == 4:
                    spawn_model_prox(modelName, self.model_5, 'robotos_name_space', self.goal_position, "world")
                rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x, self.goal_position.position.y)
                break
            else:
                pass
                

    def deleteModel(self):
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.modelName)
                break
            else:
                pass
                

    def getPosition(self, delete=False):
        if delete:
            self.deleteModel()

        time.sleep(0.5)
        self.respawnModel()
        
        
        distance = 0
        data = np.loadtxt(dirPath + '/test_narrow/ENv1/P.txt')
        for i in range(len(data)-1):

            self.respawnModel(i, data[i][0], data[i][1], 2)


        # time.sleep(0.2)       



        distance1 = 0
        data = np.loadtxt(dirPath + '/test_narrow/ENv1/GAP.txt')
        for i in range(len(data)-1):
            self.respawnModel(i, data[i][0], data[i][1], 0)
            distance1 = distance1 + math.sqrt((data[i+1][0] - data[i][0]) ** 2 + (data[i+1][1] - data[i][1]) ** 2)

        # print ("start")

        distance2 = 0
        data = np.loadtxt(dirPath + '/test_narrow/ENv1/GA.txt')
        for j in range(len(data)-1):
            self.respawnModel(j, data[j][0], data[j][1], 1)
            distance2 = distance2 + math.sqrt((data[j+1][0] - data[j][0]) ** 2 + (data[j+1][1] - data[j][1]) ** 2)

        distance3 = 0
        # data = np.loadtxt(dirPath + '/Train_point/ENV1/sac.txt')
        # for j in range(len(data)-1):
        #     self.respawnModel(j, data[j][0], data[j][1], 3)
        #     distance3 = distance3 + math.sqrt((data[j+1][0] - data[j][0]) ** 2 + (data[j+1][1] - data[j][1]) ** 2)


        distance4 = 0
        # data = np.loadtxt(dirPath + '/Train_point/ENV2/td3.txt')
        # for j in range(len(data)-1):
        #     self.respawnModel(j, data[j][0], data[j][1], 4)
        #     distance4 = distance4 + math.sqrt((data[j+1][0] - data[j][0]) ** 2 + (data[j+1][1] - data[j][1]) ** 2)

        print ("distance:", distance, "distance1: ", distance1, "distance2: ", distance2,"distance3: ", distance3,"distance4: ", distance4)
        end_time = time.time()
        print('结束时间{}'.format(time.ctime(end_time)))

        execution_time = end_time - self.start_time
        formatted_execution_time = format(execution_time, ".3f")

        print(f"程序执行时间: {formatted_execution_time} 秒")
        return self.goal_position.position.x, self.goal_position.position.y
