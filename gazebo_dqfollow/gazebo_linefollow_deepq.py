import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding

from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.msg import LinkState


class GazeboDeepQLineEnv(gazebo_env.GazeboEnv):
    def __init__(self):
        #Launch simulation

        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)

        #set up feed vis pub
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Gazebo specific services to start/stop its behavior and
        # facilitate the overall RL environment
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        # Define end conditions
        #self.theta_threshold_radians = 12 * 2 * math.pi / 360
        #self.x_threshold = 15

        self.reward_range = (-np.inf, np.inf)

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected

        self.runtime = 0

        # Setup the environment
        self._seed()
        self.action_space = spaces.Discrete(3) # F, L , R
        self.observation_space = spaces.Discrete(3)

        # State
        self.current_vel = 0
        self.data = None

        # Round state to decrease state space size
        self.num_dec_places = 2

        # record the previous line state observation (initially set to 'line straight ahead')
        self.remeber = [0,1,0]

    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''

        print("*** PROCESSING IMAGE ****")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # cv2.imshow("raw", cv_image)

        #NUM_BINS = 10
        state = [0, 0, 0]
        done = False

        colour_drop = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        ret2, gray = cv2.threshold(colour_drop, 128, 255, cv2.THRESH_BINARY)

        cv2.imshow("Gray", colour_drop)
        cv2.waitKey(1)

        cv2.imshow("Binnary", gray)
        cv2.waitKey(1)

        width = 319
        height = 239

        top_idx = 200
        bottom_idx = height

        left_view = [0, int(2*(width/5))] # left two fifths of view
        mid_view = [int(width/5),int(4*width/5)]# middle fifth of view
        right_view = [int(4*width/5), width] # right two fifths of view

        views = np.stack([left_view,mid_view,right_view])

        lt = np.mean(np.array(gray[left_view[0] : left_view[1],top_idx:bottom_idx])) # LEFT TURN
        nt = np.mean(np.array(np.array(gray[mid_view[0] : mid_view[1],top_idx:bottom_idx])))  # NO TURN
        rt = np.mean(np.array(np.array(gray[right_view[0] : right_view[1],top_idx:bottom_idx]))) # RIGTH TURN

        turns = [lt,nt,rt]
        path = min(turns)
        print(path)

        s = 0



        if path < 255:
            self.timeout = 0
            if turns.count(path) > 1:
                turns = self.rollback # if no clear path is found, assume previous valid line state
            else:
                s = turns.index(path)
                state[s] = 1 # set state
                self.rollback = turns # update rollback with new valid state
        else:
            self.timeout += 1
            print("timeout")
            if self.timeout == 50:
                done = True

        print(state)
        #cv2.circle(cv_image, (views[s, 0], 200), 20, [0, 255, 0], -1)

        print("*** DONE PROCESSING IMAGE ***")
        return state, done, cv_image

    def callback(self, data):
        self.data = data

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        print("===== STEPPING =====")

        vel_cmd = Twist()
        if action == 0:  # LEFT TURN
            vel_cmd.linear.x = 0
            vel_cmd.angular.z = 0.5
        elif action == 1:  # FORWARD
            vel_cmd.linear.x = 0.5
            vel_cmd.angular.z = 0
        elif action == 2: # RIGHT TURN
            vel_cmd.linear.x = 0
            vel_cmd.angular.z = -0.5


        # Unpause simulation to make observations
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        #action_msg.data = self.current_vel
        self.vel_pub.publish(vel_cmd)

        # Define states
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        # Pause
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        state, done, image = self.process_image(data)

        left = [1,0,0]
        mid = [0,1,0]
        right = [0,0,1]

        # Set the rewards for your action
        if not done:
            if action == 0:  # CHOSE LEFT TURN
                if state == left:
                    reward = 500
                else:
                    reward = 0
            elif action == 1: # CHOSE FORWARD
                if state == mid:
                    reward = 500
                else:
                    reward = 0
            else:
                if state == right:
                    reward = 500
                else:
                    reward = 0
        else:
            reward = -500

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.6
        cv2.putText(image, str(state), (10,20), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, str(action), (10,45), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("image", image)
        cv2.waitKey(1)
        print("===== DONE STEPPING =====")

        #self.runtime += 1

        #if self.runtime > 1000:
           # done = True

        return state, reward, done, {}

    def reset(self):
        print("===== RESETTING =====")

        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        self.runtime = 0
        state, done, image = self.process_image(data)

        print("====== DONE RESETTING =======")

        return state
