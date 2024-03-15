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
        self.action_space = spaces.Discrete(6) # F, L , R
        self.observation_space = spaces.Discrete(10)

        # State
        self.current_vel = 0
        self.data = None

        # Round state to decrease state space size
        self.num_dec_places = 2

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

        NUM_BINS = 10
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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

        span_0 = [0, int(width / NUM_BINS)]
        span_1 = [int(width / NUM_BINS), int(2 * width / NUM_BINS)]
        span_2 = [int(2 * width / NUM_BINS), int(3 * width / NUM_BINS)]
        span_3 = [int(3 * width / NUM_BINS), int(4 * width / NUM_BINS)]
        span_4 = [int(4 * width / NUM_BINS), int(5 * width / NUM_BINS)]
        span_5 = [int(5 * width / NUM_BINS), int(6 * width / NUM_BINS)]
        span_6 = [int(6 * width / NUM_BINS), int(7 * width / NUM_BINS)]
        span_7 = [int(7 * width / NUM_BINS), int(8 * width / NUM_BINS)]
        span_8 = [int(8 * width / NUM_BINS), int(9 * width / NUM_BINS)]
        span_9 = [int(9 * width / NUM_BINS), width]

        spans = np.stack([span_0, span_1, span_2, span_3, span_4, span_5, span_6, span_7, span_8, span_9])

        skew_0 = np.mean(np.array([gray[top_idx:bottom_idx, span_0]]))
        skew_1 = np.mean(np.array([gray[top_idx:bottom_idx, span_1]]))
        skew_2 = np.mean(np.array([gray[top_idx:bottom_idx, span_2]]))
        skew_3 = np.mean(np.array([gray[top_idx:bottom_idx, span_3]]))
        skew_4 = np.mean(np.array([gray[top_idx:bottom_idx, span_4]]))
        skew_5 = np.mean(np.array([gray[top_idx:bottom_idx, span_5]]))
        skew_6 = np.mean(np.array([gray[top_idx:bottom_idx, span_6]]))
        skew_7 = np.mean(np.array([gray[top_idx:bottom_idx, span_7]]))
        skew_8 = np.mean(np.array([gray[top_idx:bottom_idx, span_8]]))
        skew_9 = np.mean(np.array([gray[top_idx:bottom_idx, span_9]]))

        all_skew = [skew_0, skew_1, skew_2, skew_3, skew_4, skew_5, skew_6, skew_7, skew_8, skew_9]
        min_skew = min(all_skew)
        print(min_skew)

        s = 0

        roam_penalty = 0

        if min_skew < 255:
            self.timeout = 0
            if all_skew.count(min_skew) > 1:
                state = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
                s = 4

            else:
                s = all_skew.index(min_skew)
                state[s] = 1
        else:
            self.timeout += 1
            roam_penalty += -10*self.timeout # enforce an additional reward penalty if the car is off line, scales with time off line
            print("timeout")
            if self.timeout == 30:
                done = True

        print(state)
        cv2.circle(cv_image, (spans[s, 0], 200), 20, [0, 255, 0], -1)

        print("*** DONE PROCESSING IMAGE ***")
        return state, done, cv_image, roam_penalty

    def callback(self, data):
        self.data = data

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        print("===== STEPPING =====")

        vel_cmd = Twist()
        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.8
            vel_cmd.angular.z = 0.0
        elif action == 1:  #HARD LEFT
            vel_cmd.linear.x = 0.6
            vel_cmd.angular.z = 1.2
        elif action == 2: # SOFT LEFT
            vel_cmd.linear.x = 0.5
            vel_cmd.angular.z = 0.6
        elif action == 3:  # HARD RIGHT
            vel_cmd.linear.x = 0.6
            vel_cmd.angular.z = -1.2
        elif action == 4: # SOFT RIGHT
            vel_cmd.linear.x = 0.5
            vel_cmd.angular.z = -0.6
        elif action == 5: # Gentle Roll
            vel_cmd.linear.x = 0.3
            vel_cmd.angular.z = 0

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

        state, done, image, roam_penalty = self.process_image(data)

        mid_state = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        mid_state2 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

        far_left = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        far_left2 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        mid_left1 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        mid_left2 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

        far_right = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        far_right2 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        mid_right1 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        mid_right2 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

        # Set the rewards for your action
        if not done:
            if action == 0:  # FORWARD
                if state == mid_state or state == mid_state2:
                    reward = 20
                else:
                    reward = 1
            elif action == 1:  #  HARD LEFT
                if state == far_left or state == far_left2:
                    reward = 20
                elif state == far_right or state == far_right2:
                    reward = -40
                elif state == mid_left1 or state == mid_left2:
                    reward = 2
                elif state == mid_right1 or state == mid_right2:
                    reward = 0
                else:
                    reward = 5
            elif action == 2: # SOFT LEFT
                if state == mid_state or state == mid_state2:
                    reward = 3
                elif state == mid_left1 or state == mid_left2:
                    reward = 20
                else:
                    reward = 0
            elif action == 3: # HARD RIGHT
                if state == far_right or state == far_right2:
                    reward = 40
                elif state == mid_right1 or state == mid_right2:
                    reward = 2
                elif state==far_left or far_left2:
                    reward=-40
                else:
                    reward = 0
            elif action == 4: #SOFT RIGHT
                if state == mid_state or state == mid_state2:
                    reward = 3
                elif state == mid_right1 or state == mid_right2:
                    reward = 20
                else:
                    reward = 0
            elif action == 5: # GENTLE ROLL
                if state == far_left or state == far_right:
                    reward = 5
                else:
                    reward = -5
        else:
            reward = 1
    
        reward += roam_penalty

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.6
        cv2.putText(image, str(state), (10,20), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, str(action), (10,45), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("image", image)
        cv2.waitKey(1)
        print("===== DONE STEPPING =====")

        self.runtime += 1

        if self.runtime > 1000:
            done = True

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
        state, done, image, roam = self.process_image(data)

        print("====== DONE RESETTING =======")

        return state
