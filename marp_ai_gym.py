#!/usr/bin/python3

from gymnasium import spaces
import gymnasium as gym
import math
import matplotlib.pyplot as plt
import numpy as np
import random

COLLISION_REWARD = -60
INVALID_ACTION_REWARD = -20
IDLE_REWARD = -1
DEST_REACH_REWARD = 50
COMEOUT_FROM_DEST_REWARD = -10
MOVING_AWAY_REWARD = -2
CYCLIC_REWARD = -3
RECENT_POSES_SIZE = 3
MAX_TIMESTEP = 25


class MarpAIGym(gym.Env):
    def __init__(self, render_flag=False):
        super(MarpAIGym, self).__init__()
        self.graph = self.create_graph()
        self.action_space = spaces.Discrete(25)  # 5 actions for each AMR
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, -1, -1] + [-100] * 20, dtype=np.float32),
            high=np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 1, 10, 1, 1] + [10] * 20, dtype=np.float32),
            shape=(34,),
            dtype=np.float32,
        )  # amr poses, dest, distance to goal, waypoints, 30 elements
        self.renderflag = render_flag
        self.episode_count = 0
        self.valid_waypoints = list(self.graph.keys())
        self.picked_amr1_pose, self.picked_amr1_dest, self.picked_amr2_pose, self.picked_amr2_dest = (
            self.pick_start_dest(self.valid_waypoints)
        )
        self.solved_counter = 0
        random.seed(1) # Set the seed to make random values deterministic.

    def create_graph(self):
        return {
            (0, 2): [(0, 2), (1, 2)],
            (1, 2): [(1, 2), (0, 2), (2, 2)],
            (2, 2): [(2, 2), (1, 2), (3, 2), (2, 3), (2, 1)],
            (3, 2): [(3, 2), (2, 2), (4, 2)],
            (4, 2): [(4, 2), (3, 2), (5, 2)],
            (5, 2): [(5, 2), (4, 2)],
            (2, 3): [(2, 3), (2, 2)],
            (2, 1): [(2, 1), (2, 2), (2, 0)],
            (2, 0): [(2, 0), (2, 1)],
        }

    def init_val(self):
        self.amr1_last_pose = (-100, -100)
        self.amr2_last_pose = (-100, -100)
        if self.solved_counter >= 200 or self.episode_count % 10000 == 0:
            print("=============================================================================")
            print(f"generating new data, episdode count: {self.episode_count}, solve counter: {self.solved_counter}")
            self.picked_amr1_pose, self.picked_amr1_dest, self.picked_amr2_pose, self.picked_amr2_dest = (
                self.pick_start_dest(self.valid_waypoints)
            )
            print(f"amr1: {self.picked_amr1_pose} -> {self.picked_amr1_dest}")
            print(f"amr2: {self.picked_amr2_pose} -> {self.picked_amr2_dest}")
            self.solved_counter = 0
        self.amr1_pose, self.amr1_dest, self.amr2_pose, self.amr2_dest = (
            self.picked_amr1_pose,
            self.picked_amr1_dest,
            self.picked_amr2_pose,
            self.picked_amr2_dest,
        )

        self.amr1_options = self.pad_waypoints(self.graph[self.amr1_pose])
        self.amr2_options = self.pad_waypoints(self.graph[self.amr2_pose])
        self.step_count = 0
        self.episode_total_score = 0
        self.amr1_last_distance_to_goal = 0.0
        self.amr1_distance_to_goal = self.dist(
            self.amr1_pose[0], self.amr1_pose[1], self.amr1_dest[0], self.amr1_dest[1]
        )
        self.amr2_last_distance_to_goal = 0.0
        self.amr2_distance_to_goal = self.dist(
            self.amr2_pose[0], self.amr2_pose[1], self.amr2_dest[0], self.amr2_dest[1]
        )
        self.amr1_reached = False
        self.amr2_reached = False
        self.amr1_recent_poses = []
        self.amr2_recent_poses = []
        self.episode_count += 1

    def pick_start_dest(self, valid_waypoints, difficulty=2):
        junctions_wp = [wp for wp in valid_waypoints if len(self.graph[wp]) == 5]
        nj_wp = [wp for wp in valid_waypoints if wp not in junctions_wp] # Non-junction waypoints
        x_nj_wp = {wp[0]:[] for wp in nj_wp} # Non-junction waypoints along X
        y_nj_wp = {wp[1]:[] for wp in nj_wp} # Non-junction waypoints along Y
        for wp in nj_wp: # Populate X and Y NJ Waypoints
            x,y = wp
            x_nj_wp[x].append(wp)
            y_nj_wp[y].append(wp)

        # print("Printing pick start dest")
        # print(junctions_wp)
        # print(nj_wp)
        # print(x_nj_wp)
        # print(y_nj_wp)

        # Filter x and y values that have more than 1 waypoint
        x_nj_wp_mtone = {k: v for (k, v) in x_nj_wp.items() if len(v) >= 2}
        y_nj_wp_mtone = {k: v for (k, v) in y_nj_wp.items() if len(v) >= 2}
        if difficulty == 1: # Difficulty 1 - Any Start and End on the same line
            # Randomise amr 1 along X and amr 2 along Y
            amr1_start, amr1_dest = random.sample(x_nj_wp[random.choice(list(x_nj_wp_mtone))], 2)
            amr2_start, amr2_dest = random.sample(y_nj_wp[random.choice(list(y_nj_wp_mtone))], 2)
        elif difficulty == 2: # Difficulty 2 - Start and End Perpendicular to each other
            randJunction = random.sample(junctions_wp,1)
            junctionX, junctionY = randJunction[0]

            # Randomise amr 1 dest and amr 2 start along X and amr 2 dest and amr 1 start along Y
            amr1_dest, amr2_start = random.sample(x_nj_wp_mtone[junctionX], 2)
            amr1_start, amr2_dest = random.sample(y_nj_wp_mtone[junctionY], 2)
        elif difficulty == 3: # Difficulty 3 - Same axis, order of start pos same as dest pos
            randJunction = random.sample(junctions_wp,1)
            junctionX, junctionY = randJunction[0]
            
            randAxis = random.randint(0, 1)
            
            # TODO: Randomise start and dest to be either left or right
            if randAxis == 0: # X-axis
                dest1, dest2 = random.sample([wp for wp in y_nj_wp_mtone[junctionX] if wp[0] > junctionX] + randJunction, 2)
                start1, start2 = random.sample([wp for wp in y_nj_wp_mtone[junctionX] if wp[0] < junctionX] + randJunction, 2)
            else: # Y-axis
                dest1, dest2 = random.sample([wp for wp in x_nj_wp_mtone[junctionY] if wp[1] > junctionY] + randJunction, 2)
                start1, start2 = random.sample([wp for wp in x_nj_wp_mtone[junctionY] if wp[1] < junctionY] + randJunction, 2)

            if dest1 > dest2: # Reorder the destinations
                tempDest = dest1
                dest1 = dest2
                dest2 = tempDest
            if start1 < start2: # Reorder the start
                tempStart = start1
                start1 = start2
                start2 = tempStart
            amr1_start, amr1_dest = [start1, dest1]
            amr2_start, amr2_dest = [start2, dest2]
        elif difficulty == 4:
            randJunction = random.sample(junctions_wp,1)
            junctionX, junctionY = randJunction[0]
            
            randAxis = random.randint(0, 1)
            
            # TODO: Randomise start and dest to be either left or right
            if randAxis == 0: # X-axis
                dest1, dest2 = random.sample([wp for wp in y_nj_wp_mtone[junctionX] if wp[0] > junctionX] + randJunction, 2)
                start1, start2 = random.sample([wp for wp in y_nj_wp_mtone[junctionX] if wp[0] < junctionX] + randJunction, 2)
            else: # Y-axis
                dest1, dest2 = random.sample([wp for wp in x_nj_wp_mtone[junctionY] if wp[1] > junctionY] + randJunction, 2)
                start1, start2 = random.sample([wp for wp in x_nj_wp_mtone[junctionY] if wp[1] < junctionY] + randJunction, 2)

            if dest1 > dest2: # Reorder the destinations
                tempDest = dest1
                dest1 = dest2
                dest2 = tempDest
            if start1 > start2: # Reorder the start
                tempStart = start1
                start1 = start2
                start2 = tempStart
            amr1_start, amr1_dest = [start1, dest1]
            amr2_start, amr2_dest = [start2, dest2]
        elif difficulty == 5:
            amr1_start, amr1_dest = random.sample(valid_waypoints, 2)
            crossing_candidates = [wp for wp in valid_waypoints if wp not in {amr1_start, amr1_dest}]
            amr2_start, amr2_dest = random.sample(crossing_candidates, 2)
        
        return amr1_start, amr1_dest, amr2_start, amr2_dest

    def pad_waypoints(self, waypoints, max_size=5, pad_value=(-100, -100)):
        # pad waypoints (the invalid actions) with (-100, -100) until max_size is reached
        return waypoints + [pad_value] * (max_size - len(waypoints))

    def reset(self, seed=None):
        self.init_val()

        amr1_direction = np.array(self.amr1_dest) - np.array(self.amr1_pose)
        amr2_direction = np.array(self.amr2_dest) - np.array(self.amr2_pose)

        # Normalize to get unit vectors
        amr1_unit_vector = amr1_direction / np.linalg.norm(amr1_direction) if np.linalg.norm(amr1_direction) > 0 else np.array([0, 0])
        amr2_unit_vector = amr2_direction / np.linalg.norm(amr2_direction) if np.linalg.norm(amr2_direction) > 0 else np.array([0, 0])
        print(amr1_unit_vector, amr2_unit_vector)
        # Return the initial observation
        combined_array = np.concatenate(
            (
                list(self.amr1_pose),
                list(self.amr1_dest),
                list(self.amr2_pose),
                list(self.amr2_dest),
                [self.amr1_distance_to_goal],
                list(amr1_unit_vector),
                [self.amr2_distance_to_goal],
                list(amr2_unit_vector),
                [coord for waypoint in self.amr1_options for coord in waypoint],
                [coord for waypoint in self.amr2_options for coord in waypoint],
            )
        )
        return combined_array, {}

    def dist(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def calculate_reward(self, amr1_next, amr2_next):
        terminated = False
        truncated = False
        reward = 0

        # Ignore movement if selecting the padded value (-100, -100), and apply penalty
        if amr1_next == (-100, -100):
            reward += INVALID_ACTION_REWARD
        else:
            self.amr1_last_pose = self.amr1_pose
            self.amr1_pose = amr1_next
            if self.amr1_pose != self.amr1_dest and self.amr1_pose in self.amr1_recent_poses:
                # print(f"amr 1 cyclic {self.amr1_pose}, recentposes: {self.amr1_recent_poses}")
                # print("amr 1 cyclic")
                reward += CYCLIC_REWARD
            if len(self.amr1_recent_poses) == RECENT_POSES_SIZE:
                self.amr1_recent_poses.pop(0)
            self.amr1_recent_poses.append(self.amr1_pose)

        if amr2_next == (-100, -100):
            reward += INVALID_ACTION_REWARD
        else:
            self.amr2_last_pose = self.amr2_pose
            self.amr2_pose = amr2_next
            if self.amr2_pose != self.amr2_dest and self.amr2_pose in self.amr2_recent_poses:
                # print(f"amr 2 cyclic {self.amr2_pose}, recentposes: {self.amr2_recent_poses}")
                # print("amr 2 cyclic")
                reward += CYCLIC_REWARD
            if len(self.amr2_recent_poses) == RECENT_POSES_SIZE:
                self.amr2_recent_poses.pop(0)
            self.amr2_recent_poses.append(self.amr2_pose)

        if self.amr1_reached and self.amr1_pose != self.amr1_dest:
            # print("amr 1 comeout from dest")
            reward += COMEOUT_FROM_DEST_REWARD
        if self.amr2_reached and self.amr2_pose != self.amr2_dest:
            # print("amr 2 comeout from dest")
            reward += COMEOUT_FROM_DEST_REWARD

        # calculate distance to goal
        self.amr1_last_distance_to_goal = self.amr1_distance_to_goal
        self.amr1_distance_to_goal = self.dist(
            self.amr1_pose[0], self.amr1_pose[1], self.amr1_dest[0], self.amr1_dest[1]
        )
        self.amr2_last_distance_to_goal = self.amr2_distance_to_goal
        self.amr2_distance_to_goal = self.dist(
            self.amr2_pose[0], self.amr2_pose[1], self.amr2_dest[0], self.amr2_dest[1]
        )
        if self.amr1_distance_to_goal > self.amr1_last_distance_to_goal:
            # print("amr 1 moving away")
            reward += MOVING_AWAY_REWARD
        if self.amr2_distance_to_goal > self.amr2_last_distance_to_goal:
            # print("amr 2 moving away")
            reward += MOVING_AWAY_REWARD

        # terminate on collision
        if self.amr1_pose == self.amr2_pose:
            # print("collision, terminate")
            terminated = True
            reward += COLLISION_REWARD
        # swap = collision
        if self.amr1_last_pose == self.amr2_pose and self.amr1_pose == self.amr2_last_pose:
            # print("swap, terminate")
            terminated = True
            reward += COLLISION_REWARD

        # reward for idling
        if self.amr1_pose == self.amr1_last_pose:
            reward += IDLE_REWARD
        if self.amr2_pose == self.amr2_last_pose:
            reward += IDLE_REWARD

        # reward for reaching destination
        if not self.amr1_reached and self.amr1_pose == self.amr1_dest:
            self.amr1_reached = True
            # print("solved amr1")
            reward += DEST_REACH_REWARD
        if not self.amr2_reached and self.amr2_pose == self.amr2_dest:
            self.amr2_reached = True
            # print("solved amr2")
            reward += DEST_REACH_REWARD

        # both amrs reached destination, terminate
        if self.amr1_reached and self.amr2_reached:
            print("solved both, terminating")
            self.solved_counter += 1
            terminated = True

        # truncated on step count exceeding threshold
        if self.step_count >= MAX_TIMESTEP:
            truncated = True
        return terminated, truncated, reward

    def get_all_state(self):
        amr1_direction = np.array(self.amr1_dest) - np.array(self.amr1_pose)
        amr2_direction = np.array(self.amr2_dest) - np.array(self.amr2_pose)

        # Normalize to get unit vectors
        amr1_unit_vector = amr1_direction / np.linalg.norm(amr1_direction) if np.linalg.norm(amr1_direction) > 0 else np.array([0, 0])
        amr2_unit_vector = amr2_direction / np.linalg.norm(amr2_direction) if np.linalg.norm(amr2_direction) > 0 else np.array([0, 0])
        print(amr1_unit_vector, amr2_unit_vector)
        combined_array = np.concatenate(
            (
                list(self.amr1_pose),
                list(self.amr1_dest),
                list(self.amr2_pose),
                list(self.amr2_dest),
                [self.amr1_distance_to_goal],
                list(amr1_unit_vector),
                [self.amr2_distance_to_goal],
                list(amr2_unit_vector),
                [coord for waypoint in self.amr1_options for coord in waypoint],
                [coord for waypoint in self.amr2_options for coord in waypoint],
            )
        )
        observations = combined_array
        return (observations, self.reward, self.terminated, self.truncated, {})

    def step(self, action):
        self.step_count += 1

        # Convert the flat action back to amr1 and amr2 actions
        amr1_action = action // 5  # Integer division to get amr1's action
        amr2_action = action % 5  # Modulo operation to get amr2's action

        # amr1_action, amr2_action = action
        amr1_next = self.amr1_options[amr1_action]
        amr2_next = self.amr2_options[amr2_action]

        self.terminated, self.truncated, self.reward = self.calculate_reward(
            amr1_next, amr2_next
        )  # return bool and float

        self.amr1_options = self.pad_waypoints(self.graph.get(self.amr1_pose, []))
        self.amr2_options = self.pad_waypoints(self.graph.get(self.amr2_pose, []))
        if self.renderflag:
            self.render()

        self.episode_total_score += self.reward
        # print(f"Step {self.step_count}: Reward = {self.reward}, Total Score = {self.episode_total_score}")

        return self.get_all_state()

    def render(self):
        """
        Renders the current state of the environment using matplotlib.
        Displays the AMR1 and AMR2 positions and destinations, and the graph of possible waypoints.
        Updates the same plot window.
        """
        if not hasattr(self, "_initialized_render"):
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self._initialized_render = True

        self.ax.cla()  # Clear axes to avoid overlaying multiple plots

        # Plot the graph waypoints
        for node, neighbors in self.graph.items():
            self.ax.plot(node[0], node[1], "go")  # plot node (green circle)
            for neighbor in neighbors:
                self.ax.plot([node[0], neighbor[0]], [node[1], neighbor[1]], "b-")  # plot edges (blue lines)

        # Plot AMR1 and AMR2 current positions
        self.ax.plot(self.amr1_pose[0], self.amr1_pose[1], "ro", markersize=15, label="AMR1")  # red circle
        self.ax.plot(self.amr2_pose[0], self.amr2_pose[1], "bo", markersize=15, label="AMR2")  # blue circle

        # Plot AMR1 and AMR2 destinations
        self.ax.plot(self.amr1_dest[0], self.amr1_dest[1], "rx", markersize=12, label="AMR1 Dest")  # red cross
        self.ax.plot(self.amr2_dest[0], self.amr2_dest[1], "bx", markersize=12, label="AMR2 Dest")  # blue cross

        # Label the axes and add a title
        self.ax.set_xlim(0, 5)
        self.ax.set_ylim(0, 3)
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")
        self.ax.set_title("AMR1 & AMR2 Environment")

        # Display the legend
        self.ax.legend()

        # Update the plot
        plt.draw()
        plt.pause(0.5)  # Pause to allow for updates


if __name__ == "__main__":
    env = MarpAIGym(render_flag=True)
    print(env.reset())
    print(env.step(random.randint(0, 24)))
    print(env.step(random.randint(0, 24)))
