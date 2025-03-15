#!/usr/bin/python3

from gymnasium import spaces
import gymnasium as gym
import math
import matplotlib.pyplot as plt
import numpy as np
import random

COLLISION_REWARD = -200
INVALID_ACTION_REWARD = -10
IDLE_REWARD = -1
BOTH_IDLE_REWARD = -5
DEST_REACH_REWARD = 200
DEST_STAY_REWARD = 20 
SUCCESS_REWARD = 200
COMEOUT_FROM_DEST_REWARD = -30
DISTANCE_REWARD_MODIFIER = 2
RECENT_POSES_SIZE = 3
CYCLIC_REWARD = -2

MAX_TIMESTEP = 100
GRID_SIZE = 9

# Curriculum Learning Parameters
SCORE_TARGET_UP = -350
SCORE_TARGET_DOWN = -100
CURRICULUM_INTERVAL = 3000

class MarpAIGym(gym.Env):
    def __init__(self, render_flag=False):
        super(MarpAIGym, self).__init__()
        self.grid_size = GRID_SIZE
        self.graph = self.create_graph()
        self.action_space = spaces.Discrete(25)  # 5 actions for each AMR
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, -1, -1] + [-100] * 20, dtype=np.float32),
             high=np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 1, 10, 1, 1] + [10] * 20, dtype=np.float32),
             shape=(34,),
             dtype=np.float32,
         )  # amr poses, dest, distance to goal, waypoints, 34 elements
        self.renderflag = render_flag
        self.current_level = 0
        self.episode_count = 0
        self.all_episode_scores = []
        self.init_val()

    def create_graph(self):
        """
        Creates an 11x11 grid with specific nodes removed.
        """
        graph = {}

        # Nodes to be completely removed (per user request)
        self.REMOVED_NODES = {
            (0,0), (0,2), (0,4), (0,6), (0,8),
            (2,8), (4,8), (6,8), (8,8), 
            (8,6), (8,4), (8,2), (8,0),
            (6,0), (4,0), (2,0),
            # Consecutive removed nodes (3-link and 2-link gaps)
            (3,6), (3,7), (3,8),  # Example of a 3-node gap
            (7,4), (7,5),  # Example of a 2-node gap
            (4,1), (4,2), (1,4), (2,4), #Force Deadlock
            (1,2), (2,1), (2,2), #Force Deadlock
            (8,5), (6,2), (5,6)
        }

        # Generate connectivity while considering removed nodes
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) in self.REMOVED_NODES:
                    continue  # Skip removed nodes

                neighbors = []
                if (x > 0) and ((x - 1, y) not in self.REMOVED_NODES):
                    neighbors.append((x - 1, y))
                if (x < self.grid_size - 1) and ((x + 1, y) not in self.REMOVED_NODES):
                    neighbors.append((x + 1, y))
                if (y > 0) and ((x, y - 1) not in self.REMOVED_NODES):
                    neighbors.append((x, y - 1))
                if (y < self.grid_size - 1) and ((x, y + 1) not in self.REMOVED_NODES):
                    neighbors.append((x, y + 1))

                graph[(x, y)] = neighbors

        return graph

    def init_val(self):
        self.amr1_last_pose = (-100, -100)
        self.amr1_pose = (0, 3)
        self.amr2_last_pose = (-100, -100)
        self.amr2_pose = (3, 0)
        
        self.set_amr_destinations()
        
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
        self.extra_timestep_counter = 0
        self.extra_timestep_active = False

    def set_amr_destinations(self):
        """Set destinations based on the current curriculum level."""
        predefined_positions_1 = [(2,5), (3,5), (4,5)]
        predefined_positions_2 = [(5,2), (5,3), (5,4)]
        predefined_positions_3 = [(4,5), (4,4), (5,4), (5,5)]
        predefined_positions_4 = [(2,5), (3,5), (4,5), (4,4), (5,2), (5,3), (5,4), (5,5)]
        level_distribution = np.random.rand()
        
        if self.current_level == 0:
            self.amr1_dest, self.amr2_dest = (2, 3), (3, 2)

        if self.current_level == 1:
             if level_distribution < 0.3:
                 self.amr1_dest, self.amr2_dest = (2, 3), (3, 2) 
             else:
                self.amr1_dest, self.amr2_dest = (4, 3), (3, 4) 
 
        elif self.current_level == 2:
            if level_distribution < 0.5:
                self.amr1_dest, self.amr2_dest = (4, 3), (3, 4)  # Level 1 settings
            else:
                self.amr1_dest, self.amr2_dest = (3, 4), (4, 3)  # New for Level 2
        
        elif self.current_level == 3:
            if level_distribution < 0.5:
                # Follow Level 2 logic 
                if np.random.rand() < 0.5:
                    self.amr1_dest, self.amr2_dest = (4, 3), (3, 4)  
                else:
                    self.amr1_dest, self.amr2_dest = (3, 4), (4, 3)  
            else:
                self.amr1_dest = random.choice(predefined_positions_1)  # New for Level 3
                self.amr2_dest = random.choice(predefined_positions_2)
        
        elif self.current_level == 4:
            if level_distribution < 0.5:
                # Follow Level 3 logic 
                if np.random.rand() < 0.5:
                    if np.random.rand() < 0.5:
                        self.amr1_dest, self.amr2_dest = (4, 3), (3, 4)  
                    else:
                        self.amr1_dest, self.amr2_dest = (3, 4), (4, 3)  
                else:
                    self.amr1_dest = random.choice(predefined_positions_1)  
                    self.amr2_dest = random.choice(predefined_positions_2)
            else:
                self.amr1_dest, self.amr2_dest = random.sample(predefined_positions_3, 2)  # New for Level 4
        
        elif self.current_level == 5:
            if level_distribution < 0.5:
                # Follow Level 4 logic 
                if np.random.rand() < 0.5:
                    if np.random.rand() < 0.5:
                        if np.random.rand() < 0.5:
                            self.amr1_dest, self.amr2_dest = (4, 3), (3, 4)  
                        else:
                            self.amr1_dest, self.amr2_dest = (3, 4), (4, 3)  
                    else:
                        self.amr1_dest = random.choice(predefined_positions_1)  
                        self.amr2_dest = random.choice(predefined_positions_2)
                else:
                    self.amr1_dest, self.amr2_dest = random.sample(predefined_positions_3, 2)  
        
            else:
                self.amr1_dest, self.amr2_dest = random.sample(predefined_positions_4, 2)  # New for Level 5
        
        
        elif self.current_level == 6:
            if level_distribution < 0.5:
                # Follow Level 5 logic 
                if np.random.rand() < 0.5:
                    if np.random.rand() < 0.5:
                        if np.random.rand() < 0.5:
                            if np.random.rand() < 0.5:
                                self.amr1_dest, self.amr2_dest = (4, 3), (3, 4)  # Level 1 settings
                            else:
                                self.amr1_dest, self.amr2_dest = (3, 4), (4, 3)  # Level 2 settings
                        else:
                            self.amr1_dest = random.choice(predefined_positions_1)  
                            self.amr2_dest = random.choice(predefined_positions_2)
                    else:
                        self.amr1_dest, self.amr2_dest = random.sample(predefined_positions_3, 2)   
                else:
                    self.amr1_dest, self.amr2_dest = random.sample(predefined_positions_4, 2)  
            else:
                self.amr1_dest, self.amr2_dest = random.sample(
                    [(x, y) for x in range(4,7) for y in range(4,7) if (x, y) not in self.REMOVED_NODES], 2
                )  # New for Level 6     
        
    def get_random_destinations(self, range_x=None, range_y=None, exclude_x=None, exclude_y=None):
        """Generate valid random destinations ensuring AMR1 and AMR2 do not overlap."""
        valid_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)
                           if (x, y) not in self.REMOVED_NODES]
        
        if range_x:
            valid_positions = [(x, y) for x, y in valid_positions if range_x[0] <= x <= range_x[1]]
        if range_y:
            valid_positions = [(x, y) for x, y in valid_positions if range_y[0] <= y <= range_y[1]]
        if exclude_x:
            valid_positions = [(x, y) for x, y in valid_positions if not (exclude_x[0] <= x <= exclude_x[1])]
        if exclude_y:
            valid_positions = [(x, y) for x, y in valid_positions if not (exclude_y[0] <= y <= exclude_y[1])]
        
        return random.sample(valid_positions, 2)
    
    def pad_waypoints(self, waypoints, max_size=5, pad_value=(-100, -100)):
        # pad waypoints with (-100, -100) until max_size is reached
        return waypoints + [pad_value] * (max_size - len(waypoints))

    def reset(self, seed=None):
        """Reset the environment and adjust difficulty if needed."""
        global SCORE_TARGET_UP, SCORE_TARGET_DOWN, CURRICULUM_INTERVAL
        print(f"---\nCurrent Level: {self.current_level}")
        self.all_episode_scores.append(self.episode_total_score)
        self.episode_count += 1
        
        if self.episode_count % CURRICULUM_INTERVAL == 0:
            avg_score = np.mean(self.all_episode_scores[-CURRICULUM_INTERVAL:])
            if avg_score > SCORE_TARGET_UP:
                self.increase_difficulty()
            elif avg_score < SCORE_TARGET_DOWN:
                self.decrease_difficulty()
                
        self.init_val()
        
        amr1_direction = np.array(self.amr1_dest) - np.array(self.amr1_pose)
        amr2_direction = np.array(self.amr2_dest) - np.array(self.amr2_pose)
 
        # Normalize to get unit vectors
        amr1_unit_vector = amr1_direction / np.linalg.norm(amr1_direction) if np.linalg.norm(amr1_direction) > 0 else np.array([0, 0])
        amr2_unit_vector = amr2_direction / np.linalg.norm(amr2_direction) if np.linalg.norm(amr2_direction) > 0 else np.array([0, 0])

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
        
        # check for cyclic movement
        #  self.amr1_pose in self.amr1_recent_poses:
        
        #  self.amr2_pose in self.amr2_recent_poses:

    
        # calculate distance to goal
        self.amr1_last_distance_to_goal = self.amr1_distance_to_goal
        self.amr1_distance_to_goal = self.dist(
            self.amr1_pose[0], self.amr1_pose[1], self.amr1_dest[0], self.amr1_dest[1]
        )
        self.amr2_last_distance_to_goal = self.amr2_distance_to_goal
        self.amr2_distance_to_goal = self.dist(
            self.amr2_pose[0], self.amr2_pose[1], self.amr2_dest[0], self.amr2_dest[1]
        )

        reward += DISTANCE_REWARD_MODIFIER*(self.amr1_last_distance_to_goal - self.amr1_distance_to_goal)
        reward += DISTANCE_REWARD_MODIFIER*(self.amr2_last_distance_to_goal - self.amr2_distance_to_goal)

        # terminate on collision
        if self.amr1_pose == self.amr2_pose:
            print("collision, terminate")
            terminated = True
            reward += COLLISION_REWARD
        # swap = collision
        if self.amr1_last_pose == self.amr2_pose and self.amr1_pose == self.amr2_last_pose:
            print("swap, terminate")
            terminated = True
            reward += COLLISION_REWARD

        # reward for idling
        if self.amr1_pose == self.amr1_last_pose and self.amr2_pose == self.amr2_last_pose:
            reward += BOTH_IDLE_REWARD
        else:
            if self.amr1_pose == self.amr1_last_pose:
                reward += IDLE_REWARD
            if self.amr2_pose == self.amr2_last_pose:
                reward += IDLE_REWARD

        # reward for reaching destination
        if not self.amr1_reached and self.amr1_pose == self.amr1_dest:
            self.amr1_reached = True
            print("solved amr1")
            reward += DEST_REACH_REWARD
        if not self.amr2_reached and self.amr2_pose == self.amr2_dest:
            self.amr2_reached = True
            print("solved amr2")
            reward += DEST_REACH_REWARD
            
        if self.amr1_reached:
            if self.amr1_pose != self.amr1_dest:
                # print("amr 1 comeout from dest")
                reward += COMEOUT_FROM_DEST_REWARD
            else: reward += DEST_STAY_REWARD
        if self.amr2_reached:   
            if self.amr2_pose != self.amr2_dest:
                # print("amr 2 comeout from dest")
                reward += COMEOUT_FROM_DEST_REWARD
            else: reward += DEST_STAY_REWARD
        
        # If both AMRs have reached their destinations at least once, start the extra countdown
        if self.amr1_reached and self.amr2_reached:
            if not self.extra_timestep_active:
                self.extra_timestep_active = True  # Start countdown only once
                self.extra_timestep_counter = 0  # Reset counter

            self.extra_timestep_counter += 1
            
        if self.extra_timestep_counter >= 10:
            print("Both AMRs have reached at some point, ending episode after extra time")
            terminated = True

        # both amrs reached and stayed at destination, terminate
        if self.amr1_pose == self.amr1_dest and self.amr2_pose == self.amr2_dest:
            reward += SUCCESS_REWARD
            print("solved both, terminating")
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

    def increase_difficulty(self):
        if self.current_level < 6:
            self.current_level += 1
            print(f"Increased difficulty to Level: {self.current_level}") 
        else:
            print("Maximum level reached.")
    
    def decrease_difficulty(self):
        if self.current_level > 0:
            self.current_level -= 1
            print(f"Decreased difficulty to Level: {self.current_level}")
        else:
            print("Minimum level reached.")
    
    def render(self):
        """
        Renders the environment and displays:
        - Removed nodes (black squares)
        - AMR1 and AMR2 positions
        - Destination points
        - Connected nodes and edges
        """
        if not hasattr(self, "_initialized_render"):
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self._initialized_render = True

        self.ax.cla()

        # Plot all nodes
        for node in self.graph.keys():
            if node in self.REMOVED_NODES:
                self.ax.plot(node[0], node[1], "ks", markersize=10, label="Removed Node" if "Removed Node" not in self.ax.get_legend_handles_labels()[1] else "")
            else:
                self.ax.plot(node[0], node[1], "go", markersize=6, label="Connected Node" if "Connected Node" not in self.ax.get_legend_handles_labels()[1] else "")

        # Draw valid edges
        for node, neighbors in self.graph.items():
            for neighbor in neighbors:
                self.ax.plot([node[0], neighbor[0]], [node[1], neighbor[1]], "b-", alpha=0.5)

        # Plot AMR1 and AMR2 positions
        self.ax.plot(self.amr1_pose[0], self.amr1_pose[1], "ro", markersize=12, label="AMR1")
        self.ax.plot(self.amr2_pose[0], self.amr2_pose[1], "bo", markersize=12, label="AMR2")

        # Plot AMR1 and AMR2 destinations
        self.ax.plot(self.amr1_dest[0], self.amr1_dest[1], "rx", markersize=14, label="AMR1 Dest")
        self.ax.plot(self.amr2_dest[0], self.amr2_dest[1], "bx", markersize=14, label="AMR2 Dest")

        self.ax.set_xlim(-1, self.grid_size)
        self.ax.set_ylim(-1, self.grid_size)
        self.ax.set_xticks(range(self.grid_size))
        self.ax.set_yticks(range(self.grid_size))
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")
        self.ax.set_title("AMR1 & AMR2 Navigation")

        self.ax.legend()
        plt.draw()
        plt.pause(0.5)


if __name__ == "__main__":
    env = MarpAIGym(render_flag=True)
    print(env.reset())
    print(env.step(random.randint(0, 24)))
    print(env.step(random.randint(0, 24)))
