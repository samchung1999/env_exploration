import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import time
import gym
from gym import spaces
from copy import deepcopy
import os
import sys
import math
import heapq
abspath = os.path.dirname(os.path.abspath(__file__))
class ExploreEnv_v1(gym.Env):
    def __init__(self, map_name='train_map_l1', random_obstacle=False, training=False, render=False):
        self.explore_map_size = 24
        self.local_map_size_half = 12
        self.episode_limit = 30 
        self.occupied_pixel = 255
        self.unknown_pixel = 128
        self.free_pixel = 0
        self.explored_pixel = 255
        self.agent_pixel = 128
        self.laser_range_max = 50  
        self.laser_angle_resolution = 0.05 * np.pi  
        self.laser_angle_half = 0.75 * np.pi  
        self.orientation = 0  
        self.position = np.zeros(2, dtype=np.int32)  
        self.robot_position = np.zeros(2, dtype=np.int32)  
        self.path_data = []
        self.explored_coordinates=[]
        self.target_position_data = []
        self.move = [[0, 1],  
                     [1, 0],  
                     [0, -1],  
                     [-1, 0]]  
        self.explore_rate = 0  
        self.episode_steps = 0  
        self.ground_truth_map = np.load(abspath + '/map/{}.npy'.format(map_name))  
        self.ground_truth_map.flags.writeable = False  
        self.real_map = deepcopy(self.ground_truth_map)
        self.map = np.ones_like(self.real_map) * self.unknown_pixel
        self.grid_num = (self.real_map != self.unknown_pixel).sum()  
        self.global_map = np.zeros((self.explore_map_size, self.explore_map_size), dtype=np.uint8)
        self.local_map = np.zeros((self.explore_map_size, self.explore_map_size), dtype=np.uint8)
        self.count=0
        self.random_obstacle = random_obstacle
        self.num_obstacles = 4
        if training:
            self.max_explore_rate = 0.99
        else:
            self.max_explore_rate = 1.0
        if render:
            plt.ion()
            self.Dim0 = []
            self.Dim1 = []
        self.free_space=self.get_free_positions()
        self.free_space_num=self.free_space.shape[0]
        self.action_dim   = 100
        self.s_map_dim = (2, self.explore_map_size, self.explore_map_size)
        self.s_sensor_dim = (round(2 * self.laser_angle_half / self.laser_angle_resolution) + 2,)
        self.action_space = spaces.Discrete(self.action_dim)
        self.observation_space = spaces.Dict({"s_map": spaces.Box(low=0, high=255, shape=self.s_map_dim, dtype=np.uint8),  
                                              "s_sensor": spaces.Box(low=0, high=1.0, shape=self.s_sensor_dim, dtype=np.float32)})  
        print("init {}".format(map_name))
    def get_free_positions(self): 
        free_positions = np.argwhere(self.real_map == self.free_pixel)
        return free_positions
    def update_map(self, ):
        self.map[self.position[0], self.position[1]] = self.real_map[self.position[0], self.position[1]]
        laser = []
        for theta in np.arange(self.orientation * 0.5 * np.pi - self.laser_angle_half, self.orientation * 0.5 * np.pi + self.laser_angle_half + 1e-5, self.laser_angle_resolution):
            for r in range(1, self.laser_range_max + 1):
                dim0 = int(round(self.position[0] + r * np.sin(theta)))
                dim1 = int(round(self.position[1] + r * np.cos(theta)))
                self.map[dim0, dim1] = self.real_map[dim0, dim1]
                if self.real_map[dim0, dim1] == self.occupied_pixel:
                    break
            laser.append(np.sqrt((dim0 - self.position[0]) ** 2 + (dim1 - self.position[1]) ** 2))
        return np.array(laser, dtype=np.float32)
    def get_state(self):
        laser = self.update_map()
        min_row = max(self.position[0] - self.local_map_size_half, 0)
        max_row = min(self.position[0] + self.local_map_size_half, self.map.shape[0])
        min_col = max(self.position[1] - self.local_map_size_half, 0)
        max_col = min(self.position[1] + self.local_map_size_half, self.map.shape[1])
        local_map_rows = max_row - min_row
        local_map_cols = max_col - min_col
        if local_map_rows != self.local_map_size_half * 2 or local_map_cols != self.local_map_size_half * 2:
            self.local_map = np.zeros((self.local_map_size_half * 2, self.local_map_size_half * 2), dtype=np.uint8)
            self.local_map[:local_map_rows, :local_map_cols] = self.map[min_row:max_row, min_col:max_col]
        else:
            self.local_map = self.map[min_row:max_row, min_col:max_col]
        explore_map = (self.map != self.unknown_pixel) * self.explored_pixel
        explore_rate = explore_map.sum() / (self.grid_num * self.explored_pixel) 
        self.explored_coordinates=[]
        self.explored_coordinates = np.argwhere(explore_map == self.explored_pixel)
        nonzero_index = np.nonzero(explore_map)
        dim0_min = nonzero_index[0].min()
        dim0_max = nonzero_index[0].max()
        dim1_min = nonzero_index[1].min()
        dim1_max = nonzero_index[1].max()
        global_map = explore_map[dim0_min:dim0_max + 1, dim1_min:dim1_max + 1]
        global_map = cv2.resize(global_map, dsize=(self.explore_map_size, self.explore_map_size), interpolation=cv2.INTER_NEAREST) 
        position_0 = int((self.position[0] - dim0_min) * self.explore_map_size / (dim0_max - dim0_min))
        position_1 = int((self.position[1] - dim1_min) * self.explore_map_size / (dim1_max - dim1_min))
        global_map[np.clip(position_0 - 1, 0, self.explore_map_size):np.clip(position_0 + 2, 0, self.explore_map_size),
        np.clip(position_1 - 1, 0, self.explore_map_size):np.clip(position_1 + 2, 0, self.explore_map_size)] = self.agent_pixel
        self.global_map = global_map.astype(np.uint8)
        s_map = np.stack([self.global_map, self.local_map], axis=0) 
        s_sensor = np.concatenate([laser / self.laser_range_max, np.array([self.orientation / 4], dtype=np.float32)])
        s = {"s_map": s_map,
             "s_sensor": s_sensor}
        return s, explore_rate
    def get_info(self):
        return {"explore_rate": self.explore_rate, "position": self.position, 'episode_steps': self.episode_steps}
    def random_init_obstacle(self):
        self.real_map = deepcopy(self.ground_truth_map)
        free_index = np.argwhere(self.real_map == self.free_pixel)
        for _ in range(self.num_obstacles):
            while True:
                obstacle_position = free_index[self.np_random.integers(len(free_index))]
                if (self.real_map[obstacle_position[0] - 1:obstacle_position[0] + 2, obstacle_position[1] - 1:obstacle_position[1] + 2]).sum() == self.free_pixel:
                    self.real_map[obstacle_position[0], obstacle_position[1]] = self.occupied_pixel
                    break
    def random_init_agent(self):
        self.orientation = self.np_random.integers(4)
        free_index = np.argwhere(self.real_map == self.free_pixel)
        agent_position = free_index[self.np_random.integers(len(free_index))]
        self.position[0] = agent_position[0]
        self.position[1] = agent_position[1]
        self.robot_position[0]=self.position[0]
        self.robot_position[1]=self.position[1]
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_steps = 0
        if self.random_obstacle:
            self.random_init_obstacle()
        self.path_data = []
        self.target_position_data = []
        self.explored_coordinates=[]        
        self.random_init_agent()
        self.map = np.ones_like(self.real_map) * self.unknown_pixel
        s, explore_rate = self.get_state()
        self.explore_rate = explore_rate
        info = self.get_info()
        self.count=0
        return s, info
    def searching(self):
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN, (self.f_value(self.s_start), self.s_start))
        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)
            if s == self.s_goal: 
                break
            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)
                if s_n not in self.g:
                    self.g[s_n] = math.inf
                if new_cost < self.g[s_n]:
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))
        return self.extract_path(self.PARENT), self.CLOSED
    def searching_repeated_astar(self, e):
        path, visited = [], []
        while e >= 1:
            p_k, v_k = self.repeated_searching(self.s_start, self.s_goal, e)
            path.append(p_k)
            visited.append(v_k)
            e -= 0.5
        return path, visited
    def repeated_searching(self, s_start, s_goal, e):
        g = {s_start: 0, s_goal: float("inf")}
        PARENT = {s_start: s_start}
        OPEN = []
        CLOSED = []
        heapq.heappush(OPEN, (g[s_start] + e * self.heuristic(s_start), s_start))
        while OPEN:
            _, s = heapq.heappop(OPEN)
            CLOSED.append(s)
            if s == s_goal:
                break
            for s_n in self.get_neighbor(s):
                new_cost = g[s] + self.cost(s, s_n)
                if s_n not in g:
                    g[s_n] = math.inf
                if new_cost < g[s_n]: 
                    g[s_n] = new_cost
                    PARENT[s_n] = s
                    heapq.heappush(OPEN, (g[s_n] + e * self.heuristic(s_n), s_n))
        return self.extract_path(PARENT), CLOSED
    def get_neighbor(self, s):
        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]
    def cost(self, s_start, s_goal):
        if self.is_collision(s_start, s_goal):
            return math.inf
        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])
    def is_collision(self, s_start, s_end):
        if s_start in self.obs or s_end in self.obs:
            return True
        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
            if s1 in self.obs or s2 in self.obs:
                return True
        return False
    def f_value(self, s):
        return self.g[s] + self.heuristic(s)
    def extract_path(self, PARENT):
        path = [self.s_goal]
        s = self.s_goal
        while True:
            try:
                s = PARENT[s]
                path.append(s)
            except:
                s = self.s_start
            if s == self.s_start:
                break
        return list(path)
    def heuristic(self, s):
        heuristic_type = self.heuristic_type 
        goal = self.s_goal 
        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])
    def path_init(self, s_start, s_goal):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = "manhattan"
        self.OPEN = []  
        self.CLOSED = []  
        self.PARENT = dict() 
        self.g = dict()  
        self.u_set = self.move  
        self.obstacle_positions = np.argwhere(self.real_map == self.occupied_pixel)#
        self.obs = set(map(tuple, self.obstacle_positions))
    def quantize(self,number, original_range, new_range):
        interval_size = original_range / new_range
        quantized_number = number / interval_size
        quantized_number=int(quantized_number)
        return quantized_number
    def get_unexplored_positions(self):
        free_positions = self.free_space
        unexplored_positions = [pos for pos in free_positions if self.map[tuple(pos)] == self.unknown_pixel]
        return unexplored_positions
    def step(self, action):
        self.episode_steps += 1
        current_position = tuple(self.position)                         
        unexplored_positions = self.get_unexplored_positions()
        action_new=self.quantize(action,100,self.free_space_num)
        target_position = tuple(self.free_space[action_new])                 
        self.path_init(current_position,target_position)
        self.path, visited = self.searching()
        self.path.reverse()                                                           
        for point in self.path:
            x,y=point
            if self.real_map[self.position[0], self.position[1]] == self.occupied_pixel:
                explore_rate = self.explore_rate  
                s, explore_rate01 = self.get_state()  
                dead=True
            else:
                dead=False
                self.position[0] = x
                self.position[1] = y
                self.path_data.append(point)
                s, explore_rate = self.get_state() 
        if explore_rate > self.explore_rate:  
            r = np.clip((explore_rate ** 2 - self.explore_rate ** 2) * 10, 0, 1.0)
        else:
            r = -0.005
        if dead:
            r = r-0.01
        if explore_rate >= self.max_explore_rate: 
            terminal = True
            r += 1.0
        elif self.episode_steps == self.episode_limit:
            terminal = True
        else:
            terminal = False
        self.explore_rate = explore_rate 
        info = self.get_info()
        print(self.count)
        self.count+=1
        print(explore_rate)
        return s, r, terminal, False, info
    def render(self, mode='human'):
        print(self.explored_coordinates)
        sns.heatmap(self.real_map, cmap='Greys', cbar=False) 
        plt.gca().invert_yaxis()
        plt.show()
        
        
        