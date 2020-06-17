import gym
import numpy as np
import pandas as pd
import math

from simulator import Simulator

class FactoryEnv(gym.Env):
    def __init__(self, is_train):
        self.is_train = is_train
        self.simulator = Simulator()

        self.order_data = pd.read_csv("data/order.csv")
        for i in range(40):
            self.order_data.loc[91+i,:] = ['0000-00-00', 0, 0, 0, 0]        

        self.submission = pd.read_csv("data/sample_submission.csv")
    
        self.work_time = [28, 98] * 17 + [42]
        self.action_plus = [(0.0, 0.0), (5.8, 0.0), (0.0, 5.8), (5.8, 5.8)]

        self.MOL_queue = np.zeros([49, 4])

    def save_csv(self):
        PRTs = self.submission[["PRT_1", "PRT_2", "PRT_3", "PRT_4"]].values
        PRTs = (PRTs[:-1]-PRTs[1:])[24*23:]
        PRTs[-1] = [0., 0., 0., 0.]
        PRTs = np.ceil(PRTs * 1.1)+1
        PAD = np.zeros((24*23+1, 4))
        PRTs = np.append(PRTs, PAD, axis=0).astype(int)
        self.submission.loc[:, "PRT_1":"PRT_4"] = PRTs

        self.submission.to_csv("test.csv", index=False)

    def reset(self):
        self.now_stock = np.array(pd.read_csv("data/stock.csv"), dtype=np.float32)[0]

        self.step_count = 0
        self.work_index = 0
        self.remain_time = 0

        self.line_A_yield = 0.0
        self.line_B_yield = 0.0

        self.line_A_MOL = []
        self.line_B_MOL = []

        state = np.concatenate([[self.step_count], [0]*20, self.now_stock[8:]])/1000000

        return state

    def step1(self, action):
        action_list = [(1, 1), (2, 2), (3, 3), (4, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        
        self.line_A_MOL.append(action_list[action][0])
        self.line_B_MOL.append(action_list[action][1])
        
    def step2(self, action):
        if self.remain_time == 0:
            self.remain_time = self.work_time[self.work_index] - 1
            self.work_index += 1
        else:
            self.remain_time -= 1

        if self.step_count == 552:
            self.line_A_yield = 3.2
            self.line_B_yield = 3.2

        def process():
            self.now_stock[4:8] += self.MOL_queue[0]
            if self.step_count > 551:
                self.MOL_queue[-1][self.line_A_MOL[math.floor((self.work_index-1)/2)]-1] = self.line_A_yield * 0.975
                self.MOL_queue[-1][self.line_B_MOL[math.floor((self.work_index-1)/2)]-1] = self.line_B_yield * 0.975

            self.MOL_queue[:-1] = self.MOL_queue[1:]
            self.MOL_queue[-1] = [0, 0, 0, 0]

            if self.step_count > 551:
                self.now_stock[self.line_A_MOL[math.floor((self.work_index-1)/2)]-1] -= self.line_A_yield
                self.now_stock[self.line_B_MOL[math.floor((self.work_index-1)/2)]-1] -= self.line_B_yield
            
        if self.work_index % 2 == 0:
            self.line_A_yield = self.action_plus[action][0]
            self.line_B_yield = self.action_plus[action][1]

            process()

        self.submission.loc[self.step_count, "PRT_1":"PRT_4"] = self.now_stock[:4]
        
        # done, reward
        if self.step_count == 2183:
            done = True
            score, _ = self.simulator.get_score(self.submission)
            reward = (20000000 - score) / 20000000
            print(f"reward : {reward}")
        else: 
            reward = 0
            done = False

        # write
        if self.work_index % 2 != 0:
            self.submission.loc[self.step_count, "Event_A"] = f"CHECK_{self.line_A_MOL[math.floor((self.work_index-1)/2)]}"
            self.submission.loc[self.step_count, "MOL_A"] = 0.0
            self.submission.loc[self.step_count, "Event_B"] = f"CHECK_{self.line_B_MOL[math.floor((self.work_index-1)/2)]}"
            self.submission.loc[self.step_count, "MOL_B"] = 0.0
        else:
            self.submission.loc[self.step_count, "Event_A"] = "PROCESS"
            self.submission.loc[self.step_count, "Event_B"] = "PROCESS"
            if self.step_count > 551:
                self.submission.loc[self.step_count, "MOL_A"] = round(self.line_A_yield, 1)
                self.submission.loc[self.step_count, "MOL_B"] = round(self.line_B_yield, 1)
            else:
                self.submission.loc[self.step_count, "MOL_A"] = 0.
                self.submission.loc[self.step_count, "MOL_B"] = 0.
 
        # state t+1
        self.step_count += 1
        state = np.concatenate([[self.step_count], np.array(self.order_data.loc[self.step_count//24:(self.step_count//24+4), 'BLK_1':'BLK_4']).reshape(-1), self.now_stock[8:]+400*np.sum(self.MOL_queue, axis=0)+self.now_stock[4:8]*400])/1000000

        info = {}            
        return state, reward, done, info