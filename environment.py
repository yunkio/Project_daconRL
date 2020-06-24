import numpy as np
import pandas as pd
import os
from pathlib import Path

class FactoryEnv:
    def __init__(self):
        order = pd.read_csv("data/order.csv")
        for i in range(40):
            order.loc[91+i,:] = ['0000-00-00', 0, 0, 0, 0]
        self.order = order    
        self.order_dates = list(order[order.iloc[:,1:5].apply(sum, axis=1) != 0].index)
        self.order_dates.append(91) # 리스트 마지막일때 오류 방지
        self.order_stack = np.cumsum(list(map(int, order.iloc[:,1:5].apply(sum, axis=1)))) # 일별 누적 주문량
        self.submission = pd.read_csv("data/sample_submission.csv")
        self.max_count = pd.read_csv('data/max_count.csv')
        self.stock = pd.read_csv("data/stock.csv")
        self.cut_yield = pd.read_csv("data/cut_yield.csv")
        self.blk_dict = {'BLK_1' : 506, 'BLK_2' : 506, 'BLK_3' : 400, 'BLK_4' : 400}
        self.PRT = list()
        
        
        self.queue = np.empty((0,3), float)
        self.total_step = len(self.submission)
        self.step_count = 0
        self.order_month = 202004
        self.prev_score = 1
        self.prev_action = 0
        self.prev_change = 0
        
        self.day_process_n = 0
        
        self.p = 0
        self.q = 0
        self.c_t = 0
        self.c_n = 0
        self.s_t = 0
        self.s_n = 0
        
        self.mask = np.zeros([11], np.bool)
        

        self.process = 0        # 생산 가능 여부, 0 이면 28 시간 검사 필요
        self.change = 0         # change 가능 여부
        self.check = 1          # check 가능 여부
        self.stop = 0           # stop 가능 여부
        
        self.process_mode = 0   # 생산 물품 번호 1~4, stop시 0
        self.process_time = 0   # 생산시간이 얼마나 지속되었는지 검사, PROCESS +1, CHANGE +1, 최대 140
        self.check_time = 28    # 28시간 검사를 완료했는지 검사, CHECK Event시 -1, processtime_time >=98 이면 28
        self.change_time = 0
        self.stop_time = 0
        self.action_map = {0:'CHECK_1', 1:'CHECK_2', 2:'CHECK_3', 3:'CHECK_4',
                          4:'CHANGE_1', 5:'CHANGE_2', 6:'CHANGE_3', 7:'CHANGE_4',
                          8:'STOP',
                          9:'PROCESS_0', 10:'PROCESS_6.66'}
        
#     def update_mask(self):
#         self.mask[:] = False
#         if self.process == 0:
#             if self.check_time == 28:
#                 self.mask[:4] = True
#             if self.check_time < 28:
#                 self.mask[self.process_mode] = True
#         if self.process == 1:
#             self.mask[4] = True
#             if self.process_time > 98:
#                 self.mask[:4] = True
                
#     def save_csv(self):
#         PRTs = self.submission[["PRT_1", "PRT_2", "PRT_3", "PRT_4"]].values
#         PRTs = (PRTs[:-1]-PRTs[1:])[24*23:]
#         PRTs[-1] = [0., 0., 0., 0.]
#         PRTs = np.ceil(PRTs * 1.1)+1
#         PAD = np.zeros((24*23+1, 4))
#         PRTs = np.append(PRTs, PAD, axis=0).astype(int)
#         self.submission.loc[:, "PRT_1":"PRT_4"] = PRTs

#         self.submission.to_csv("test.csv", index=False)    
        
    def reset(self):
        self.step_count = 0
        
    def get_process_mode(self, data):
        if 'CHECK' in data:
            return int(data[-1])
        elif 'CHANGE' in data:
            return int(data[-1])
        else:
            return np.nan         
    
    def cal_queue(self):
    # process마다 queue를 계산하는 함수
        if len(self.queue) > 0:
            self.queue = self.queue - np.array([0,0,1])    
            self.queue,self.stock = self.queue_to_stock(self.queue,self.stock)    

    def queue_to_stock(self, queue, stock):
    # cal_queue 함수의 계산을 도와주는 함수
        if len(queue) > 0:
            if queue[0,2] == 0.0:
                stock.iloc[0, int(queue[0,0])+4] += queue[0,1]
                queue = np.delete(queue, (0), axis=0)
                
                if len(queue) > 0:
                    queue, stock = self.queue_to_stock(queue, stock)    
            return queue, stock
    
    def cal_order(self):
        # stock과 order를 계산해서 처리하고 초과분과 부족분을 return하는 함수
        over = 0
        under = 0
        order_time = (self.step_count // 24) * 24 + 6
        order_day = self.step_count // 24 # order 데이터의 row 번호
        
        
        if order_day in self.order_dates: #납품하는 날일 경우
            
            if self.step_count == order_time: #납품해야하는 시간일 경우
                self.order_month = int(''.join(self.order.loc[order_day, 'time'].split("-")[0:2])) # month
                for i in range(1,5):
                    blk = 'BLK_' + str(i)
                    mol = 'MOL_' + str(i)
                    mol2blk = self.blk_dict[blk]
                    ratio = (self.cut_yield[self.cut_yield['date'] == int(self.order_month)][blk].iloc[0]) / 100 # cut_yield 

                    # 창고에 blk이 더 많으면 stock에서 뺌
                    if self.stock.loc[0,blk] >= self.order.loc[order_day, blk]:
                        self.stock.loc[0,blk] -= int(self.order.loc[order_day, blk])
                        self.order.loc[order_day, blk] = 0

                    # 창고에 blk이 더 적으면
                    elif self.stock.loc[0,blk] < self.order.loc[order_day, blk]:
                        self.order.loc[order_day, blk] -= int(self.stock.loc[0,blk])
                        self.stock.loc[0,blk] = 0
                        need_mol = np.ceil(self.order.loc[order_day, blk] / mol2blk / ratio)
                        # 소지량만큼 order에서 뺀 뒤 필요한 mol 갯수를 계산

                        # mol이 충분히 있을 경우 blk으로 자르고 남은 blk은 창고로
                        if self.stock.loc[0,mol] >= need_mol:
                            self.stock.loc[0,mol] -= int(need_mol)
                            self.stock.loc[0,blk] = int(need_mol*ratio*mol2blk) - int(self.order.loc[order_day, blk])
                            self.order.loc[order_day, blk] = 0

                        # mol이 부족할 경우 일단 있는 mol을 전부 자르고 부족분 처리하고 나머지는 이월
                        elif self.stock.loc[0,mol] < need_mol:
                            self.order.loc[order_day, blk] -= int(self.stock.loc[0,mol] * ratio * mol2blk)
                            self.stock.loc[0,mol] = 0

                            under += int(self.order.loc[order_day, blk])
                            
                            self.order.loc[self.order_dates[self.order_dates.index(order_day) + 1], blk] += int(self.order.loc[order_day, blk])
                            self.order.loc[order_day, blk] = 0

                    # 초과분        
                    over += int(self.stock.loc[0,blk])
                    over += int(self.stock.loc[0,mol] * ratio * mol2blk)
        self.p += over
        self.q += under

#     def get_state(self, prev_action = 0):
#         s = list()
        
#         if prev_action in range(0,4): # Check
#             s_action = 0
#             s_time = self.check_time

#         elif prev_action in range(4,8): # Change
#             s_action = 1
#             s_time = self.change_time

#         elif prev_action == 8: # Stop
#             s_action = 2
#             s_time = self.stop_time

#         elif prev_action > 8: # Process
#             s_action = 3
#             s_time = self.process_time

            
#         order_len = 30 # 30일 후의 주문까지 state로 고려
#         date = self.step_count // 24
#         order_state = list(self.order.loc[date : date+order_len, 'BLK_1':'BLK_4'].values.flatten())
        
#         s.append(s_action)
#         # 0 : Check, 1 : Change, 2 : Stop, 3 : Process (1)
#         s.append(self.process_mode)
#         # 0~3 : 1~4 (1)
#         s.append(s_time)
#         # action에 따른 연속 시간 (1)
#         s += list(self.stock.values[0])
#         # 재고량 (12)
#         s += order_state
#         # 주문량 (124)
        
#         return s

    def get_sum_groupby(self, a, key_pos=0, value_pos=1):
        n  = np.unique(a[:,key_pos])
        nv = np.array( [ np.sum(a[a[:,key_pos]==i,value_pos]) for i in n] )
        r  = np.column_stack((n,nv)) # Stack 1-D arrays as columns into a 2-D array
        return r
    
    
    
    
    def get_state(self, prev_action = 0):
        s = list()
        
        if prev_action in range(0,4): # Check
            s_action = 0
            s_time = self.check_time

        elif prev_action in range(4,8): # Change
            s_action = 1
            s_time = self.change_time

        elif prev_action == 8: # Stop
            s_action = 2
            s_time = self.stop_time

        elif prev_action > 8: # Process
            s_action = 3
            s_time = self.process_time

            
        # 주문량 state 구하기
        order_len = 30 # 30일 후의 주문까지 state로 고려
        date = self.step_count // 24
        order_state = self.order.loc[date : date+order_len, 'BLK_1':'BLK_4'].values
        order_len_list = [3, 15, 30] # 몇 일 후의 order를 볼껀지
        order_state_list = np.empty((0,4), float)
        for i in order_len_list:
            order_state_list = np.vstack([order_state_list, order_state[date:date+i].sum(axis=0)])
            
        # 생산 대기중인 MOL 수량 구하기
        queue_mol = np.array([0,0,0,0])
        sum_queue = self.get_sum_groupby(self.queue)
        for line, mol in sum_queue:
            queue_mol[int(line)] += mol

        # 재고량 구하기 (blk 기준)
        stock_mol = self.stock.values[0][4:8]
        stock_blk = self.stock.values[0][8:]
        stock_state = [0, 0, 0, 0]

        for i in range(4):
            blk = 'BLK_' + str(i+1)
            mol = 'MOL_' + str(i+1)
            mol2blk = self.blk_dict[blk]
            ratio = (self.cut_yield[self.cut_yield['date'] == int(self.order_month)][blk].iloc[0]) / 100 
            stock_state[i] += stock_blk[i]
            stock_state[i] += stock_mol[i] * mol2blk * ratio


        s += [s_action, self.process_mode, self.check_time, self.change_time, self.stop_time, self.process_time, self.check, self.change, self.stop, self.process, self.c_n, self.s_n]
        # (12)
        s += list(queue_mol)
        # 생산 대기량 (4)
        s += list(stock_state)
        # 재고량 (4)
        s += list(order_state_list.flatten())
        # 주문량 (12)
        
        return s

#     def get_reward(self, prev_score):
#         return prev_score - self.get_score()
    
    def get_score(self):
#         N = self.order_stack[self.step_count // 24]
#         M = self.step_count
        f1 = self.score_func(self.p, 32550830)
        # F(p, 10N)
        f2 = self.score_func(self.c_t, 2184) / (1 + 0.1*self.c_n)
        # F(c, M) / (1+0.1 x c_n)
        f3 = self.score_func(self.s_t, 2184) / (1 + 0.1*self.s_n)
        # F(s, M) / (1+0.1 x s_n)
        f4 = self.score_func(self.q, 32550830)
        # F(q, 10N)
        return f1, f2, f3, f4
    
    def score_func(self, x, a):
        if x < a:
            return 1 - (x / a)
        else:
            return 0.0
    
    def step(self, action):
        done = False
        self.step_count += 1
        if self.step_count % 24 == 0:
            self.day_process_n = 0 # 일별 생산량 초기화

        # Check  
        if action in range(0,4):
            if self.check_time == 28:
                self.stop_time = 0
                self.process_time = 0
            self.process_mode = action # 0~3
            self.prev_change = action
            self.check_time -= 1
            self.change = 0
            self.check = 1
            self.stop = 0
            self.process = 0
            
            if self.check_time == 0:
                self.process = 1
                self.check = 0
                self.check_time = 28
                              
        # Change           
        elif action in range(4,8): 
            # 기록하기 (점수)
            if self.change_time == 0:
                self.c_n += 1     
            self.c_t += 1
            
            self.change_time += 1
            self.process_time += 1
            self.change = 1
            self.check = 0
            self.stop = 0
            self.process = 0
            change_done = False
            
            if action in [4,5]: # Change 1~2
                self.process = 0
                if self.process_mode in (0,1):
                    if self.change_time == 6 : # Change 끝났으면
                        change_done = True                        
                elif self.process_mode in (2,3):
                    if self.change_time == 13 :
                        change_done = True                       
            elif action in [6,7]: # Change 3~4
                if self.process_mode in (0,1):
                    if self.change_time == 13:
                        change_done = True
                elif self.process_mode in (2,3):
                    if self.change_time == 6:
                        change_done = True
            
            if change_done == True:
                self.change = 0
                self.process = 1
                self.change_time = 0
                self.process_mode = action - 4
                self.prev_change = self.process_mode
                change_done = False
                        
        # Stop    
        elif action == 8: 
            if self.stop_time == 0:
                self.s_n += 1
                self.process_time = 0
            self.s_t += 1

            self.stop_time += 1
            self.change = 0
            self.check = 0
            self.stop = 1
            self.process = 0
            
            if self.stop_time >= 28:
                self.check = 1              
            if self.stop_time == 192:
                self.stop = 0
        
        # Process
        elif action > 8: 
            if self.process_time == 0:
                self.stop_time = 0
            self.process_time += 1
            self.change = 1
            self.check = 0
            self.stop = 0
            self.process = 1
            
            if self.process_time >= 98:
                self.check = 1
                self.stop = 1

            if self.process_time >= 125:
                self.change = 0

            if self.process_time == 140:
                self.process = 0
            
            if action == 9:
                process_n = 0
            elif action == 10:
                process_n = 6.66
            
            self.day_process_n += process_n # 일별 생산량
            new_queue = [self.process_mode, process_n * 2, 48] # 라인 2개에서 하므로
            self.queue = np.vstack([self.queue, new_queue])
            self.PRT.append(new_queue[0:2] + [self.step_count])

        self.cal_queue()
        self.cal_order()
        
        s_p,s_c,s_s,s_q = self.get_score()

        score = 50 * s_p + 20 * s_c + 10 * s_s + 20 * s_q 
        if (s_p > 0.2) & (s_c > 0.2) & (s_s > 0.2) & (s_q > 0.2) :
            reward = score - self.prev_score
        else :
            reward = -50
            score = 0
            done = True
          
        state = self.get_state(self.prev_action)
        self.prev_score = score
        self.prev_action = action
        
        if self.step_count == 2185:
            done = True
        return state, reward, done