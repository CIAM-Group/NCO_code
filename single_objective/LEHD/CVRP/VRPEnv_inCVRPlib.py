import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from tqdm import tqdm


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    # BATCH_IDX: torch.Tensor
    # POMO_IDX: torch.Tensor
    problems: torch.Tensor
    # shape: (batch, pomo)
    first_node: torch.Tensor
    current_node: torch.Tensor
    # shape: (batch, pomo)
    # ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class VRPEnv:
    def __init__(self, **env_params):
        ####################################
        self.env_params = env_params
        self.problem_size = None

        self.data_path = env_params['data_path']
        self.sub_path = env_params['sub_path']
        ####################################
        self.batch_size = None
        self.problems = None
        self.first_node = None

        self.loc_all=[] # the first node is depot, others are customers. shape (B,V+1,2)
        self.demand_all=[] # the first one is depot's demand, which is set to zero. shape (B,V+1)
        self.capacity_all=[] # shape (B)
        self.cost_all=[] # shape (B)
        self.solution_all=[] # shape (B,V,2)
        self.duration_all =[]
        self.start_capacity=None

        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        self.selected_student_list = None

        self.test_in_vrplib = env_params['test_in_vrplib']
        self.vrplib_path = env_params['vrplib_path']
        self.vrplib_cost = None
        self.vrplib_name = None
        self.vrplib_problems = None
        self.problem_max_min = None
        self.episode = None

        # shape: (batch, pomo, 0~problem)

    def load_problems(self, episode, batch_size, ):
        self.episode = episode
        self.batch_size = batch_size

        if not self.test_in_vrplib:

            self.problems_nodes = self.raw_data_nodes[episode:episode + batch_size]
            # shape (B,V+1,2)
            self.Batch_demand = self.raw_data_demand[episode:episode + batch_size]
            # shape (B,V+1)

            self.Batch_capacity = self.raw_data_capacity[episode:episode + batch_size]

            self.solution = self.raw_data_node_flag[episode:episode + batch_size]
            # shape (B,V,2)
            self.Batch_capacity = self.Batch_capacity[:,None].repeat(1,self.solution.shape[1]+1)
            # shape (B,V+1)

            self.problems = torch.cat((self.problems_nodes,self.Batch_demand[:,:,None],
                                       self.Batch_capacity[:,:,None]),dim=2)
            # shape (B,V+1,4)

            if self.sub_path:
                self.problems, self.solution = self.sampling_subpaths(self.problems, self.solution)

        else:

            self.cvrp_node_coords = self.cvrp_node_coords_all[episode]
            self.cvrp_demands = self.cvrp_demands_all[episode]
            self.cvrp_capacitys=self.cvrp_capacitys_all[episode]
            self.vrplib_cost=self.vrplib_cost_all[episode]
            self.vrplib_name=self.vrplib_name_all[episode]

            problem_nodes = self.cvrp_node_coords
            problem_demands = self.cvrp_demands
            problem_capacitys = self.cvrp_capacitys

            problem_size = len(problem_nodes)

            problem_nodes = np.array(problem_nodes).reshape(1,problem_size,2)
            problem_demands = np.array(problem_demands).reshape(1, problem_size, 1)

            problem_nodes = torch.from_numpy(problem_nodes).cuda().float()

            self.problem_max_min = [torch.max(problem_nodes),torch.min(problem_nodes)]
            problem_nodes = (problem_nodes - self.problem_max_min[1])/(self.problem_max_min[0]-self.problem_max_min[1])

            problem_demands = torch.from_numpy(problem_demands).cuda().float()

            capacity_repeat = torch.tensor([problem_capacitys]).cuda().float().unsqueeze(0).unsqueeze(1).repeat(1,problem_size,1)
            self.raw_data_capacity = capacity_repeat
            self.problems = torch.cat((problem_nodes,problem_demands,capacity_repeat),dim=2)

            self.solution = None

        self.problem_size = self.problems.shape[1]-1



    def vrp_whole_and_solution_subrandom_inverse(self, solution):

        clockwise_or_not = torch.rand(1)[0]

        if clockwise_or_not >= 0.5:
            solution = torch.flip(solution, dims=[1])
            index = torch.arange(solution.shape[1]).roll(shifts=1)
            solution[:, :, 1] = solution[:, index, 1]

        # 1.
        # find the number of subtours in each instance.
        # the total number of subpaths in all instances:     all_subtour_num，
        # The longest length in a subpath among all instances:  max_subtour_length
        batch_size = solution.shape[0]
        problem_size = solution.shape[1]

        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)
        all_subtour_num = torch.sum(visit_depot_num)

        fake_solution = torch.cat((solution[:, :, 1], torch.ones(batch_size)[:, None]), dim=1)

        start_from_depot = fake_solution.nonzero()

        start_from_depot_1 = start_from_depot[:, 1]

        start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)

        sub_tours_length = start_from_depot_2 - start_from_depot_1

        max_subtour_length = torch.max(sub_tours_length)

        # 2。
        # For each subpath, take it out separately, pandding 0 to length max_subtour_length
        #For each instance, padding 0 to max_subtour_num number of subpaths
        # 3.
        # Put all subpaths of all instances into the same array

        start_from_depot2 = solution[:, :, 1].nonzero()
        start_from_depot3 = solution[:, :, 1].roll(shifts=-1, dims=1).nonzero()

        repeat_solutions_node = solution[:, :, 0].repeat_interleave(visit_depot_num, dim=0)
        double_repeat_solution_node = repeat_solutions_node.repeat(1, 2)

        x1 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             >= start_from_depot2[:, 1][:, None]
        x2 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             <= start_from_depot3[:, 1][:, None]

        x3 = (x1 * x2).long()

        sub_tourss = double_repeat_solution_node * x3

        x4 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             < (start_from_depot2[:, 1][:, None] + max_subtour_length)

        x5 = x1 * x4

        sub_tours_padding = sub_tourss[x5].reshape(all_subtour_num, max_subtour_length)

        # 4.
        # For each row, a random number of [0,100] is generated, greater than 50 is positive and less than 50 is inverse

        clockwise_or_not = torch.rand(len(sub_tours_padding))

        clockwise_or_not_bool = clockwise_or_not.le(0.5)

        # 5.
        # For each row, randomly flip

        sub_tours_padding[clockwise_or_not_bool] = torch.flip(sub_tours_padding[clockwise_or_not_bool], dims=[1])

        # 6。
        # Map the subtours to the original solution matrix dimension
        sub_tourss_back = sub_tourss

        sub_tourss_back[x5] = sub_tours_padding.ravel()

        solution_node_flip = sub_tourss_back[sub_tourss_back.gt(0.1)].reshape(batch_size, problem_size)

        solution_flip = torch.cat((solution_node_flip.unsqueeze(2), solution[:, :, 1].unsqueeze(2)), dim=2)

        return solution_flip

    def vrp_whole_and_solution_subrandom_shift_V2inverse(self, solution):
        '''
        For each instance, shift randomly so that different end_with depot nodes can reach the last digit.
        '''

        problem_size = solution.shape[1]
        batch_size = solution.shape[0]

        start_from_depot = solution[:, :, 1].nonzero()
        end_with_depot = start_from_depot.clone()
        end_with_depot[:, 1] = end_with_depot[:, 1] - 1
        end_with_depot[end_with_depot.le(-0.5)] = solution.shape[1] - 1
        end_with_depot[:,1] = torch.roll(end_with_depot[:,1],dims=0,shifts=-1)
        visit_depot_num = solution[:,:,1].sum(1)
        min_length = torch.min(visit_depot_num)

        first_node_index = torch.randint(low=0, high=min_length, size=[1])[0]  # in [0,N)

        temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k=1)
        visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()

        temp_index = np.dot(visit_depot_num_numpy, temp_tri)
        temp_index_torch = torch.from_numpy(temp_index).long().cuda()

        pick_end_with_depot_index = temp_index_torch + first_node_index
        pick_end_with_depot_ = end_with_depot[pick_end_with_depot_index][:,1]
        first_index= pick_end_with_depot_
        end_indeex = pick_end_with_depot_+problem_size

        index = torch.arange(2*problem_size)[None,:].repeat(batch_size,1)
        x1 = index > first_index[:,None]
        x2 = index<= end_indeex[:,None]
        x3 = x1.int()*x2.int()
        double_solution = solution.repeat(1,2,1)
        solution = double_solution[x3.gt(0.5)[:,:,None].repeat(1,1,2)].reshape(batch_size,problem_size,2)

        return solution


    def sampling_subpaths(self, problems, solution, length_fix=False):
        # problems shape (B,V+1,4)
        # solution shape (B,V,2)

        # step：
        # 1.Extract subtour

        problems_size = problems.shape[1] - 1

        batch_size = problems.shape[0]
        embedding_size = problems.shape[2]

        # the first node of subpath: uniform sampling, from 0 to N
        # 1.1
        length_of_subpath = torch.randint(low=4, high=problems_size + 1, size=[1])[0]  # in [4,V]

        solution = self.vrp_whole_and_solution_subrandom_inverse(solution)
        solution = self.vrp_whole_and_solution_subrandom_shift_V2inverse(solution)
        # 1.3
        #  Find the points that start from deopt, and then subtract 1 to get the point that ends with depot

        start_from_depot = solution[:, :, 1].nonzero()

        end_with_depot = start_from_depot
        end_with_depot[:, 1] = end_with_depot[:, 1] - 1
        end_with_depot[end_with_depot.le(-0.5)] = solution.shape[1] - 1

        # 1.4
        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)

        p = torch.rand(len(visit_depot_num))
        select_end_with_depot_node_index = p * visit_depot_num
        select_end_with_depot_node_index = torch.floor(select_end_with_depot_node_index).long()

        temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k=1)
        visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()

        temp_index = np.dot(visit_depot_num_numpy, temp_tri)
        temp_index_torch = torch.from_numpy(temp_index).long().cuda()
        select_end_with_depot_node_index_ = select_end_with_depot_node_index + temp_index_torch

        # This is the point at which each instance is randomly selected with an end with depot
        select_end_with_depot_node = end_with_depot[select_end_with_depot_node_index_, 1]

        # 1.5
        double_solution = torch.cat((solution, solution), dim=1)

        select_end_with_depot_node = select_end_with_depot_node + problems_size

        indexx = torch.arange(length_of_subpath).repeat(batch_size, 1)
        offset = select_end_with_depot_node - length_of_subpath + 1

        indexxxx = indexx + offset[:, None]

        sub_tour = double_solution[:, indexxxx, :]

        sub_tour = sub_tour.view(-1, length_of_subpath, 2)

        index_1 = torch.arange(0, batch_size * batch_size, batch_size)
        index_2 = torch.arange(batch_size)
        index_3 = index_1 + index_2
        sub_solution = sub_tour[index_3, :, :]

        # Calculate the capacity of the first point

        offset_index = problems.shape[0]
        start_index = indexxxx[:,0]

        x1 = torch.arange(double_solution[:offset_index,:,1].shape[1])<=start_index[:offset_index][:,None]

        start_capacity = 0
        before_is_via_depot_all = double_solution[:offset_index,:,1]*x1
        before_is_via_depot = before_is_via_depot_all.nonzero()

        visit_depot_num_2 = torch.sum(before_is_via_depot_all, dim=1)

        select_end_with_depot_node_index_2 = visit_depot_num_2-1

        temp_tri_2 = np.triu(np.ones((len(visit_depot_num_2), len(visit_depot_num_2))), k=1)
        visit_depot_num_numpy_2 = visit_depot_num_2.clone().cpu().numpy()

        temp_index_2 = np.dot(visit_depot_num_numpy_2, temp_tri_2)
        temp_index_torch_2 = torch.from_numpy(temp_index_2).long().cuda()

        select_end_with_depot_node_index_2 = select_end_with_depot_node_index_2 + temp_index_torch_2
        before_is_via_depot_index = before_is_via_depot[select_end_with_depot_node_index_2]

        before_start_index = before_is_via_depot_index[:,1]
        x2 = torch.arange(double_solution[:offset_index, :, 1].shape[1]) <start_index[:offset_index][:, None]
        x3 = torch.arange(double_solution[:offset_index, :, 1].shape[1]) >=before_start_index[:, None]
        x4 = x2 * x3
        double_solution_demand = problems[:offset_index,:,2][torch.arange(offset_index)[:,None].repeat(1,double_solution.shape[1]),double_solution[:offset_index,:,0] ]
        before_demand = double_solution_demand*x4
        self.satisfy_demand = before_demand.sum(1)

        problems[:offset_index,:,3] = problems[:offset_index,:,3] - self.satisfy_demand[:,None]
        # -----------------------------
        # 2. Update the subtour's index
        # -----------------------------

        # 2.1
        sub_solution_node = sub_solution[:, :, 0]

        new_sulution_ascending, rank = torch.sort(sub_solution_node, dim=-1, descending=False)  # 升序
        _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)  # 升序
        sub_solution[:, :, 0] = new_sulution_rank+1

        # 2.2

        index_2, _ = torch.cat((new_sulution_ascending, new_sulution_ascending, new_sulution_ascending, new_sulution_ascending), dim=1). \
            type(torch.long).sort(dim=-1, descending=False)

        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, index_2.shape[1])  # shape: [B, 2current_step]
        temp = torch.arange((embedding_size), dtype=torch.long)[None, :].expand(batch_size, embedding_size)  # shape: [B, current_step]
        index_3 = temp.repeat([1, length_of_subpath])

        new_data = problems[index_1, index_2, index_3].view(batch_size, length_of_subpath, embedding_size)
        new_data = torch.cat((problems[:, 0, :].unsqueeze(dim=1), new_data), dim=1)

        return new_data, sub_solution

    def shuffle_data(self):
        # shuffle the training set data
        index = torch.randperm(len(self.raw_data_nodes)).long()
        self.raw_data_nodes = self.raw_data_nodes[index]
        self.raw_data_capacity = self.raw_data_capacity[index]
        self.raw_data_demand = self.raw_data_demand[index]
        self.raw_data_cost = self.raw_data_cost[index]
        self.raw_data_node_flag = self.raw_data_node_flag[index]


    def load_raw_data(self,episode=1000000):
        def tow_col_nodeflag(node_flag):
            tow_col_node_flag = []
            V = int(len(node_flag) / 2)
            for i in range(V):
                tow_col_node_flag.append([node_flag[i], node_flag[V + i]])
            return tow_col_node_flag

        # Because the dataset is too large, I split it into two reads

        if self.env_params['mode']=='train':

            self.raw_data_nodes_1 = []
            self.raw_data_capacity_1 = []
            self.raw_data_demand_1 = []
            self.raw_data_cost_1 = []
            self.raw_data_node_flag_1 = []
            for line in tqdm(open( self.data_path, "r").readlines()[0:int(0.5 * episode)], ascii=True):
                line = line.split(",")

                depot_index = int(line.index('depot'))
                customer_index = int(line.index('customer'))
                capacity_index = int(line.index('capacity'))
                demand_index = int(line.index('demand'))
                cost_index = int(line.index('cost'))
                node_flag_index = int(line.index('node_flag'))

                depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
                customer = [[float(line[idx]), float(line[idx + 1])] for idx in range(customer_index + 1, capacity_index, 2)]

                loc = depot + customer

                capacity = int(float(line[capacity_index + 1]))
                if int(line[demand_index + 1]) == 0:
                    demand = [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
                else:
                    demand = [0] + [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
                # Include depot's demand, which is 0, in the first

                cost = float(line[cost_index + 1])
                node_flag = [int(line[idx]) for idx in range(node_flag_index + 1, len(line))]
                node_flag = tow_col_nodeflag(node_flag)
                self.raw_data_nodes_1.append(loc)
                self.raw_data_capacity_1.append(capacity)
                self.raw_data_demand_1.append(demand)
                self.raw_data_cost_1.append(cost)
                self.raw_data_node_flag_1.append(node_flag)

            self.raw_data_nodes_1 = torch.tensor(self.raw_data_nodes_1, requires_grad=False)
            # shape (B,V+1,2)  customer num + depot
            self.raw_data_capacity_1 = torch.tensor(self.raw_data_capacity_1, requires_grad=False)
            # shape (B )
            self.raw_data_demand_1 = torch.tensor(self.raw_data_demand_1, requires_grad=False)
            # shape (B,V+1) customer num + depot
            self.raw_data_cost_1 = torch.tensor(self.raw_data_cost_1, requires_grad=False)
            # shape (B )
            self.raw_data_node_flag_1 = torch.tensor(self.raw_data_node_flag_1, requires_grad=False)
            # shape (B,V,2)

            self.raw_data_nodes_2 = []
            self.raw_data_capacity_2 = []
            self.raw_data_demand_2 = []
            self.raw_data_cost_2 = []
            self.raw_data_node_flag_2 = []
            for line in tqdm(open(self.data_path, "r").readlines()[int(0.5 * episode):int(episode)], ascii=True):
                line = line.split(",")
                depot_index = int(line.index('depot'))
                customer_index = int(line.index('customer'))
                capacity_index = int(line.index('capacity'))
                demand_index = int(line.index('demand'))
                cost_index = int(line.index('cost'))
                node_flag_index = int(line.index('node_flag'))

                depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]

                customer = [[float(line[idx]), float(line[idx + 1])] for idx in range(customer_index + 1, capacity_index, 2)]

                loc = depot + customer

                capacity = int(float(line[capacity_index + 1]))

                if int(line[demand_index + 1]) == 0:
                    demand = [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
                else:
                    demand = [0] + [int(line[idx]) for idx in range(demand_index + 1, cost_index)]

                cost = float(line[cost_index + 1])
                node_flag = [int(line[idx]) for idx in range(node_flag_index + 1, len(line))]

                node_flag = tow_col_nodeflag(node_flag)
                self.raw_data_nodes_2.append(loc)
                self.raw_data_capacity_2.append(capacity)
                self.raw_data_demand_2.append(demand)
                self.raw_data_cost_2.append(cost)
                self.raw_data_node_flag_2.append(node_flag)

            self.raw_data_nodes_2 = torch.tensor(self.raw_data_nodes_2, requires_grad=False)
            # shape (B,V+1,2)  customer num + depot
            self.raw_data_capacity_2 = torch.tensor(self.raw_data_capacity_2, requires_grad=False)
            # shape (B )
            self.raw_data_demand_2 = torch.tensor(self.raw_data_demand_2, requires_grad=False)
            # shape (B,V+1) customer num + depot
            self.raw_data_cost_2 = torch.tensor(self.raw_data_cost_2, requires_grad=False)
            # shape (B )
            self.raw_data_node_flag_2 = torch.tensor(self.raw_data_node_flag_2, requires_grad=False)
            # shape (B,V,2)

            self.raw_data_nodes = torch.cat((self.raw_data_nodes_1,self.raw_data_nodes_2),dim=0)
            self.raw_data_capacity = torch.cat((self.raw_data_capacity_1, self.raw_data_capacity_2), dim=0)
            self.raw_data_demand = torch.cat((self.raw_data_demand_1, self.raw_data_demand_2), dim=0)
            self.raw_data_cost = torch.cat((self.raw_data_cost_1, self.raw_data_cost_2), dim=0)
            self.raw_data_node_flag = torch.cat((self.raw_data_node_flag_1, self.raw_data_node_flag_2), dim=0)


        if self.env_params['mode'] == 'test':
            if not self.test_in_vrplib:
                self.raw_data_nodes = []
                self.raw_data_capacity = []
                self.raw_data_demand = []
                self.raw_data_cost = []
                self.raw_data_node_flag = []
                for line in tqdm(open(self.data_path, "r").readlines()[0:episode], ascii=True):
                    line = line.split(",")

                    depot_index = int(line.index('depot'))
                    customer_index = int(line.index('customer'))
                    capacity_index = int(line.index('capacity'))
                    demand_index = int(line.index('demand'))
                    cost_index = int(line.index('cost'))
                    node_flag_index = int(line.index('node_flag'))

                    depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
                    customer = [[float(line[idx]), float(line[idx + 1])] for idx in range(customer_index + 1, capacity_index, 2)]

                    loc = depot + customer
                    capacity = int(float(line[capacity_index + 1]))
                    if int(line[demand_index + 1]) ==0:
                        demand = [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
                    else:
                        demand = [0] + [int(line[idx]) for idx in range(demand_index + 1, cost_index)]

                    cost = float(line[cost_index + 1])
                    node_flag = [int(line[idx]) for idx in range(node_flag_index + 1, len(line))]

                    node_flag = tow_col_nodeflag(node_flag)

                    self.raw_data_nodes.append(loc)
                    self.raw_data_capacity.append(capacity)
                    self.raw_data_demand.append(demand)
                    self.raw_data_cost.append(cost)
                    self.raw_data_node_flag.append(node_flag)

                self.raw_data_nodes = torch.tensor(self.raw_data_nodes, requires_grad=False)
                # shape (B,V+1,2)  customer num + depot
                self.raw_data_capacity = torch.tensor(self.raw_data_capacity, requires_grad=False)
                # shape (B )
                self.raw_data_demand = torch.tensor(self.raw_data_demand, requires_grad=False)
                # shape (B,V+1) customer num + depot
                self.raw_data_cost = torch.tensor(self.raw_data_cost, requires_grad=False)
                # shape (B )
                self.raw_data_node_flag = torch.tensor(self.raw_data_node_flag, requires_grad=False)
                # shape (B,V,2)
            else:
                self.cvrp_node_coords_all, self.cvrp_demands_all, self.cvrp_capacitys_all, \
                self.vrplib_cost_all, self.vrplib_name_all = self.make_vrplib_data(self.vrplib_path, episode)


        print(f'load raw dataset done!', )

    def make_vrplib_data(self, filename,episode):

        node_coords = []
        demands = []
        capacitys = []
        costs = []
        names = []

        for line in tqdm(open(filename, "r").readlines(), ascii=True):
            line = line.split(", ")

            name_index = int(line.index('[\'name\''))
            depot_index = int(line.index('\'depot\''))
            customer_index = int(line.index('\'customer\''))
            capacity_index = int(line.index('\'capacity\''))
            demand_index = int(line.index('\'demand\''))
            cost_index = int(line.index('\'cost\''))

            depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
            customer = [[float(line[idx]), float(line[idx + 1])] for idx in
                        range(customer_index + 1, demand_index, 2)]

            loc = depot + customer

            capacity = int(float(line[capacity_index + 1]))
            demand = [int(line[idx]) for idx in range(demand_index + 1, capacity_index)]
            cost = float(line[cost_index + 1])

            node_coords.append(loc)
            demands.append(demand)
            capacitys.append(capacity)
            costs.append(cost)
            names.append(line[name_index+1][1:-1])

        # Each row of data represents an instance, and the size of each instance is different
        node_coords = np.array(node_coords)
        demands = np.array(demands)
        capacitys = np.array(capacitys)
        costs = np.array(costs)
        names = np.array(names)

        return node_coords, demands, capacitys, costs, names



    def reset(self, mode, sample_size = 1):
        self.selected_count = 0

        if mode == 'train':
            self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
            self.selected_teacher_flag = torch.zeros((self.batch_size, 0), dtype=torch.long)
            self.selected_student_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
            self.selected_student_flag= torch.zeros((self.batch_size, 0), dtype=torch.long)
            self.current_node=self.problems[:,0,:]
            self.first_node = self.current_node
            self.step_state = Step_State(problems=self.problems, first_node=self.first_node[:, None, :],
                                         current_node=self.current_node[:, None, :])
        if mode == 'test':

            self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
            self.selected_teacher_flag = torch.zeros((self.batch_size, 0), dtype=torch.long)
            self.selected_student_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
            self.selected_student_flag = torch.zeros((self.batch_size, 0), dtype=torch.long)
            self.current_node = self.problems[:, 0, :]
            self.first_node = self.current_node
            self.step_state = Step_State(problems=self.problems, first_node=self.first_node[:, None, :],
                                         current_node=self.current_node[:, None, :])

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        reward_student = None
        done = False
        return self.step_state, reward, reward_student, done

    def step(self, selected, selected_student,selected_flag_teacher,selected_flag_student,use_bs=False,
             beam_select_list=None,beam_select_flag=None, beam_width=None,select_new_beam_index=None):

        self.selected_count += 1

        gather_index = selected[:, None, None].expand((len(selected), 1, 4)) # shape [B,1,4]

        # --------------------
        #---------------------
        # Update capacity
        # 1. If flag = 1, the vehicle returns to depot and capacity is refilled
        is_depot = selected_flag_teacher==1
        self.problems[is_depot, :, 3] =  self.raw_data_capacity.ravel()[0].item()

        # 2. If capacity is less than demand, capacity is also refilled and the flag of the current access node is changed to 1

        self.current_node_temp = self.problems.gather(index=gather_index, dim=1).squeeze(1)
        demands = self.current_node_temp[:,2]
        smaller_ = self.problems[:, 0, 3] < demands

        selected_flag_teacher[smaller_] = 1
        self.problems[smaller_, :, 3] =  self.raw_data_capacity.ravel()[0].item()
        # print('----------------')
        # 3. Subtract the demand of the currently visited node regardless of whether the vehicle is returned to depot to refill

        self.problems[:,:,3] =  self.problems[:,:,3]- demands[:,None]

        # --------------------
        #---------------------

        self.current_node = self.problems.gather(index=gather_index, dim=1).squeeze(1)
        # shape [B,4]


        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, None]), dim=1)

        self.selected_teacher_flag = torch.cat((self.selected_teacher_flag, selected_flag_teacher[:, None]), dim=1)

        self.selected_student_list = torch.cat((self.selected_student_list, selected_student[:, None]), dim=1)

        self.selected_student_flag = torch.cat((self.selected_student_flag, selected_flag_student[:, None]), dim=1)

        self.step_state.current_node = self.current_node[:, None, :]

        self.step_state.first_node[:, 0, 2]= 0
        self.step_state.current_node[:, 0, 2] = 0
        self.first_node[:, 2] = 0
        self.current_node[:, 2] = 0
        self.step_state.first_node[:, 0, 3] = self.problems[:,1,3].clone()
        self.step_state.current_node[:, 0, 3] = self.problems[:, 1, 3].clone()
        self.first_node[:, 3] = self.problems[:, 1, 3].clone()
        self.current_node[:, 3] = self.problems[:, 1, 3].clone()
        # returning values
        done = (self.selected_count == self.problems.shape[1]-1)
        if done:
            reward, reward_student = self._get_travel_distance()  # note the minus sign!
        else:
            reward, reward_student = None, None

        return self.step_state, reward, reward_student, done

    def make_dir(self,path_destination):
        isExists = os.path.exists(path_destination)
        if not isExists:
            os.makedirs(path_destination)
        return

    def cal_length(self, problems, order_node, order_flag):
        # problems:   [B,V+1,2]
        # order_node: [B,V]
        # order_flag: [B,V]
        order_node_ = order_node.clone()

        order_flag_ = order_flag.clone()

        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)

        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0

        roll_node = order_node_.roll(dims=1, shifts=1)

        problem_size = problems.shape[1] - 1

        order_gathering_index = order_node_.unsqueeze(2).expand(-1, problem_size, 2)
        order_loc = problems.gather(dim=1, index=order_gathering_index)

        roll_gathering_index = roll_node.unsqueeze(2).expand(-1, problem_size, 2)
        roll_loc = problems.gather(dim=1, index=roll_gathering_index)

        flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, problem_size, 2)
        flag_loc = problems.gather(dim=1, index=flag_gathering_index)

        order_lengths = ((order_loc - flag_loc) ** 2)

        order_flag_[:,0]=0
        flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, problem_size, 2)
        flag_loc = problems.gather(dim=1, index=flag_gathering_index)

        roll_lengths = ((roll_loc - flag_loc) ** 2)

        length = (order_lengths.sum(2).sqrt() + roll_lengths.sum(2).sqrt()).sum(1)

        return length

    def _get_travel_distance(self):

        # optimal distance
        if self.test_in_vrplib:
            travel_distances = self.vrplib_cost
            self.problems[:,:,:2] =  self.problems[:,:,:2] * (self.problem_max_min[0] - self.problem_max_min[1]) + self.problem_max_min[1]
        else:
            # teacher length
            problems = self.problems[:,:,[0,1]]
            order_node = self.solution[:,:,0]
            order_flag = self.solution[:,:,1]
            travel_distances = self.cal_length( problems, order_node, order_flag)
            # self.drawPic_VRP(problems[0,:,:], order_node[0],order_flag[0],name='teather')

        # trained model's distance
        problems = self.problems[:, :, [0, 1]]
        order_node = self.selected_student_list.clone()
        order_flag = self.selected_student_flag.clone()

        travel_distances_student = self.cal_length(problems, order_node, order_flag)

        # draw figure， validate the result.
        # self.drawPic_VRP(problems[0,:,:], order_node[0],order_flag[0],name='student')

        return -travel_distances, -travel_distances_student

    def _get_travel_distance_2(self, problems_, solution_,need_optimal =False ):

        if self.test_in_vrplib:
            if need_optimal:
                return self.vrplib_cost, self.vrplib_name
            else:
                problems = problems_[:, :, [0, 1]].clone() * (self.problem_max_min[0] - self.problem_max_min[1]) + self.problem_max_min[1]
                order_node = solution_[:, :, 0].clone()
                order_flag = solution_[:, :, 1].clone()
                travel_distances = self.cal_length(problems, order_node, order_flag)
        else:
            problems = problems_[:, :, [0, 1]].clone()
            order_node = solution_[:, :, 0].clone()
            order_flag = solution_[:, :, 1].clone()
            travel_distances = self.cal_length(problems, order_node, order_flag)

        return travel_distances

    def destroy_solution(self, problem, complete_solution):


        self.problems, self.solution, first_node_index,length_of_subpath,double_solution = self.sampling_subpaths_repair(
            problem, complete_solution, mode=self.env_params['mode'])


        partial_solution_length = self._get_travel_distance_2(self.problems, self.solution,
                                                              need_optimal=False)

        return partial_solution_length,first_node_index,length_of_subpath,double_solution


    def sampling_subpaths_repair(self, problems, solution, length_fix=False, mode='test', repair=True):
        # problems shape (B,V+1,4)
        # solution shape (B,V,2) index从1开始

        problems_size = problems.shape[1] - 1
        # print('problems_size',problems_size)
        batch_size = problems.shape[0]
        embedding_size = problems.shape[2]

        # the first node of subpath: uniform sampling, from 0 to N
        # 1.1

        length_of_subpath = torch.randint(low=4, high=problems_size+1 , size=[1])[0]  # in [4,N]

        start_from_depot = solution[:, :, 1].nonzero()

        end_with_depot = start_from_depot
        end_with_depot[:, 1] = end_with_depot[:, 1] - 1
        end_with_depot[end_with_depot.le(-0.5)] = solution.shape[1] - 1

        # 1.4
        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)

        p = torch.rand(len(visit_depot_num))
        select_end_with_depot_node_index = p * visit_depot_num
        select_end_with_depot_node_index = torch.floor(select_end_with_depot_node_index).long()

        temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k=1)
        visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()

        temp_index = np.dot(visit_depot_num_numpy, temp_tri)
        temp_index_torch = torch.from_numpy(temp_index).long().cuda()

        select_end_with_depot_node_index_ = select_end_with_depot_node_index + temp_index_torch

        select_end_with_depot_node = end_with_depot[select_end_with_depot_node_index_, 1]
        # 1.5
        double_solution = torch.cat((solution, solution), dim=1)

        select_end_with_depot_node = select_end_with_depot_node + problems_size

        indexx = torch.arange(length_of_subpath).repeat(batch_size, 1)
        offset = select_end_with_depot_node - length_of_subpath + 1

        indexxxx = indexx + offset[:, None]


        sub_solu_index1 = torch.arange(batch_size)[:,None].repeat(1,2*length_of_subpath)
        sub_solu_index2 =indexxxx.repeat_interleave(2,dim=1)
        sub_solu_index3 = torch.arange(double_solution.shape[2])[None,:].repeat(batch_size,length_of_subpath)
        sub_solution = double_solution[sub_solu_index1,sub_solu_index2,sub_solu_index3].reshape(batch_size,length_of_subpath,2)

        offset_index = problems.shape[0]
        start_index = indexxxx[:, 0]


        x1 = torch.arange(double_solution[:offset_index, :, 1].shape[1]) <= start_index[:offset_index][:, None]

        start_capacity = 0
        before_is_via_depot_all = double_solution[:offset_index, :, 1] * x1
        before_is_via_depot = before_is_via_depot_all.nonzero()

        visit_depot_num_2 = torch.sum(before_is_via_depot_all, dim=1)

        select_end_with_depot_node_index_2 = visit_depot_num_2 - 1

        temp_tri_2 = np.triu(np.ones((len(visit_depot_num_2), len(visit_depot_num_2))), k=1)
        visit_depot_num_numpy_2 = visit_depot_num_2.clone().cpu().numpy()

        temp_index_2 = np.dot(visit_depot_num_numpy_2, temp_tri_2)
        temp_index_torch_2 = torch.from_numpy(temp_index_2).long().cuda()

        select_end_with_depot_node_index_2 = select_end_with_depot_node_index_2 + temp_index_torch_2
        before_is_via_depot_index = before_is_via_depot[select_end_with_depot_node_index_2]

        before_start_index = before_is_via_depot_index[:, 1]
        x2 = torch.arange(double_solution[:offset_index, :, 1].shape[1]) < start_index[:offset_index][:, None]
        x3 = torch.arange(double_solution[:offset_index, :, 1].shape[1]) >= before_start_index[:, None]
        x4 = x2 * x3
        double_solution_demand = problems[:offset_index, :, 2][
            torch.arange(offset_index)[:, None].repeat(1, double_solution.shape[1]), double_solution[:offset_index, :, 0]]

        before_demand = double_solution_demand * x4

        self.satisfy_demand = before_demand.sum(1)

        problems[:offset_index, :, 3] = problems[:offset_index, :, 3] - self.satisfy_demand[:, None]

        # -----------------------------
        # 2.
        # -----------------------------
        # 2.1
        sub_solution_node = sub_solution[:, :, 0]
        new_sulution_ascending, rank = torch.sort(sub_solution_node, dim=-1, descending=False)  # 升序
        _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)  # 升序
        sub_solution[:, :, 0] = new_sulution_rank + 1
        # 2.2
        index_2, _ = torch.cat((new_sulution_ascending, new_sulution_ascending, new_sulution_ascending, new_sulution_ascending), dim=1). \
            type(torch.long).sort(dim=-1, descending=False)

        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, index_2.shape[1])  # shape: [B, 2current_step]
        temp = torch.arange((embedding_size), dtype=torch.long)[None, :].expand(batch_size, embedding_size)  # shape: [B, current_step]
        index_3 = temp.repeat([1, length_of_subpath])

        new_data = problems[index_1, index_2, index_3].view(batch_size, length_of_subpath, embedding_size)
        new_data = torch.cat((problems[:, 0, :].unsqueeze(dim=1), new_data), dim=1)
        if repair == True:
            return new_data, sub_solution,start_index,length_of_subpath,double_solution
        else:
            return new_data, sub_solution


    def valida_solution_legal(self, problem, solution,capacity_=50):

        capacitys = {100: 50,
                     200: 80,
                     500: 100,
                     1000: 250}

        problem_size = solution.shape[1]
        capacity = capacitys[problem_size]
        # print('capacity',capacity)
        coor = problem[:, :, [0, 1]]
        demand = problem[:, :, 2]

        order_node = solution[:, :, 0].clone()
        order_flag = solution[:, :, 1].clone()



        if_begin_flag_legal = (order_flag[:,0]!=1).any()

        # 0.
        if if_begin_flag_legal:
            assert False, 'e1: wrong begin_flag_legal!'

        # 1. Determine whether each index of the solution node list is unique
        uniques = torch.unique(order_node[0])
        if len(uniques) != problem.shape[1] - 1:
            assert False, 'e2: wrong node list!'


        # 2. Find the demand for each sub tour and determine whether it exceeds capacity

        batch_size = solution.shape[0]


        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)

        all_subtour_num = torch.sum(visit_depot_num)

        fake_solution = torch.cat((solution[:, :, 1], torch.ones(batch_size)[:, None]), dim=1)

        start_from_depot = fake_solution.nonzero()

        start_from_depot_1 = start_from_depot[:, 1]

        start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)

        sub_tours_length = start_from_depot_2 - start_from_depot_1

        max_subtour_length = torch.max(sub_tours_length)

        # 2。


        start_from_depot2 = solution[:, :, 1].nonzero()
        start_from_depot3 = solution[:, :, 1].roll(shifts=-1, dims=1).nonzero()

        repeat_solutions_node = solution[:, :, 0].repeat_interleave(visit_depot_num, dim=0)
        double_repeat_solution_node = repeat_solutions_node.repeat(1, 2)

        x1 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             >= start_from_depot2[:, 1][:, None]
        x2 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             <= start_from_depot3[:, 1][:, None]

        x3 = (x1 * x2).long()

        sub_tourss = double_repeat_solution_node * x3

        x4 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             < (start_from_depot2[:, 1][:, None] + max_subtour_length)

        x5 = x1 * x4

        sub_tours_padding = sub_tourss[x5].reshape(all_subtour_num, max_subtour_length)

        ########################----------
        ########################----------

        demands = torch.repeat_interleave(demand, repeats=visit_depot_num, dim=0)

        index = torch.arange(sub_tours_padding.shape[0])[:, None].repeat(1, sub_tours_padding.shape[1])
        sub_tours_demands = demands[index, sub_tours_padding].sum(dim=1)
        if_legal = (sub_tours_demands > capacity)

        if if_legal.any():
            assert False, 'e3: wrong capacity!'

        return