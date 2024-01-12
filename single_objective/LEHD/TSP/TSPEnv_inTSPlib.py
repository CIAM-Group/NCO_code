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
    data: torch.Tensor
    first_node: torch.Tensor
    current_node: torch.Tensor



class TSPEnv:
    def __init__(self, **env_params):

        ####################################
        self.env_params = env_params
        self.problem_size = None
        self.data_path = env_params['data_path']
        self.sub_path = env_params['sub_path']
        ####################################
        self.batch_size = None
        self.problems = None  # shape: [B,V,2]
        self.first_node = None  # shape: [B,V]

        self.raw_data_nodes = []
        self.raw_data_tours = []

        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        self.selected_student_list = None

        self.test_in_tsplib = env_params['test_in_tsplib']
        self.tsplib_path = env_params['tsplib_path']
        self.tsplib_cost = None
        self.tsplib_name = None
        self.tsplib_problems = None
        self.problem_max_min = None
        self.episode = None

    def load_problems(self, episode, batch_size):
        self.episode = episode

        self.batch_size = batch_size

        if not self.test_in_tsplib:
            self.problems, self.solution = self.raw_data_nodes[episode:episode + batch_size], self.raw_data_tours[episode:episode + batch_size]
            # shape: [B,V,2]  ;  shape: [B,V]

            if self.sub_path:
                self.problems, self.solution = self.sampling_subpaths(self.problems, self.solution,mode='train')

            if_inverse = True
            if_inverse_index = torch.randint(low=0, high=100, size=[1])[0]  # in [4,N]
            if if_inverse_index < 50:
                if_inverse = False

            if if_inverse:
                self.solution = torch.flip( self.solution , dims=[1])


        else:
            self.tsplib_problems, self.tsplib_cost, self.tsplib_name = self.make_tsplib_data(self.tsplib_path,episode)
            self.tsplib_cost = torch.tensor(self.tsplib_cost)
            self.problems = torch.from_numpy(self.tsplib_problems.reshape(1,-1,2)).cuda().float()
            self.problem_max_min = [torch.max(self.problems),torch.min(self.problems)]
            self.problems = (self.problems - self.problem_max_min[1])/(self.problem_max_min[0]-self.problem_max_min[1])
            self.solution = None
        self.problem_size = self.problems.shape[1]



    def sampling_subpaths(self, problems, solution, length_fix=False, mode='test', repair=False):

        problems_size = problems.shape[1]
        batch_size = problems.shape[0]
        embedding_size = problems.shape[2]

        first_node_index = torch.randint(low=0, high=problems_size, size=[1])[0]  # in [0,N)

        # length of subpath: uniform sampling, from 4 to N
        if mode == 'test':

            length_of_subpath = torch.randint(low=4, high=problems_size + 1, size=[1])[0]  # in [4,N]
        else:
            if length_fix:
                length_of_subpath = problems_size
            else:
                length_of_subpath = torch.randint(low=4, high=problems_size + 1, size=[1])[0]  # in [4,N]
                # length_of_subpath = problems_size

        # -----------------------------
        # new_sulution
        # -----------------------------
        double_solution = torch.cat([solution, solution], dim=-1)
        new_sulution = double_solution[:, first_node_index: first_node_index + length_of_subpath]
        new_sulution_ascending, rank = torch.sort(new_sulution, dim=-1, descending=False)  # 升序
        _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)  # 升序

        # -----------------------------
        # new_problems
        # -----------------------------
        index_2, _ = torch.cat((new_sulution_ascending, new_sulution_ascending), dim=1).type(torch.long).sort(dim=-1,
                                                                                                              descending=False)  # shape: [B, 2current_step]
        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, index_2.shape[1])  # shape: [B, 2current_step]
        temp = torch.arange((embedding_size), dtype=torch.long)[None, :].expand(batch_size, embedding_size)  # shape: [B, current_step]
        index_3 = temp.repeat([1, length_of_subpath])

        new_data = problems[index_1, index_2, index_3].view(batch_size, length_of_subpath, 2)

        if repair == True:
            return new_data, new_sulution_rank, first_node_index, length_of_subpath, double_solution
        else:
            return new_data, new_sulution_rank


    def shuffle_data(self):
        # 打乱训练集数据
        index = torch.randperm(len(self.raw_data_nodes)).long()
        self.raw_data_nodes = self.raw_data_nodes[index]
        self.raw_data_tours = self.raw_data_tours[index]


    def load_raw_data(self, episode,begin_index=0):

        print('load raw dataset begin!')

        self.raw_data_nodes = []
        self.raw_data_tours = []
        for line in tqdm(open(self.data_path, "r").readlines()[0+begin_index:episode+begin_index], ascii=True):
            line = line.split(" ")
            num_nodes = int(line.index('output') // 2)
            nodes = [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]

            self.raw_data_nodes.append(nodes)
            tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]#[:-1]

            self.raw_data_tours.append(tour_nodes)

        self.raw_data_nodes = torch.tensor(self.raw_data_nodes,requires_grad=False)
        self.raw_data_tours = torch.tensor(self.raw_data_tours,requires_grad=False)
        print(f'load raw dataset done!', )  # 读1024个 TSP100 instance 用时 4s

    def make_dataset(self, filename, episode, batch_size, num_samples):
        nodes_coords = []
        tour = []
        for line in tqdm(open(filename, "r").readlines()[episode:episode + batch_size], ascii=True):
            line = line.split(" ")
            num_nodes = int(line.index('output') // 2)
            nodes_coords.append(
                [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]
            )

            tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]#[:-1]
            tour.append(tour_nodes)

        nodes_coords = torch.tensor(nodes_coords)
        tour = torch.tensor(tour)
        return nodes_coords, tour

    def make_tsplib_data(self, filename,episode):
        instance_data = []
        cost = []
        instance_name = []
        for line in open(filename, "r").readlines()[episode:episode+1]:
            line = line.rstrip("\n")
            line = line.replace('[', '')
            line = line.replace(']', '')
            line = line.replace('\'', '')
            line = line.split(sep=',')

            line_data = np.array(line[2:], dtype=float).reshape(-1, 2)
            instance_data.append(line_data)
            cost.append(np.array(line[1], dtype=float))
            instance_name.append(np.array(line[0], dtype=str))
        instance_data = np.array(instance_data)  # 每一行的数据表示一个instance，每一个instance的size不一样
        cost = np.array(cost)
        instance_name = np.array(instance_name)
        # print(instance_data.shape)

        return instance_data, cost, instance_name


    def destroy_solution(self, problem, complete_solution):


        self.problems, self.solution,first_node_index,length_of_subpath,double_solution = self.sampling_subpaths(
            problem, complete_solution, mode=self.env_params['mode'],repair=True)

        partial_solution_length = self._get_travel_distance_2(self.problems, self.solution,
                                                                      need_optimal=False)
        return partial_solution_length,first_node_index,length_of_subpath,double_solution


    def reset(self, mode,):
        self.selected_count = 0

        # shape: (batch, pomo)
        if mode == 'train':
            self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
            self.selected_student_list = torch.zeros((self.batch_size, 0), dtype=torch.long)

        if mode == 'test':

            self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long)  # [B*(V-1),0]
            self.selected_student_list = torch.zeros((self.batch_size, 0), dtype=torch.long)  # [B*(V-1),0]

        self.step_state = Step_State(data=self.problems, first_node=self.first_node,
                                     current_node=self.current_node)

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        reward_student = None
        done = False
        return self.step_state, reward, reward_student, done

    def step(self, selected, selected_student):

        self.selected_count += 1

        gather_index = selected[:, None, None].expand((len(selected), 1, 2))

        self.current_node = self.problems.gather(index=gather_index, dim=1).squeeze(1)

        if self.selected_count == 1:
            self.first_node = self.current_node

        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, None]), dim=1)  # shape: [B, current_step]

        self.selected_student_list = torch.cat((self.selected_student_list, selected_student[:, None]), dim=1)

        self.step_state.current_node = self.current_node[:, None, :]
        if self.selected_count == 1:
            self.step_state.first_node = self.step_state.current_node

        done = (self.selected_count == self.problems.shape[1])
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



    def _get_travel_distance(self):

        if self.test_in_tsplib:
            travel_distances = self.tsplib_cost
            self.problems =  self.problems * (self.problem_max_min[0] - self.problem_max_min[1]) + self.problem_max_min[1]

        else:

            gathering_index = self.solution.unsqueeze(2).expand(self.batch_size, self.problems.shape[1], 2)

            seq_expanded = self.problems

            ordered_seq = seq_expanded.gather(dim=1, index=gathering_index)

            rolled_seq = ordered_seq.roll(dims=1, shifts=-1)

            segment_lengths = ((ordered_seq - rolled_seq) ** 2)

            segment_lengths = segment_lengths.sum(2).sqrt()

            travel_distances = segment_lengths.sum(1)


        # trained model's distance
        gathering_index_student = self.selected_student_list.unsqueeze(2).expand(-1, self.problems.shape[1], 2)
        ordered_seq_student = self.problems.gather(dim=1, index=gathering_index_student)
        rolled_seq_student = ordered_seq_student.roll(dims=1, shifts=-1)
        segment_lengths_student = ((ordered_seq_student - rolled_seq_student) ** 2)
        segment_lengths_student = segment_lengths_student.sum(2).sqrt()
        # shape: (batch,problem)
        travel_distances_student = segment_lengths_student.sum(1)
        # shape: (batch)
        return travel_distances, travel_distances_student


    def _get_travel_distance_2(self, problems, solution, need_optimal = False):
        if self.test_in_tsplib:
            if need_optimal:
                return self.tsplib_cost,self.tsplib_name
            else:
                problems_copy = problems.clone().detach() * (self.problem_max_min[0] - self.problem_max_min[1]) + \
                                self.problem_max_min[1]

                gathering_index = solution.unsqueeze(2).expand(problems_copy.shape[0], problems_copy.shape[1], 2)

                seq_expanded = problems_copy

                ordered_seq = seq_expanded.gather(dim=1, index=gathering_index)

                rolled_seq = ordered_seq.roll(dims=1, shifts=-1)

                segment_lengths = ((ordered_seq - rolled_seq) ** 2)

                segment_lengths = segment_lengths.sum(2).sqrt()

                travel_distances = segment_lengths.sum(1)

                return travel_distances
        else:
            gathering_index = solution.unsqueeze(2).expand(problems.shape[0], problems.shape[1], 2)

            seq_expanded = problems

            ordered_seq = seq_expanded.gather(dim=1, index=gathering_index)

            rolled_seq = ordered_seq.roll(dims=1, shifts=-1)

            segment_lengths = ((ordered_seq - rolled_seq) ** 2)

            segment_lengths = segment_lengths.sum(2).sqrt()

            travel_distances = segment_lengths.sum(1)

        return travel_distances