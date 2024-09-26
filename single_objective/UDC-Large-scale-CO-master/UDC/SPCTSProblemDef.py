import torch
import numpy as np

K_n = {
    20: 2,
    50: 3,
    100: 4,
    500: 9,
    1000: 12,
    5000: 20,
    10000: 38
}


def get_random_problems(batch_size, problem_size, coords=None, test=False):
    if coords is not None:
        problems = coords
    else:
        problems = torch.rand(size=(batch_size, problem_size, 2))

    prizes = torch.rand(size=(batch_size, problem_size - 1)) * 4 / problem_size
    prize = torch.cat((torch.zeros((batch_size, 1)), prizes), dim=1)
    stochastic_prize = torch.rand(batch_size, problem_size) * prize * 2
    if test == True:
        K = K_n[problem_size]
    else:
        K = np.random.randint(9, 12 + 1)  # one scalar
    beta = torch.rand(size=(batch_size, problem_size - 1)) * 3 * K / problem_size
    c = torch.cat((torch.zeros((batch_size, 1)), beta), dim=1)  # (n+1,)
    # problems.shape: (batch, problem, 2)
    problems = torch.cat((problems, stochastic_prize.unsqueeze(-1), prize.unsqueeze(-1), c.unsqueeze(-1)), dim=2)
    return problems


def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems
