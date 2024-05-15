import torch


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


def augment_xy_data_by_N_fold(problems, N, depot=None):
    x = problems[:, :, [0]]
    y = problems[:, :, [1]]

    if depot is not None:
        x_depot = depot[:, [0]]
        y_depot = depot[:, [1]]
    idx = torch.rand(N - 1)

    for i in range(N - 1):

        problems = torch.cat((problems, SR_transform(x, y, idx[i])), dim=0)
        if depot is not None:
            depot = torch.cat((depot, SR_transform(x_depot, y_depot, idx[i])), dim=0)

    if depot is not None:
        return problems, depot

    return problems


def SR_transform(x, y, idx):
    if idx < 0.5:
        phi = idx * 4 * math.pi
    else:
        phi = (idx - 0.5) * 4 * math.pi

    x = x - 1 / 2
    y = y - 1 / 2

    x_prime = torch.cos(phi) * x - torch.sin(phi) * y
    y_prime = torch.sin(phi) * x + torch.cos(phi) * y

    if idx < 0.5:
        dat = torch.cat((x_prime + 1 / 2, y_prime + 1 / 2), dim=2)
    else:
        dat = torch.cat((y_prime + 1 / 2, x_prime + 1 / 2), dim=2)
    return dat


def augment(input, N_aug=8):
    if isinstance(input, dict):
        if N_aug == 8:
            if input['loc'].size() == input['depot'].size():
                input['loc'], input['depot'] = augment_xy_data_by_8_fold(input['loc']), augment_xy_data_by_8_fold(input['depot'])
            else:
                input['loc'], input['depot'] = augment_xy_data_by_8_fold(input['loc']), augment_xy_data_by_8_fold(input['depot'].view(-1, 1, 2))
        else:
            if input['loc'].size() == input['depot'].size():
                input['loc'], input['depot'] = input['loc'], input['depot']
            else:
                input['loc'], input['depot'] = input['loc'], input['depot'].view(-1, 1, 2)
    else:
        if N_aug == 8:
            input = augment_xy_data_by_8_fold(input)
        else:
            input = augment_xy_data_by_N_fold(input, N_aug)
    return input
