import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import math


class EmbeddingNet(nn.Module):

    def __init__(
            self,
            embedding_dim,
            seq_length,
    ):
        super(EmbeddingNet, self).__init__()
        self.embedding_dim = embedding_dim

        # Two ways for generalizing CPEs:
        # -- 1. Use the target size CPE directly (default)
        self.pattern = self.Cyclic_Positional_Encoding(seq_length, embedding_dim)
        # -- 2. Use the original size CPE: reuse the wavelength of the original size but make it compatible with the target size (by duplicating or discarding)
        # way 1 works well for most of cases
        # original_size, target_size = 100, 150
        # self.pattern = self.Cyclic_Positional_Encoding(original_size, embedding_dim, target_size=target_size)

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def basesin(self, x, T, fai=0):
        return np.sin(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)

    def basecos(self, x, T, fai=0):
        return np.cos(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)

    # implements the CPE
    def Cyclic_Positional_Encoding(self, n_position, emb_dim, mean_pooling=True, target_size=None):

        Td_set = np.linspace(np.power(n_position, 1 / (emb_dim // 2)), n_position, emb_dim // 2, dtype='int')
        x = np.zeros((n_position, emb_dim))

        for i in range(emb_dim):
            Td = Td_set[i // 3 * 3 + 1] if (i // 3 * 3 + 1) < (emb_dim // 2) else Td_set[-1]
            fai = 0 if i <= (emb_dim // 2) else 2 * np.pi * ((-i + (emb_dim // 2)) / (emb_dim // 2))
            longer_pattern = np.arange(0, np.ceil((n_position) / Td) * Td, 0.01)
            if i % 2 == 1:
                x[:, i] = self.basecos(longer_pattern, Td, fai)[np.linspace(0, len(longer_pattern), n_position, dtype='int', endpoint=False)]
            else:
                x[:, i] = self.basesin(longer_pattern, Td, fai)[np.linspace(0, len(longer_pattern), n_position, dtype='int', endpoint=False)]

        pattern = torch.from_numpy(x).type(torch.FloatTensor)
        pattern_sum = torch.zeros_like(pattern)

        # for generalization (way 2): reuse the wavelength of the original size but make it compatible with the target size (by duplicating or discarding)
        if target_size is not None:
            pattern = pattern[np.ceil(np.linspace(0, n_position - 1, target_size))]
            pattern_sum = torch.zeros_like(pattern)
            n_position = target_size

        # averaging the adjacient embeddings if needed (optional, almost the same performance)
        arange = torch.arange(n_position)
        pooling = [0] if not mean_pooling else [-2, -1, 0, 1, 2]
        time = 0
        for i in pooling:
            time += 1
            index = (arange + i + n_position) % n_position
            pattern_sum += pattern.gather(0, index.view(-1, 1).expand_as(pattern))
        pattern = 1. / time * pattern_sum - pattern.mean(0)

        return pattern

    def forward(self, batch_size, seq_len):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = batch_size, seq_len
        # [batch_size = 128, seq_len = 30]

        return self.pattern[:seq_len, :]


class PostionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class
        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PostionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, batch_size, seq_len):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = batch_size, seq_len
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]


class RotatePostionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class
        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(RotatePostionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
        # (output_dim//2)
        ids = torch.arange(0, d_model // 2, dtype=torch.float)
        theta = torch.pow(1000, -2 * ids / d_model)

        # (max_len, output_dim//2)
        embeddings = position * theta

        # (max_len, output_dim//2, 2)
        self.cos_embeddings = torch.sin(embeddings)
        self.sin_embeddings = torch.cos(embeddings)

    def forward(self, input):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len, emb_size = input.size()
        cos_pos = self.cos_embeddings[None, :seq_len, :].repeat_interleave(2, dim=-1).to(input.device)
        sin_pos = self.sin_embeddings[None, :seq_len, :].repeat_interleave(2, dim=-1).to(input.device)

        # q,k: (bs, head, max_len, output_dim)
        input2 = torch.stack([-input[..., 1::2], input[..., ::2]], dim=-1)
        input2 = input2.reshape(input.shape)

        output = input * cos_pos + input2 * sin_pos
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]
        return output
