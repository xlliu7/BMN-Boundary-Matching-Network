import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init


class BMSampling(nn.Module):
    '''Implements the BM layer'''

    def __init__(self, opt):
        super(BMSampling, self).__init__()
        self.temporal_dim = opt['temporal_scale']
        self.batch_size = opt['bmn_batch_size']
        # self.feat_dim = opt['bmn_feat_dim']
        self.num_sample_point = opt['num_sample_point']
        self.num_prop_per_loc = opt['num_prop_per_loc']
        self.roi_expand_ratio = opt['roi_expand_ratio']
        self.smp_weight = self.get_pem_smp_weight()

    def get_pem_smp_weight(self):
        T = self.temporal_dim
        N = self.num_sample_point
        D = self.num_prop_per_loc
        w = torch.zeros([T, N, D, T])  # T * N * D * T
        # In each temporal location i, there are D predefined proposals,
        # with length ranging between 1 and D
        # the j-th proposal is [i, i+j+1], 0<=j<D
        # however, a valid proposal should meet i+j+1 < T
        for i in range(T-1):
            for j in range(min(T-1-i, D)):
                xmin = (i)
                xmax = (j + 1)
                # proposals[j, i, :] = [xmin, xmax]
                length = xmax - xmin
                xmin_ext = xmin - length * self.roi_expand_ratio
                xmax_ext = xmax + length * self.roi_expand_ratio
                bin_size = (xmax_ext - xmin_ext) / (N - 1)
                points = [xmin_ext + ii *
                          bin_size for ii in range(N)]
                for k, xp in enumerate(points):
                    if xp < 0 or xp > T - 1:
                        continue
                    left, right = int(np.floor(xp)), int(np.ceil(xp))
                    left_weight = 1 - (xp - left)
                    right_weight = 1 - (right - xp)
                    w[left, k, j, i] += left_weight
                    w[right, k, j, i] += right_weight
        return w.view(T, -1).float()

    def _apply(self, fn):
        self.smp_weight = fn(self.smp_weight)

    def forward(self, X):
        input_size = X.size()
        assert(input_size[-1] == self.temporal_dim)
        # assert(len(input_size) == 3 and
        X_view = X.view(-1, input_size[-1])
        # feature [bs*C, T]
        # smp_w    [T, N*D*T]
        # out      [bs*C, N*D*T] --> [bs, C, N, D, T]
        result = torch.matmul(X_view, self.smp_weight)
        return result.view(self.batch_size, input_size[1], self.num_sample_point, self.num_prop_per_loc, self.temporal_dim)


class BMN(nn.Module):
    def __init__(self, opt):
        super(BMN, self).__init__()
        self.feat_dim = opt["bmn_feat_dim"]
        self.temporal_dim = opt["temporal_scale"]
        self.batch_size = opt["bmn_batch_size"]
        # self.c_hidden = opt["tem_hidden_dim"]
        self.bmn_best_loss = 10000000
        self.output_dim = 3
        self.num_sample_point = opt['num_sample_point']

        self.base = nn.Sequential(
            nn.Conv1d(self.feat_dim, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.ReLU()
        )

        self.tem = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 2, 3, padding=1),
            nn.Sigmoid()
        )

        self.bm_layer = BMSampling(opt)

        self.pem_c3d = nn.Sequential(
            nn.Conv3d(128, 512, (self.num_sample_point, 1, 1),
                      (self.num_sample_point, 1, 1)),
            nn.ReLU(),
        )
        self.pem_c2d = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        base = self.base(x)
        tem_out = self.tem(base)

        pem_feature = self.bm_layer(base)  # bs * c * N * D * T
        y = self.pem_c3d(pem_feature)
        y = y.squeeze(2)
        y = self.pem_c2d(y)
        return tem_out, y


if __name__ == '__main__':
    opt = {
        'bmn_feat_dim': 400,
        'temporal_scale': 100,
        'bmn_batch_size': 4,
        'num_sample_point': 32,    # the symbol N in the original paper
        'num_prop_per_loc': 100,   # the symbol D in the original paper
        'roi_expand_ratio': 0.25
    }

    model = BMN(opt)
    x = torch.from_numpy(np.random.rand(
        opt['bmn_batch_size'], opt['bmn_feat_dim'], opt['temporal_scale'])).float()
    x = torch.autograd.Variable(x.cuda())
    model.cuda()
    # pdb.set_trace()
    s_ = time.time()
    tem_out, pem_out = model(x)
    print(time.time() - s_)
    print(tem_out.shape, pem_out.shape)
