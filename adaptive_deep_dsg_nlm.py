import torch.nn as nn
from denoisers.utils import *
from denoisers.dsg_nlm import DSGNLM

class ADDSGNLM(nn.Module):
    def __init__(self):
        super(ADDSGNLM, self).__init__()  # 修正了super()的使用
        h_list = [0.9804]*4
        s_list = [0.8]*3
        self.h = nn.Parameter(torch.tensor(h_list, dtype=torch.float32), requires_grad=True)
        self.s = nn.Parameter(torch.tensor(s_list, dtype=torch.float32), requires_grad=True)
    def forward(self, y):
        laplacian_data = laplacian_filter(y)

        x = DSGNLM(noisy_img=y, guide_img=y, patch_rad=3, window_rad=4, sigma=self.h[0]*laplacian_data)
        for i in range(3):
            step_size = torch.clamp(self.s[i],0.6,1)
            x = (1-step_size)*x+step_size*y

            laplacian_data = laplacian_filter(x)
            x = DSGNLM(noisy_img=x, guide_img=y, patch_rad=3, window_rad=4, sigma=self.h[i+1]*laplacian_data)
        return x
