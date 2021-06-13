import argparse
import numpy as np
import os
import time
from tqdm import tqdm
import torch
import torchsparse
from model import MinkUNet

has_cuda = torch.cuda.is_available()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='semantic-kitti')
    parser.add_argument('--measure-steps', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    args, opts = parser.parse_known_args()
    dataset_name = args.dataset.lower()
    device = args.device
    measure_steps = args.measure_steps
        
    torch.manual_seed(1)

    sk_file_dir = '../../pclibs/sparseconv/data/semantic-kitti'
    sk_selected = ['000060.bin']

    net = MinkUNet(num_classes=19, run_up=True).to(device).eval()
    
    
    if dataset_name == 'semantic-kitti':
        all_fns = [os.path.join(sk_file_dir, x) for x in sk_selected]
    else:
        raise NotImplementedError
    
    time_lis = []
    with torch.no_grad():
        for idx in range(1 + measure_steps):
            if idx==1:
                with open("mm_time.txt", "w") as fp:
                    pass
            for fn in tqdm(all_fns):
                if dataset_name == 'semantic-kitti':
                    pc = np.fromfile(fn, dtype=np.float32).reshape(-1,4)
                    pc[:, :3] = np.floor(pc[:, :3] / 0.05)
                inds = torchsparse.utils.sparse_quantize(pc[:, :3])
                pc = pc[inds]
                coord = np.zeros((pc.shape[0], 4))
                coord[:, :3] = pc[:, :3]
                coord[:, -1] = 0
                sample_feat = torch.randn(pc.shape[0], 4).float()
                sample_coord = torch.from_numpy(coord).int()
                sample_coord[:, -1] = 0
                x = torchsparse.SparseTensor(sample_feat, sample_coord.int()).to(device)
                if has_cuda:
                    torch.cuda.synchronize()
                st = time.time()
                y1 = net(x)
                if has_cuda:
                    torch.cuda.synchronize()    
                ed = time.time()
                if idx >= 1:
                    time_lis.append(ed-st)
    print(f"MinkUNet latency {np.mean(time_lis)} ± {np.std(time_lis)}")