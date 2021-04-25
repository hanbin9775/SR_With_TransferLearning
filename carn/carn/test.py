import os
import json
import time
import importlib
import argparse
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.utils.data as data
from glob import glob
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--sample_dir", type=str)
    parser.add_argument("--test_data_dir", type=str, default="dataset/Urban100")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--shave", type=int, default=20)

    return parser.parse_args()


def save_image(tensor, filename):
    tensor = tensor.cpu()
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)

    
class TestDataset(data.Dataset):
    def __init__(self, dirname, scale):
        super(TestDataset, self).__init__()

        self.lr = glob(os.path.join(dirname, "*.png"))
        self.lr.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        lr = Image.open(self.lr[index])
        lr = lr.convert("RGB")
        filename = self.lr[index].split("/")[-1]

        return self.transform(lr), filename

    def __len__(self):
        return len(self.lr)


def sample(net, device, dataset, cfg):
    scale = cfg.scale
    for lr, name in tqdm(dataset):
        t1 = time.time()
        lr = lr.unsqueeze(0).to(device)
        sr = net(lr, cfg.scale).detach().squeeze(0)
        lr = lr.squeeze(0)
        t2 = time.time()

        sr_dir = os.path.join(cfg.sample_dir, cfg.test_data_dir.split("/")[-1])

        os.makedirs(sr_dir, exist_ok=True)

        sr_im_path = os.path.join(sr_dir, name)
        save_image(sr, sr_im_path)


def main(cfg):
    module = importlib.import_module("model.{}".format(cfg.model))
    net = module.Net(multi_scale=True, 
                     group=cfg.group)
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))

    state_dict = torch.load(cfg.ckpt_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # name = k[7:] # remove "module."
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    dataset = TestDataset(cfg.test_data_dir, cfg.scale)
    sample(net, device, dataset, cfg)
 

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
