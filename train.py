from __future__ import print_function
import os
# from dataset
import torch
from torch.utils import data
import torch.nn.functional as F
from models import *
import torchvision
from utils import Visualizer, view_model
import torch
import numpy as np
import random
import time
from config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from test import *
import  dataset
from dataset import sampler
from torch.utils.data.sampler import BatchSampler



def save_model(model, save_path, name, iter_cnt):
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


if __name__ == '__main__':

    opt = Config()
    if opt.display:
        visualizer = Visualizer()
    device = torch.device("cuda")

    trn_dataset = dataset.load(
            name = opt.dataset,
            root = opt.train_root,
            mode = 'train',
            transform = dataset.utils.make_transform(
                is_train = True,
                is_inception = (opt.backbone == 'bn_inception')
            ))
    balanced_sampler = sampler.BalancedSampler(trn_dataset, batch_size=opt.sz_batch, images_per_class = opt.IPC)
    batch_sampler = BatchSampler(balanced_sampler, batch_size = opt.sz_batch, drop_last = True)
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        num_workers = opt.nb_workers,
        pin_memory = True,
        batch_sampler = batch_sampler
    )

    ev_dataset = dataset.load(
        name=opt.dataset,
        root=opt.test_root,
        mode='eval',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=(opt.backbone == 'bn_inception')
        ))

    dl_ev = torch.utils.data.DataLoader(
        ev_dataset,
        batch_size=opt.sz_batch,
        shuffle=False,
        num_workers=opt.nb_workers,
        pin_memory=True
    )

    print(len(dl_tr.dataset))
    print(len(dl_ev.dataset))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()


    model = resnet50()

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(2048, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(2048, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(2048, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(2048, opt.num_classes)

    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)
    evaluate(model, dl_ev, eval_nmi=True, recall_list=[1, 2, 4, 8])

    for i in range(opt.max_epoch):
        scheduler.step()

        model.train()
        for ii, data in tqdm(enumerate(dl_tr)):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(dl_tr) + ii

            if iters % opt.print_freq == 0:
                evaluate(model, dl_ev, eval_nmi=True, recall_list=[1, 2, 4, 8])

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, i)

        model.eval()
        evaluate(model, dl_ev, eval_nmi=True, recall_list=[1, 2, 4, 8])