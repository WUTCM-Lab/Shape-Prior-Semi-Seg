import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from itertools import cycle
from data.build_dataset import build_dataset
from models.build_model import build_model
from models.dc_gan import DCGAN_D
from utils.evaluate import evaluate
from opt import args
from utils.loss import BceDiceLoss
import math
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def DeepSupSeg(pred, gt):
    criterion = BceDiceLoss()
    loss = criterion(pred, gt)
    return loss

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1-float(iter)/max_iter)**power)

def adjust_lr_rate(argsimizer, iter, total_batch):
    lr = lr_poly(args.lr, iter, args.nEpoch*total_batch, args.power)
    argsimizer.param_groups[0]['lr'] = lr
    return lr

def train():
    """load data"""
    train_l_data, _ , valid_data = build_dataset(args)
    train_l_dataloader = DataLoader(train_l_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_sign = False
    if valid_data is not None:
        valid_sign = True
        valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
        val_total_batch = int(len(valid_data) / 1)
    
    """load model"""
    model = build_model(args)

    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mt, weight_decay=args.weight_decay)

    # train
    print('\n---------------------------------')
    print('Start training')
    print('---------------------------------\n')

    F1_best, F1_second_best, F1_third_best = 0, 0, 0
    best = 0
    for epoch in range(args.nEpoch):
        model.train()
      
        print("Epoch: {}".format(epoch))
        total_batch = math.ceil(len(train_l_data) / args.batch_size)
        bar = tqdm(enumerate(train_l_dataloader), total=total_batch)
        for batch_id, data_l in bar:
            itr = total_batch * epoch + batch_id
            img, gt = data_l['image'], data_l['label']
            if args.GPUs:
                img = img.cuda()
                gt = gt.cuda()
            optim.zero_grad()
            mask = model(img)
            loss = DeepSupSeg(mask, gt) 
            loss.backward()
            optim.step()
            adjust_lr_rate(optim, itr, total_batch)

        if valid_sign == True:
            recall, specificity, precision, F1, F2, \
            ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice = evaluate(model, valid_dataloader, val_total_batch)

            print("Valid Result:")
            print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f' \
                % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice))

            if dice > best:
                best = dice
            print("Best Dice:: ", best)

            if (F1 > F1_best):
                F1_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/best.pth")
            elif(F1 > F1_second_best):
                F1_second_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/second_best.pth")
            elif(F1 > F1_third_best):
                F1_third_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/third_best.pth")

def train_semi():
    """load data"""
    train_l_data, train_u_data, valid_data = build_dataset(args)
    train_l_dataloader = DataLoader(train_l_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_u_dataloader = DataLoader(train_u_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_sign = False
    if valid_data is not None:
        valid_sign = True
        valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
        val_total_batch = int(len(valid_data) / 1)
    """load model"""
    model = build_model(args)

    netD = DCGAN_D(64, 100, 1, 64, 1, 0)
    netD.cuda()
    netD_weight = torch.load("models/pretrain/GAN/netD_epoch_10000.pth")
    netD.load_state_dict(netD_weight)
    netD.eval()

    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mt, weight_decay=args.weight_decay)

    # train
    print('\n---------------------------------')
    print('Start training_semi')
    print('---------------------------------\n')
    F1_best, F1_second_best, F1_third_best = 0, 0, 0
    best = 0
    for epoch in range(args.nEpoch):
        model.train()
        print("Epoch: {}".format(epoch))
        loader = iter(zip(cycle(train_l_dataloader), train_u_dataloader))
        bar = tqdm(range(len(train_u_dataloader)))
        for batch_id in bar:
            data_l, data_u = next(loader)
            total_batch = len(train_u_dataloader)
            itr = total_batch * epoch + batch_id
            img_l, gt = data_l['image'], data_l['label']
            img_u = data_u
            if args.GPUs:
                img_l = img_l.cuda()
                gt = gt.cuda()
                img_u = img_u.cuda()
            optim.zero_grad()
            pred_l = model(img_l)
            mask = pred_l[0]
            loss_l_seg = DeepSupSeg(mask, gt)
            loss_l = loss_l_seg
            pred_u = model(img_u)
            _, predboud, inpimg2, inpimg3, inpimg4, inpimg5, mask_boud = pred_u
            loss_u_seg = DeepSupSeg(predboud, mask_boud)
            shape_u_1 = F.interpolate(predboud, size = (64, 64), mode = 'bilinear', align_corners = False)
            shape_u_2 = F.interpolate(inpimg2, size = (64, 64), mode = 'bilinear', align_corners = False)
            shape_u_3 = F.interpolate(inpimg3, size = (64, 64), mode = 'bilinear', align_corners = False)
            shape_u_4 = F.interpolate(inpimg4, size = (64, 64), mode = 'bilinear', align_corners = False)
            shape_u_5 = F.interpolate(inpimg5, size = (64, 64), mode = 'bilinear', align_corners = False)
            loss_u_shape = (netD(shape_u_1) + netD(shape_u_2) + netD(shape_u_3) + netD(shape_u_4) + netD(shape_u_5)) / 5
            loss_u = loss_u_seg + 0.1 * loss_u_shape
            loss = 2 * loss_l + loss_u
            loss.backward()
            optim.step()
            adjust_lr_rate(optim, itr, total_batch)
        model.eval()
        if valid_sign == True:
            recall, specificity, precision, F1, F2, \
            ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice = evaluate(model, valid_dataloader, val_total_batch)

            print("Valid Result:")
            print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f' \
                % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean,dice))
            
            if dice > best:
                best = dice
            print("Best Dice:: ", best)

            if (F1 > F1_best):
                F1_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/best.pth")
            elif(F1 > F1_second_best):
                F1_second_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/second_best.pth")
            elif(F1 > F1_third_best):
                F1_third_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/third_best.pth")

def test():
  
    print('loading data......')
    test_data = build_dataset(args)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
    total_batch = int(len(test_data) / 1)
    model = build_model(args)
    model.eval()

    recall, specificity, precision, F1, F2, \
            ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice = evaluate(model, test_dataloader, total_batch)
    
    print("Test Result:")
    print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f' \
                % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean,dice))

if __name__ == '__main__':

    checkpoint_name = os.path.join(args.root, 'semi/checkpoint/' + args.ckpt_name)
    if not os.path.exists(checkpoint_name):
        os.makedirs(checkpoint_name)
    else:
        pass
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPUs
    if args.manner == 'full':
        print('---{}-Seg Train---'.format(args.dataset))
        train()
    elif args.manner =='semi':
        print('---{}-seg Semi-Train--'.format(args.dataset))
        train_semi()
    elif args.manner == 'test':
        print('---{}-Seg Test---'.format(args.dataset))
        test()
    print('Done')

