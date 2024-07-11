from torch.utils.data import Dataset
from torchvision import transforms
import os
from utils.mytransforms import *
import json


class BUSIDataSet(Dataset):
    def __init__(self, root, expID, mode='train', ratio=10, sign='label', transform=None):
        super(BUSIDataSet, self).__init__()
        self.mode = mode
        self.sign = sign
        if mode == 'train':
            if sign =='label':
                if expID == 1:
                    imgfile = os.path.join(root,'data/splits/BUSI/72/labeled.txt')
                elif expID == 2:
                    imgfile = os.path.join(root,'data/splits/BUSI/144/labeled.txt')
                elif expID == 3:
                    imgfile = os.path.join(root,'data/splits/BUSI/288/labeled.txt')
                with open(imgfile,'r') as f:
                    imglist = f.read().splitlines()
                    self.imglist = [os.path.join(root, 'data/BUSI',img) for img in imglist]
            else:
                if expID == 1:
                    imgfile = os.path.join(root,'data/splits/BUSI/72/unlabeled.txt')
                elif expID == 2:
                    imgfile = os.path.join(root,'data/splits/BUSI/144/unlabeled.txt')
                elif expID == 3:
                    imgfile = os.path.join(root,'data/splits/BUSI/288/unlabeled.txt')
                with open(imgfile,'r') as f:
                    imglist = f.read().splitlines()
                    self.imglist = [os.path.join(root, 'data/BUSI',img) for img in imglist]
        elif mode == 'valid':
            imgfile = os.path.join(root,'data/splits/BUSI/val.txt')
            with open(imgfile,'r') as f:
                    imglist = f.read().splitlines()
                    self.imglist = [os.path.join(root, 'data/BUSI',img) for img in imglist]
        elif mode == 'test':
            imgfile = os.path.join(root, 'data/BUSI/leftImg/val')
            self.imglist = [file for file in self.get_all_files(imgfile)]

        if transform is None:
            if mode == 'train' and sign == 'label':
               transform = transforms.Compose([
                   Resize((320, 320)),
                   RandomHorizontalFlip(),
                   RandomVerticalFlip(),
                   RandomRotation(90),
                   RandomZoom((0.9, 1.1)),
                   RandomCrop((256, 256)),
                   ToTensor()
               ])
            elif mode == 'train' and sign == 'unlabel':
                transform = transforms.Compose([
                    transforms.Resize((320, 320)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(90),
                    transforms.RandomCrop((256, 256)),
                    transforms.ToTensor()
                ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                   Resize((320, 320)),
                   ToTensor()
                ])
        self.transform = transform

    def __getitem__(self, index):
        if self.mode == 'train' and self.sign == 'unlabel':
            img_path = self.imglist[index]
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                return self.transform(img)
        else:
            img_path = self.imglist[index]
            gt_path = img_path.replace('leftImg', 'gtFine')
            img = Image.open(img_path).convert('RGB')
            gt = Image.open(gt_path).convert('L')
            data = {'image': img, 'label': gt}
            if self.transform:
                data = self.transform(data)
            data['name'] = self.imglist[index].split('/')[-1]
            return data

    def __len__(self):
        return len(self.imglist)
    
    def get_all_files(self,directory):
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.abspath(os.path.join(root, file))
                file_paths.append(file_path)
        return file_paths
