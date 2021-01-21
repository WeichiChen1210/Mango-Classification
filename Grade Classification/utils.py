import random
import pandas as pd
from PIL import Image
from os.path import join

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

grades = {'A': 0, 'B': 1, 'C': 2}
# original dataset
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

sample_num = [28585.0, 11140.0, 5275.0]

class AverageMeter(object):    
    def __init__(self):
        self.item_list = []

    def reset(self):
        self.item_list.clear()
    
    def get_avg(self):        
        return sum(self.item_list) / len(self.item_list)

    def update(self, items):
        self.item_list.append(items)
    
    def __len__(self):
        return len(self.item_list)

def acc_logging(acc_list, epoch, is_train, _print, writer):
    if is_train:
        mode = 'train'
    else:
        mode = 'test'
    
    scalar_dict = {}
    log_str = ''
    
    log_str = 'Epoch {} - {} acc: '.format(epoch+1, mode)
    num_outputs = len(acc_list)-1
    writer_str = 'classifier'

    for classifier_idx in range(num_outputs):
        log_str += '{}/{}: {:.3f}, '.format(classifier_idx+1, num_outputs, acc_list[classifier_idx])
        scalar_dict[writer_str+str(classifier_idx+1)] = acc_list[classifier_idx]
    
    log_str += 'Ensemble: {:.3f}'.format(acc_list[-1])
    scalar_dict['Ensemble'] = acc_list[-1]

    writer.add_scalars('{}_acc'.format(mode), scalar_dict, epoch+1)
    _print(log_str)

def get_dataloader(dataset, args, is_train=True, sampler=None):
    loader = None
    # training dataset
    if is_train:
        if sampler is None:
            loader = DataLoader(dataset,
                                batch_size=args.batchsize,
                                num_workers=args.num_workers,
                                shuffle=True,
                                pin_memory=True)
        # training and validation
        elif sampler == 'subset':
            n_train = dataset.__len__()
            split = int(n_train * 0.9)
            train_loader = DataLoader(dataset, batch_size=args.batchsize, num_workers=args.num_workers, 
                                      sampler=SubsetRandomSampler(range(split)), pin_memory=True)
            val_loader = DataLoader(dataset, batch_size=args.batchsize, num_workers=args.num_workers,
                                    sampler=SubsetRandomSampler(range(split, n_train)), pin_memory=True)
            return train_loader, val_loader
        
        # weighted sampler
        elif sampler == 'weighted':
            weights = 1. / torch.tensor(sample_num, dtype=torch.float)
            sample_weights = weights[dataset.labels]
            sampler = WeightedRandomSampler(sample_weights, dataset.__len__(), replacement=True)
            loader = DataLoader(dataset, batch_size=args.batchsize, num_workers=args.num_workers, sampler=sampler, pin_memory=True)
    # test dataset
    else:
        loader = DataLoader(dataset, 
                            batch_size=args.batchsize,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True)
    return loader

class MangoDataset(Dataset):
    def __init__(self, is_train=True, transform=None, path=None):
        self.train = is_train
        
        # prepare transform
        if transform is None:
            self.transform = self.get_transform()
        else:
            self.transform = transform

        # read from csv
        dir_path, csv_path = '', ''
        if path is None:
            dir_path = './dataset/crop/'
        else:
            dir_path = path
        if is_train:
            csv_path = join(dir_path, 'train.csv')
        else:
            csv_path = join(dir_path, 'dev.csv')
        
        df = pd.read_csv(csv_path)
        names = df['image_id'].tolist()
        labels = df['grade'].tolist()
        names = [join(csv_path[:-4]+'/', name) for name in names]
        labels = [int(grades[label]) for label in labels]
        
        self.names = names
        self.labels = labels
        
        self.indices = [[], [], []]
        for idx, label in enumerate(self.labels):
            self.indices[label].append(idx)

    def get_transform(self):
        if self.train:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.ColorJitter(saturation=0.4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation((-15, 15)),
                transforms.RandomVerticalFlip(p=0.5),
                # transforms.FiveCrop((224, 224)),
                # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                # transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean, std)(crop) for crop in crops])),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)                
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        return transform
            
    def __getitem__(self, idx):
        label = int(self.labels[idx])
        img = Image.open(self.names[idx])
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)