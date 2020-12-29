import os
import argparse
from os.path import join
from tqdm import tqdm
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

from evaluate import test_network
from models import networks
from utils import MangoDataset as test_Dataset, get_dataloader
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class MangoDataset(Dataset):
    def __init__(self, path=None):
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # read from csv
        dir_path, csv_path = '', ''
        csv_path = path + 'test.csv'
        dir_path = csv_path[:-4] + '/'
        
        # dir_path = path if path is not None else '../dataset/preprocessed/test/'
        # csv_path = '../dataset/preprocessed/test.csv'
        
        df = pd.read_csv(csv_path)
        self.raw_names = df['image_id'].tolist()
        self.names = [join(dir_path, name) for name in self.raw_names]
                    
    def __getitem__(self, idx):
        img = Image.open(self.names[idx])
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.names)

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu-no', type=int, help='cpu: -1, gpu: 0 ~ n ', default=0)
    parser.add_argument('--network', type=str, help='the type of network', default='resnet18')
    parser.add_argument('--batchsize', type=int, help='batch size', default=128)
    parser.add_argument('--num-workers', type=int, help='number of workers for data loader', default=8)
    parser.add_argument('--load-path', type=str, help='trained model load path', default=None)
    parser.add_argument('--save-path',type=str, help='model save path', default='./results/')
    parser.add_argument('--data-path',type=str, help='dataset path', default='./dataset/crop/')
    parser.add_argument('--inference', dest='inference', action='store_true', help='inference mode')
    parser.add_argument('--test', dest='inference', action='store_false', help='testing mode')
    parser.set_defaults(inference=False)
    parser.add_argument('--attention', dest='attention', action='store_true', help='use attention or not')
    parser.set_defaults(attention=False)

    return parser

def inference(args, network):
    num2label = ['A', 'B', 'C']
    print('Loading test set...')
    dataset = MangoDataset(args.data_path)
    dataloader = DataLoader(dataset, args.batchsize, shuffle=False, num_workers=args.num_workers)

    network.cuda()
    network.eval()

    prediction = []
    iterator = tqdm(dataloader, desc='Inferencing', bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}')
    with torch.no_grad():
        for inputs in iterator:
            inputs = inputs.cuda(non_blocking=True)
            outputs, _ = network(inputs)

            _, predicted = torch.max(outputs[-1].data, 1)
            predicted = predicted.cpu().tolist()
            prediction.extend([num2label[idx] for idx in predicted])

    names = dataset.raw_names.copy()
    d_predict = {'image_id': names, 'label': prediction}
    df_pred = pd.DataFrame(data=d_predict)
    os.makedirs(args.save_path, exist_ok=True)
    df_pred.to_csv(args.save_path + 'prediction.csv', index=False)

def enemble_inference(args, network):
    num2label = ['A', 'B', 'C']
    print('Loading test set...')
    dataset = MangoDataset(args.data_path)
    dataloader = DataLoader(dataset, args.batchsize, shuffle=False, num_workers=args.num_workers)

    network.cuda()
    network.eval()

    prediction = []
    iterator = tqdm(dataloader, desc='Inferencing', bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}')
    with torch.no_grad():
        for inputs in iterator:
            inputs = inputs.cuda(non_blocking=True)
            outputs = network(inputs)

            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().tolist()
            prediction.extend([num2label[idx] for idx in predicted])

    names = dataset.raw_names.copy()
    d_predict = {'image_id': names, 'label': prediction}
    df_pred = pd.DataFrame(data=d_predict)
    df_pred.to_csv(args.save_path + 'prediction.csv', index=False)

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    print('Loading model...')
    args.save_path = args.load_path

    if 'ensemble' in args.network:
        network_list = ['resnet18', 'resnet18']
        model_path = ['20201211_211503/resnet18_checkpoint.pth', '20201211_220231/resnet18_checkpoint.pth']
        ensemble_list = []
        for net, path in zip(network_list, model_path):
            model = networks[net](False, num_classes=3)
            model.load_state_dict(torch.load(os.path.join('./trained_models/', path))['state_dict'])
            model.cuda()
            ensemble_list.append(model)
        
        network = networks[args.network](ensemble_list, 3)
        checkpoint = torch.load(os.path.join(args.load_path, 'checkpoint.pth'))
        network.fc.load_state_dict(checkpoint['state_dict'])
    else:
        network = networks[args.network](pretrained=False, use_attention=args.attention, num_classes=3)
        checkpoint = torch.load(os.path.join(args.load_path, args.network+'_checkpoint.pth'))
        network.load_state_dict(checkpoint['state_dict'])
    

    if args.inference:
        if 'ensemble' in args.network:
            enemble_inference(args, network)
        else:
            inference(args, network)
    else:
        print('Preparing dataset...')
        dataset = test_Dataset(False, path=args.data_path)
        dataloader = get_dataloader(dataset, args, False)

        print('Testing...')
        test_acc = test_network(args, network, dataloader, mode='test')
        print(test_acc)