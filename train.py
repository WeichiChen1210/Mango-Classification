import os
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm
from tensorboardX import SummaryWriter

from parameter import get_parameter
from evaluate import test_network
from utils import AverageMeter, acc_logging, MangoDataset, get_dataloader
from loss import get_criterion

from models import networks

def train_network(args, network=None, train_loader=None, val_loader=None, test_loader=None, logging=None):
    device = torch.device("cuda" if args.gpu_no >= 0 else "cpu")
    _print = logging
    writer = SummaryWriter(log_dir=os.path.join(args.save_path, 'events/'))
    
    # DataParallel
    if args.dataparallel:
        network = nn.DataParallel(network)
    network = network.cuda()
    
    # loss
    criterion_ce = get_criterion(args, 'CE')
    criterion_kl = get_criterion(args, 'KL')
    criterion_tri = get_criterion(args, 'tri')
    
    # optimizer and scheduler
    optimizer = torch.optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)    
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestone, gamma=args.lr_gamma)

    # acc recorder
    best_acc = 0.0

    # num of classifier
    if 'res' in args.network:
        num_classifier = 5
    elif 'vgg' in args.network:
        num_classifier = 6

    _print("--" * 30)
    _print("Start training")

    for epoch in range(args.start_epoch, args.epoch):
        tic = time.time()
        print("---------- Epoch %d ----------"% (epoch+1))
        network.train()
        torch.backends.cudnn.benchmark = True
    
        # for recording loss and accuracy
        correct = [0.0 for _ in range(num_classifier)]  # record num of correct predictions
        predicted = [0.0 for _ in range(num_classifier)]    # record predictions
        total = 0.0 # total num of samples
        total_loss = AverageMeter()
        ce = AverageMeter()
        kl = AverageMeter()
        hints = AverageMeter()
        triplet = AverageMeter()
        ordinal = AverageMeter()
    
        train_iter = tqdm(train_loader, desc='Training: ', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        for inputs, targets in train_iter:  
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda()

            outputs, features = network(inputs, True)
            
            # compute ensemble results
            ensemble = sum(outputs)/len(outputs)
            ensemble.detach_()

            ### compute loss
            ce_loss = torch.FloatTensor([0.]).cuda(non_blocking=True)
            kl_loss = torch.FloatTensor([0.]).cuda(non_blocking=True)
            hints_loss = torch.FloatTensor([0.]).cuda(non_blocking=True)
            tri_loss = torch.FloatTensor([0.]).cuda(non_blocking=True)
            ord_loss = torch.FloatTensor([0.]).cuda(non_blocking=True)

            # CrossEntropy loss
            ce_loss += criterion_ce(outputs[-1], targets)
            # triplet loss
            tri_loss = criterion_tri.batch_all_triplet_loss(targets, features[-1]) * args.tri_coef
            # ordinal loss
            ord_loss = criterion_tri.batch_all_ordinal_loss(targets, features[-1]) * args.ord_coef

            # detach teacher's output
            teacher_output = outputs[-1].detach()
            teacher_feature = features[-1].detach()
            
            ###  shallow classifiers
            for index in range(len(outputs)-1): # output 3 ~ 1
                # CrossEntropy loss
                ce_loss += criterion_ce(outputs[index], targets) * (1 - args.loss_coef)
                
                ## loss with deepest classifier
                kl_loss += criterion_kl(outputs[index], teacher_output)
                hints_loss += torch.dist(features[index], teacher_feature, p=2)
                
                # triplet loss
                tri_loss += criterion_tri.batch_all_triplet_loss(targets, features[index])
                # ordinal loss
                ord_loss += criterion_tri.batch_all_ordinal_loss(targets, features[index])

            kl_loss = kl_loss * args.loss_coef
            hints_loss = hints_loss * args.feature_loss_coef
            tri_loss = tri_loss * args.tri_coef
            ord_loss = ord_loss * args.ord_coef
            loss = ce_loss + kl_loss + hints_loss + tri_loss + ord_loss

            # backward    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update loss
            total_loss.update(loss.item())
            ce.update(ce_loss.item())
            kl.update(kl_loss.item())
            hints.update(hints_loss.item())
            triplet.update(tri_loss.item())
            ordinal.update(ord_loss.item())
            outputs.append(ensemble)
            
            # compute accuracy
            for classifier_index in range(len(outputs)):
                _, predicted[classifier_index] = torch.max(outputs[classifier_index].data, 1)
                correct[classifier_index] += float(predicted[classifier_index].eq(targets.data).cpu().sum())
            total += float(targets.size(0))
            
            criterion_tri.reset()

        writer.add_scalars('Loss', {'ce': ce.get_avg(),
                                    'kl': kl.get_avg(),
                                    'hints': hints.get_avg(),
                                    'triplet': triplet.get_avg(),
                                    'ordinal': ordinal.get_avg(),
                                    'total': total_loss.get_avg()}, epoch+1)
        # compute acc of this epoch
        acc = [100 * correct_num / total for correct_num in correct]
        
        # Testing
        test_acc = test_network(args, network, test_loader, mode='test')

        # logging
        _print('Epoch {} - train loss {:.3f}'.format(epoch+1, total_loss.get_avg()))
        acc_logging(acc, epoch, True, _print, writer)
        acc_logging(test_acc, epoch, False, _print, writer)

        # adjust learning rate
        _print('lr: {}'.format(optimizer.param_groups[0]['lr']))
        if scheduler is not None:
            scheduler.step()        
        
        # save model
        if args.dataparallel:
            state_dict = network.module.state_dict()
        else:
            state_dict = network.state_dict()
        
        model_dict = {'epoch': epoch, 'state_dict': state_dict, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
        if test_acc[-1] > best_acc and (epoch + 1) > (args.epoch / 2):
            torch.save(model_dict, os.path.join(args.save_path, args.network+'_best.pth'))
            best_acc = test_acc[-1]
        
        torch.save(model_dict, os.path.join(args.save_path, args.network+'_checkpoint.pth'))
        
        _print("epoch time: %.2fs"%(time.time() - tic))
        _print("##" * 20)
    
    _print("End at %s"%time.ctime())
    _print("--" * 30)
    _print("Best acc %.3f"%(best_acc))

    writer.export_scalars_to_json(os.path.join(args.save_path, 'events/scalars.json'))
    writer.close()
        
    return network

if __name__ == "__main__":
    args = get_parameter()
    
    ### make dir
    args.save_path = os.path.join(args.save_path, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(args.save_path)
    
    # save parameters
    with open(os.path.join(args.save_path, 'params.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write("%s: %s\n"%(key, value))

    ### logging
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(args.save_path, 'log.log'),
                        filemode='w',
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    _print = logging.info
    
    # dataset
    print('Loading dataset...')
    num_classes = 3
    train_dataset = MangoDataset(True, path=args.data_path)
    test_dataset = MangoDataset(False, path=args.data_path)
    # dataloader
    train_loader = get_dataloader(train_dataset, args, True)
    test_loader = get_dataloader(test_dataset, args, False)

    # network and training
    print('Preparing model...')
    network = networks[args.network](pretrained=args.pretrained, num_classes=num_classes)
    network = train_network(args, network=network, train_loader=train_loader, test_loader=test_loader, logging=_print)
