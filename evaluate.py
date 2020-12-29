import torch
import torch.nn as nn
from tqdm import tqdm

def test_network(args, network=None, dataloader=None, mode='val'):
    device = torch.device("cuda" if args.gpu_no >= 0 else "cpu")
    
    network.cuda()
    network.eval()

    if mode == 'val':
        mode = 'Validating: '
    else:
        mode = 'Testing: '
    
    num_classifier = 0
    if 'res' in args.network:
        num_classifier = 5
    elif 'vgg' in args.network:
        num_classifier = 6
    
    correct = [0.0 for _ in range(num_classifier)]
    predicted = [0.0 for _ in range(num_classifier)]
    total = 0.0

    test_iter = tqdm(dataloader, desc=mode, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    with torch.no_grad():            
        for images, labels in test_iter:
            images, labels = images.to(device), labels.to(device)
                
            outputs, _ = network(images)
            ensemble = sum(outputs) / len(outputs)
            outputs.append(ensemble)
                
            for classifier_index in range(len(outputs)):
                _, predicted[classifier_index] = torch.max(outputs[classifier_index].data, 1)
                correct[classifier_index] += float(predicted[classifier_index].eq(labels.data).cpu().sum())
                
            total += float(labels.size(0))
    acc = [100 * correct_num / total for correct_num in correct]

    return acc