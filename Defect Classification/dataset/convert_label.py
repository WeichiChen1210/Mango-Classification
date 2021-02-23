"""
This generates the files needed for YOLOv4.
"""
from os import makedirs
from os.path import join, exists, isfile
import csv
import cv2
from tqdm import tqdm
import argparse

from preprocess import adjust_saturation, sharpen

label2idx = {'乳汁吸附': 0, '機械傷害': 1, '炭疽病': 2, '著色不佳': 3, '黑斑病': 4} # convert labels to idxs
# wrtie label to idx to file
if not isfile('./label_idx.txt'):
    with open('./label_idx.txt', 'w') as f:
        out_str = ''
        for label, idx in label2idx.items():
            out_str += label + ',' + str(idx) + '\n'
        f.write(out_str)

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='val', choices=['train', 'val', 'test'], help='train|val|test')
    parser.add_argument('--preprocess', action='store_true', help='preprocess data or not')

    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    label_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}            # for counting number of labels
    csv_path = './' + args.mode + '.csv'                    # path to read positions and labels
    src_img_root = './' + args.mode + '/'                   # path to read imgs

    target_name_file = args.mode + '.txt'                   # file to store image names
    target_shape_file = args.mode + '.shapes'               # shape file
    target_img_folder = 'images/' + args.mode + '/'         # target image folder path in YOLOv4 repository
    target_label_path = 'labels/' + args.mode + '/'         # target label folder path in YOLOv4 repository
    target_data_root = '../YOLOv4/data/'            # the data root

    if args.mode == 'test':
        path = join(target_data_root, 'images/test/')
        if not exists(path):
            makedirs(path, exist_ok=True)
        
        csv_path = './Test_UploadSheet.csv'
        # read csv file
        lines = []
        with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
            rows = csv.reader(f)
            lines = [row for row in rows]
        lines = [[word for word in line if len(word) > 0] for line in lines[1:]]
        print('total', len(lines), 'data')
        
        # write name file
        with open(join(target_data_root, 'test.txt'), 'w') as f:
            out_str = []
            for line in lines:
                out_str.append(join('./'+target_img_folder, line[0]))
            out_str = '\n'.join(out_str)
            f.write(out_str)
        
        # copy images
        with open(join(target_data_root, 'test.shapes'), 'w') as f:
            out_str = []
            iterator = tqdm(lines)
            for line in iterator:
                name = line[0]
                img = cv2.imread(join(src_img_root, name))
                img_h, img_w, _ = img.shape
                out_str.append('{} {}'.format(img_w, img_h))

                if args.preprocess:
                    hsv = adjust_saturation(img, 10)
                    sharp = sharpen(hsv)
                    cv2.imwrite(join(target_data_root, target_img_folder, name), sharp)
                else:
                    cv2.imwrite(join(target_data_root, target_img_folder, name), img)
            out_str = '\n'.join(out_str)
            f.write(out_str)

    else:
        for target in [target_img_folder, target_label_path]:
            path = join(target_data_root, target)
            if not exists(path):
                makedirs(path, exist_ok=True)

        # read csv file
        lines = []
        with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
            rows = csv.reader(f)
            lines = [row for row in rows]
        lines = [[word for word in line if len(word) > 0] for line in lines]
        print('total', len(lines), 'data')

        # write name file
        with open(join(target_data_root, target_name_file), 'w') as f:
            out_str = []
            for line in lines:
                out_str.append(join('./'+target_img_folder, line[0]))
            out_str = '\n'.join(out_str)
            f.write(out_str)

        # process lines to name, positions and label
        shapes = []
        names = []
        iterator = tqdm(lines)
        for line in iterator:
            name = line.pop(0)  # pop to get img name
            img_name = join(src_img_root, name)
            names.append(img_name)

            # parse to get positions
            labels = []
            for i in range(0, len(line), 5):
                x, y, w, h = int(float(line[i])), int(float(line[i+1])), int(float(line[i+2])), int(float(line[i+3]))
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                if w < 0:
                    w = 0
                if h < 0:
                    w = 0
                class_label = label2idx[line[i+4][3:]]
                label = tuple([class_label, x, y, w, h])
                labels.append(label)
            assert len(labels) == int(len(line) / 5), 'not equal'
            
            # create shape file
            img = cv2.imread(img_name)
            img_h, img_w, _ = img.shape
            shape_str = '{} {}'.format(img_w, img_h)
            shapes.append(shape_str)

            # copy image
            if args.preprocess:
                hsv = adjust_saturation(img, 10)
                sharp = sharpen(hsv)
                cv2.imwrite(join(target_data_root, target_img_folder, name), sharp)
            else:
                cv2.imwrite(join(target_data_root, target_img_folder, name), img)

            # calculate proportions for yolo label format
            img_labels = []
            for label in labels:
                id, xmin, ymin, w, h = label
                """
                if id not in [2, 3]:
                    continue
                if id == 2:
                    id = 0
                if id == 3:
                    id = 1
                """
                x = (xmin + w / 2) * 1.0 / img_w
                y = (ymin + h / 2) * 1.0 / img_h
                w_pro = w * 1.0 / img_w
                h_pro = h * 1.0 / img_h
                out_str = '{} {} {} {} {}'.format(id, round(x, 6), round(y, 6), round(w_pro, 6), round(h_pro, 6))
                img_labels.append(out_str)
            img_labels = '\n'.join(img_labels)

            # write label file
            with open(join(target_data_root, target_label_path, name[:-3] + 'txt'), 'w') as f:
                f.write(img_labels)
            
        # write shape file
        shapes = '\n'.join(shapes)
        with open(join(target_data_root, target_shape_file), 'w') as f:
            f.write(shapes)
    