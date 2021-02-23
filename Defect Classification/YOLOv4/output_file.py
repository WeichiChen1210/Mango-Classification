import os
import pandas as pd

def output_csv(predictions):
    df = pd.read_csv('./Test_UploadSheet.csv')
    names = df['image_id'].tolist()

    result_dict = dict()
    
    for img_name, labels in predictions:
        if labels is None:
            result_dict[img_name] = (0, 0, 0, 0, 0)
        else:
            label_list = [False, False, False, False, False]
            for value in labels:
                """
                if value == 0:
                    value = 2
                elif value == 1:
                    value = 3
                """
                if not label_list[value]:
                    label_list[value] = True
            label_list = [int(value) for value in label_list]
            result_dict[img_name] = tuple(label_list)
    
    defects = {'D1': [], 'D2': [], 'D3': [], 'D4': [], 'D5': []}
    for name in names:
        label = result_dict[name]
        for idx, l in enumerate(label):
            key = 'D{}'.format(idx+1)
            defects[key].append(l)
    data = {'image_id': names, 'D1': defects['D1'], 'D2': defects['D2'], 'D3': defects['D3'], 'D4': defects['D4'], 'D5': defects['D5']}
    out_df = pd.DataFrame.from_dict(data)
    out_df.to_csv('./test.csv', index=False)
    print(out_df)