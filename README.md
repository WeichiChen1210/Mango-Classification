# Mango Grade and Defect Classification by AI CUP 2020
This is a competition held by AI CUP 2020.

There are two task in the competition:

1. Grade Classification
    
    Classify input mango images into 3 grades, from the best to the worst. We apply self-distillation framework and propose triplet loss and ordinal loss for this task.
2. Defect Classification

    Classify input mango images into 5 kinds of defect. Since the test set does not provide defect positions, we apply YOLOv4 to locate and classify defects.