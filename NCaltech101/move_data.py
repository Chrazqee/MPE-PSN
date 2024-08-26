import os
import shutil
# class_num = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

root = '/home/chrazqee/datasets/ncaltech-101/frames_number_10_split_by_number'
class_num = os.listdir(root)

for cn in class_num:
    source = os.path.join(root, cn)
    target = os.path.join(root, 'test', cn)
    if not os.path.exists(target):
        os.makedirs(target)

    img_num = len(os.listdir(source))
    for i in range(1, int(img_num * 0.1) + 1):
        os.symlink(os.path.join(source, f'image_{i:04}.npz'), os.path.join(target, f'image_{i:04}.npz'))

    target = os.path.join(root, 'train', cn)
    if not os.path.exists(target):
        os.makedirs(target)
    for i in range(int(img_num * 0.1) + 1, img_num + 1):
        os.symlink(os.path.join(source, f'image_{i:04}.npz'), os.path.join(target, f'image_{i:04}.npz'))
