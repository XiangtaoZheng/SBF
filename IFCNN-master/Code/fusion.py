import os
import cv2
import time
import torch
from model import myIFCNN

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']="0"
os.environ['CUDA_LAUNCH_BLOCKING']='1'

from torchvision import transforms
from torch.autograd import Variable

from PIL import Image
import numpy as np

from utils.myTransforms import denorm, norms, detransformcv2

# we use fuse_scheme to choose the corresponding model, 
# choose 0 (IFCNN-MAX) for fusing multi-focus, infrare-visual and multi-modal medical images, 2 (IFCNN-MEAN) for fusing multi-exposure images
fuse_scheme = 0
if fuse_scheme == 0:
    model_name = 'IFCNN-MAX'
elif fuse_scheme == 1:
    model_name = 'IFCNN-SUM'
elif fuse_scheme == 2:
    model_name = 'IFCNN-MEAN'
else:
    model_name = 'IFCNN-MAX'

# load pretrained model
model = myIFCNN(fuse_scheme=fuse_scheme)
model.load_state_dict(torch.load('snapshots/'+ model_name + '.pth'))
model.eval()
model = model.cuda()

from utils.myDatasets import ImageSequence

is_save = True  # Whether to save the results
is_gray = False  # Color (False) or Gray (True)
is_folder = False  # one parameter in ImageSequence
# mean = [0.485, 0.456, 0.406]  # Color (False) or Gray (True)
# std = [0.229, 0.224, 0.225]
mean = [0.043107, 0.050852, 0.055146]  # dominant
std = [0.015259, 0.015895, 0.017594]

parent_folder = '/nvme2/user9/OHS_MS/testnofused'

# （11group：group_0 to group_10）
all_groups = [f'group_{i}' for i in range(11)]
first_group = 'group_0'

# dominant
first_group_path = os.path.join(parent_folder, first_group)

# copy dominant group to fused file manually

for i in range(1, 11):
    group_name = f'group_{i}'
    dataset2_path = os.path.join(parent_folder, group_name)
    dataset3_path = '/nvme2/user9/OHS_MS/testfusedifcnn'
    output_path = os.path.join(dataset3_path, f'fused_{i}')

    print(f"fusion {first_group} and {group_name} -> {output_path}")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for category in os.listdir(first_group_path):
        cat_path1 = os.path.join(first_group_path, category)
        cat_path2 = os.path.join(dataset2_path, category)

        if not os.path.isdir(cat_path1) or not os.path.isdir(cat_path2):
            continue

        output_cat_path = os.path.join(output_path, category)
        if not os.path.exists(output_cat_path):
            os.makedirs(output_cat_path)

        images1 = [f for f in os.listdir(cat_path1) if f.endswith('.bmp') or f.endswith('.png')]
        images2 = [f for f in os.listdir(cat_path2) if f.endswith('.bmp') or f.endswith('.png')]

        min_count = min(len(images1), len(images2))
        images1 = sorted(images1)[:min_count]
        images2 = sorted(images2)[:min_count]

        for img_name1, img_name2 in zip(images1, images2):

            path1 = os.path.join(cat_path1, img_name1)
            path2 = os.path.join(cat_path2, img_name2)

            if not os.path.exists(path1) or not os.path.exists(path2):
                continue

            output_filename = img_name1
            if output_filename.endswith('.bmp'):
                output_filename = output_filename.replace('.bmp', '.png')
            saveroot = os.path.join(output_cat_path, output_filename)

            paths = [path1, path2]

            filename = f'{category}_{img_name1}'
            seq_loader = ImageSequence(is_folder, 'RGB', transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]), *paths)
            imgs = seq_loader.get_imseq()

            with torch.no_grad():
                vimgs = []
                for idx, img in enumerate(imgs):
                    img.unsqueeze_(0)
                    vimgs.append(Variable(img.cuda()))
                vres = model(*vimgs)
                res = denorm(mean, std, vres[0]).clamp(0, 1) * 255
                res_img = res.cpu().data.numpy().astype('uint8')
                img_result = Image.fromarray(res_img.transpose([1, 2, 0]))

            if is_save:
                if is_gray:
                    img_result.convert('L').save(saveroot, format='PNG', compress_level=0)
                else:
                    img_result.save(saveroot, format='PNG', compress_level=0)

    print(f"fusion {first_group} and {group_name}")

print("Fusion complete!")
