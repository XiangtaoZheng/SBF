import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from mycode import dataset
import numpy as np
from PIL import Image
import os

src_data_folder = "/nvme2/user9/OHS_MS/test"
class_names = os.listdir(src_data_folder)
class_names = sorted(class_names)

image_data = dataset.MyImageFolder(r'/nvme2/user9/OHS_MS/test',
                                   'ohs', 'tif')
pic_test = DataLoader(image_data, batch_size=1, shuffle=None)

# band number
num_bands = 32

num_groups = (num_bands + 2) // 3  

base_save_path = r'/nvme2/user9/OHS_MS/testnofused'

for group_idx in range(num_groups):
    group_folder = os.path.join(base_save_path, f'group_{group_idx}')
    for class_name in class_names:
        class_folder = os.path.join(group_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)

group_counters = {}

for class_idx in range(len(class_names)):
    group_counters[class_idx] = [0] * num_groups

for i, data_ in tqdm(enumerate(pic_test)):

    img_test = torch.squeeze(data_[0], 0)
    label = data_[1].item()

    img_test_np = img_test.numpy()

    if img_test_np.shape[0] == num_bands:
        channel_first = True
    elif img_test_np.shape[2] == num_bands:
        channel_first = False
    else:
        channel_first = True
        print(f"Warning: Unable to determine the shape of the image: {img_test_np.shape}")

    for group_idx in range(num_groups):

        start_idx = group_idx * 3
        if start_idx + 3 <= num_bands:
            band_indices = [start_idx, start_idx + 1, start_idx + 2]
        else:
            band_indices = [num_bands - 3, num_bands - 2, num_bands - 1]

        if channel_first:
            # [C, H, W]
            selected_bands = img_test_np[band_indices, :, :]
        else:
            # [H, W, C]
            selected_bands = img_test_np[:, :, band_indices]

        image_numpy = (selected_bands * 255.0).astype(np.uint8)

        # to PIL
        if image_numpy.shape[0] == 3:
            # [3, H, W] to [H, W, 3]
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
        elif image_numpy.shape[2] == 3:
            pass
        else:
            print(f"Warning: The image shape is incorrect: {image_numpy.shape}")
            continue

        image_pil = Image.fromarray(image_numpy)

        group_counters[label][group_idx] += 1
        s = group_counters[label][group_idx]

        save_path = os.path.join(
            base_save_path,
            f'group_{group_idx}',
            class_names[label],
            f'{class_names[label]}_{s}_0.bmp'
        )
        image_pil.save(save_path)

print("Band extraction completed!")
print(f"A total of {num_groups} band combination folders have been created:")
for group_idx in range(num_groups):
    start_idx = group_idx * 3
    if start_idx + 3 <= num_bands:
        bands = f"{start_idx}-{start_idx + 2}"
    else:
        bands = f"{num_bands - 3}-{num_bands - 1}"
    print(f"  Group {group_idx}: Bands {bands}")
