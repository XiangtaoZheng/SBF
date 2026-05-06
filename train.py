import os
import torchvision
import torch.nn as nn
import torch.optim as optim
import gc
import torch
from tqdm import tqdm
from torchvision import transforms, datasets
import json
import math


def get_band_means_stds(all_means, all_stds, model_idx, total_models):
    """
    根据模型索引获取对应的均值和方差
    total_models: 总共需要训练多少个模型
    model_idx: 当前模型的索引（从1开始）
    """
    n = len(all_means)

    # 如果波段数能被3整除，均匀分配
    if n % 3 == 0:
        band_per_model = 3
        start_idx = (model_idx - 1) * band_per_model
        end_idx = start_idx + band_per_model

        # 确保不越界
        if end_idx > n:
            end_idx = n

        model_means = all_means[start_idx:end_idx]
        model_stds = all_stds[start_idx:end_idx]
    else:
        # 如果不能被3整除，最后一个模型使用最后3个波段
        if model_idx < total_models:
            # 前几个模型均匀分配
            band_per_model = 3
            start_idx = (model_idx - 1) * band_per_model
            end_idx = start_idx + band_per_model
        else:
            # 最后一个模型使用最后3个波段
            start_idx = n - 3
            end_idx = n

        model_means = all_means[start_idx:end_idx]
        model_stds = all_stds[start_idx:end_idx]

    return model_means, model_stds


def train_model(base_dir_train, save_path, device, model_idx=None, all_means=None, all_stds=None, total_models=11):
    """训练单个模型，返回模型信息但不保留模型对象"""
    print(f"\n{'=' * 60}")
    print(f"Training Model {model_idx if model_idx else ''} on dataset: {base_dir_train}")
    print(f"Saving to: {save_path}")
    print(f"{'=' * 60}")

    # 获取当前模型对应的均值和方差
    if all_means is not None and all_stds is not None and model_idx is not None:
        model_means, model_stds = get_band_means_stds(all_means, all_stds, model_idx, total_models)
        print(f"Model {model_idx} using means: {model_means}")
        print(f"Model {model_idx} using stds: {model_stds}")

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomChoice([
               transforms.RandomRotation(degrees=(0,0)),
               transforms.RandomRotation(degrees=(90,90)),
               transforms.RandomRotation(degrees=(180,180)),
               transforms.RandomRotation(degrees=(270,270))
                                   ]),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(model_means, model_stds)]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(model_means, model_stds)])}

    batch_size = 64

    train_dataset = datasets.ImageFolder(root=base_dir_train, transform=data_transform['train'])
    train_num = len(train_dataset)

    # 如果是第一个模型，保存类别索引
    if model_idx == 1:
        scene_list = train_dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in scene_list.items())
        json_str = json.dumps(cla_dict, indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=base_dir_train, transform=data_transform['val'])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    print(f"Using {train_num} images for training, {val_num} images for validation.")

    net = torchvision.models.swin_v2_b(weights=True)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.0001)

    epochs = 60
    best_acc = 0.0
    train_steps = len(train_loader)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0

        for step, data in enumerate(tqdm(train_loader, desc=f"Model {model_idx} - Epoch {epoch + 1}")):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        net.eval()
        acc = 0.0
        with torch.no_grad():
            for val_data in tqdm(validate_loader, desc="Validating"):
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print(
            f'[Model {model_idx} Epoch {epoch + 1}] Train Loss: {running_loss / train_steps:.3f}, Val Accuracy: {val_accurate:.3f}')

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            print(f'Model saved with accuracy: {best_acc:.3f}')
        if val_accurate == 1:
            break
    print(f'Finished Training Model {model_idx}')

    return {
        'model_path': save_path,
        'accuracy': best_acc,
        'model_idx': model_idx,
        'means': model_means,
        'stds': model_stds
    }


def main():
    """主函数：训练多个模型并进行权重优化"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # 这里使用示例值，需要替换为实际计算的值
    all_means = [0.093717, 0.106012, 0.098394, 0.254145, 0.224985, 0.216232, 0.230480, 0.260408, 0.259068,
                 0.229231, 0.241858, 0.252416, 0.226055, 0.249143, 0.241474, 0.210571, 0.228882, 0.236008,
                 0.291387, 0.226586, 0.304752, 0.304627, 0.289448, 0.243154, 0.253498, 0.243998, 0.226220,
                 0.282287, 0.294700, 0.233003, 0.234874, 0.210449, 0.184403
]
    all_stds = [0.032271, 0.034158, 0.032637, 0.035663, 0.039308, 0.039440, 0.042536, 0.047685, 0.047473,
                0.045605, 0.051311, 0.054917, 0.052689, 0.059189, 0.059367, 0.047386, 0.055134, 0.056277,
                0.060127, 0.051464, 0.063112, 0.060267, 0.059465, 0.051482, 0.050392, 0.050228, 0.045531,
                0.053061, 0.057810, 0.046762, 0.045986, 0.043450, 0.037206
]

    print(f"Total bands: {len(all_means)}")
    print(f"First few means: {all_means[:6]}")
    print(f"First few stds: {all_stds[:6]}")

    # 配置数据路径和模型保存路径
    model_save_paths = [
        '/nvme2/user9/OHS_MS/model_0.pth',
        '/nvme2/user9/OHS_MS/model_1.pth',
        '/nvme2/user9/OHS_MS/model_2.pth',
        '/nvme2/user9/OHS_MS/model_3.pth',
        '/nvme2/user9/OHS_MS/model_4.pth',
        '/nvme2/user9/OHS_MS/model_5.pth',
        '/nvme2/user9/OHS_MS/model_6.pth',
        '/nvme2/user9/OHS_MS/model_7.pth',
        '/nvme2/user9/OHS_MS/model_8.pth',
        '/nvme2/user9/OHS_MS/model_9.pth',
        '/nvme2/user9/OHS_MS/model_10.pth',
    ]
    base_dirs = [
        '/nvme2/user9/OHS_MS/trainfusedifcnn/fused_0',
        '/nvme2/user9/OHS_MS/trainfusedifcnn/fused_1',
        '/nvme2/user9/OHS_MS/trainfusedifcnn/fused_2',
        '/nvme2/user9/OHS_MS/trainfusedifcnn/fused_3',
        '/nvme2/user9/OHS_MS/trainfusedifcnn/fused_4',
        '/nvme2/user9/OHS_MS/trainfusedifcnn/fused_5',
        '/nvme2/user9/OHS_MS/trainfusedifcnn/fused_6',
        '/nvme2/user9/OHS_MS/trainfusedifcnn/fused_7',
        '/nvme2/user9/OHS_MS/trainfusedifcnn/fused_8',
        '/nvme2/user9/OHS_MS/trainfusedifcnn/fused_9',
        '/nvme2/user9/OHS_MS/trainfusedifcnn/fused_10',
    ]

    total_models = len(base_dirs)
    print(f"Total models to train: {total_models}")

    # 训练所有模型
    trained_models_info = []

    for i, (base_dir, save_path) in enumerate(zip(base_dirs, model_save_paths)):
        model_idx = i + 1

        # 检查模型是否已经存在
        if os.path.exists(save_path):
            print(f"\nModel {model_idx} already exists at {save_path}. Skipping training...")
            # 即使模型已存在，也记录它使用的均值和方差
            model_means, model_stds = get_band_means_stds(all_means, all_stds, model_idx, total_models)
            model_info = {
                'model_path': save_path,
                'accuracy': 0.0,  
                'model_idx': model_idx,
                'means': model_means,
                'stds': model_stds
            }
            trained_models_info.append(model_info)
            continue

        torch.cuda.empty_cache()
        gc.collect()

        # 训练新模型
        model_info = train_model(base_dir, save_path, device,
                                 model_idx=model_idx,
                                 all_means=all_means,
                                 all_stds=all_stds,
                                 total_models=total_models)
        trained_models_info.append(model_info)

        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'=' * 60}")
    print("All models trained/loaded successfully!")
    print(f"{'=' * 60}")

    # 显示模型信息
    for info in trained_models_info:
        print(f"Model {info['model_idx']}: {info['model_path']} - Accuracy: {info['accuracy']:.3f}")
        print(f"  Means: {info['means']}")
        print(f"  Stds: {info['stds']}")


if __name__ == '__main__':
    main()
