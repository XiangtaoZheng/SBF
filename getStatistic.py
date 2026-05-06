import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from mycode import dataset
import os
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def getStat(dataset_path):
    """
    计算指定数据集中3波段图像的均值和标准差

    Args:
        dataset_path: 数据集路径

    Returns:
        mean_list: 每个波段的均值列表
        std_list: 每个波段的标准差列表
    """
    train_data = ImageFolder(root=dataset_path, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)

    # 初始化为3个波段（假设所有图像都是3通道RGB图像）
    mean = torch.zeros(3)
    std = torch.zeros(3)

    for X, _ in train_loader:
        for d in range(3):  # 3个波段
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()

    mean.div_(len(train_data))
    std.div_(len(train_data))

    return list(mean.numpy()), list(std.numpy())


def natural_sort_key(name):
    """
    自然排序键函数，确保数字部分按数值大小排序
    例如：group_1, group_2, group_10 而不是 group_1, group_10, group_2
    """
    # 尝试提取名称中的数字部分
    match = re.search(r'(\d+)', name)
    if match:
        # 返回一个元组：第一部分是非数字前缀（小写），第二部分是数字（转换为整数）
        return (name[:match.start()].lower(), int(match.group(1)))
    # 如果没有数字，按原字符串排序
    return (name.lower(), 0)


def get_all_datasets_stats(base_path):
    """
    获取所有数据集的统计信息

    Args:
        base_path: 包含多个数据集的根目录

    Returns:
        all_means: 所有数据集的均值数组（按数据集和波段顺序排列）
        all_stds: 所有数据集的标准差数组（按数据集和波段顺序排列）
        stats_dict: 包含每个数据集统计信息的字典（按名称排序）
    """
    # 获取所有数据集名称并按照自然排序
    dataset_names = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]
    dataset_names.sort(key=natural_sort_key)

    # 初始化结果字典和数组
    stats_dict = {}
    all_means = []
    all_stds = []

    # 按自然排序后的名称处理数据集
    for dataset_name in dataset_names:
        dataset_path = os.path.join(base_path, dataset_name)

        print(f"Processing dataset: {dataset_name}")
        try:
            mean, std = getStat(dataset_path)
            stats_dict[dataset_name] = {
                'mean': mean,
                'std': std
            }
            print(f" Statistical information of dataset {dataset_name}:")
            print(f"    Band mean: {mean}")
            print(f"    Band standard deviation: {std}")

            # 将当前数据集的统计信息添加到总数组中
            all_means.extend(mean)
            all_stds.extend(std)

        except Exception as e:
            print(f"  An error occurred while processing dataset {dataset_name}: {e}")
            # 如果出错，添加3个NaN值作为占位符
            all_means.extend([float('nan')] * 3)
            all_stds.extend([float('nan')] * 3)

    return all_means, all_stds, stats_dict


if __name__ == '__main__':
    # 主函数：处理多个数据集
    base_path = '/nvme2/user9/OHS_MS/trainfusedifcnn/'  # 包含多个数据集的根目录

    if os.path.exists(base_path):
        all_means, all_stds, all_stats = get_all_datasets_stats(base_path)

        # 输出整合后的数组
        print("\n=== Integrated statistical information ===")
        print("The mean array of all datasets (in the order of dataset and band)：")
        mean_str = ', '.join([f"{m:.6f}" for m in all_means])
        print(mean_str)

        print("\nStandard deviation array of all datasets (in the order of dataset and band)：")
        std_str = ', '.join([f"{s:.6f}" for s in all_stds])
        print(std_str)

        print("\n=== Detailed statistical information of each dataset ===")
        sorted_keys = sorted(all_stats.keys(), key=natural_sort_key)
        for dataset_name in sorted_keys:
            stats = all_stats[dataset_name]
            print(f"\n{dataset_name}:")
            for i in range(3):  # 3个波段
                print(f"  band {i + 1}: mean={stats['mean'][i]:.6f}, std={stats['std'][i]:.6f}")
    else:
        print(f"Error: The path {base_path} does not exist")
