import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
from skimage.metrics import structural_similarity as ssim
from skimage import color
import cv2
import torch.nn.functional as F
import torch

# 尝试导入pyiqa库
try:
    import pyiqa
    PYIQA_AVAILABLE = True
except ImportError:
    print("警告: pyiqa库未安装")
    print("请安装: pip install pyiqa")
    PYIQA_AVAILABLE = False
    pyiqa = None

warnings.filterwarnings('ignore')


# ==================== 图像质量评估函数 ====================

def compute_niqe_score(img):
    """
    使用pyiqa计算图像的NIQE (Natural Image Quality Evaluator) 得分

    参数:
        img: 输入图像 (H, W, 3)

    返回:
        niqe_score: NIQE得分 (越低表示质量越好)
    """
    # 创建NIQE评估器
    niqe_metric = pyiqa.create_metric('niqe', device='cpu')

    # 转换为RGB格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb / 255.0

    # 从(H, W, C)转换为(C, H, W)，然后添加批次维度
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float()

    # 检查图像尺寸 - NIQE要求图像至少96x96
    _, _, h, w = img_tensor.shape
    if h < 96 or w < 96:
        # 使用双线性插值上采样到至少96x96
        scale_h = max(96, h)
        scale_w = max(96, w)

        # 如果两个维度都需要调整，保持宽高比
        if scale_h != h or scale_w != w:
            # 计算保持宽高比的尺寸
            if h < 96 and w < 96:
                # 两个维度都太小，等比例放大
                scale_factor = max(96 / h, 96 / w)
                new_h = int(h * scale_factor)
                new_w = int(w * scale_factor)
            elif h < 96:
                # 只有高度太小
                new_h = 96
                new_w = w
            else:  # w < 96
                # 只有宽度太小
                new_h = h
                new_w = 96

            # 进行上采样
            img_tensor = F.interpolate(
                img_tensor,
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            )

    # 计算NIQE得分
    niqe_score = niqe_metric(img_tensor).item()
    return niqe_score


def compute_piqe_score(img):
    """
    使用pyiqa计算图像的PIQE (Perception-based Image Quality Evaluator) 得分

    参数:
        img: 输入图像 (H, W, 3)

    返回:
        piqe_score: PIQE得分 (越低表示质量越好)
    """
    # 创建PIQE评估器
    piqe_metric = pyiqa.create_metric('piqe', device='cpu')
    # 转换为RGB格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb / 255.0

    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float()

    # 计算PIQE得分
    piqe_score = piqe_metric(img_tensor).item()
    return piqe_score


# ==================== 图像加载函数 ====================

def load_images_from_groups(base_dirs):
    """
    从多个组文件夹加载所有图像

    参数:
        base_dirs: 列表，包含每个组的文件夹路径

    返回:
        group_images: 列表，每个元素是一个组的图像列表
        image_filenames: 列表，每个组的图像文件名列表
    """
    group_images = []
    image_filenames = []

    for group_dir in base_dirs:
        print(f"加载组: {os.path.basename(group_dir)}")

        # 查找所有类别文件夹
        class_dirs = [os.path.join(group_dir, d) for d in os.listdir(group_dir)
                      if os.path.isdir(os.path.join(group_dir, d))]

        images = []
        filenames = []

        for class_dir in class_dirs:
            # 查找所有图像文件
            image_files = [f for f in os.listdir(class_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                try:
                    # 读取图像并转换为numpy数组
                    img = Image.open(img_path)
                    img_array = np.array(img)

                    # 确保是三通道图像
                    if len(img_array.shape) == 2:
                        # 灰度图转三通道
                        img_array = np.stack([img_array] * 3, axis=-1)
                    elif img_array.shape[2] == 4:
                        # RGBA转RGB
                        img_array = img_array[:, :, :3]
                    elif img_array.shape[2] == 1:
                        # 单通道转三通道
                        img_array = np.stack([img_array[:, :, 0]] * 3, axis=-1)

                    # 添加到列表
                    images.append(img_array)
                    filenames.append(os.path.join(os.path.basename(class_dir), img_file))
                except Exception as e:
                    print(f"警告: 无法读取图像 {img_path}: {e}")

        group_images.append(images)
        image_filenames.append(filenames)

        print(f"  -> 已加载 {len(images)} 张图像")

    return group_images, image_filenames


# ==================== 质量指标计算函数 ====================

def compute_self_variance(group_images):
    """
    计算每组图像的自身方差

    参数:
        group_images: 列表，每个元素是一个组的图像列表

    返回:
        self_variances: 列表，每个组的平均方差
    """
    n_groups = len(group_images)
    self_variances = np.zeros(n_groups)

    for i in range(n_groups):
        if len(group_images[i]) == 0:
            continue

        total_variance = 0

        for img in group_images[i]:
            channel_variances = []
            for c in range(3):  # 遍历RGB三个通道
                channel_data = img[:, :, c].flatten()
                channel_variances.append(np.var(channel_data))
            total_variance += np.mean(channel_variances)

        self_variances[i] = total_variance / len(group_images[i])

    return self_variances


def compute_quality_metrics(img):
    """
    计算单张图像的综合质量指标
    包括方差、NIQE和PIQE

    参数:
        img: 输入图像 (H, W, 3)

    返回:
        metrics: 包含方差、NIQE、PIQE的字典
    """

    # 计算NIQE
    niqe_score = compute_niqe_score(img)

    # 计算PIQE
    piqe_score = compute_piqe_score(img)

    return {
        'niqe': niqe_score,
        'piqe': piqe_score
    }


def compute_group_qualities(group_images,group_variances):
    """
    计算每组图像的平均质量指标

    参数:
        group_images: 列表，每个元素是一个组的图像列表

    返回:
        group_qualities: 列表，每个组的平均质量指标
    """
    n_groups = len(group_images)
    group_qualities = []

    for i in range(n_groups):
        if len(group_images[i]) == 0:
            group_qualities.append({
                'variance': 0,
                'niqe': 0,
                'piqe': 0
            })
            continue

        # 采样以减少计算量
        sample_size = min(100, len(group_images[i]))
        indices = np.random.choice(len(group_images[i]), sample_size, replace=False)

        total_niqe = 0
        total_piqe = 0

        for idx in indices:
            img = group_images[i][idx]
            metrics = compute_quality_metrics(img)
            total_niqe += metrics['niqe']
            total_piqe += metrics['piqe']

        # 计算平均值
        avg_niqe = total_niqe / sample_size
        avg_piqe = total_piqe / sample_size

        group_qualities.append({
            'variance': group_variances[i],
            'niqe': avg_niqe,
            'piqe': avg_piqe
        })

    return group_qualities


# ==================== 差异指标计算函数 ====================

def compute_structural_similarity(img1, img2):
    """
    计算两个图像之间的结构相似性指数（SSIM）

    参数:
        img1, img2: 输入图像 (H, W, 3)

    返回:
        ssim_value: SSIM值 (0-1之间，1表示完全相同)
    """
    # 转换为灰度图像计算SSIM
    gray1 = color.rgb2gray(img1)
    gray2 = color.rgb2gray(img2)

    # 计算SSIM
    ssim_value = ssim(gray1, gray2, data_range=1.0)

    return ssim_value


def compute_mse(img1, img2):
    """
    计算两个图像之间的NMSE

    参数:
        img1, img2: 输入图像 (H, W, 3)

    返回:
        mse: MSE值
    """
    # 确保图像是float类型
    img1_float = img1.astype(np.float64)
    img2_float = img2.astype(np.float64)

    # 计算MSE
    mse = np.mean((img1_float - img2_float) ** 2)
    nmse = 2 * mse / (np.mean(img1_float ** 2) + np.mean(img2_float ** 2))

    return nmse


def compute_group_ssim_and_mse(group_images):
    """
    计算每组图像与其他组的平均SSIM和MSE

    参数:
        group_images: 列表，每个元素是一个组的图像列表

    返回:
        avg_ssim_matrix: 矩阵，每对组之间的平均SSIM
        avg_mse_matrix: 矩阵，每对组之间的平均MSE
        group_avg_ssim: 列表，每个组与其他组的平均SSIM
        group_avg_mse: 列表，每个组与其他组的平均MSE
    """
    n_groups = len(group_images)
    avg_ssim_matrix = np.zeros((n_groups, n_groups))
    avg_mse_matrix = np.zeros((n_groups, n_groups))
    group_avg_ssim = np.zeros(n_groups)
    group_avg_mse = np.zeros(n_groups)

    # 随机采样以减少计算量
    sample_size = min(100, min([len(g) for g in group_images if len(g) > 0]))

    for i in range(n_groups):
        if len(group_images[i]) == 0:
            continue

        for j in range(i + 1, n_groups):
            if len(group_images[j]) == 0:
                continue

            # 从每组中随机选择图像进行差异计算
            indices_i = np.random.choice(len(group_images[i]),
                                         min(sample_size, len(group_images[i])),
                                         replace=False)
            indices_j = np.random.choice(len(group_images[j]),
                                         min(sample_size, len(group_images[j])),
                                         replace=False)

            total_ssim = 0
            total_mse = 0
            count = 0

            for idx_i in indices_i:
                img_i = group_images[i][idx_i]

                for idx_j in indices_j:
                    img_j = group_images[j][idx_j]

                    # 计算SSIM
                    ssim_val = compute_structural_similarity(img_i, img_j)
                    total_ssim += ssim_val

                    # 计算MSE
                    mse_val = compute_mse(img_i, img_j)
                    total_mse += mse_val

                    count += 1

            if count > 0:
                avg_ssim = total_ssim / count
                avg_mse = total_mse / count

                avg_ssim_matrix[i, j] = avg_ssim
                avg_ssim_matrix[j, i] = avg_ssim
                avg_mse_matrix[i, j] = avg_mse
                avg_mse_matrix[j, i] = avg_mse

    # 计算每个组与其他组的平均SSIM和MSE
    for i in range(n_groups):
        valid_ssim = [avg_ssim_matrix[i, j] for j in range(n_groups)
                      if j != i and avg_ssim_matrix[i, j] > 0]
        valid_mse = [avg_mse_matrix[i, j] for j in range(n_groups)
                     if j != i and avg_mse_matrix[i, j] > 0]

        if valid_ssim:
            group_avg_ssim[i] = np.mean(valid_ssim)
        if valid_mse:
            group_avg_mse[i] = np.mean(valid_mse)

    return avg_ssim_matrix, avg_mse_matrix, group_avg_ssim, group_avg_mse


# ==================== 标准化和归一化函数 ====================

def z_score_normalization(data):
    """
    对数据进行Z-score标准化

    参数:
        data: 输入数据

    返回:
        normalized_data: Z-score标准化后的数据
    """
    mean_val = np.mean(data)
    std_val = np.std(data)
    if std_val > 0:
        normalized_data = (data - mean_val) / std_val
    else:
        normalized_data = np.zeros_like(data)
    return normalized_data


def min_max_normalization(data):
    """
    对数据进行min-max归一化到[0,1]范围

    参数:
        data: 输入数据

    返回:
        normalized_data: 归一化后的数据
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val > min_val:
        normalized_data = (data - min_val) / (max_val - min_val + 1e-10)
    else:
        normalized_data = np.zeros_like(data)
    return normalized_data


def sigmoid_normalization(data):
    """
    对数据进行Sigmoid归一化到(0,1)范围

    参数:
        data: 输入数据

    返回:
        normalized_data: Sigmoid归一化后的数据
    """

    normalized_data = 1 / (1 + np.exp(-data))

    return normalized_data

# ==================== 主导得分计算函数 ====================

def compute_dominance_scores(group_qualities, group_avg_ssim, group_avg_mse):
    """
    计算主导图像适宜性得分
    对质量指标使用Z-score标准化后相加得到综合质量指标，然后归一化
    对差异指标直接归一化

    参数:
        group_qualities: 列表，每个组的质量指标
        group_avg_ssim: 列表，每个组与其他组的平均SSIM
        group_avg_mse: 列表，每个组与其他组的平均MSE

    返回:
        dominance_scores: 列表，每个组的主导图像得分
        normalized_metrics: 列表，每个组的归一化指标
    """
    n_groups = len(group_qualities)
    dominance_scores = np.zeros(n_groups)
    epsilon = 0.2

    # 提取各质量指标
    variances = np.array([q['variance'] for q in group_qualities])
    niqe_scores = np.array([q['niqe'] for q in group_qualities])
    piqe_scores = np.array([q['piqe'] for q in group_qualities])

    # 计算SSIM差异 (1-SSIM)
    ssim_differences = 1 - group_avg_ssim

    # 对质量指标进行Z-score标准化
    z_variances = z_score_normalization(variances)
    z_niqe = z_score_normalization(niqe_scores)
    z_piqe = z_score_normalization(piqe_scores)

    norm_variances = sigmoid_normalization(z_variances)
    norm_niqe = sigmoid_normalization(z_niqe)
    norm_piqe = sigmoid_normalization(z_piqe)
    combined_quality_raw = norm_variances + norm_niqe + norm_piqe
    norm_combined_quality = combined_quality_raw / 3

    norm_mse = group_avg_mse
    # 计算综合差异指标 (等权重组合SSIM差异和MSE)
    combined_difference = (ssim_differences + norm_mse) / 2

    # 计算主导得分 S_i = D_i / (Q_i + ε)
    for i in range(n_groups):
        if norm_combined_quality[i] >= 0 and combined_difference[i] >= 0:
            safe_quality = norm_combined_quality[i] + epsilon
            dominance_scores[i] = combined_difference[i] / safe_quality

    # 收集归一化后的指标
    normalized_metrics = []
    for i in range(n_groups):
        normalized_metrics.append({
            'z_variance': norm_variances[i],
            'norm_niqe': norm_niqe[i],
            'norm_piqe': norm_piqe[i],
            'combined_quality_raw': combined_quality_raw[i],
            'norm_combined_quality': norm_combined_quality[i],
            'ssim_differences': ssim_differences[i],
            'norm_mse': norm_mse[i],
            'combined_difference': combined_difference[i],
            'dominance_score': dominance_scores[i]
        })

    return dominance_scores, normalized_metrics


# ==================== 分析函数 ====================

def analyze_dominance_patterns(group_qualities, group_avg_ssim, group_avg_mse,
                               dominance_scores, normalized_metrics):
    """
    分析主导图像选择模式

    参数:
        group_qualities: 列表，每个组的质量指标
        group_avg_ssim: 列表，每个组与其他组的平均SSIM
        group_avg_mse: 列表，每个组与其他组的平均MSE
        dominance_scores: 列表，每个组的主导图像得分
        normalized_metrics: 列表，每个组的归一化指标

    返回:
        analysis: 分析结果字典
    """
    n_groups = len(group_qualities)

    # 找到最佳和最差主导图像组
    valid_scores = [(i, dominance_scores[i]) for i in range(n_groups)
                    if dominance_scores[i] > 0]

    if not valid_scores:
        return None

    best_group, best_score = max(valid_scores, key=lambda x: x[1])

    # 提取原始指标
    variances = np.array([q['variance'] for q in group_qualities])
    niqe_scores = np.array([q['niqe'] for q in group_qualities])
    piqe_scores = np.array([q['piqe'] for q in group_qualities])

    # 计算SSIM差异
    ssim_differences = 1 - group_avg_ssim

    # 提取归一化指标
    combined_qualities = np.array([m['norm_combined_quality'] for m in normalized_metrics])
    combined_differences = np.array([m['combined_difference'] for m in normalized_metrics])

    analysis = {
        'best_group': best_group,
        'best_score': best_score,
        'variances': variances,
        'niqe_scores': niqe_scores,
        'piqe_scores': piqe_scores,
        'ssim_values': group_avg_ssim,
        'ssim_differences': ssim_differences,
        'mse_values': group_avg_mse,
        'combined_qualities': combined_qualities,
        'combined_differences': combined_differences,
        'dominance_scores': dominance_scores
    }

    print("\n" + "=" * 70)
    print("=== 主导图像分析结果===")
    print("=" * 70)
    print(f"最佳主导组: 组 {best_group} (得分: {best_score:.6f})")

    return analysis


# ==================== 可视化函数 ====================

def plot_metrics(analysis):
    """
    绘制归一化指标对比图

    参数:
        analysis: 分析结果字典
    """
    n_groups = len(analysis['dominance_scores'])
    groups = list(range(n_groups))

    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))

    # 归一化主导得分
    norm_scores = min_max_normalization(analysis['dominance_scores'])
    norm_qualities = min_max_normalization(analysis['combined_qualities'])
    norm_differences = min_max_normalization(analysis['combined_differences'])
    print(norm_differences)
    print(norm_qualities)
    print(norm_scores)

    # 绘制所有归一化指标
    ax.plot(groups, norm_scores, 'k-', linewidth=3, label='Dominance Score (S_i)')
    ax.plot(groups, norm_qualities, 'm-', linewidth=2, label='Image Quality (Q_i)')
    ax.plot(groups, norm_differences, 'c-', linewidth=2, label='Inter-group Difference (D_i)')

    # 设置标签和标题
    ax.set_xlabel('Group Index', fontsize=22, fontweight='bold')
    ax.set_ylabel('Normalized Value', fontsize=22, fontweight='bold')
    ax.set_title(
        'Normalized Comparison of All Metrics',
        fontsize=24, fontweight='bold')

    # 添加网格和图例
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=16, loc='best')

    # 设置x轴刻度
    ax.set_xticks(groups)

    plt.tight_layout()

    # 保存图形
    plt.savefig('dominance_analysis_hsrs4.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存为 'dominance_analysis_hsrs4.png'")

    plt.show()


# ==================== 主函数 ====================

def main():
    # 输入路径
    base_dirs = [
        '/nvme2/user9/hsrs/trainnofused/group_0',
        '/nvme2/user9/hsrs/trainnofused/group_1',
        '/nvme2/user9/hsrs/trainnofused/group_2',
        '/nvme2/user9/hsrs/trainnofused/group_3',
        '/nvme2/user9/hsrs/trainnofused/group_4',
        '/nvme2/user9/hsrs/trainnofused/group_5',
        '/nvme2/user9/hsrs/trainnofused/group_6',
        '/nvme2/user9/hsrs/trainnofused/group_7',
        '/nvme2/user9/hsrs/trainnofused/group_8',
        '/nvme2/user9/hsrs/trainnofused/group_9',
        '/nvme2/user9/hsrs/trainnofused/group_10',
        '/nvme2/user9/hsrs/trainnofused/group_11',
        '/nvme2/user9/hsrs/trainnofused/group_12',
        '/nvme2/user9/hsrs/trainnofused/group_13',
        '/nvme2/user9/hsrs/trainnofused/group_14',
        '/nvme2/user9/hsrs/trainnofused/group_15',
    ]

    print("开始加载图像数据...")
    print("=" * 70)

    # 1. 加载图像
    group_images, image_filenames = load_images_from_groups(base_dirs)

    # 检查每个组是否有足够的图像
    for i, images in enumerate(group_images):
        if len(images) == 0:
            print(f"错误: 组 {i} 没有找到图像")
            return

    print("\n" + "=" * 70)
    print("图像加载完成!")
    print(f"总组数: {len(group_images)}")

    # 2. 计算组方差（使用所有图像）
    print("\n计算各组方差...")
    print("=" * 70)
    print("计算组方差...")
    group_variances = compute_self_variance(group_images)

    # 输出方差结果
    print("\n各组方差:")
    print(f"{'组':<6} {'方差':<12}")
    print("-" * 30)
    for i, var in enumerate(group_variances):
        print(f"{i:<6} {var:<12.6f}")

    # 3. 计算其他质量指标（NIQE和PIQE）
    print("\n计算各组NIQE和PIQE指标...")
    print("=" * 70)
    print("计算NIQE和PIQE...")
    group_qualities = compute_group_qualities(group_images, group_variances)

    # 输出质量结果
    print("\n各组质量指标:")
    print(f"{'组':<6} {'方差':<12} {'NIQE':<12} {'PIQE':<12}")
    print("-" * 50)
    for i, q in enumerate(group_qualities):
        print(f"{i:<6} {q['variance']:<12.6f} {q['niqe']:<12.6f} {q['piqe']:<12.6f}")

    # 3. 计算组间SSIM和MSE
    print("\n计算组间SSIM和MSE...")
    print("=" * 70)
    avg_ssim_matrix, avg_mse_matrix, group_avg_ssim, group_avg_mse = \
        compute_group_ssim_and_mse(group_images)

    # 输出SSIM和MSE结果
    print("\n各组与其他组的平均SSIM和MSE:")
    print(f"{'组':<6} {'平均SSIM':<12} {'平均MSE':<12}")
    print("-" * 50)
    for i in range(len(group_avg_ssim)):
        print(f"{i:<6} {group_avg_ssim[i]:<12.6f} {group_avg_mse[i]:<12.6f}")

    # 4. 计算主导得分（对质量指标使用Z-score标准化后相加，对差异指标直接归一化）
    print("\n计算主导得分（质量指标Z-score标准化后相加，差异指标直接归一化）...")
    print("=" * 70)
    dominance_scores, normalized_metrics = compute_dominance_scores(
        group_qualities, group_avg_ssim, group_avg_mse)

    # 输出得分结果
    print("\n主导得分 (S_i = D_i / (Q_i + ε)):")
    print(f"{'组':<6} {'质量Q_i':<12} {'差异D_i':<12} {'得分S_i':<12}")
    print("-" * 50)
    for i, score in enumerate(dominance_scores):
        if score > 0:
            q_i = normalized_metrics[i]['norm_combined_quality']
            d_i = normalized_metrics[i]['combined_difference']
            print(f"{i:<6} {q_i:<12.6f} {d_i:<12.6f} {score:<12.6f}")

    # 5. 分析主导模式
    print("\n分析主导图像选择模式...")
    analysis = analyze_dominance_patterns(
        group_qualities, group_avg_ssim, group_avg_mse,
        dominance_scores, normalized_metrics)

    # 6. 绘制图表
    if analysis:
        print("\n生成图表...")
        plot_metrics(analysis)

    # 7. 详细结果输出
    print("\n" + "=" * 70)
    print("详细结果（质量指标Z-score标准化后相加，差异指标直接归一化）:")
    print("-" * 70)
    print(f"{'组':<6} {'方差':<10} {'NIQE':<10} {'PIQE':<10} "
          f"{'SSIM':<10} {'MSE':<10} {'质量Q_i':<10} {'差异D_i':<10} {'得分S_i':<10} {'排名':<6}")
    print("-" * 70)

    # 计算排名
    valid_scores = [(i, dominance_scores[i]) for i in range(len(dominance_scores))
                    if dominance_scores[i] > 0]
    sorted_scores = sorted(valid_scores, key=lambda x: x[1], reverse=True)
    rank_dict = {group: rank + 1 for rank, (group, _) in enumerate(sorted_scores)}

    for i in range(len(dominance_scores)):
        if dominance_scores[i] > 0:
            rank = rank_dict.get(i, len(dominance_scores) + 1)
            q_i = normalized_metrics[i]['norm_combined_quality']
            d_i = normalized_metrics[i]['combined_difference']
            print(f"{i:<6} {group_qualities[i]['variance']:<10.6f} "
                  f"{group_qualities[i]['niqe']:<10.6f} "
                  f"{group_qualities[i]['piqe']:<10.6f} "
                  f"{group_avg_ssim[i]:<10.6f} "
                  f"{group_avg_mse[i]:<10.6f} "
                  f"{q_i:<10.6f} "
                  f"{d_i:<10.6f} "
                  f"{dominance_scores[i]:<10.6f} "
                  f"{rank:<6}")

    print("-" * 70)

    # 8. 总结表格
    print("\n总结:")
    print("-" * 70)
    if analysis:
        print(f"最佳组: 组 {analysis['best_group']} (S_i = {analysis['best_score']:.6f})")

    print("\n程序执行完成!")


if __name__ == "__main__":
    main()