from tqdm import tqdm
import torch
import numpy as np
import os
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound, ProbabilityOfImprovement, ExpectedImprovement, GPHedge
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


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


def load_model(model_path, device):
    """加载单个模型"""
    model = torchvision.models.swin_v2_b(weights=True)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model


def create_test_dataset(data_dir, pipeline):
    """创建测试数据集和加载器"""
    test_dataset = datasets.ImageFolder(root=data_dir, transform=pipeline)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_dataset, test_loader


def plot_confusion_matrix(cm, label_names, title, save_path, figsize=(10, 8)):
    """绘制并保存混淆矩阵的通用函数"""
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    plt.imshow(cm_normalized, cmap='Blues')
    plt.title(title, fontsize=14)
    plt.xlabel("Predict label", fontsize=12)
    plt.ylabel("Truth label", fontsize=12)
    plt.yticks(range(len(label_names)), label_names, fontsize=10)
    plt.xticks(range(len(label_names)), label_names, rotation=270, fontsize=10)
    plt.tight_layout()
    plt.colorbar()

    # 添加数值标签
    mask = cm_normalized > 0.005
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            if mask[j, i]:
                color = (1, 1, 1) if cm_normalized[j, i] > 0.5 else (0, 0, 0)
                value = format('%.2f' % cm_normalized[j, i])
                plt.text(i, j, value, verticalalignment='center',
                         horizontalalignment='center', color=color, fontsize=12)

    # 保存图像
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'混淆矩阵已保存至: {save_path}')


class DynamicBayesianOptimizer:
    """动态贝叶斯优化器类 - 每N张图像迭代一次"""

    def __init__(self, num_models, init_weights=None, init_points=5, n_iter_per_update=2, update_frequency=30):
        """
        初始化动态贝叶斯优化器

        Args:
            num_models: 模型数量
            init_weights: 初始权重，如果为None则使用等权重
            init_points: 初始采样点数
            n_iter_per_update: 每次更新的迭代次数
            update_frequency: 每多少张图像更新一次权重
        """
        self.num_models = num_models
        self.init_points = init_points
        self.n_iter_per_update = n_iter_per_update
        self.update_frequency = update_frequency
        self.history = []  # 保存历史数据
        self.iteration = 0
        self.sample_buffer = []  # 缓冲存储样本

        # 初始化权重
        if init_weights is None:
            self.current_weights = np.ones(num_models) / num_models
        else:
            self.current_weights = np.array(init_weights)

        # 定义优化空间
        self.pbounds = {f'w{i}': (0, 1) for i in range(1, num_models + 1)}

        # 创建acquisition函数
        self.acquisition_function = GPHedge(
            base_acquisitions=[
                UpperConfidenceBound(kappa=2),
                ProbabilityOfImprovement(xi=0.01),
                ExpectedImprovement(xi=0.01)
            ],
            random_state=42
        )

        # 创建优化器
        self.optimizer = BayesianOptimization(
            f=self._dummy_eval,
            pbounds=self.pbounds,
            acquisition_function=self.acquisition_function,
            verbose=0,
            random_state=42,
        )

    def _dummy_eval(self, **kwargs):
        """虚拟评估函数，实际不直接使用"""
        return 0.0

    def add_sample(self, sample_data):
        """
        添加样本到缓冲区

        Args:
            sample_data: 字典，包含每个模型的预测概率
        """
        # 确保概率值
        for i in range(self.num_models):
            if f'model_{i}' in sample_data:
                # 确保概率和为1
                prob = sample_data[f'model_{i}']
                prob = prob / prob.sum() if prob.sum() > 0 else prob
                sample_data[f'model_{i}'] = prob

        self.sample_buffer.append(sample_data)
        self.history.append(sample_data)

        # 如果缓冲区达到更新频率，进行优化
        if len(self.sample_buffer) >= self.update_frequency:
            self._update_weights()
            self.sample_buffer = []  # 清空缓冲区

    def _update_weights(self):
        """执行贝叶斯优化更新权重"""
        if len(self.sample_buffer) == 0:
            return

        self.iteration += 1

        # 获取类别数（从第一个样本的第一个模型预测中）
        if len(self.sample_buffer) > 0 and 'model_0' in self.sample_buffer[0]:
            self.num_classes = len(self.sample_buffer[0]['model_0'])

        # 如果还在初始采样阶段
        if len(self.history) <= self.init_points:
            # 随机探索
            random_weights = np.random.dirichlet(np.ones(self.num_models))
            params = {f'w{i + 1}': random_weights[i] for i in range(self.num_models)}

            # 计算随机权重的性能（使用负熵）
            score = self._evaluate_weights(random_weights)

            # 注册到优化器
            self.optimizer.register(params, score)

            if len(self.history) == self.init_points:
                print(f"\n完成初始采样 ({self.init_points} 个样本)")
        else:
            # 基于缓冲区样本创建目标函数
            def batch_eval(**weights_dict):
                """基于缓冲区样本的评估函数"""
                weights = np.array([weights_dict[f'w{i}'] for i in range(1, self.num_models + 1)])
                weights = np.clip(weights, 0, 1)

                # 归一化权重
                if weights.sum() == 0:
                    weights = np.ones(self.num_models) / self.num_models
                else:
                    weights = weights / weights.sum()

                # 使用负熵作为目标
                return self._evaluate_weights(weights)

            # 创建新的优化器实例，避免历史数据干扰
            self.optimizer = BayesianOptimization(
                f=batch_eval,
                pbounds=self.pbounds,
                acquisition_function=self.acquisition_function,
                verbose=0,
                random_state=42,
            )

            # 使用当前权重作为初始点
            current_params = {f'w{i + 1}': self.current_weights[i] for i in range(self.num_models)}
            current_score = batch_eval(**current_params)
            self.optimizer.register(current_params, current_score)

            # 执行贝叶斯优化迭代
            self.optimizer.maximize(
                init_points=0,  # 不使用额外的随机点
                n_iter=self.n_iter_per_update,
            )

            # 获取最佳权重
            if len(self.optimizer._space) > 0:
                best_params = self.optimizer.max['params']
                new_weights = np.array([best_params[f'w{i}'] for i in range(1, self.num_models + 1)])
                new_weights = np.clip(new_weights, 0, 1)

                if new_weights.sum() > 0:
                    new_weights = new_weights / new_weights.sum()

                # 平滑更新权重
                alpha = 0.05  # 学习率
                self.current_weights = alpha * new_weights + (1 - alpha) * self.current_weights
                self.current_weights = self.current_weights / self.current_weights.sum()

                print(f"批量更新 {self.iteration}: 权重更新为 {np.round(self.current_weights, 4)}")
                print(f"目标函数值（负熵）: {self.optimizer.max['target']:.6f}")

        return self.current_weights

    def _evaluate_weights(self, weights):
        """使用负熵作为目标函数（无监督优化）"""
        if len(self.sample_buffer) == 0:
            return 0.0

        total_negative_entropy = 0.0

        for data in self.sample_buffer:
            # 计算加权集成预测
            weighted_proba = np.zeros(self.num_classes)  # 需要知道类别数

            for i in range(self.num_models):
                weighted_proba += weights[i] * data[f'model_{i}']

            # 确保概率和为1
            weighted_proba = weighted_proba / weighted_proba.sum()

            # 计算负熵：-Σ p_i * log(p_i)
            epsilon = 1e-10  # 避免log(0)
            entropy = -np.sum(weighted_proba * np.log(weighted_proba + epsilon))
            negative_entropy = -entropy  # 我们希望最大化负熵（最小化不确定性）

            total_negative_entropy += negative_entropy

        # 返回平均负熵
        return total_negative_entropy / len(self.sample_buffer)

    def finalize(self):
        """处理剩余的缓冲区样本"""
        if len(self.sample_buffer) > 0:
            self._update_weights()


def predict_image_with_dynamic_bo():
    device = torch.device('cuda:2' if torch.cuda.is_available() else "cpu")
    CPU = torch.device('cpu')

    # 模型路径列表
    model_paths = [
        '/nvme2/user9/hscohs/model_0.pth',
        '/nvme2/user9/hscohs/model_1.pth',
        '/nvme2/user9/hscohs/model_2.pth',
        '/nvme2/user9/hscohs/model_3.pth',
        '/nvme2/user9/hscohs/model_4.pth',
        '/nvme2/user9/hscohs/model_5.pth',
        '/nvme2/user9/hscohs/model_6.pth',
        '/nvme2/user9/hscohs/model_7.pth',
        '/nvme2/user9/hscohs/model_8.pth',
        '/nvme2/user9/hscohs/model_9.pth',
        '/nvme2/user9/hscohs/model_10.pth',
    ]

    # 数据路径列表
    data_dirs = [
        '/nvme2/user9/hscohs/testfusedifcnn/fused_0',
        '/nvme2/user9/hscohs/testfusedifcnn/fused_1',
        '/nvme2/user9/hscohs/testfusedifcnn/fused_2',
        '/nvme2/user9/hscohs/testfusedifcnn/fused_3',
        '/nvme2/user9/hscohs/testfusedifcnn/fused_4',
        '/nvme2/user9/hscohs/testfusedifcnn/fused_5',
        '/nvme2/user9/hscohs/testfusedifcnn/fused_6',
        '/nvme2/user9/hscohs/testfusedifcnn/fused_7',
        '/nvme2/user9/hscohs/testfusedifcnn/fused_8',
        '/nvme2/user9/hscohs/testfusedifcnn/fused_9',
        '/nvme2/user9/hscohs/testfusedifcnn/fused_10',
    ]

    # 加载均值和方差数组
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
    total_models = len(model_paths)

    # 初始权重
    initial_weights = [ ]

    # 创建动态贝叶斯优化器
    dynamic_bo = DynamicBayesianOptimizer(
        num_models=total_models,
        init_weights=initial_weights,
        init_points=5,  # 初始采样点
        n_iter_per_update=2,  # 每次更新的迭代次数
        update_frequency=30  # 每n张图像更新一次
    )

    # 为每个模型创建对应的数据预处理管道
    pipelines = []
    for model_idx in range(1, total_models + 1):
        model_means, model_stds = get_band_means_stds(all_means, all_stds, model_idx, total_models)
        print(f"Model {model_idx - 1} using means: {model_means}")
        print(f"Model {model_idx - 1} using stds: {model_stds}")

        pipeline = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=model_means, std=model_stds),
        ])
        pipelines.append(pipeline)

    # 加载所有模型
    models = [load_model(path, device) for path in model_paths]

    # 创建所有测试数据集和加载器
    test_datasets = []  # 保存测试数据集
    test_loaders = []
    for data_dir, pipeline in zip(data_dirs, pipelines):
        test_dataset, test_loader = create_test_dataset(data_dir, pipeline)
        test_datasets.append(test_dataset)
        test_loaders.append(test_loader)

    # 初始化结果
    Y_true = torch.tensor([], device=device)
    Y_pred = torch.tensor([], device=device)
    all_weights_history = []  # 保存权重变化历史

    # 初始化每个模型的预测结果统计
    model_predictions = []  # 存储每个模型的所有预测
    for _ in range(total_models):
        model_predictions.append([])

    # 获取当前权重
    current_weights = dynamic_bo.current_weights.copy()
    print(f"\n初始权重: {np.round(current_weights, 4)}")
    print(f"更新频率: 每{dynamic_bo.update_frequency}张图像更新一次权重")

    # 并行处理所有数据加载器
    for batch_idx, batch_data in enumerate(tqdm(zip(*test_loaders), total=len(test_loaders[0]))):
        batch_images = []

        # 处理每个模态的数据
        for i, (images, labels) in enumerate(batch_data):
            # 第一个模态用于获取真实标签
            if i == 0:
                Y_true = torch.cat((Y_true, labels.to(device)), dim=0)
                true_label = labels.item()

            # 处理图像
            img = torch.squeeze(images, 0).unsqueeze(dim=0).float().to(device)
            batch_images.append(img)

        # 模型预测
        with torch.no_grad():
            model_probs = []

            for model, img in zip(models, batch_images):
                output = model(img)
                prob = F.softmax(output, dim=1)
                model_probs.append(prob.cpu().numpy()[0])  # [num_classes]

            # 收集每个模型的预测结果
            for i in range(total_models):
                pred_class = np.argmax(model_probs[i])
                model_predictions[i].append(pred_class)

            # 收集模型预测数据
            sample_data = {}
            for i in range(total_models):
                sample_data[f'model_{i}'] = model_probs[i]
            sample_data['true_label'] = true_label

            # 添加到优化器缓冲区
            dynamic_bo.add_sample(sample_data)

            # 使用当前权重进行集成预测
            weighted_proba = np.zeros_like(model_probs[0])
            for i in range(total_models):
                weighted_proba += current_weights[i] * model_probs[i]

            # 获取预测结果
            pred = np.argmax(weighted_proba)
            Y_pred = torch.cat((Y_pred, torch.tensor([pred], device=device)), dim=0)

            # 更新当前权重（从优化器获取最新权重）
            current_weights = dynamic_bo.current_weights.copy()

            # 保存权重历史
            all_weights_history.append(current_weights.copy())

    # 处理剩余的缓冲区样本
    dynamic_bo.finalize()

    # 获取最终权重
    current_weights = dynamic_bo.current_weights.copy()

    # 转移到CPU并转换为numpy
    Y_true_np = Y_true.to(CPU).numpy()
    Y_pred_np = Y_pred.to(CPU).numpy()

    # 计算评估指标
    f1 = f1_score(Y_true_np, Y_pred_np, average='macro')
    acc = accuracy_score(Y_true_np, Y_pred_np)
    kappa = cohen_kappa_score(Y_true_np, Y_pred_np)
    cm = confusion_matrix(Y_true_np, Y_pred_np)

    # 计算并显示每个模型的准确率
    print('\n' + '-' * 60)
    print("各模型单独准确率：")
    model_accuracies = []
    model_f1_scores = []
    model_cms = []

    for i in range(total_models):
        # 将模型预测列表转换为numpy数组
        model_pred_np = np.array(model_predictions[i])
        # 计算该模型的准确率
        model_acc = accuracy_score(Y_true_np, model_pred_np)
        # 计算该模型的F1分数
        model_f1 = f1_score(Y_true_np, model_pred_np, average='macro')
        # 计算该模型的混淆矩阵
        model_cm = confusion_matrix(Y_true_np, model_pred_np)

        model_accuracies.append(model_acc)
        model_f1_scores.append(model_f1)
        model_cms.append(model_cm)

        print(f"模型 {i:2d}: 准确率 = {model_acc:.4f}, F1 = {model_f1:.4f}, 权重 = {current_weights[i]:.4f}")
    print('-' * 60)

    print('\n' + '=' * 60)
    print('最终评估结果:')
    print(f'最终权重: {np.round(current_weights, 4)}')
    print(f'总迭代次数: {dynamic_bo.iteration}')
    print(f'集成模型 ACC: {acc:.4f}')
    print(f'集成模型 F1: {f1:.4f}')
    print(f'集成模型 Kappa: {kappa:.4f}')
    print(f'集成模型混淆矩阵:\n{cm}')
    print('=' * 60)

    # 保存权重历史
    all_weights_history = np.array(all_weights_history)
    np.save('dynamic_weights_history2.npy', all_weights_history)

    # 获取类别标签名称
    label_names = test_datasets[0].classes

    # 绘制集成模型的混淆矩阵
    ensemble_save_path = '/nvme2/user9/cm/swinv2bf2hscohs_ensemble2.png'
    plot_confusion_matrix(cm, label_names,
                          f'Ensemble Model Confusion Matrix (ACC={acc:.4f})',
                          ensemble_save_path)

    # 绘制各个基模型的混淆矩阵
    print("\n正在绘制各个基模型的混淆矩阵...")
    base_model_cm_dir = '/nvme2/user9/cm/base_models2/'
    os.makedirs(base_model_cm_dir, exist_ok=True)

    for i in range(total_models):
        model_cm = model_cms[i]
        model_acc = model_accuracies[i]

        # 保存路径
        model_save_path = os.path.join(base_model_cm_dir, f'model_{i}_confusion_matrix.png')

        # 绘制混淆矩阵
        plot_confusion_matrix(model_cm, label_names,
                              f'Base Model {i+1} Confusion Matrix (ACC={model_acc:.4f})',
                              model_save_path)

        print(f"  模型 {i} 的混淆矩阵已保存至: {model_save_path}")

    print(f"\n所有基模型的混淆矩阵已保存至: {base_model_cm_dir}")

    return acc, f1, kappa, current_weights


if __name__ == '__main__':
    predict_image_with_dynamic_bo()
