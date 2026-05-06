import os
import json

import numpy
import torchvision
import torch
import numpy as np
from tqdm import tqdm
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound, ProbabilityOfImprovement, ExpectedImprovement, GPHedge
from torchvision import transforms, datasets
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import gc


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


def load_model_for_optimization(model_path, device):
    """专门为优化阶段加载模型"""
    model = torchvision.models.swin_v2_b(weights=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  
    return model


def optimize_weights(model_paths, test_base_dirs, device, all_means, all_stds, init_points=100,
                     n_iter=400):
    """使用贝叶斯优化优化模型权重"""
    print("\n" + "=" * 60)
    print("Starting Bayesian Optimization for Model Weights")
    print(f"Total models: {len(model_paths)}")
    print(f"Total bands in means/std arrays: {len(all_means)}")
    print("=" * 60)

    # 预计算各模型预测结果
    def precompute_model_predictions(model_paths, test_dirs, all_means, all_stds):
        """为每个模型预存预测logits，逐个加载模型"""
        precomputed = {}
        total_models = len(model_paths)

        # 为每个样本预先决定随机变换
        test_dataset = datasets.ImageFolder(root=test_dirs[0])
        n_samples = len(test_dataset)

        np.random.seed(42)  
        rotations = np.random.choice([0, 90, 180, 270], size=n_samples)
        h_flips = np.random.choice([True, False], size=n_samples)
        v_flips = np.random.choice([True, False], size=n_samples)

        for model_idx, (model_path, test_dir) in enumerate(zip(model_paths, test_dirs), 1):
            print(f"\nPrecomputing predictions for Model {model_idx}...")

            # 获取当前模型对应的均值和方差
            model_means, model_stds = get_band_means_stds(all_means, all_stds, model_idx, total_models)
            print(f"Model {model_idx} using means: {model_means}")
            print(f"Model {model_idx} using stds: {model_stds}")

            pipeline = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=model_means, std=model_stds),
            ])

            # 加载测试数据集
            test_dataset = datasets.ImageFolder(root=test_dir, transform=pipeline)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=64, shuffle=False, num_workers=8)

            model = load_model_for_optimization(model_path, device)

            predictions = []
            labels_list = []

            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc=f"Model {model_idx}")):
                    inputs = inputs.to(device)

                    # 应用相同的随机变换到当前批次
                    batch_start = batch_idx * test_loader.batch_size

                    for i in range(inputs.size(0)):
                        sample_idx = batch_start + i
                        if sample_idx < n_samples:
                            if rotations[sample_idx] == 90:
                                inputs[i] = torch.rot90(inputs[i], 1, [1, 2])
                            elif rotations[sample_idx] == 180:
                                inputs[i] = torch.rot90(inputs[i], 2, [1, 2])
                            elif rotations[sample_idx] == 270:
                                inputs[i] = torch.rot90(inputs[i], 3, [1, 2])

                            if h_flips[sample_idx]:
                                inputs[i] = torch.flip(inputs[i], [2])

                            if v_flips[sample_idx]:
                                inputs[i] = torch.flip(inputs[i], [1])

                    outputs = model(inputs)
                    predictions.append(F.softmax(outputs, dim=1).cpu().numpy())

                    if model_idx == 1:
                        labels_list.append(labels.numpy())

            # 保存预测结果
            precomputed[f'm{model_idx}'] = np.concatenate(predictions, axis=0)

            # 保存真实标签（只做一次）
            if model_idx == 1:
                Y_true = np.concatenate(labels_list, axis=0)

            # 清理当前模型
            del model
            torch.cuda.empty_cache()
            gc.collect()

        return precomputed, Y_true

    # 执行预计算
    precomputed, Y_true = precompute_model_predictions(model_paths, test_base_dirs, all_means, all_stds)
    print(f"\nPrecomputed predictions for {len(model_paths)} models")
    print(f"True labels shape: {Y_true.shape}")

    # 定义贝叶斯优化目标函数
    def model_ensemble_eval(**weights_dict):
        """权重自动归一化处理"""
        weights = np.array([weights_dict[f'w{i}'] for i in range(1, len(model_paths) + 1)])
        weights = np.clip(weights, 0, 1)  # 约束每个权重在[0,1]

        # 归一化权重
        if weights.sum() == 0:
            weights = np.ones(len(weights)) / len(weights)
        else:
            weights = weights / weights.sum()

        # 多模型加权集成
        weighted_proba = np.zeros_like(precomputed['m1'])
        for i in range(len(model_paths)):
            weighted_proba += weights[i] * precomputed[f'm{i + 1}']

        y_pred = np.argmax(weighted_proba, axis=1)

        return accuracy_score(Y_true, y_pred)

    # 定义优化空间
    pbounds = {f'w{i}': (0, 1) for i in range(1, len(model_paths) + 1)}

    # 创建acquisition函数
    acquisition_function = GPHedge(
        base_acquisitions=[
            UpperConfidenceBound(kappa=2),
            ProbabilityOfImprovement(xi=0.01),
            ExpectedImprovement(xi=0.01)
        ],
        random_state=42
    )

    # 创建优化器
    optimizer = BayesianOptimization(
        f=model_ensemble_eval,
        pbounds=pbounds,
        acquisition_function=acquisition_function,
        verbose=2,
        random_state=42,
    )

    # 执行优化
    print(f"\nStarting optimization with {init_points} initial points and {n_iter} iterations...")
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
    )

    # 获取最佳权重
    best_params = optimizer.max['params']
    final_weights = np.array([best_params[f'w{i}'] for i in range(1, len(model_paths) + 1)])
    final_weights = np.clip(final_weights, 0, 1)
    if final_weights.sum() > 0:
        final_weights = final_weights / final_weights.sum()  # 归一化

    print(f"\n{'=' * 60}")
    print("Optimization Results:")
    print(f"Best Accuracy: {optimizer.max['target']:.4f}")
    print(f"Optimized Weights: {np.round(final_weights, 4)}")
    np.savetxt('weight.txt', np.round(final_weights, 4), delimiter=',')
    print(f"{'=' * 60}")

    return final_weights, optimizer.max['target']


def main():
    """主函数：优化模型权重并进行集成预测"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

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

    # 从训练信息文件中加载配置
    if os.path.exists('training_info.json'):
        with open('training_info.json', 'r') as f:
            training_info = json.load(f)

        model_paths = training_info['model_paths']
        base_dirs = training_info['base_dirs']

        print(f"Loaded {len(model_paths)} models from training_info.json")
    else:
        # 如果没有训练信息文件，使用硬编码的路径
        print("Warning: training_info.json not found, using hardcoded paths")

        model_paths = [
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


    # 检查所有模型文件是否存在
    missing_models = []
    for i, model_path in enumerate(model_paths):
        if not os.path.exists(model_path):
            missing_models.append((i + 1, model_path))

    if missing_models:
        print(f"\nWarning: {len(missing_models)} model(s) not found:")
        for idx, path in missing_models:
            print(f"  Model {idx}: {path}")

        proceed = input("\nContinue with available models? (y/n): ")
        if proceed.lower() != 'y':
            print("Exiting...")
            return

        # 只保留存在的模型和对应的数据路径
        available_indices = [i for i, path in enumerate(model_paths) if os.path.exists(path)]
        model_paths = [model_paths[i] for i in available_indices]
        base_dirs = [base_dirs[i] for i in available_indices]

        print(f"\nProceeding with {len(model_paths)} available models")

    # 优化权重
    print(f"\n{'=' * 60}")
    print("Starting weight optimization...")
    print(f"{'=' * 60}")

    try:
        optimized_weights, best_accuracy = optimize_weights(
            model_paths, base_dirs, device, all_means, all_stds, init_points=100, n_iter=400)

        # 保存权重
        weights_dict = {
            'weights': optimized_weights.tolist(),
            'accuracy': float(best_accuracy),
            'model_paths': model_paths,
            'data_dirs': base_dirs,
            'optimization_success': True,
            'normalization_info': {
                'total_bands': len(all_means),
                'means_source': 'all_means.npy',
                'stds_source': 'all_stds.npy'
            }
        }

        with open('optimized_weights.json', 'w') as f:
            json.dump(weights_dict, f, indent=4)

        print(f"\nOptimized weights saved to 'optimized_weights.json'")

    except Exception as e:
        print(f"Error during optimization: {e}")
        print("Continuing with equal weights...")

        # 如果优化失败，使用等权重
        optimized_weights = np.ones(len(model_paths)) / len(model_paths)
        best_accuracy = 0.0

        # 保存等权重信息
        weights_dict = {
            'weights': optimized_weights.tolist(),
            'accuracy': 0.0,
            'model_paths': model_paths,
            'data_dirs': base_dirs,
            'optimization_success': False,
            'note': 'Optimization failed, using equal weights',
            'normalization_info': {
                'total_bands': len(all_means),
                'means_source': 'all_means.npy',
                'stds_source': 'all_stds.npy'
            }
        }

        with open('equal_weights.json', 'w') as f:
            json.dump(weights_dict, f, indent=4)

        print(f"\nEqual weights saved to 'equal_weights.json'")


if __name__ == '__main__':
    main()
