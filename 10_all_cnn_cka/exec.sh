# Train All-CNN-C (depth 1) with input/intermediate dropout
python train/train_cifar10.py --config_fname configs/all_cnn_depth_1_v1_seed_0.yaml
python train/train_cifar10.py --config_fname configs/all_cnn_depth_1_v1_seed_1.yaml

# Train All-CNN-C (depth 1) without dropout
python train/train_cifar10.py --config_fname configs/all_cnn_depth_1_v2_seed_0.yaml
python train/train_cifar10.py --config_fname configs/all_cnn_depth_1_v2_seed_1.yaml

# Train All-CNN-C (depth 2) with input/intermediate dropout
python train/train_cifar10.py --config_fname configs/all_cnn_depth_2_v1.yaml

# Train All-CNN-C (depth 2) without dropout
python train/train_cifar10.py --config_fname configs/all_cnn_depth_2_v2.yaml
