import os
import jax
import jax.numpy as jnp
import numpy as np

from train.models import All_CNN_C_Features
from train.utils import get_dataloaders
from mle_logging import load_log, load_model


def CKA(X, Y, kernel="linear", sigma_frac=0.4):
    """Centered Kernel Alignment."""
    if kernel == "linear":
        K, L = linear_kernel(X, Y)
    elif kernel == "rbf":
        K, L = rbf_kernel(X, Y, sigma_frac)
    return HSIC(K, L) / jnp.sqrt(HSIC(K, K) * HSIC(L, L))


@jax.jit
def linear_kernel(X, Y):
    K = X @ X.T
    L = Y @ Y.T
    return K, L


@jax.jit
def rbf_kernel(X, Y, sigma_frac=0.4):
    """Compute radial basis function kernels."""
    # Define helper for euclidean distance
    def euclidean_dist_matrix(X, Y):
        """Compute matrix of pairwise, squared Euclidean distances."""
        norms_1 = (X ** 2).sum(axis=1)
        norms_2 = (Y ** 2).sum(axis=1)
        return jnp.abs(norms_1.reshape(-1, 1) + norms_2 - 2 * jnp.dot(X, Y.T))

    # Define σ as a fraction of the median distance between examples
    dist_X = euclidean_dist_matrix(X, X)
    dist_Y = euclidean_dist_matrix(Y, Y)
    sigma_x = sigma_frac * jnp.percentile(dist_X, 0.5)
    sigma_y = sigma_frac * jnp.percentile(dist_Y, 0.5)
    K = jnp.exp(-dist_X / (2 * sigma_x ** 2))
    L = jnp.exp(-dist_Y / (2 * sigma_y ** 2))
    return K, L


@jax.jit
def HSIC(K, L):
    """Hilbert-Schmidt Independence Criterion."""
    m = K.shape[0]
    H = jnp.eye(m) - 1 / m * jnp.ones((m, m))
    numerator = jnp.trace(K @ H @ L @ H)
    return numerator / (m - 1) ** 2


def get_cka_matrix(activations_1, activations_2, kernel="linear"):
    """Loop over layer combinations & construct CKA matrix."""
    num_layers_1 = len(activations_1)
    num_layers_2 = len(activations_2)
    cka_matrix = np.zeros((num_layers_1, num_layers_2))
    symmetric = num_layers_1 == num_layers_2

    for i in range(num_layers_1):
        if symmetric:
            for j in range(i, num_layers_2):
                X, Y = activations_1[i], activations_2[j]
                cka_temp = CKA(X, Y, kernel)
                cka_matrix[num_layers_1 - i - 1, j] = cka_temp
                cka_matrix[i, num_layers_1 - j - 1] = cka_temp
        else:
            for j in range(num_layers_2):
                X, Y = activations_1[i], activations_2[j]
                cka_temp = CKA(X, Y)
                cka_matrix[num_layers_1 - i - 1, j] = cka_temp
    return cka_matrix


def main(fname, kernel):
    log = load_log(os.path.join("experiments/", fname))
    # Get CIFAR-10 dataloader
    train_loader, test_loader = get_dataloaders(2056)
    # Get batch data & compute final activations
    for batch_idx, (data, target) in enumerate(test_loader):
        batch_images = jnp.array(data)
        break

    cnn_vars = load_model(log.meta.model_ckpt, model_type="jax")
    model = All_CNN_C_Features(**log.meta.config_dict["model_config"])
    activations_final = model.apply(cnn_vars, batch_images, train=False)

    # Reload the trained final checkpoint
    all_cka_matrices = []
    ckpt_list = [log.meta.init_ckpt]
    ckpt_list = ckpt_list + log.meta.every_k_ckpt_list

    # Loop over all checkpoint and construct CKA of ckpt vs final ckpt
    for ckpt_path in ckpt_list:
        cnn_vars = load_model(ckpt_path, model_type="jax")
        activations_ckpt = model.apply(cnn_vars, batch_images, train=False)
        cka_matrix = get_cka_matrix(activations_ckpt, activations_final, kernel)
        all_cka_matrices.append(cka_matrix)
        print(ckpt_path)

    stacked_rsm = np.stack(all_cka_matrices, axis=0)
    np.save(f"rsm_{fname}_{kernel}.npy", stacked_rsm)

    reload = np.load(f"rsm_{fname}_{kernel}.npy")
    print(reload.shape)


if __name__ == "__main__":
    experiment_dirs = ["all_cnn_depth_1_v2_seed_0", "all_cnn_depth_2_v2"]
    kernel_types = ["linear"]
    for kernel in kernel_types:
        for e_dir in experiment_dirs:
            main(e_dir, kernel)
