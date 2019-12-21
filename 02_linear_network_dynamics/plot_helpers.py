import numpy as np
import matplotlib.pyplot as plt

def plot_singular_values(s):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(np.arange(1, 5), s)
    ax.set_ylabel(r"$a_i(t)$", fontsize=15)
    ax.set_xlabel("Epochs", fontsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Singular Value Modes", fontsize=18)
    return


def plot_figure3(s_vals_deep_ana, s_vals_deep_emp,
                 s_vals_shallow_ana, s_vals_shallow_emp, s):
    # Plot the results for the single hidden-layer network & the shallow network
    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    for i in range(4):
        axs[0].plot(s_vals_deep_ana[:, i], label="Analytical")
        axs[0].plot(s_vals_deep_emp[:, i], label="Simulated")
        axs[0].axhline(s[i], linestyle="--")

    for i in range(4):
        axs[1].plot(s_vals_shallow_ana[:, i], label="Analytical")
        axs[1].plot(s_vals_shallow_emp[:, i], label="Simulated")
        axs[1].axhline(s[i], linestyle="--")

    # Plot Cosmetics
    for i in range(2):
        axs[i].set_ylabel(r"$a_i(t)$", fontsize=15)
        axs[i].set_xlabel("Epochs", fontsize=15)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)

    axs[0].set_title("1-Hidden-Layer Net", fontsize=18)
    axs[1].set_title("Shallow Net", fontsize=18)
    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    fig.suptitle("Toy Example - Singular Value Mode Convergence", fontsize=22)
    return


def plot_singular_val_dynamics(log_s_vals, log_loss, s, title, save_fname=None):
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    # Plot the singular value dynamics
    for i in range(log_s_vals.shape[1]):
        axs[0].plot(log_s_vals[:, i])
        axs[1].plot(log_s_vals[:, i] - s[i])

    # Plot the Learning Loss/MSE
    axs[2].plot(log_loss)

    # Plot timepoint of convergence
    diff = log_s_vals - s
    conv_t = [np.where(np.abs(diff[:, i]) < 0.05)[0][0] for i in range(diff.shape[1])]
    axs[3].plot(conv_t)
    axs[3].set_xlabel("Singular Value Mode", fontsize=15)
    axs[3].set_ylabel("Epoch of Convergence", fontsize=15)
    axs[3].spines['top'].set_visible(False)
    axs[3].spines['right'].set_visible(False)

    for i in range(3):
        axs[i].set_xlabel("Epochs", fontsize=15)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)

    axs[0].set_ylabel(r"$a_i(t)$", fontsize=15)
    axs[1].set_ylabel(r"$a_i(t) - s_i$", fontsize=15)
    axs[2].set_ylabel(r"MSE", fontsize=15)
    axs[3].set_ylabel(r"$t: |a_i(t) - s_i| < 0.05$", fontsize=15)

    axs[0].set_title("Singular Value Modes", fontsize=18)
    axs[1].set_title("Convergence", fontsize=18)
    axs[2].set_title("Training Loss", fontsize=18)
    axs[3].set_title("Time of Convergence", fontsize=18)
    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    fig.suptitle(title, fontsize=22)
    if save_fname is not None:
        plt.savefig(save_fname, dpi=300)
    return axs


def plot_learning_loss(train_losses, test_losses=None, legend_titles=["Shallow, Linear", "ReLU"]):
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    for i in range(len(train_losses)):
        axs.plot(train_losses[i], label="Train Loss: " + legend_titles[i])
    axs.legend()
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    return
