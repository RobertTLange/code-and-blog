import numpy as np
from dual_ad_regression import DataLoader, substract_constant


def regression_loss_dual(data_loader, batch_id, b_current):
    # Transform the coefficients into dual numbers
    b_dual = DualTensor(b_current, deriv_dim=True)
    # Select the current batch
    X, y = data_loader.get_batch_idx(batch_id, get_batch=True)
    # Compute output & calculate the error
    out = dot_product(b_dual, X)
    print(out.dual.shape)
    temp = substract_constant(out, y)
    mse = dual_dot_product(temp, temp)
    print(mse.dual.shape)
    # Normalize MSE to get mean!
    mse.real /= len(y)
    return mse

class DualTensor(object):
    # Class object for dual representation of a vector
    def __init__(self, real, dual=None, deriv_dim=False):
        self.real = real
        if dual is not None:
            total_dim = dual.shape[0]
            self.dual = dual
        else:
            try:
                total_dim = real.shape[0]
                self.dual = np.zeros((total_dim, total_dim))
                if deriv_dim:
                    np.fill_diagonal(self.dual, 1)
            except:
                self.dual = None

def dot_product(b_dual, x):
    real_part = np.dot(x, b_dual.real)
    dual_part = np.dot(x, b_dual.dual)
    return DualTensor(real_part, dual_part)

def dual_dot_product(dual_1, dual_2):
    real_part = np.dot(dual_1.real, dual_2.real)
    dual_part = np.dot(dual_1.real, dual_2.dual) + np.dot(dual_2.real, dual_1.dual)
    return DualTensor(real_part, dual_part)

def train_regression(n, d, n_epoch, batch_size, b_init, l_rate):
    # Generate the data for a coefficient vector!
    data_loader = DataLoader(n, d, batch_size)
    b_hist, func_val_hist = [b_init], [regression_loss_dual(data_loader, 0,
                                        b_init)]

    for epoch in range(n_epoch):
        # Shuffle the batch identities at beginning of each epoch
        data_loader.shuffle_arrays()
        for batch_id in range(data_loader.num_batches):
            # Calculate the forward AD - real = func, dual = deriv
            current_dual = regression_loss_dual(data_loader, batch_id, b_hist[-1])
            # Perform grad step & append results to the placeholder list
            b_hist.append(b_hist[-1] - l_rate*np.array(current_dual.dual).flatten())
            func_val_hist.append(current_dual.real)

        if epoch % 1 == 0:
            print("beta true: {} | beta value: {} | fct min: {}".format(data_loader.b_true.flatten(), b_hist[-1], current_dual.real))
    return b_hist[1:], func_val_hist[1:]

if __name__ == "__main__":
    np.random.seed(1)
    train_regression(1000, 4, 1, 32, np.array([0, 0, 0, 0, 0]).astype(float), 0.01)
    # Test example from the blog post
    # bivariate_f_dual(5, 2, coeffs=[3, 2])
