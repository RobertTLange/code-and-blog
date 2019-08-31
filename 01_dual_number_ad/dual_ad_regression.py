import numpy as np

class MAD_DualNumber(object):
    # Small class object for dual number - Two real numbers (real & dual part)
    def __init__(self, real, dual=None, total_dim=None, deriv_dim=None):
        self.real = real
        if dual is not None:
            self.dual = dual
            self.total_dim = len(dual)
        else:
            self.total_dim = total_dim
            self.dual = np.zeros(total_dim)
            if deriv_dim is not None:
                self.dual[deriv_dim] = 1

def add_duals(*dual_list):
    # Operator non-"overload": Add a list of dual numbers
    real_part, dual_part = dual_list[0].real, dual_list[0].dual
    for dual_numb in dual_list[1:]:
        real_part += dual_numb.real
        dual_part += dual_numb.dual
    return MAD_DualNumber(real_part, dual_part)

def add_constant(dual_numb, constant):
    # Add a dual number and a real number
    return MAD_DualNumber(dual_numb.real + constant, dual_numb.dual)

def substract_duals(*dual_list):
    # Operator non-"overload": Substract a list of dual numbers
    real_part, dual_part = dual_list[0].real, dual_list[0].dual
    for dual_numb in dual_list[1:]:
        real_part -= dual_numb.real
        dual_part -= dual_numb.dual
    return MAD_DualNumber(real_part, dual_part)

def substract_constant(dual_numb, constant):
    # Add a dual number and a real number
    return MAD_DualNumber(dual_numb.real - constant, dual_numb.dual)

def multiply_duals(*dual_list):
    # Operator non-"overload": Multiply a list of dual numbers
    real_part, dual_part = dual_list[0].real, dual_list[0].dual
    for dual_numb in dual_list[1:]:
        real_part_new = real_part * dual_numb.real
        dual_part_new = real_part * dual_numb.dual + dual_part * dual_numb.real
        real_part, dual_part = real_part_new, dual_part_new
    return MAD_DualNumber(real_part, dual_part)

def multiply_constant(dual_numb, constant):
    # Operator non-"overload": Multiply a list of dual numbers
    real_part_new = constant * dual_numb.real
    dual_part_new = dual_numb.dual * constant
    return MAD_DualNumber(real_part_new, dual_part_new)

def divide_duals(*dual_list):
    # Operator non-"overload": Divide a list of dual numbers
    real_part, dual_part = dual_list[0].real, dual_list[0].dual
    for dual_numb in dual_list[1:]:
        real_part_new = real_part / dual_numb.real
        dual_part_new = (dual_part * dual_numb.real - real_part * dual_numb.dual)/(dual_numb.real*dual_numb.real)
        real_part, dual_part = real_part_new, dual_part_new
    return MAD_DualNumber(real_part, dual_part)

def divide_constant(dual_numb, constant):
    # Operator non-"overload": Divide a list of dual numbers
    real_part_new = dual_numb.real / constant
    dual_part_new = (dual_numb.dual * constant)/(constant*constant)
    real_part, dual_part = real_part_new, dual_part_new
    return MAD_DualNumber(real_part, dual_part)

def power(dual_numb, power):
    # Operator non-"overload": Potentiate a dual number
    if power == 0:
        dual_numb = MAD_DualNumber(1, total_dim=dual_numb.total_dim)
    else:
        dual_numb = multiply_duals(*power*[dual_numb])
    return dual_numb

class DataLoader(object):
    # Small class object for dual number - Two real numbers (real & dual part)
    def __init__(self, n, d, batch_size, binary=False):
        self.total_dim = d + 1
        self.X, self.y = self.generate_regression_data(n, d, binary)

        # Set batch_id for different indices
        self.num_batches = np.ceil(n/batch_size).astype(int)
        self.batch_ids = np.array([np.repeat(i, batch_size) for i in range(self.num_batches)]).flatten()[:n]

    def generate_regression_data(self, n, d, binary=False):
        self.b_true = self.generate_coefficients(d)
        X = np.random.normal(0, 1, n*d).reshape((n, d))
        noise = np.random.normal(0, 1, n).reshape((n, 1))
        inter = np.ones(n).reshape((n, 1))
        X = np.hstack((inter, X))
        y = np.matmul(X, self.b_true) + noise
        if binary:
            y[y > 0] = 1
            y[y < 0] = 0
        return X, y

    def generate_coefficients(self, d, intercept=True):
        b_random = np.random.randint(-5, 5, d + intercept)
        return b_random.reshape((d + intercept, 1))

    def shuffle_arrays(self):
        assert len(self.X) == len(self.y)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]

    def get_batch_idx(self, batch_id, get_batch=False):
        # Subselect the current batch processed in forward mode differentiation!
        idx = np.where(self.batch_ids == batch_id)[0]

        if get_batch:
            return self.X[idx, :], self.y[idx].flatten()
        else:
            return idx

    def get_datapoint(self, idx):
        return self.X[idx, :], self.y[idx]


def regression_loss_dual(data_loader, batch_id, b_current):
    # Perform on individual points & sum up the gradients
    # Transform the coefficients into dual numbers
    d = len(b_current)
    b_dual = [MAD_DualNumber(b_current[i], total_dim=d, deriv_dim=i) for i in range(d)]
    # Select the current batch
    idx = data_loader.get_batch_idx(batch_id)
    # Init mse + grad dual object
    mse_total = MAD_DualNumber(0, total_dim=d)
    # Loop over all datapoints
    for point_idx in idx:
        x, y = data_loader.get_datapoint(point_idx)
        out = dual_dot_product(b_dual, x)
        mse = power(substract_constant(out, y), 2)
        mse_total = add_duals(mse_total, mse)
    return mse


def dual_dot_product(b_dual, x):
    # Initialize a dual to which the different coeff products are added
    out = MAD_DualNumber(0, total_dim=len(b_dual))
    for dim in range(len(b_dual)):
        out = add_duals(out, multiply_constant(b_dual[dim], x[dim]))
    return out


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

def bivariate_f_dual(x, y, coeffs=[3, 2]):
    # Simple test function with 3*x^2 - 2*y^3
    x = MAD_DualNumber(x, total_dim=2, deriv_dim=0)
    y = MAD_DualNumber(y, total_dim=2, deriv_dim=1)
    c_dual = [MAD_DualNumber(c, total_dim=2) for c in coeffs]

    temp_1 = multiply(c_dual[0], power(x, 2))
    temp_2 = multiply(c_dual[1], power(y, 3))
    full = substract(temp_1, temp_2)
    return

if __name__ == "__main__":
    np.random.seed(1)
    train_regression(500, 3, 20, 5, np.array([0, 0, 0, 0]).astype(float), 0.05)
    # Test example from the blog post
    # bivariate_f_dual(5, 2, coeffs=[3, 2])
