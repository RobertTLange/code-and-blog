import numpy as np
from dual_ad_regression_vec import DualTensor, dot_product, dual_dot_product
from dual_ad_regression import DataLoader, substract_constant, add_duals


def mlp_loss_dual(out, y):
    temp = substract_constant(out, y)
    mse = dual_dot_product(temp, temp)
    # Normalize MSE to get mean!
    mse.real /= len(y)
    return mse


def train_mlp(mlp_net, n, d, n_epoch, batch_size, l_rate):
    # Generate the data for a coefficient vector!
    data_loader = DataLoader(n, d, batch_size)

    for epoch in range(n_epoch):
        # Shuffle the batch identities at beginning of each epoch
        data_loader.shuffle_arrays()
        for batch_id in range(data_loader.num_batches):
            # Select the current batch
            X, y = data_loader.get_batch_idx(batch_id, get_batch=True)
            # Perform forward pass
            out = mlp_net.forward(X)
            # Calculate the forward AD - real = func, dual = deriv
            current_dual = regression_loss_dual(data_loader, batch_id, b_hist[-1])
            print("fct min: {}".format(current_dual.real))
    return


def sigmoid(dual_tensor):
    real_part = 1/(1+np.exp(-dual_tensor.real))
    deriv = np.multiply(real_part, 1-real_part)
    dual_part_W = np.tensordot(deriv, dual_tensor.W.dual.T, axes=1)
    dual_part_b = np.dot(deriv, dual_tensor.b.dual)
    return MLP_Tensor(real_part, dual_part_W, dual_part_b)


def dot_product_tensors(dual_1, dual_2):
    real_part = np.dot(dual_1.real, dual_2.real)
    dual_part = np.dot(dual_1.real, dual_2.dual)
    return MLP_Tensor(real_part, W_dual=dual_part, b_dual=dual_1.b.dual)


def add_tensors(dual_1, dual_2):
    real_part = dual_1.real + dual_2.real
    dual_part = dual_1.b.dual + dual_2.dual
    return MLP_Tensor(real_part, W_dual=dual_1.W.dual, b_dual=dual_part)


class Dual_LinearLayer():
    def  __init__(self, input_dim, output_dim, act_func=None):
        # Initialize weight matrix and bias vector real and dual parts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = DualTensor(np.zeros((input_dim, output_dim)),
                            np.zeros((input_dim, input_dim, output_dim)))
        self.b = DualTensor(np.zeros(output_dim),
                            np.zeros((output_dim, output_dim)))
        self.act_func = act_func
        self.reset_grad()

    def apply(self, x):
        out_mult = dot_product_tensors(x, self.W)
        out_add = add_tensors(out_mult, self.b)
        if self.act_func is not None:
            out = self.act_func(out_add)
        return out

    def reset_grad(self):
        # Set multivariate derivate dual 1 - Each channel corresponds to
        # Specific in-out neuron connectivity/synapse
        self.W.dual = np.zeros((self.input_dim, self.input_dim, self.output_dim))
        for i in range(self.input_dim):
            for j in range(self.output_dim):
                self.W.dual[i, i, j] = 1

        self.b.dual = np.zeros((self.output_dim, self.output_dim))
        np.fill_diagonal(self.b.dual, 1)


class MLP_Tensor():
    def  __init__(self, x, W_dual=None, b_dual=None,
                  input_dim=None, output_dim=None):
        # Initialize weight matrix and bias vector real and dual parts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.real = x
        self.W = DualTensor(None, None)
        self.b = DualTensor(None, None)

        if b_dual is not None:
            self.b.dual = b_dual
        else:
            if output_dim is None:
                output_dim = self.W.dual.shape[2]
            self.b.dual = np.zeros((output_dim, output_dim))
            np.fill_diagonal(self.b.dual, 1)

        if W_dual is not None:
            self.W.dual = W_dual
        else:
            self.W.dual = np.zeros((input_dim, input_dim, output_dim))

            for i in range(self.input_dim):
                for j in range(self.output_dim):
                    self.W.dual[i, i, j] = 1

class MLP():
    def __init__(self, input_dim, hidden_units, output_dim, act_func):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim

        self.L_1 = Dual_LinearLayer(input_dim, hidden_units, act_func)
        self.L_2 = Dual_LinearLayer(hidden_units, hidden_units, act_func)
        self.L_3 = Dual_LinearLayer(hidden_units, output_dim)
        self.layers = [self.L_1, self.L_2, self.L_3]

    def forward(self, x):
        # Before starting of forward pass clear up the gradients
        self.zero_grad()
        x_in = MLP_Tensor(x, input_dim=self.input_dim,
                          output_dim=self.hidden_units)
        print(x_in.real.shape, x_in.W.dual.shape, x_in.b.dual.shape)
        out_1 = self.L_1.apply(x_in)
        print(out_1.real.shape, out_1.W.dual.shape, out_1.b.dual.shape)
        out_2 = self.L_2.apply(out_1)
        out_3 = self.L_3.apply(out_2)
        return out_3

    def zero_grad(self):
        # Make sure that dual coeff/deriv cleaned after update/before forward!
        for layer in self.layers:
            layer.reset_grad()

    def backward(self, loss, l_rate):
        # Perform "mini-backward" pass!
        latest_W_grad = loss.W_dual
        latest_b_grad = loss.b_dual
        for layer in reversed(self.layers):
            # Update the chained gradient expression
            latest_W_grad *= layer.W.dual
            latest_b_grad *= layer.b.dual
            # Perform the SGD update
            layer.W.real -= l_rate*latest_W_grad
            layer.b.real -= l_rate*latest_b_grad
        return

if __name__ == "__main__":
    mlp_net = MLP(4, 10, 1, sigmoid)
    train_mlp(mlp_net, 200, 3, 10, 5, 0.05)
