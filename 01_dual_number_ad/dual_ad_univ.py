import unittest

class DualNumber(object):
    # Small class object for dual number - Two real numbers (real & dual part)
    def __init__(self, real, dual):
        self.real = real
        self.dual = dual

def add(*dual_list):
    # Operator non-"overload": Add a list of dual numbers
    real_part, dual_part = 0, 0
    for dual_numb in dual_list:
        real_part += dual_numb.real
        dual_part += dual_numb.dual
    return DualNumber(real_part, dual_part)

def substract(*dual_list):
    # Operator non-"overload": Substract a list of dual numbers
    real_part, dual_part = 0, 0
    for dual_numb in dual_list:
        real_part -= dual_numb.real
        dual_part -= dual_numb.dual
    return DualNumber(real_part, dual_part)

def multiply(*dual_list):
    # Operator non-"overload": Multiply a list of dual numbers
    real_part, dual_part = dual_list[0].real, dual_list[0].dual
    for dual_numb in dual_list[1:]:
        real_part_new = real_part * dual_numb.real
        dual_part_new = real_part * dual_numb.dual + dual_part * dual_numb.real
        real_part, dual_part = real_part_new, dual_part_new
    return DualNumber(real_part, dual_part)

def divide(*dual_list):
    # Operator non-"overload": Divide a list of dual numbers
    real_part, dual_part = dual_list[0].real, dual_list[0].dual
    for dual_numb in dual_list[1:]:
        real_part_new = real_part / dual_numb.real
        dual_part_new = (dual_part * dual_numb.real - real_part * dual_numb.dual)/(dual_numb.real*dual_numb.real)
        real_part, dual_part = real_part_new, dual_part_new
    return DualNumber(real_part, dual_part)

def power(dual_numb, power):
    # Operator non-"overload": Potentiate a dual number
    if power == 0:
        dual_numb = DualNumber(1, 0)
    else:
        dual_numb = multiply(*power*[dual_numb])
    return dual_numb

class test(unittest.TestCase):
    # Test the different arithmetic functions for dual number calculus
    def __init__(self, *args, **kwargs):
        super(test, self).__init__(*args, **kwargs)
        self.a_dual = DualNumber(2, 1)
        self.b_dual = DualNumber(3, 4)
        self.c_dual = DualNumber(5, 7)

    def test_add(self):
        sum_a_b = add(self.a_dual, self.b_dual)
        sum_a_b_c = add(self.a_dual, self.b_dual, self.c_dual)
        self.assertEqual(sum_a_b.real, 5)
        self.assertEqual(sum_a_b.dual, 5)
        self.assertEqual(sum_a_b_c.real, 10)
        self.assertEqual(sum_a_b_c.dual, 12)

    def test_mult(self):
        mult_a_b = multiply(self.a_dual, self.b_dual)
        mult_a_b_c = multiply(self.a_dual, self.b_dual, self.c_dual)
        self.assertEqual(mult_a_b.real, 6)
        self.assertEqual(mult_a_b.dual, 11)
        self.assertEqual(mult_a_b_c.real, 30)
        self.assertEqual(mult_a_b_c.dual, 97)

    def test_potentiate(self):
        pow_a_3 = power(self.a_dual, 3)
        self.assertEqual(pow_a_3.real, 8)
        self.assertEqual(pow_a_3.dual, 12)


def polynomial_f_primal(x, coeffs=[1, 4, 5]):
    # Primal polynomial function for plotting
    out = 0
    for degree in range(len(coeffs)):
        out += coeffs[degree]*x**degree
    return out


def polynomial_f_dual(x, coeffs=[1, 4, 5]):
    # Dual polynomial function for optimization
    x = DualNumber(x, 1)
    c_dual = [DualNumber(c, 0) for c in coeffs]
    # Initialize a dual to which the different polyonmial degrees are added
    out = DualNumber(0, 0)
    for degree in range(len(coeffs)):
        out = add(out, multiply(c_dual[degree], power(x, degree)))
    return out


def gradient_descent(n_iters, func_dual, x_init, l_rate):
    # Initialize placeholders for optimization trajectory
    x_hist, func_val_hist = [x_init], [func_dual(x_init.real)]
    # Run the SGD loop
    for iter in range(n_iters):
        # Evaluate and get current gradiate simultaneously
        current_dual = func_dual(x_hist[-1])
        # Print current state of optimization
        if iter % 5 == 0:
            print("Current x value: {} | fct min: {}".format(x_hist[-1], current_dual.real))
        # Perform grad step & append results to the placeholder list
        x_hist.append(x_hist[iter] - l_rate*current_dual.dual)
        func_val_hist.append(current_dual.real)
    return x_hist[1:], func_val_hist[1:]


if __name__ == "__main__":
    # unittest.main()
    x_hist, func_val_hist = gradient_descent(50, polynomial_f_dual, 20, 0.01)
