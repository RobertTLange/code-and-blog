import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import functools
from helpers.es_helpers import (init_cma_es, eigen_decomposition,
                                check_termination, init_logger, update_logger)
from helpers.es_helpers import batch_hump_camel, batch_himmelblau, batch_rosenbrock


def ask(rng, params, memory):
    """ Propose parameters to evaluate next. """
    C, B, D = eigen_decomposition(memory["C"], memory["B"], memory["D"])
    x = sample(rng, memory, B, D, params["n_dim"], params["pop_size"])
    memory["C"], memory["B"], memory["D"] = C, B, D
    return x, memory

@functools.partial(jit, static_argnums=(4, 5))
def sample(rng, memory, B, D, n_dim, pop_size):
    """ Jittable Multivariate Gaussian Sample Helper. """
    z = jax.random.normal(rng, (n_dim, pop_size)) # ~ N(0, I)
    y = B.dot(jnp.diag(D)).dot(z)                 # ~ N(0, C)
    y = jnp.swapaxes(y, 1, 0)
    x = memory["mean"] + memory["sigma"] * y      # ~ N(m, σ^2 C)
    return x

def tell_cma_strategy(x, fitness, params, memory):
    """ Update the surrogate ES model. """
    # Update/increase the generation counter
    memory["generation"] = memory["generation"] + 1
    # Sort new results, extract parents, store best performer
    concat_p_f = jnp.hstack([jnp.expand_dims(fitness, 1), x])
    sorted_solutions = concat_p_f[concat_p_f[:, 0].argsort()]
    # Update mean, isotropic/anisotropic paths, covariance, stepsize
    mean, y_k, y_w = update_mean(sorted_solutions, params, memory)
    memory["mean"] = mean
    p_sigma, C_2, C, B, D = update_p_sigma(y_w, params, memory)
    memory["p_sigma"], memory["C"], memory["B"], memory["D"] = p_sigma, C, B, D
    p_c, norm_p_sigma, h_sigma = update_p_c(y_w, params, memory)
    memory["p_c"] = p_c
    C = update_covariance(y_k, h_sigma, C_2, params, memory)
    memory["C"] = C
    sigma = update_sigma(norm_p_sigma, params, memory)
    memory["sigma"] = sigma
    return memory

# JIT-compiled version for tell interface
tell = jit(tell_cma_strategy)

def update_mean(sorted_solutions, params, memory):
    """ Update mean of strategy. """
    x_k = sorted_solutions[:, 1:]       # ~ N(m, σ^2 C)
    y_k_temp = (x_k - memory["mean"])   # ~ N(0, σ^2 C)
    y_w_temp = jnp.sum(y_k_temp.T * params["weights_truncated"], axis=1)
    mean = memory["mean"] + params["c_m"] * y_w_temp
    # Comple z-scoring for later updates
    y_k = y_k_temp/ memory["sigma"]
    y_w = y_w_temp/ memory["sigma"]
    return mean, y_k, y_w

def update_p_sigma(y_w, params, memory):
    """ Update evolution path for covariance matrix. """
    C, B, D = eigen_decomposition(memory["C"], memory["B"], memory["D"])
    C_2 = B.dot(jnp.diag(1 / D)).dot(B.T)  # C^(-1/2) = B D^(-1) B^T
    p_sigma_new = (1 - params["c_sigma"]) * memory["p_sigma"] + jnp.sqrt(
        (1 - (1 - params["c_sigma"])**2) *
        params["mu_eff"]) * C_2.dot(y_w)
    _B, _D = None, None
    return p_sigma_new, C_2, C, _B, _D

def update_p_c(y_w, params, memory):
    """ Update evolution path for sigma/stepsize. """
    norm_p_sigma = jnp.linalg.norm(memory["p_sigma"])
    h_sigma_cond_left = norm_p_sigma / jnp.sqrt(
        1 - (1 - params["c_sigma"]) ** (2 * (memory["generation"] + 1)))
    h_sigma_cond_right = (1.4 + 2 / (memory["mean"].shape[0] + 1)) * params["chi_n"]
    h_sigma = 1.0 * (h_sigma_cond_left < h_sigma_cond_right)
    p_c = (1 - params["c_c"]) * memory["p_c"] + h_sigma * jnp.sqrt((1 -
          (1 - params["c_c"])**2) * params["mu_eff"]) * y_w
    return p_c, norm_p_sigma, h_sigma

def update_covariance(y_k, h_sigma, C_2, params, memory):
    """ Update cov. matrix estimator using rank 1 + μ updates. """
    w_io = params["weights"] * jnp.where(params["weights"] >= 0, 1,
                                        memory["mean"].shape[0]/
            (jnp.linalg.norm(C_2.dot(y_k.T), axis=0) ** 2 + 1e-20))
    delta_h_sigma = (1 - h_sigma) * params["c_c"] * (2 - params["c_c"])
    rank_one = jnp.outer(memory["p_c"], memory["p_c"])
    rank_mu = jnp.sum(
        jnp.array([w * jnp.outer(y, y) for w, y in zip(w_io, y_k)]), axis=0)
    C = ((1 + params["c_1"] * delta_h_sigma - params["c_1"]
          - params["c_mu"] * jnp.sum(params["weights"])) * memory["C"]
         + params["c_1"] * rank_one + params["c_mu"] * rank_mu)
    return C

def update_sigma(norm_p_sigma, params, memory):
    """ Update stepsize sigma. """
    sigma = (memory["sigma"] * jnp.exp((params["c_sigma"] / params["d_sigma"])
                                      * (norm_p_sigma / params["chi_n"] - 1)))
    return sigma

def run_cma_es(rng, fitness_fct, num_generations, num_params, sigma_init,
               pop_size, elite_size, print_every_gen, top_k = 2):
    """ Run CMA-ES Pipeline for a number of generations. """
    mean_init = jnp.zeros(num_params)
    params, memory = init_cma_es(mean_init, sigma_init, pop_size, elite_size)
    evo_logger = init_logger(top_k, num_params)

    for generation in range(num_generations):
        rng, rng_input = jax.random.split(rng)
        x, memory = ask(rng_input, params, memory)
        value = fitness_fct(x)
        memory = tell(x, value, params, memory)
        evo_logger = update_logger(evo_logger, x, value, memory, top_k)
        if (generation + 1) % print_every_gen == 0:
            print("# Gen: {} | Fitness: {:.2f} | Params: {}".format(generation+1,
                                                      evo_logger["top_values"][0],
                                                      evo_logger["top_params"][0]))
    return evo_logger


if __name__ == '__main__':
    # Run the JAX CMA-ES on 3 2-d toy problems
    rng = jax.random.PRNGKey(0)
    print(10*"=" + " Start 6 Hump Camel Funct. " + 10*"=")
    camel_log = run_cma_es(rng, batch_hump_camel, 50, 2, 1, 4, 2, 10)
    print(10*"=" + " Start Himmelblau Function " + 10*"=")
    himmel_log = run_cma_es(rng, batch_himmelblau, 50, 2, 1, 4, 2, 10)
    print(10*"=" + " Start Rosenbrock Function " + 10*"=")
    rosen_log = run_cma_es(rng, batch_rosenbrock, 50, 2, 1, 4, 2, 10)
