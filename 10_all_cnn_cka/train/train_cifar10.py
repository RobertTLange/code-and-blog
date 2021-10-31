import math
import numpy as np
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from mle_logging import MLELogger
from mle_logging.utils import load_yaml_config
from models import All_CNN_C
from utils import get_dataloaders


class TrainState(train_state.TrainState):
    batch_stats: Any


def train(train_config, model_config, log_config):
    """Execute model training and evaluation loop."""
    log = MLELogger(
        **log_config,
        seed_id=train_config["seed_id"],
        config_dict={
            "train_config": train_config.toDict(),
            "model_config": model_config.toDict(),
            "log_config": log_config.toDict(),
        }
    )
    train_loader, test_loader = get_dataloaders(
        train_config["batch_size"],
        train_config["use_augment"],
        train_config["use_cutout"],
    )
    rng = jax.random.PRNGKey(train_config["seed_id"])
    step = 0

    # Initialize model and optimizer
    rng, init_rng = jax.random.split(rng)

    def create_train_state(rng, model_config, train_config):
        """Creates initial `TrainState`."""
        cnn = All_CNN_C(**model_config)
        cnn_params = cnn.init(rng, jnp.ones([1, 32, 32, 3]), train=False)
        # LRate schedule - decay after k epochs by constant multiplicative
        scale_dict = {}
        for ep in train_config["decay_after_epochs"]:
            step_id = ep * math.ceil(
                train_config["train_data_size"] / train_config["batch_size"]
            )
            scale_dict[step_id] = train_config["lrate_decay"]
        schedule_fn = optax.piecewise_constant_schedule(
            init_value=-train_config["lrate"], boundaries_and_scales=scale_dict
        )

        tx = optax.chain(
            optax.clip_by_global_norm(train_config["max_norm"]),
            optax.trace(
                decay=train_config["momentum"], nesterov=train_config["nesterov"]
            ),
            optax.scale_by_schedule(schedule_fn),
        )

        return TrainState.create(
            apply_fn=cnn.apply,
            params=cnn_params["params"],
            tx=tx,
            batch_stats=cnn_params["batch_stats"],
        )

    state = create_train_state(init_rng, model_config, train_config)

    @jax.jit
    def apply_model(rng, state, images, labels):
        """Compute grads, loss and accuracy (single batch)."""

        def loss_fn(params):
            logits, new_model_state = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                images,
                mutable=["batch_stats"],
                rngs={"dropout": rng},
            )
            one_hot = jax.nn.one_hot(labels, 10)
            l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(state.params))
            cent_loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot)
            loss = jnp.mean(cent_loss + train_config["w_decay"] * l2_loss)
            return loss, (logits, new_model_state)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params)
        logits, _ = aux[1]
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        return grads, aux, accuracy

    @jax.jit
    def update_model(state, grads, new_model_state):
        return state.apply_gradients(
            grads=grads, batch_stats=new_model_state["batch_stats"]
        )

    def eval_step(state, image, labels):
        variables = {"params": state.params, "batch_stats": state.batch_stats}
        logits = state.apply_fn(variables, image, train=False, mutable=False)
        one_hot = jax.nn.one_hot(labels, 10)
        l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(state.params))
        cent_loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot)
        loss = jnp.mean(cent_loss + 0.001 * l2_loss)
        acc = jnp.mean(jnp.argmax(logits, -1) == labels)
        return loss, acc

    # Store initial model checkpoint
    log.save_init_model({"params": state.params, "batch_stats": state.batch_stats})

    # Loop over training epochs and batches
    for epoch in range(train_config["num_epochs"]):
        epoch_loss, epoch_acc = [], []
        for batch_idx, (data, target) in enumerate(train_loader):
            rng, rng_drop = jax.random.split(rng)
            batch_images = jnp.array(data)
            batch_labels = jnp.array(target)
            grads, aux, acc = apply_model(rng_drop, state, batch_images, batch_labels)
            logits, new_model_state = aux[1]
            state = update_model(state, grads, new_model_state)
            epoch_loss.append(aux[0])
            epoch_acc.append(acc)
            step += 1

        # Mean training loss/acc and compute test performance
        train_loss = np.mean(epoch_loss)
        train_acc = np.mean(epoch_acc)

        test_loss, test_acc = 0, 0
        for batch_idx, (data, target) in enumerate(test_loader):
            batch_images = jnp.array(data)
            batch_labels = jnp.array(target)
            test_perf = eval_step(state, batch_images, batch_labels)
            test_loss += test_perf[0]
            test_acc += test_perf[1]
        test_loss /= 2
        test_acc /= 2

        # Update the logger
        log.update(
            {"num_updates": step, "num_epochs": epoch + 1},
            {
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
            },
            model={"params": state.params, "batch_stats": state.batch_stats},
            save=True,
        )
    return state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("--config_fname", required=True, type=str)
    args = vars(parser.parse_args())

    config = load_yaml_config(args["config_fname"], return_dotmap=True)
    train(config.train_config, config.model_config, config.log_config)
