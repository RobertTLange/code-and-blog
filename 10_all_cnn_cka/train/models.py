import jax.numpy as jnp
from flax import linen as nn


def conv_bn_relu_block(x, features, kernel_size, strides, train, padding="SAME"):
    x = nn.Conv(
        features=features,
        kernel_size=kernel_size,
        strides=strides,
        use_bias=False,
        padding=padding,
    )(x)
    x = nn.BatchNorm(
        use_running_average=not train, momentum=0.9, epsilon=1e-5, dtype=jnp.float32
    )(x)
    x = nn.relu(x)
    return x


class All_CNN_C(nn.Module):
    """All-CNN-C architecture as in Springenberg et al. (2015)."""

    num_classes: int
    depth: int
    dropout_input: float
    dropout_hidden: float

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Block 0: Dropout on 20% of input image
        x = nn.Dropout(rate=self.dropout_input)(x, deterministic=not train)

        # Block 1 a): 3 × 3 conv. 96-BN-ReLu ×(3n − 1)
        for i in range(3 * self.depth - 1):
            x = conv_bn_relu_block(x, 96, (3, 3), (1, 1), train)
        # Block 1 b): 3 × 3 conv. 96 stride 2-BN-ReLu
        x = conv_bn_relu_block(x, 96, (3, 3), (2, 2), train)
        x = nn.Dropout(rate=self.dropout_hidden)(x, deterministic=not train)

        # Block 2 a): 3 × 3 conv. 192-BN-ReLu ×(3n − 1)
        for i in range(3 * self.depth - 1):
            x = conv_bn_relu_block(x, 192, (3, 3), (1, 1), train)
        # Block 2 b): 3 × 3 conv. 192 stride 2-BN-ReLu
        x = conv_bn_relu_block(x, 192, (3, 3), (2, 2), train)
        x = nn.Dropout(rate=self.dropout_hidden)(x, deterministic=not train)

        # Block 3 a): 3 × 3 conv. 192 BN-ReLu ×(n − 1)
        for i in range(self.depth - 1):
            x = conv_bn_relu_block(x, 192, (3, 3), (1, 1), train)
        # Block 3 b): 3×3 conv. 192 valid padding-BN-ReLu
        x = conv_bn_relu_block(x, 192, (3, 3), (1, 1), train, "VALID")

        # Block 4: 1 × 1 conv. 192-BN-ReLu ×n
        x = conv_bn_relu_block(x, 192, (1, 1), (1, 1), train)

        # Block 5: 1 × 1 conv. num_outputs-BN-ReLu ×n
        x = conv_bn_relu_block(x, self.num_classes, (1, 1), (1, 1), train)

        # Global average pooling -> logits
        x = nn.avg_pool(x, window_shape=(6, 6), strides=None, padding="VALID")
        x = jnp.squeeze(x, axis=(1, 2))
        return x


class All_CNN_C_Features(nn.Module):
    """All-CNN-C architecture as in Springenberg et al. (2015)."""

    num_classes: int
    depth: int
    dropout_input: float
    dropout_hidden: float

    @nn.compact
    def __call__(self, x, train: bool = True):
        all_features = []
        # Block 0: Dropout on 20% of input image
        x = nn.Dropout(rate=self.dropout_input)(x, deterministic=not train)

        # Block 1 a): 3 × 3 conv. 96-BN-ReLu ×(3n − 1)
        for i in range(3 * self.depth - 1):
            x = conv_bn_relu_block(x, 96, (3, 3), (1, 1), train)
            all_features.append(x.reshape((x.shape[0], -1)))
        # Block 1 b): 3 × 3 conv. 96 stride 2-BN-ReLu
        x = conv_bn_relu_block(x, 96, (3, 3), (2, 2), train)
        x = nn.Dropout(rate=self.dropout_hidden)(x, deterministic=not train)
        all_features.append(x.reshape((x.shape[0], -1)))

        # Block 2 a): 3 × 3 conv. 192-BN-ReLu ×(3n − 1)
        for i in range(3 * self.depth - 1):
            x = conv_bn_relu_block(x, 192, (3, 3), (1, 1), train)
            all_features.append(x.reshape((x.shape[0], -1)))
        # Block 2 b): 3 × 3 conv. 192 stride 2-BN-ReLu
        x = conv_bn_relu_block(x, 192, (3, 3), (2, 2), train)
        x = nn.Dropout(rate=self.dropout_hidden)(x, deterministic=not train)
        all_features.append(x.reshape((x.shape[0], -1)))

        # Block 3 a): 3 × 3 conv. 192 BN-ReLu ×(n − 1)
        for i in range(self.depth - 1):
            x = conv_bn_relu_block(x, 192, (3, 3), (1, 1), train)
            all_features.append(x.reshape((x.shape[0], -1)))
        # Block 3 b): 3×3 conv. 192 valid padding-BN-ReLu
        x = conv_bn_relu_block(x, 192, (3, 3), (1, 1), train, "VALID")
        all_features.append(x.reshape((x.shape[0], -1)))

        # Block 4: 1 × 1 conv. 192-BN-ReLu ×n
        x = conv_bn_relu_block(x, 192, (1, 1), (1, 1), train)
        all_features.append(x.reshape((x.shape[0], -1)))

        # Block 5: 1 × 1 conv. num_outputs-BN-ReLu ×n
        x = conv_bn_relu_block(x, self.num_classes, (1, 1), (1, 1), train)
        all_features.append(x.reshape((x.shape[0], -1)))

        # Global average pooling -> logits
        x = nn.avg_pool(x, window_shape=(6, 6), strides=None, padding="VALID")
        x = jnp.squeeze(x, axis=(1, 2))
        all_features.append(x.reshape((x.shape[0], -1)))
        return all_features


class LogisticRegression(nn.Module):
    """Logistic Regression for Linear Probes."""

    num_classes: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=self.num_classes)(x)
