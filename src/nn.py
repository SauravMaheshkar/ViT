from typing import Callable

import tensorflow as tf  # type: ignore

from src.attention import MultiHeadSelfAttention


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout_rate: float = 0.1,
        random_seed: int = 42,
        activation_fn: Callable = tf.nn.gelu,
        ln_eps: float = 1e-6,
        **kwargs
    ) -> None:
        super(TransformerBlock, self).__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.random_seed = random_seed
        self.activation_fn = activation_fn
        self.ln_eps = ln_eps

        self.att: tf.keras.layers.Layer = MultiHeadSelfAttention(
            self.embed_dim, self.num_heads
        )
        self.mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.mlp_dim, activation=self.activation_fn),
                tf.keras.layers.Dropout(self.dropout_rate, seed=self.random_seed),
                tf.keras.layers.Dense(self.embed_dim),
                tf.keras.layers.Dropout(self.dropout_rate, seed=self.random_seed),
            ]
        )
        self.layernorm1: tf.keras.layers.Layer = tf.keras.layers.LayerNormalization(
            epsilon=self.ln_eps
        )
        self.layernorm2: tf.keras.layers.Layer = tf.keras.layers.LayerNormalization(
            epsilon=self.ln_eps
        )
        self.dropout1: tf.keras.layers.Layer = tf.keras.layers.Dropout(
            dropout_rate, seed=self.random_seed
        )
        self.dropout2: tf.keras.layers.Layer = tf.keras.layers.Dropout(
            dropout_rate, seed=self.random_seed
        )

    @tf.function
    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        inputs_norm: tf.Tensor = self.layernorm1(inputs)
        attn_output: tf.Tensor = self.att(inputs_norm)
        attn_output = self.dropout1(attn_output, training=training)
        out1: tf.Tensor = attn_output + inputs

        out1_norm: tf.Tensor = self.layernorm2(out1)
        mlp_output: tf.Tensor = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output, training=training)
        return mlp_output + out1


class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_classes: int,
        d_model: int,
        num_heads: int,
        mlp_dim: int,
        channels: int = 3,
        dropout_rate: float = 0.1,
        random_seed: int = 42,
        activation_fn: Callable = tf.nn.gelu,
        ln_eps: float = 1e-6,
        **kwargs
    ) -> None:
        super(VisionTransformer, self).__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.channels = channels
        self.dropout_rate = dropout_rate
        self.random_seed = random_seed
        self.activation_fn = activation_fn
        self.ln_eps = ln_eps

        self.patch_dim: int = channels * patch_size ** 2
        self.num_patches: int = (image_size // patch_size) ** 2

        self.rescale: tf.keras.layers.Layer = tf.keras.layers.Rescaling(1.0 / 255)
        self.pos_emb = self.add_weight(
            "pos_emb", shape=(1, self.num_patches + 1, self.d_model)
        )
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, self.d_model))
        self.patch_proj: tf.keras.layers.Layer = tf.keras.layers.Dense(self.d_model)
        self.enc_layers = [
            TransformerBlock(
                d_model,
                num_heads,
                mlp_dim,
                dropout_rate,
                random_seed,
                activation_fn,
                ln_eps,
            )
            for _ in range(num_layers)
        ]
        self.mlp_head = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(epsilon=self.ln_eps),
                tf.keras.layers.Dense(mlp_dim, activation=self.activation_fn),
                tf.keras.layers.Dropout(dropout_rate, seed=self.random_seed),
                tf.keras.layers.Dense(num_classes),
            ]
        )

    @tf.function
    def extract_patches(self, images: tf.Tensor) -> tf.Tensor:
        batch_size: int = tf.shape(images)[0]
        patches: tf.Tensor = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    @tf.function
    def call(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        batch_size: int = tf.shape(x)[0]
        x = self.rescale(x)
        patches: tf.Tensor = self.extract_patches(x)
        x = self.patch_proj(patches)

        class_emb: tf.Tensor = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.d_model]
        )
        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb

        for layer in self.enc_layers:
            x = layer(x, training)

        # First (class token) is used for classification
        x = self.mlp_head(x[:, 0])
        return x
