from typing import Sequence

import tensorflow as tf  # type: ignore

__all__ = ["MultiHeadSelfAttention"]


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int = 8, **kwargs) -> None:
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim: int = embed_dim // num_heads
        self.query_dense: tf.keras.layers.Layer = tf.keras.layers.Dense(embed_dim)
        self.key_dense: tf.keras.layers.Layer = tf.keras.layers.Dense(embed_dim)
        self.value_dense: tf.keras.layers.Layer = tf.keras.layers.Dense(embed_dim)
        self.combine_heads: tf.keras.layers.Layer = tf.keras.layers.Dense(embed_dim)

    @tf.function
    def attention(
        self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor
    ) -> Sequence[tf.Tensor]:
        score: tf.Tensor = tf.matmul(query, key, transpose_b=True)
        dim_key: tf.Tensor = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score: tf.Tensor = score / tf.math.sqrt(dim_key)
        weights: tf.Tensor = tf.nn.softmax(scaled_score, axis=-1)
        output: tf.Tensor = tf.matmul(weights, value)
        return output, weights

    @tf.function
    def separate_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size: int = tf.shape(inputs)[0]
        query: tf.Tensor = self.query_dense(inputs)
        key: tf.Tensor = self.key_dense(inputs)
        value: tf.Tensor = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, _ = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention: tf.Tensor = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output: tf.Tensor = self.combine_heads(concat_attention)
        return output
