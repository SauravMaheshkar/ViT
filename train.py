import os
from argparse import ArgumentParser

import tensorflow as tf

from src.nn import VisionTransformer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--image-size", default=32, type=int)
    parser.add_argument("--patch-size", default=4, type=int)
    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--d-model", default=64, type=int)
    parser.add_argument("--num-heads", default=4, type=int)
    parser.add_argument("--mlp-dim", default=128, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=4096, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    args = parser.parse_args()

    num_classes = 100
    input_shape = (32, 32, 3)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = VisionTransformer(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_layers=args.num_layers,
            num_classes=100,
            d_model=args.d_model,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            channels=3,
            dropout_rate=0.1,
        )
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            metrics=["accuracy"],
        )

    model.fit(x=x_train, y=y_train, epochs=args.epochs)
    model.save_weights(os.path.join(args.logdir, "vit"))
