To train on CIFAR100 run :-

```
sudo docker run --gpus all -v $PWD:/tmp -w /tmp -it tensorflow/tensorflow:latest-gpu python train.py
```
