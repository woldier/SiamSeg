#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
# 是一种 Bash 脚本中的参数默认值赋值方式，主要作用是：如果环境变量 PORT 已经被定义并有值，那么就使用它的值；如果 PORT 没有被定义或为空，则使用默认值 29500
PORT=${PORT:-29500}
# # $(dirname $0)指的是当前脚本所在的目录, 这一步将 $(dirname $0)/.. 目录添加到 PYTHONPATH 的前面.
# 这里通过 ":" 把新的路径 $(dirname $0)/.. 添加在原有的 PYTHONPATH 之前（Linux 中路径列表使用 : 分隔）
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
# ${@:3}：表示从第 3 个参数开始的所有参数，具体到这里，就是跳过前两个参数，获取后面的所有参数。 这样通过bash 传递的从第三个起的参数将会给 train.py 如可以设置 --resume
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#torchrun --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}