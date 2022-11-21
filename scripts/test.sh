#!/usr/bin/env bash

set -x
GPUS=$1
# $@表示命令行所有参数，${@:2}表示从命令行的第二个参数开始截取到最后一个参数
PY_ARGS=${@:2}

CUDA_VISIBLE_DEVICES=${GPUS} python main.py --test ${PY_ARGS}