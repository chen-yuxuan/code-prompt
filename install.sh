#!/bin/sh
pip install --upgrade pynvml
pip install nvidia-ml-py3
pip install .
$*
