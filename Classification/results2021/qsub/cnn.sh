#!/bin/bash
#$ -cwd
#$ -l rt_G.large=1
#$ -l h_rt=24:00:00

# Lines starting with "#$" are regarded as qsub options:
# "-cwd" means output files are written in the current working directory.
#
# A more practical example of ABCI job script:
# how to specify resources, use groups and disk spaces etc
#
# "rt_C.small" is the predefined set of resources including
# CPU, GPU, memory, maximum computation time, points etc.
#
# Examples of predefined resource sets:
# rt_C.small
# 5CPU / 30GB memory / max 168h / 0.2p/h
# rt_C.large
# 20CPU / 120GB memory / max 72h / 0.6p/h
# rt_G.small
# 5CPU / 1GPU / 60GB memory / max 168h / 0.3p/h
# rt_G.large
# 20CPU / 4GPU / 240GB memory / max 72h / 0.9p/h
# See also https://docs.abci.ai/ja/03/
#
# "rt_C.small=[int]" means requiring multiple rt_C.small over [int] nodes.
# "-l h_rt=1:00:00" specify the computation time (here 1 hour)
#
# Submit this script using the following command:
# qsub -g gac50426 abci_example2.sh
#
# "-g gac50426" specify the group whose points will be consumed.
#
# In ABCI, jobs consume "points" purchased with real money.
# Each ABCI "group" has separate points.
# "show_point" command shows available groups and there points.
# Disk space also consumes points.
# Currently, we use gaa50089 for disk space and other groups for jobs.

# In ABCI, popular packages are provided as modules.
# Consider to use modules before trying to install by yourself.
# You can see the list of available modules by "module avail 2>&1 | less"
#
# In ABCI, modules are inactive by default in child nodes.
# To activate
source /etc/profile.d/modules.sh
# Below is a typical workflow to use modules.
# First, unload all modules to prevent possible contamination.
module purge

# ここから各自修正する
# Then, load necessary modules（自分がロードしたいモジュールロードする）
module load python/3.6/3.6.12 cuda/11.0/11.0.3 cudnn/8.1/8.1.1 gcc/7.4.0

# If you create a python virtual environment using python provided by a modules,
# the environment should be activated after loading the corresponding python module.
# 自分の環境を配置する
source ~/jupyter_env/bin/activate

# In ABCI, the space of home directory is limited to ~100GB.
# Large files should be saved in group directory.
# /groups/gaa50089
#
# How to use directories efficiently:
# Prepare a symbolic link from home directory to group directory.
# (e.g. ln -s /groups/gaa50089/[yourname] $HOME/work)
# then define variables like this
# $data変数はjobを投げる時に設定する、今回はforで実行したいデータセットを繰り返すあげる
PYDIR=$HOME/DDDog/NDAcquisition/Classification
mask=$mask
lens=$lens
mmmodel=$mmmodel

python $PYDIR/ClassifyCNN.py $mask $lens $mmmodel

