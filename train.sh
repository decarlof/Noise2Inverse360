eval "$(conda shell.bash hook)"
conda deactivate
conda activate "path to virtual environment"
python -m torch.distributed.run --nproc_per_node=2 main.py -gpus=0,1 -config=$1
conda deactivate