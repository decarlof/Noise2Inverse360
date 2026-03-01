eval "$(conda shell.bash hook)"
conda deactivate
conda activate "path to virtual environment"
python denoise_volume.py -gpus=0 -config=$1 -start_slice=$2 -end_slice=$3
conda deactivate