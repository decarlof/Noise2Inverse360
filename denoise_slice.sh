eval "$(conda shell.bash hook)"
conda deactivate
conda activate "path to virtual environment"
python denoise_slice.py -gpus=0 -config=$1 -slice_number=$2
conda deactivate