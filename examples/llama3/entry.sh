export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RANK=$OMPI_COMM_WORLD_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE

python $*