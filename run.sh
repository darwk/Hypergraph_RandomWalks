# Note: all paths referenced here are relative to the Docker container.
#
# Add the Nvidia drivers to the path
export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
# Tools config for CUDA, Anaconda installed in the common /tools directory
source /tools/config.sh
# Activate your environment
source activate py35
# Change to the directory in which your code is present
cd /path/to/run.sh
# Run the code. The -u option is used here to use unbuffered writes
# so that output is piped to the file as and when it is produced.
# Here, the code is the MNIST Tensorflow example.
python __main__.py --num_walks 10  --walk_length 25 --num_dimensions 16 --window_size 10 --num_neighbors 10 &> out