# Note: all paths referenced here are relative to the Docker container.
#
#source /tools/config.sh
# Activate your environment
#source activate py35
cd /storage/home/darwk/virtualenv
source python3_env/bin/activate

# Change to the directory in which your code is present
cd /storage/home/darwk/workspace/HypergraphDeepwalk/Hypergraph
# Run the code. The -u option is used here to use unbuffered writes
# so that output is piped to the file as and when it is produced.
python -u __main__.py --num_walks 10  --walk_length 25 --num_dimensions 16 --window_size 10 --num_neighbors 10 &> out
