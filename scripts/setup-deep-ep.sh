 # tar -xzf infiniband-headers.tgz -C $HOME/local/rdma/include
export CPATH=$HOME/local/rdma/include:$CPATH
export CPLUS_INCLUDE_PATH=$HOME/local/rdma/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=$HOME/local/rdma/include:$C_INCLUDE_PATH


# export NVSHMEM_DIR="$NPKG/nvshmem"
# export PATH="${NVSHMEM_DIR}/bin:$PATH"

export CPATH=$CUDA_HOME/include/cccl:$CPATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include/cccl:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=$CUDA_HOME/include/cccl:$C_INCLUDE_PATH


ln -sf $NVSHMEM_DIR/lib/libnvshmem_host.so.3 $NVSHMEM_DIR/lib/libnvshmem_host.so


# python -m pip install -v . --no-build-isolation --no-deps
# python -m pip wheel . --no-deps --no-build-isolation --no-index -w .
