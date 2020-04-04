docker run --net=host --shm-size=1024G -ti --rm -u $(id -u):$(id -g) -w /workspace -v $PWD:/workspace trimesh
