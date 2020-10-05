 docker run -ti --rm -v $PWD:/workspace -w /workspace -u $(id -u):$(id -g) -v /data/ugui0/kawana/backup:/data ubuntu:18.04-env  rsync -avP out /data/occupancy_networks/
