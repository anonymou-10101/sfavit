PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    torchrun \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=2 --master_port=56789 \
    $(dirname "$0")/tools/train.py $(dirname "$0")/configs/sfavit/upernet_sfavit_small.py \
    --launcher pytorch ${@:4} --work-dir ./workdir/upernet_sfavit_small_$(TZ=Asia/Seoul date +%Y-%m-%d_%H-%M-%S)
