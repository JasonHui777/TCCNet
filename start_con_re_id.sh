device=$1
python train.py --config_file ./configs/MSMT17/vit_conreid_stride.yml MODEL.DEVICE_ID "('${device}')" || exit 1;