
'''
NOTE: replace torchrun with torch.distributed.launch if you use older version of pytorch.
I suggest you use the same version as I do since I have not tested compatibility with older version after updating.
'''


## bisenetv1 cityscapes
export CUDA_VISIBLE_DEVICES=0
cfg_file=configs/ReHalf_U2NET_city.py
NGPUS=1
torchrun --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file


