source /mnt/data/250010236/anaconda3/bin/activate
conda activate nisnn 
cd /mnt/data/250010236/DSAINet
python /mnt/data/250010236/DSAINet/train_loso_ann_pretrain.py \
 --config /mnt/data/250010236/DSAINet/config/DSAINet_ANN2SNN.yaml \
 --dataset BCIC-IV-2a \
 --times 33 --epochs 150 --ann-epochs 50 \