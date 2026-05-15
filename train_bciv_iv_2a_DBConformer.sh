source /mnt/data/250010236/anaconda3/bin/activate
conda activate nisnn 
cd /mnt/data/250010236/DSAINet
python /mnt/data/250010236/DSAINet/train_loso.py \
 --config /mnt/data/250010236/DSAINet/config/DBConformer.yaml \
 --dataset BCIC-IV-2a
