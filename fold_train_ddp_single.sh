for i in {0..5}
do
  CUDA_VISIBLE_DEVICES=6,7 python train.py trainer.gpus=1 +trainer.precision=16 logger=wandb datamodule.fold=$i
done