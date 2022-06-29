CUDA_VISIBLE_DEVICES=6,7 python train.py trainer.gpus=1 +trainer.precision=16 logger=wandb model.net.encoder_name='efficientnet-b0'
