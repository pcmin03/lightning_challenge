CUDA_VISIBLE_DEVICES=6,7 python train.py trainer.gpus=2 +trainer.precision=16 logger=wandb +trainer.strategy=ddp
