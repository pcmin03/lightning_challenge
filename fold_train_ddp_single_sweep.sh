CUDA_VISIBLE_DEVICES=1 python train.py trainer.gpus=1 +trainer.precision=16 logger=wandb datamodule.batch_size=10 datamodule.fold=2
