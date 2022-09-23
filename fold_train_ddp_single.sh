HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=1 python  train.py trainer.gpus=1 +trainer.precision=16 logger=wandb datamodule.batch_size=5 datamodule.fold=0
