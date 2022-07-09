CUDA_VISIBLE_DEVICES=2 python train.py trainer.gpus=1 +trainer.precision=16 logger=wandb datamodule.batch_size=10 datamodule=mri_3d model=mri_3d datamodule.fold=2 callbacks=default 
