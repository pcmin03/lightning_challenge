CUDA_VISIBLE_DEVICES=6 python train.py trainer.gpus=1 +trainer.precision=16 logger=wandb datamodule.batch_size=4 datamodule=mri_3d model=mri_3d datamodule.fold=3 callbacks=default &
CUDA_VISIBLE_DEVICES=7 python train.py trainer.gpus=1 +trainer.precision=16 logger=wandb datamodule.batch_size=4 datamodule=mri_3d model=mri_3d datamodule.fold=4 callbacks=default
