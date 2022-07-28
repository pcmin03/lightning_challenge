for i in {0..5}
do
	CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py trainer.gpus=4 datamodule.fold=$i +trainer.precision=16 logger=wandb +trainer.strategy=ddp_find_unused_parameters_false model=mri_3d datamodule=mri_3d datamodule.batch_size=3
done
