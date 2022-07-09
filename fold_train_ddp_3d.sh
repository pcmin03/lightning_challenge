for i in {0..5}
do
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py trainer.gpus=4 datamodule.fold=$i +trainer.precision=16 logger=wandb +trainer.strategy=ddp model.net.encoder_name=efficientnet-b1 model.net.classes=5 model=mri_3d datamodule=mri_3d datamodule.batch_size=30 datamodule.file_name=data_3d_fold.csv
done
