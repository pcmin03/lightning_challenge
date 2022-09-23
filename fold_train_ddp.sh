for i in {0..5}
do
  HYDRA_FULL_ERROR=1 python train.py trainer.gpus=6 datamodule.fold=$i +trainer.precision=16 logger=wandb +trainer.strategy=ddp datamodule.batch_size=2 
done
