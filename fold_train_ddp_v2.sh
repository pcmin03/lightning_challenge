for i in {0..5}
do
  CUDA_VISIBLE_DEVICES=0,1,2,3, python train.py trainer.gpus=4 datamodule.fold=$i +trainer.precision=16 logger=wandb +trainer.strategy=ddp model.net.encoder_name=efficientnet-b2
done
