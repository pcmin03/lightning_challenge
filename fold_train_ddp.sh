for i in {2..9}
do
  CUDA_VISIBLE_DEVICES=1,2,3,4 python train.py trainer.gpus=4 datamodule.fold=$i +trainer.precision=16 logger=wandb +trainer.strategy=ddp model.net.encoder_name=efficientnet-b1
done
