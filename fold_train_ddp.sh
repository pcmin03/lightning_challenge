for i in {0..5}
do
  CUDA_VISIBLE_DEVICES=6,7 python train.py trainer.gpus=2 datamodule.fold=$i +trainer.precision=16 logger=wandb +trainer.strategy=ddp model.net.background=efficientnet-b1
done
