for i in {0..5}
do
  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python train.py trainer.gpus=5 datamodule.fold=$i +trainer.precision=16 logger=wandb +trainer.strategy=ddp model.net.background=efficientnet-b2 datamodule.batch_size=10
done
