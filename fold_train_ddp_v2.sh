for i in {0..5}
do
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py trainer.gpus=4 datamodule.fold=$i +trainer.precision=16 logger=wandb +trainer.strategy=ddp model.net.encoder_name=efficientnet-b1 model.net.decoder=deeplabv3plus model.net.classes=5 datamodule.batch_size=90 datamodule.file_name=25d_train_test_fold_v2.csv
done
