
#rm -rf model/sq_wavenet_new_fast_reg_small

#MODEL_DIR=model/sq_wavenet_new_fast_reg_small_1024
MODEL_DIR=model/test1
MODEL=compressedResWavenetModel


python train.py --train_data_pattern='features2/train*.tfrecord' --frame_features=True --model=$MODEL --feature_names="rgb" --feature_sizes="1024" --train_dir=$MODEL_DIR --batch_size=64 --base_learning_rate=0.0001 --num_epochs=5


