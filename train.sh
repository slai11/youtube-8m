
MODEL_DIR=model_checkpoints/new_reswavenet
MODEL=ResWavenetModel

python train.py --train_data_pattern='dataset/training_set/train*.tfrecord' --frame_features=True --model=$MODEL --feature_names="rgb" --feature_sizes="1024" --train_dir=$MODEL_DIR --batch_size=64 --base_learning_rate=0.0001 --num_epochs=5


