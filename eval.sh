MODEL_DIR=model_checkpoints/compressedreswavenet
MODEL=compressedResWavenetModel

python eval.py --eval_data_pattern='dataset/test_set/validate*.tfrecord' --frame_features=True --feature_names="rgb" --feature_sizes="1024" --batch_size=64 --model=$MODEL --train_dir=$MODEL_DIR --run_once=True --training=False
