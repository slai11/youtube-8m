MODEL_DIR=model/sq_wavenet_new_fast_reg_small_1024
MODEL=compressedResWavenetModel

python eval.py --eval_data_pattern='features/validate*.tfrecord' --frame_features=True --feature_names="rgb" --feature_sizes="1024" --batch_size=64 --model=$MODEL --train_dir=$MODEL_DIR --run_once=True --training=False
