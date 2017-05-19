MODEL_DIR=model/

python eval.py --eval_data_pattern='features/validate*.tfrecord' --frame_features=True --feature_names="rgb" --feature_sizes="1024" --batch_size=128 --model=LstmModel --train_dir=$MODEL_DIR/lstm_model --run_once=True
