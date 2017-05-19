echo "deleting archived models"

rm -rf model/lstm_model

echo "deleted, now running model"
MODEL_DIR=model/
python train.py --train_data_pattern='features2/train*.tfrecord' --frame_features=True --model=LstmModel --feature_names="rgb" --feature_sizes="1024" --train_dir=$MODEL_DIR/lstm_model --batch_size=128 --base_learning_rate=0.0002


