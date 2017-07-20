echo "deleting archived models"

#rm -rf model/sq_wavenet_new_fast_reg_small
echo "deleted, now running model"

MODEL_DIR=model/sq_wavenet_new_fast_reg_small_1024
#for i in 1 2 3 4 5
#do 
python train.py --train_data_pattern='features2/train*.tfrecord' --frame_features=True --model=finalModel --feature_names="rgb" --feature_sizes="1024" --train_dir=$MODEL_DIR --batch_size=64 --base_learning_rate=0.0001 --num_epochs=5

 # python eval.py --eval_data_pattern='features/validate*.tfrecord' --frame_features=True --feature_names="rgb" --feature_sizes="1024" --batch_size=64 --model=LstmModel --train_dir=$MODEL_DIR --run_once=True --training=False
#done

#MODEL_DIR=model/run1_1e-5_50epoch_bs64
#python train.py --train_data_pattern='features2/train*.tfrecord' --frame_features=True --model=RDCModel --feature_names="rgb" --feature_sizes="1024" --train_dir=$MODEL_DIR --batch_size=64 --base_learning_rate=0.00001 --num_epochs=50



