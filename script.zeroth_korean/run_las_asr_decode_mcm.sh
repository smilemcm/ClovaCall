


--batch_size 16
--num_workers 8
--num_gpu 1
--rnn-type LSTM
--lr 3e-4
--learning-anneal 1.1
--dropout 0.3
--teacher_forcing 1.0
--encoder_layers 3 --encoder_size 512
--decoder_layers 2 --decoder_size 512
--train-file data/zeroth_korean/train_zeroth_korean.json
--test-file-list data/zeroth_korean/test_zeroth_korean.json
--labels-path data/kor_syllable_zeroth.json
--dataset-path data/zeroth_korean
--load-model --mode "test"
--max_len 128
--cuda
--save-folder models/"zeroth_korean"/LSTM_512x3_512x2_"zeroth_korean"
--model-path models/"zeroth_korean"/LSTM_512x3_512x2_"zeroth_korean"/final.pth
--log-path log/"zeroth_korean"/LSTM_512x3_512x2_"zeroth_korean"