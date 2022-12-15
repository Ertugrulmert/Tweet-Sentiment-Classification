#!/bin/bash
node ./jsonFormat.js
python main.py -eval -logfolder -save_dir ./json_result -testdata ./json_input/data.json -outsave result.json -load_from pretrained_models/word_model/model_50_word.pt  -input word -attention -bias -lowercase -bos -eos -brnn -batch_size 32 -dropout 0.5 -emb_size 100 -end_epoch 50 -layers 3 -learning_rate_decay 0.05 -lr 0.01 -max_grad_norm 5 -rnn_size 200 -rnn_type 'LSTM' -tie_decoder_embeddings -share_embeddings -share_vocab -start_decay_after 15 -teacher_forcing_ratio 0.6  -max_train_decode_len 100
node ./jsonFormatReverse.js