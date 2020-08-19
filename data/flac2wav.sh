#!/bin/bash

# Ubuntu 기준 (root로 변경 가능한 아이디) 
# installation
# >> sudo apt install ffmpeg

# Convert all .flac files within this folder to .wav files

# find . -iname "*.flac" | wc

for flacfile in `find ./zeroth_korean/train_data_01 -iname "*.flac"`
do
    ffmpeg -y -i $flacfile -ab 64k -ar 16000 -ac 1 "${flacfile%.*}.wav"
done

for flacfile in `find ./zeroth_korean/test_data_01 -iname "*.flac"`
do
    ffmpeg -y -i $flacfile -ab 64k -ar 16000 -ac 1 "${flacfile%.*}.wav"
done
