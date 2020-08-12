# Convert all .flac files within this folder to .wav files

# find . -iname "*.flac" | wc

for wavfile in `find ./zeroth_korean/train_data_01 -iname "*.wav"`
do
    python trim_wav.py "${wavfile}" "${wavfile%.*}.trimmed.wav"
done

for wavfile in `find ./zeroth_korean/test_data_01 -iname "*.wav"`
do
    python trim_wav.py "${wavfile}" "${wavfile%.*}.trimmed.wav"
done
