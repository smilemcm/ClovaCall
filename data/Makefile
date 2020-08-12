all:

text2json:
	python text2json.py zeroth_korean/train_data_01 train_zeroth_korean.trimmed.json
	python text2json.py zeroth_korean/test_data_01 test_zeroth_korean.trimmed.json

flac2wav:
	./flac2wav.sh

trim_wav:
	./trim_wav.sh

check_vocab:
	python check_syllable.py train_zeroth_korean.json test_zeroth_korean.json kor_syllable.json

