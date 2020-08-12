all:

train:
	cd script.zeroth_korean && ./run_las_asr_trainer.sh

train.trim:
	cd script.zeroth_korean && ./run_las_asr_trainer.trimmed.sh

test:
	cd script.zeroth_korean && ./run_las_asr_decode.sh

test.trim:
	cd script.zeroth_korean && ./run_las_asr_decode.trimmed.sh
