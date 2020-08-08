all:

train:
	cd script.zeroth_korean && ./run_las_asr_trainer.sh

test:
	cd script.zeroth_korean && ./run_las_asr_decode.sh
