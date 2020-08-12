import sys
import librosa
import soundfile as sf


def trim_wav(org_wav_path, trim_wav_path):
    y, sr = librosa.load(org_wav_path, mono=True, sr=16000)
    dur_y = librosa.get_duration(y, sr)
    
    yt, index = librosa.effects.trim(y, top_db=25)
    dur_yt = librosa.get_duration(yt, sr)
    
    print("{}\tdur_y: {:.4f}\tdur_yt: {:.4f}".format(org_wav_path, dur_y, dur_yt))
    
    y, sr = sf.read(org_wav_path, dtype='int16')
    y_trimmed = y[index[0]:index[1]]
    # print(len(y_trimmed)/float(sr))
    # print(y_trimmed)
    
    sf.write(trim_wav_path, y_trimmed, sr)


if __name__ == '__main__':
    trim_wav(sys.argv[1], sys.argv[2])