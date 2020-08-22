#!env python

from flask import Flask
from flask import Response
from flask import request
from flask import jsonify

from werkzeug.utils import secure_filename

import json
import time
import os
import io
import wave

from pprint import pprint


#################################
import torch
import numpy as np
import librosa
import scipy

from data_loader import load_audio
import label_loader

from models import EncoderRNN, DecoderRNN, Seq2Seq

import shutil
from scipy.io import wavfile

##################################

app = Flask(__name__)


char2index = dict()
index2char = dict()
SOS_token = 0
EOS_token = 0
PAD_token = 0

model = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup():
    global char2index
    global index2char
    global SOS_token
    global EOS_token
    global PAD_token
    
    global model
    global device
    
    
    char2index, index2char = label_loader.load_label_json("../data/kor_syllable_zeroth.json")
    SOS_token = char2index['<s>']
    EOS_token = char2index['</s>']
    PAD_token = char2index['_']

    print(f"device: {device}")
    
    input_size = int(161)
    enc = EncoderRNN(input_size, 512, n_layers=3,
                     dropout_p=0.3, bidirectional=True, 
                     rnn_cell='LSTM', variable_lengths=False)

    dec = DecoderRNN(len(char2index), 128, 512, 512,
                     SOS_token, EOS_token,
                     n_layers=2, rnn_cell='LSTM', 
                     dropout_p=0.3, bidirectional_encoder=True)

    model = Seq2Seq(enc, dec).to(device)
    
    model_path = "../models/zeroth_korean_trimmed/LSTM_512x3_512x2_zeroth_korean_trimmed/final.pth"
    print("Loading checkpoint model %s" % model_path)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['model'])
    print('Model loaded')


def label_to_string(labels):
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token:
                break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents


def parse_audio(audio_path, sample_rate=16000, window_size=0.02, window_stride=0.01, normalize=True):
    
    y = load_audio(audio_path)

    n_fft = int(sample_rate * window_size)
    window_size = n_fft
    stride_size = int(sample_rate * window_stride)

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=stride_size, win_length=window_size, window=scipy.signal.hamming)
    spect, phase = librosa.magphase(D)

    # S = log(S+1)
    spect = np.log1p(spect)
    if normalize:
        mean = np.mean(spect)
        std = np.std(spect)
        spect -= mean
        spect /= std

    spect = torch.FloatTensor(spect)
    spect_length = torch.IntTensor([spect.size(1)])
    
    spect = spect.unsqueeze(0).unsqueeze(0)
    
    # print(spect.size())
    # print(spect_length.size(), spect_length)

    return spect, spect_length


def recognize(audio_path=None):
    rec = {}
    
    try:
        model.eval()
        with torch.no_grad():
            feats, feat_lengths = parse_audio(audio_path) 
            
            feats = feats.to(device)
            feat_lengths = feat_lengths.to(device)

            logit = model(feats, feat_lengths, None, teacher_forcing_ratio=0.0)
            logit = torch.stack(logit, dim=1).to(device)
            y_hat = logit.max(-1)[1]
            
            hyp_label = y_hat[0]
            hyp = label_to_string(hyp_label)
            rec['hyp'] = hyp
    
    except Exception as e:
        print("Execption: ", e)
        rec['hyp'] = ""
        
    return rec


def save(audio=None):
    save_dir = "audio"
    
    try:
        audio_filename = secure_filename(audio.filename)
        audio_path = os.path.join(save_dir, audio_filename)
        audio.save(audio_path)
    except Exception as e:
        print("Execption: ", e)
        return None
    finally:
        audio.close()

    return audio_path


@app.route('/predict', methods=['POST'])
def predict():
    start_time_save = time.time()
    audio_path = save(audio=request.files['audio'])
    end_time_save = time.time()
    
    start_time_reco = time.time()
    rec_result = recognize(audio_path=audio_path)
    end_time_reco = time.time()
    
    rec_result['save_time'] = float("{:.4f}".format(end_time_save - start_time_save))
    rec_result['reco_time'] = float("{:.4f}".format(end_time_reco - start_time_reco))
    rec_result['tota_time'] = float("{:.4f}".format(rec_result['save_time'] + rec_result['reco_time']))
    
    print("="*70)
    # print(f"audio_path: {audio_path}")
    pprint(f"rec_result: {rec_result}")
    print("="*70)
    
    return jsonify(rec_result)
    

if __name__ == '__main__':
    setup()
    app.run(host='0.0.0.0', port=5000, debug=True)