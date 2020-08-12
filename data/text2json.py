import sys
import os
import pathlib
import json

from pprint import pprint


def read_text(text_file_path, dir_path):
    
    result = []
    
    with open(text_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.strip().split()
            # print(data)
            
            key = data[0]
            value = " ".join(data[1:])
            # print("K: {}, V: {}".format(key, value))
            
            wav_path = os.path.join(dir_path.split('/')[1], key.split("_")[1], key.split("_")[0], key + ".trimmed.wav")
            wav = str(pathlib.Path(wav_path))
            text = value
            speaker_id = key.split("_")[0]
                 
            sample = {
                "wav": wav,
                "text": text,
                "speaker_id": speaker_id
            }
            
            result.append(sample)
            
            # print(wav, text, speaker_id)
    
    # print(text_file_path, type(text_file_path))
    # print(text_file_path.relative_to('zeroth_korean/test_data_01'))
    
    return result
    

def text2json(dir_path, json_path):
    
    # 모든 하위 디렉토리 검색하여 txt 파일 목록 리스트화
    file_ext = r"**/*.txt"
    text_file_list = list(pathlib.Path(dir_path).glob(file_ext))
    print("text_list : {}".format(text_file_list))
    print()
    
    # 모든 목록 작성 
    data_total = []
    for text_file_path in text_file_list:
        data = read_text(text_file_path, dir_path)
        data_total = data_total + data
    
    print()
    pprint(data_total)
    
    # 파일로 기록하기
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data_total, indent=4, ensure_ascii=False))
    


if __name__ == '__main__':
    
    text2json(sys.argv[1], sys.argv[2])