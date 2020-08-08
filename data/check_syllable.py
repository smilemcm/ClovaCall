import sys
import os
import pathlib
import json



def check(json_path, char2index):

    key_list = char2index.keys()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        for sample in data:
            text = sample['text']
            
            for char in text:
                if char not in key_list:
                    print(json_path, sample['wav'], text, char)


def get_vocab(kor_syllable_json):
    with open(kor_syllable_json, 'r', encoding='utf-8') as f:
        labels = json.load(f)
        
        char2index = dict()
        index2char = dict()

        for index, char in enumerate(labels):
            char2index[char] = index
            index2char[index] = char

        return char2index, index2char

def check_syllable(train_json, test_json, kor_syllable_json):
    
    char2index, index2char = get_vocab(kor_syllable_json)
    
    check(train_json, char2index)
    check(test_json, char2index)
    
    
    
    


if __name__ == '__main__':
    
    check_syllable(sys.argv[1], sys.argv[2], sys.argv[3])