
#TODO ques auto generate this object or this milk

import os
import re
import glob
import csv

###load model
import torch

from lavis.models import load_model, load_preprocess
from lavis.common.registry import registry

from omegaconf import OmegaConf
from generate import generate


import cv2
from PIL import Image as PIL_Image

import numpy as np

import time

vlm = load_model(
    name='blip2_t5_instruct',
    model_type='flant5xxl',
    checkpoint='../../pgvlm/pgvlm_weights.bin',  # replace with location of downloaded weights
    is_eval=True,
    #device="cuda" if torch.cuda.is_available() else "cpu"
    device="cpu"
)

vlm.qformer_text_input = False  # Optionally disable qformer text
model_cls = registry.get_model_class('blip2_t5_instruct')
model_type = 'flant5xxl'
preprocess_cfg = OmegaConf.load(model_cls.default_config_path(model_type)).preprocess
vis_processors, _ = load_preprocess(preprocess_cfg)
processor = vis_processors["eval"]

print('########node vlm is ready!!!########')
###load model


def gen_prompt(num,obj_name):

    obj_name = 'this object'
    
    questions = [
            'Question1: Does ' + obj_name + ' weigh a lot? Respond unknown if you are not sure. Short answer:',
            'Question2: Is ' + obj_name + ' easily breakable? Respond unknown if you are not sure. Short answer:',
            'Question3: Is ' + obj_name + ' easily bendable? Respond unknown if you are not sure. Short answer:',
            'Question4: What is ' + obj_name + ' made of? Respond unknown if you are not sure. Short answer:',
            'Question5: Would you describe ' + obj_name + ' as opaque, transparent, or translucent? Respond unknown if you are not sure. Short answer:',
            #'Question6: What does ' + obj_name + ' contain? Respond unknown if you are not sure. Short answer:',

            'Question6: What does this container contain? Respond unknown if you are not sure. Short answer:',
            'Question7: Is this container able to hold water inside easily? Respond unknown if you are not sure. Short answer:',
            'Question8: Is this container sealed shut? Respond unknown if you are not sure. Short answer:'
    ]
    #questions = [
    #        'Question1: Classify how heavy ' + obj_name + ' is heavy or light? Respond unknown if you are not sure. Short answer:',
    #        'Question2: Classify how easily ' + obj_name + ' can be broken or not? Respond unknown if you are not sure. Short answer:',
    #        'Question3: Classify ' + obj_name + ' as deformable,or undeformable? Respond unknown if you are not sure. Short answer:',
    #        'Question4: Classify ' + obj_name + ' that made of plastic, paper, glass, or ceramics? Respond unknown if you are not sure. Short answer:',
    #        'Question5: Classify ' + obj_name + ' as transparent, translucent,or opaque? Respond unknown if you are not sure. Short answer:',
    #        'Question6: Classify what are the contents of ' + obj_name + '? Respond unknown if you are not sure. Short answer:',
    #        'Question7: Classify ' + obj_name + ' as can contain liquid? Respond unknown if you are not sure. Short answer:',
    #        #'Question8: Classify this object as is sealed? Respond unknown if you are not sure. Short answer:',
    #        ##'Question8: Classify does ' + obj_name + ' have a cap? Respond unknown if you are not sure. Short answer:'
    #        'Question8: Classify Is the top of ' + obj_name + ' open? Respond unknown if you are not sure. Short answer:'
    #]
    return questions
    


def ques(num,obj_name,image,questions):
   for prompt in questions:
       question = {
               'prompt': prompt,
               'image': torch.stack([processor(image)], dim=0).to(vlm.device)
               }

       answers, scores = generate(vlm, question, length_penalty=0, repetition_penalty=1, num_captions=2)
       scores = np.exp(scores)
       array = scores.tolist()
       result = dict(zip(answers,array))
       print(prompt)
       print('answers:score(%)',result)
       keys = list(result.keys())
       values = list(result.values())
       

       write_csv(num,obj_name,prompt,keys[0],values[0],keys[1],values[1])


def write_csv(num,obj_name,prompt,key_0,values_0,key_1,values_1):
    with open('eval.csv', mode = 'a',newline = '') as file:
        writer = csv.writer(file)
        writer.writerow([num,obj_name,prompt,key_0,values_0,key_1,values_1])


current_directory = os.getcwd()
#ban pycache
folder_names = [name for name in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, name)) and name != "__pycache__"]
print(folder_names)

for image_dir in folder_names:
    print(image_dir)

    obj_name = str(image_dir)

    #get dir name and swap _ to space
    #obj_name = str(image_dir)
    match = re.search(r'([^/]+)/(open|close)$', obj_name)
    if match:
        obj_name = match.group(1)
    else:
        pass
    parts = obj_name.split('_')
    obj_name = ' '.join(parts[1:])
    num = int(parts[0])
    
    print(parts)
    #print(num)
    print(obj_name)
    
    # ディレクトリ内のファイル名を取得
    image_files = os.listdir(image_dir)
    #TODO sort
    #image_files.sort()
    image_files = sorted(image_files, key=lambda s: int(re.search(r'\d+', s).group()))
    print(image_files)
    current_index = 0
    
    while current_index < len(image_files):
        filename = image_files[current_index]
        image_path = os.path.join(image_dir, filename)
        print(image_path)
    
        image = cv2.imread(image_path)
        
        prompt = gen_prompt(num,obj_name)
        #print(prompt)
    
        if image is not None:

            cv2pil = PIL_Image.fromarray(image) #convert format cv2 to PIL
            ques(image_path,obj_name,cv2pil,prompt)
            #cv2.imshow('Image', image)
            #key = cv2.waitKey(0)

            current_index += 1

            
            #if key == 13:  # エンターキーが押されたら次の画像を表示
            #    current_index += 1
            #if key == 27:  # ESCキーが押されたら終了
            #    break
    
    #cv2.destroyAllWindows()
    
