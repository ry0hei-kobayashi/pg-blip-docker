import torch

from lavis.models import load_model, load_preprocess
from lavis.common.registry import registry

from omegaconf import OmegaConf
from generate import generate

import rospy
from sensor_msgs.msg import Image

import cv2
from PIL import Image as PIL_Image
from cv_bridge import CvBridge

import numpy as np

import time

bridge = CvBridge()

vlm = load_model(
    name='blip2_t5_instruct',
    model_type='flant5xxl',
    checkpoint='../pgvlm/pgvlm_weights.bin',  # replace with location of downloaded weights
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

#obj_name = 'the plastic bottle of water'
#obj_name = 'the plastic bottle of milk'
#obj_name = 'this pak of milk'
obj_name = 'this object'
#obj_name = 'the cracker box'

questions = [
        'Question1: Classify how heavy ' + obj_name + ' is heavy or light? Respond unknown if you are not sure. Short answer:',
        'Question2: Classify how easily ' + obj_name + ' can be broken or not? Respond unknown if you are not sure. Short answer:',
        'Question3: Classify ' + obj_name + ' as deformable,or undeformable? Respond unknown if you are not sure. Short answer:',
        'Question4: Classify ' + obj_name + ' that made of plastic, paper, glass, or ceramics? Respond unknown if you are not sure. Short answer:',
        'Question5: Classify ' + obj_name + ' as transparent, translucent,or opaque? Respond unknown if you are not sure. Short answer:',
        'Question6: Classify what are the contents of ' + obj_name + '? Respond unknown if you are not sure. Short answer:',
        'Question7: Classify ' + obj_name + ' as can contain liquid? Respond unknown if you are not sure. Short answer:',
        #'Question8: Classify this object as is sealed? Respond unknown if you are not sure. Short answer:',
        ##'Question8: Classify does ' + obj_name + ' have a cap? Respond unknown if you are not sure. Short answer:'
        'Question8: Classify Is the top of ' + obj_name + ' open? Respond unknown if you are not sure. Short answer:'
]

class VLM:
    def __init__(self):

        self.vlm = vlm
        self.processor = processor

        self.frame_count = 0
        self.start_time = time.time()

        self.sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/rgb/image_rect_color', Image, self.image_callback)

    def image_callback(self, msg):

        cv_image = bridge.imgmsg_to_cv2(msg, 'bgr8')
        bgr2rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        #show_image = cv2.imshow('vlm window',bgr2rgb)
        #cv2.waitKey(1)

        cv2pil = PIL_Image.fromarray(bgr2rgb) #convert format cv2 to PIL
        ques = self.question(cv2pil)
        
        #for fps calc
        self.frame_count += 1
        if time.time() - self.start_time >= 1:
            fps = self.frame_count / (time.time() - self.start_time)
            print(f"FPS: {fps:.3f}")
            self.frame_count = 0
            self.start_time = time.time()

    def question(self, image):
        for prompt in questions:
            question = {
                    'prompt': prompt,
                    'image': torch.stack([self.processor(image)], dim=0).to(self.vlm.device)
                    }
    
            answers, scores = generate(self.vlm, question, length_penalty=0, repetition_penalty=1, num_captions=2)
            scores = np.exp(scores)
            array = scores.tolist()
            result = dict(zip(answers,array))
            print(prompt)
            print('answers:score(%)',result)

if __name__ == "__main__":

    rospy.init_node('vlm_realtime')
    vlm = VLM()

    rospy.spin()
