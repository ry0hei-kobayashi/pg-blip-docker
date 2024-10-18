import torch
from PIL import Image
from omegaconf import OmegaConf

from lavis.models import load_model, load_preprocess, load_model_and_preprocess
from lavis.common.registry import registry

#import requests
from accelerate import dispatch_model, infer_auto_device_map, load_checkpoint_and_dispatch

from generate import generate

example_image = Image.open('images/tuna.jpg').convert("RGB")

def get_blip_model(device='cuda', dtype=torch.bfloat16, use_multi_gpus=True):
    #torch.half
    model, vis_processors, txt_processors = load_model_and_preprocess(
                    name='blip2_t5_instruct',
                    model_type='flant5xxl',
                    #load_in_8bit=True,
                    is_eval=True,
                )
    model.to(dtype)
    if use_multi_gpus:
        device_map = infer_auto_device_map(
                model, 
                max_memory={0: "10GiB", 1: "10GiB"}, 
                no_split_module_classes=['LlamaDecoderLayer', 'VisionTransformer']
        )

        device_map['llm_model.lm_head'] = device_map['llm_proj'] = device_map['llm_model.model.embed_tokens']
        print(device_map)

        model = dispatch_model(model, device_map=device_map)
        torch.cuda.empty_cache()

    else:
        model.to('cuda:0')
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    return model, (txt_processors, vis_processors)

get_blip_model()

vlm = load_model(
    name='blip2_t5_instruct',
    model_type='flant5xxl',
    checkpoint='../pgvlm/pgvlm_weights.bin',  # replace with location of downloaded weights
    is_eval=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
    #device="cpu"
)

vlm.qformer_text_input = False  # Optionally disable qformer text

model_cls = registry.get_model_class('blip2_t5_instruct')
model_type = 'flant5xxl'
preprocess_cfg = OmegaConf.load(model_cls.default_config_path(model_type)).preprocess
vis_processors, _ = load_preprocess(preprocess_cfg)
processor = vis_processors["eval"]

question_samples = {
    #'prompt': 'Question: Classify this object as transparent, translucent, or opaque? Respond unknown if you are not sure. Short answer:',
    #deformable, undeformable, or no
    'prompt': 'Question: Classify this object as deformable or, undeformable? Respond unknown if you are not sure. Short answer:',
    'image': torch.stack([processor(example_image)], dim=0).to(vlm.device)
}

answers, scores = generate(vlm, question_samples, length_penalty=0, repetition_penalty=1, num_captions=2)
print(answers, scores)
# ['opaque', 'translucent', 'transparent'] tensor([-0.0373, -4.2404, -4.4436], device='cuda:0')
