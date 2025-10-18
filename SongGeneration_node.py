# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import numpy as np
import io
import torchaudio
from .node_utils import gc_clear
from .generate import auto_prompt_type,infer_stage2,inference_lowram_final,build_model,Separator,song_infer_lowram
import time
import folder_paths

MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))

from .SongGeneration.codeclm.models import builders
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

# add checkpoints dir
SongGeneration_Weigths_Path = os.path.join(folder_paths.models_dir, "SongGeneration")
if not os.path.exists(SongGeneration_Weigths_Path):
    os.makedirs(SongGeneration_Weigths_Path)
folder_paths.add_model_folder_path("SongGeneration", SongGeneration_Weigths_Path)



class SongGeneration_Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "infer_model": (["none"] +[i for i in folder_paths.get_filename_list("SongGeneration") if i.endswith(".pt") ],),
            },
        }

    RETURN_TYPES = ("SongGeneration_Audiolm","SongGeneration_Cfg")
    RETURN_NAMES = ("model","cfg")
    FUNCTION = "main"
    CATEGORY = "SongGeneration"

    def main(self, infer_model,):
        infer_model_path=folder_paths.get_full_path("SongGeneration", infer_model) if infer_model != "none" else None
        assert infer_model_path is not None ,"模型不能为空.need infer model"
        model,cfg=build_model(os.path.join(SongGeneration_Weigths_Path, "ckpt"),infer_model_path)
        return (model,cfg)



class SongGeneration_Stage1:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {              
                "vae": (folder_paths.get_filename_list("vae"),),
                "seperate_model": (["none"] + [i for i in folder_paths.get_filename_list("SongGeneration") if i.endswith(".safetensors") and not "fix" in i.lower()],),  
                "prompt_pt":  (["none"] + [i for i in folder_paths.get_filename_list("SongGeneration") if "prompt" in i.lower()],),
                "auto_prompt_audio_type": (auto_prompt_type,),
                "model_1rvq": (["none"] + [i for i in folder_paths.get_filename_list("SongGeneration") if i.endswith(".safetensors")],),
                "demucs_pt":  (["none"] + [i for i in folder_paths.get_filename_list("SongGeneration") if i.endswith(".pth")],),
            },
             "optional": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("SongGeneration_Cond",)
    RETURN_NAMES = ("cond",)
    FUNCTION = "main"
    CATEGORY = "SongGeneration"

    def main(self,vae,seperate_model, auto_prompt_audio_type,prompt_pt,model_1rvq,demucs_pt,**kwargs):

        audio=kwargs.get("audio", None)
        model_sep_path=folder_paths.get_full_path("SongGeneration", seperate_model) if seperate_model != "none" else None
        vae_model=folder_paths.get_full_path("vae", vae)
        prompt_pt_path=folder_paths.get_full_path("SongGeneration", prompt_pt) if prompt_pt != "none" else None
    
        if audio is not None:
            
            prompt_audio_path = os.path.join(folder_paths.get_input_directory(), f"audio_{time.strftime('%m%d%H%S')}_temp.wav")
            waveform=audio["waveform"].squeeze(0)
            buff = io.BytesIO()
            torchaudio.save(buff, waveform, audio["sample_rate"], format="FLAC")
            with open(prompt_audio_path, 'wb') as f:
                f.write(buff.getbuffer())
            use_descriptions=False #不建议同时提供参考音频和描述文本

            dm_model_path=folder_paths.get_full_path("SongGeneration", demucs_pt) if demucs_pt != "none" else None
            assert dm_model_path is not None ,"使用参考音频时，需要选择htdemucs模型， if use audio need htdemucs model"
            separator = Separator(dm_model_path, os.path.join(current_node_path, "SongGeneration/third_party/demucs/ckpt/htdemucs.yaml"))

            model_1rvq_path=folder_paths.get_full_path("SongGeneration", model_1rvq) if model_1rvq != "none" else None
            assert model_1rvq_path is not None ,"使用参考音频时，需要选择model_模型， if use audio need model_odel"
            audio_tokenizer = builders.get_audio_tokenizer_model(f"Flow1dVAE1rvq_{model_1rvq_path}",os.path.join(current_node_path, f'SongGeneration/conf/stable_audio_1920_vae.json'),vae_model,'inference')
     
            seperate_tokenizer = builders.get_audio_tokenizer_model(f"Flow1dVAESeparate_{model_sep_path}",os.path.join(current_node_path, f'SongGeneration/conf/stable_audio_1920_vae.json'),vae_model,'inference')
                
        else:
            prompt_audio_path,use_descriptions,audio_tokenizer,separator,seperate_tokenizer=None,True,None,None,None

        original_item=song_infer_lowram(seperate_tokenizer,separator,audio_tokenizer,prompt_pt_path, folder_paths.get_output_directory(),prompt_audio_path,auto_prompt_audio_type,)

        gc_clear()
        print("Stage1 is done.")
        return ({"item": original_item, "use_descriptions": use_descriptions,"model_sep_path":model_sep_path,"vae_model":vae_model},)

class SongGeneration_Stage2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":  ("SongGeneration_Audiolm",),
                "cfg": ("SongGeneration_Cfg",),
                "cond": ("SongGeneration_Cond",),
                "lyric": ("STRING", {"multiline": True, "default": "[intro-short] ;\n [verse]\n 雪花舞动在无尽的天际.情缘如同雪花般轻轻逝去.希望与真挚.永不磨灭.你的忧虑.随风而逝 ;\n [chorus]\n 我怀抱着守护这片梦境.在这世界中寻找爱与虚幻.苦辣酸甜.我们一起品尝.在雪的光芒中.紧紧相拥 ;\n [inst-short] ;\n [verse]\n雪花再次在风中飘扬.情愿如同雪花般消失无踪.希望与真挚.永不消失.在痛苦与喧嚣中.你找到解脱 ;\n [chorus]\n 我环绕着守护这片梦境.在这世界中感受爱与虚假.苦辣酸甜.我们一起分享.在白银的光芒中.我们同在 ;\n [outro-short]"}),
                "description": ("STRING", {"multiline": False, "default": "female, dark, pop, sad, piano and drums, the bpm is 125"}), #OPTIONAL
                "cfg_coef": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 3.0, "step": 0.1}),
                "temp": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "record_tokens": ("BOOLEAN", {"default": True}),
                "record_window": ("INT", {"default": 50, "min": 1, "max": 1000, "step": 1}),                 
            },
        }

    RETURN_TYPES = ("SongGeneration_Cond",)
    RETURN_NAMES = ("cond",)
    FUNCTION = "main"
    CATEGORY = "SongGeneration"

    def main(self, model,cfg,cond,lyric,description,cfg_coef,temp,top_k,top_p,record_tokens,record_window):

        descriptions=description if cond.get("use_descriptions",False) else None
        
        items=infer_stage2(cond.get("item"),model,cfg.max_dur,lyric,descriptions,cfg_coef, temp,top_k,top_p,record_tokens ,record_window )
        gc_clear()
        return ({"items":items,"cfg":cfg,"model_sep_path":cond["model_sep_path"],"vae_model":cond["vae_model"]},)



class SongGeneration_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond": ("SongGeneration_Cond",),
                "gen_type": (["mixed","bgm","vocal",],), 
                "save_separate": ("BOOLEAN", {"default": False}),
            }
            }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio", )
    FUNCTION = "sampler_main"
    CATEGORY = "SongGeneration"

    def sampler_main(self,cond,gen_type,save_separate):
        cfg=cond.get("cfg")
        cfg.gen_type=gen_type
        model_sep_path=cond["model_sep_path"]
        vae_model=cond["vae_model"]
        print("start inference final,loading model")
        seperate_tokenizer = builders.get_audio_tokenizer_model(f"Flow1dVAESeparate_{model_sep_path}",os.path.join(current_node_path, f'SongGeneration/conf/stable_audio_1920_vae.json'),vae_model,'inference')
        seperate_tokenizer = seperate_tokenizer.eval().cuda()
        audio=inference_lowram_final(cfg,seperate_tokenizer,cfg.max_dur,cond.get("items"),folder_paths.get_output_directory(),save_separate)
        del seperate_tokenizer
        gc_clear()
        return (audio,)



NODE_CLASS_MAPPINGS = {
    "SongGeneration_Loader":SongGeneration_Loader,
    "SongGeneration_Stage1": SongGeneration_Stage1,
    "SongGeneration_Stage2": SongGeneration_Stage2,
    "SongGeneration_Sampler": SongGeneration_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SongGeneration_Loader": "SongGeneration_Loader",
    "SongGeneration_Stage1": "SongGeneration_Stage1",
    "SongGeneration_Stage2": "SongGeneration_Stage2",
    "SongGeneration_Sampler": "SongGeneration_Sampler",
}
