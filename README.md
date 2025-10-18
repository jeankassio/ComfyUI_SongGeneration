# ComfyUI_SongGeneration
 [SongGeneration](https://github.com/tencent-ailab/SongGeneration):High-Quality Song Generation with Multi-Preference Alignment (SOTA),you can try VRAM>12G

# Update
* 10/18  修改加载流程，支持最新的full ，new，large模型，large模型12GVram可能会OOM，修复高版本transformer 的函数错误/Modify the loading process to support the latest full, new, and large models, and fix function errors in higher versions of transformers   
*  07/29，支持bgm和人声（vocal，目前还是有bgm底噪）单独输出，选择mixed为合成全部，模型加载方式更合理，去掉诸多debug打印，新增save_separate按钮，开启则保存三个音频（bgm，vocal，mixed）；
* Test env（插件测试环境）：window11，python3.11， torch2.6 ，cu124， VR12G,（transformers 4.45.1）


# 1. Installation

In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_SongGeneration.git
```

# 2. Requirements  
* window平台最难装的就是[fairseq](https://github.com/facebookresearch/fairseq)库，python3.11的建议用轮子安装[liyaodev/fairseq](https://github.com/liyaodev/fairseq/releases/tag/v0.12.3.1)；
* 如果缺失库，打开requirements_orgin.txt文件，看是少了哪个，手动安装；
* The most difficult thing to install on the Windows platform is the Fairseq library. It is recommended to install it on wheels for version 3.11 [liyaodev/fairseq](https://github.com/liyaodev/fairseq/releases/tag/v0.12.3.1)；
* If the library is missing, open the ’requirements_orgin.txt‘ file and see which one is missing, then manually install it；  

```
pip install -r requirements.txt
```

# 3.Model
* 3.1.1 download  ckpt  from [tencent/SongGeneration](https://huggingface.co/tencent/SongGeneration/tree/main)   国内建议魔搭[AI-ModelScope/SongGeneration](https://www.modelscope.cn/models/AI-ModelScope/SongGeneration/files)    
* 3.1.2  [new base](https://huggingface.co/lglg666/SongGeneration-base-new),[large ](https://huggingface.co/lglg666/SongGeneration-large),[full](https://huggingface.co/lglg666/SongGeneration-base-full)    
* 3.1.3 new prompt,[emb](https://github.com/tencent-ailab/SongGeneration/tree/main/tools)   
* 3.1.4 download htdemucs.pth [tencent/SongGeneration](https://huggingface.co/tencent/SongGeneration/tree/main/third_party/demucs/ckpt)  
* 文件结构如下,修改了加载流程，原来的结构也能用：
```
--  ComfyUI/models/SongGeneration/
    |-- htdemucs.pth #150M
    |--prompt.pt  # 3M
    |--new_prompt.pt  # 3M
    |--model_2.safetensors
    |--model_2_fixed.safetensors
    |--new_model.pt  # rename from model.pt
    |--large_model.pt  #  rename from model.pt
    |-- ckpt/  # 24.4G all 整个文件夹的大小
        |--encode-s12k.pt  # 3.68G
        |--models--lengyue233--content-vec-best/
--  ComfyUI/models/vae/
    |--autoencoder_music_1320k.ckpt  
```
# 4 Example
![](https://github.com/smthemex/ComfyUI_SongGeneration/blob/main/example_workflows/SongGeneration.png)

# 5 Citation
```
@article{lei2025levo,
  title={LeVo: High-Quality Song Generation with Multi-Preference Alignment},
  author={Lei, Shun and Xu, Yaoxun and Lin, Zhiwei and Zhang, Huaicheng and Tan, Wei and Chen, Hangting and Yu, Jianwei and Zhang, Yixuan and Yang, Chenyu and Zhu, Haina and Wang, Shuai and Wu, Zhiyong and Yu, Dong},
  journal={arXiv preprint arXiv:2506.07520},
  year={2025}
}
```



