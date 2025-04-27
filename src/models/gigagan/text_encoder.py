import torch
from torch import nn, Tensor
import torch.nn.functional as F

from beartype import beartype
from beartype.typing import List, Tuple, Dict, Iterable
import json
import argparse
from einops import pack, repeat, unpack

from src.models.vad.model import PretrainedLMModel
from src.models.gigagan.transformer import Transformer
from src.models.gigagan.open_clip import OpenClipAdapter
from src.models.gigagan.utils import (exists, set_requires_grad_)

from transformers import (RobertaConfig, RobertaTokenizer)

class TextEncoder(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        dim,
        depth,
        clip: OpenClipAdapter | None = None,
        dim_head = 64,
        heads = 8,
        vad_model_path,
        vad_config_path
    ):
        print("Creating text encoder")
        super().__init__()
        self.dim = dim

        # load the valence arousal dominance model
        self.vad_model, self.vad_tokenizer = self.load_vad_model(vad_model_path, vad_config_path)

        if not exists(clip):
            clip = OpenClipAdapter()

        self.clip = clip
        set_requires_grad_(clip, False)

        self.learned_global_token = nn.Parameter(torch.randn(dim))

        self.project_in = nn.Linear(clip.dim_latent, dim) if clip.dim_latent != dim else nn.Identity()

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads
        )

    def load_vad_model(self, model_path, config_path):
        model_name = 'roberta-large'
        cache_path = './../ckpt/roberta-large'

        config = RobertaConfig.from_pretrained(
            model_name, 
            cache_dir=cache_path+'/model/config/')

        with open(config_path) as config_file:
            args = json.load(config_file)
            args = argparse.Namespace(**args)

        config.args = args

        tokenizer = RobertaTokenizer.from_pretrained(model_name, cache_dir=cache_path+'/vocab/')
        model = PretrainedLMModel(config, cache_path, model_name)

        model.to(torch.device(args.device))
        
        # 1. set path and load states
        print("Loading VAD Model from:", model_path)
        state = torch.load(model_path)
        
        # 2. load (override) pre-trained model (without head)
        model_dict = model.state_dict()
        ckpt__dict = state['state_dict']
        ckpt__dict_head_removed = {k: v for k, v in ckpt__dict.items() if k not in ['head.bias', 'head.weight']}
        model_dict.update(ckpt__dict_head_removed) 
        model.load_state_dict(model_dict, strict=False)
        
        print("Loading Model from:", model_path, "...Finished.")
        return model, tokenizer

    @beartype
    def forward(
        self,
        texts: List[str] | None = None,
        text_encodings: Tensor | None = None
    ):
        assert exists(texts) ^ exists(text_encodings)

        if not exists(text_encodings):
            with torch.no_grad():
                self.clip.eval()
                _, text_encodings = self.clip.embed_texts(texts)

        mask = (text_encodings != 0.).any(dim = -1)

        text_encodings = self.project_in(text_encodings)

        mask_with_global = F.pad(mask, (1, 0), value = True)

        global_tokens = torch.empty((), device=self.vad_model.args.device)
        for text in texts:
            token_out = self.vad_tokenizer(text)
            input_ids = torch.tensor([token_out['input_ids']], dtype=torch.long, device=self.vad_model.args.device)
            attention_masks = torch.tensor([token_out['attention_mask']], dtype=torch.long, device=self.vad_model.args.device)

            lm_logits, cls_logits = self.vad_model(input_ids, attention_mask=attention_masks,n_epoch=1)

            if global_tokens.dim() == 0:
                global_tokens = cls_logits
            else:
                torch.cat((global_tokens, cls_logits))
        
        global_tokens = repeat(global_tokens, 'b d -> b (repeat d)', repeat=22)
        global_tokens = global_tokens[:, :64]

        
        global_tokens = global_tokens.to(text_encodings.device)

        text_encodings, ps = pack([global_tokens, text_encodings], 'b * d')

        text_encodings = self.transformer(text_encodings, mask = mask_with_global)

        global_tokens, text_encodings = unpack(text_encodings, ps, 'b * d')

        return global_tokens, text_encodings, mask