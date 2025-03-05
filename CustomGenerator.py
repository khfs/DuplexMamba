#!/usr/bin/env python3

import os
import sys
import torch
import logging
import speechbrain as sb
import numpy as np
from hyperpyyaml import load_hyperpyyaml

logger = logging.getLogger(__name__)

os.environ['WANDB__SERVICE_WAIT'] = '999999'

IDLE = "idle"
PREFILL = "prefilling"
GENERATION = "generating"

class ASR(sb.core.Brain):
    def __init__(self, modules=None, opt_class=None, hparams=None, run_opts=None, checkpointer=None):
        super().__init__(
            modules=modules, 
            opt_class=opt_class, 
            hparams=hparams, 
            run_opts=run_opts, 
            checkpointer=checkpointer
        )

        self.tokenizer = hparams["Mamba"].tokenizer
        add_special_tokens_(
            hparams["Mamba"].model, self.tokenizer, {"sep_token": "<|endofuser|>", "additional_special_tokens": ["<|incomplete|>", "<|ignore|>"]}
        )
        self.prepare_prompt()
        self.load_model()

        if isinstance(self.modules.Mamba, torch.nn.parallel.DistributedDataParallel):
            self.embedding_layer = self.modules.Mamba.module.model.backbone.embeddings
        else:
            self.embedding_layer = self.modules.Mamba.model.backbone.embeddings

    def prepare_prompt(self):
        self.prefix_prompt = self.tokenizer("<|user|>\nPlease answer the questions in the user's input speech.\n<|beginofspeech|>", return_tensors="pt").input_ids
        self.prefix_prompt = self.prefix_prompt.to(self.device)
        self.endofspeech = self.tokenizer("<|endofspeech|>", return_tensors="pt").input_ids
        self.endofspeech = self.endofspeech.to(self.device)
        self.sep_token = self.tokenizer(self.tokenizer.sep_token, return_tensors="pt").input_ids
        self.sep_token = self.sep_token.to(self.device)
        self.assistant_prompt = self.tokenizer("\n<|assistant|>\n", return_tensors="pt").input_ids
        self.assistant_prompt = self.assistant_prompt.to(self.device)
        self.prompt_suffix = self.tokenizer("<|endofspeech|>" + self.tokenizer.sep_token + "\n<|assistant|>\n", return_tensors="pt").input_ids
        self.prompt_suffix = self.prompt_suffix.to(self.device)

    def load_model(self):
        logger.info("Loading stage4-110_ckpt")

        pretrained_ckpts = self.hparams.pretrained_checkpointer.list_checkpoints()
        pretrained_ckpt = sb.utils.checkpoints.average_checkpoints(
            pretrained_ckpts, recoverable_name="model",
        )
        self.hparams.model.load_state_dict(pretrained_ckpt, strict=True)

        logger.info("Loading completed")

    def prefill(self, words):
        if self.modules.Mamba.status != PREFILL:
            raise RuntimeError(f"Model status is '{self.modules.Mamba.status}', expected 'prefilling'.")
        if self.modules.Mamba.model_kwargs is None:
            raise RuntimeError(f"Model model_kwargs is '{self.modules.Mamba.model_kwargs}', expected not None.")
        tokens = self.tokenizer(words, return_tensors="pt").input_ids.to(self.device)
        tokens_feats = self.embedding_layer(tokens)
        _ = self.modules.Mamba.duplex_generate(inputs_embeds=tokens_feats, max_new_tokens=2)
        
    def audio_encoder(self, wav=None):
        sig = sb.dataio.dataio.read_audio(wav).to(self.device)
        wav = sig.unsqueeze(0)
        wav_len = torch.tensor([1.], device=wav.device)

        # compute features
        feats = self.hparams.compute_features(wav) # (B, T, 80)
        feats = self.modules.normalize(feats, wav_len)

        # forward modules
        src = self.modules.CNN(feats) # (B, L, 20, 32) -> (B, L, 640)

        enc_out, pred = self.modules.Transformer(
            src, tgt=None, wav_len=wav_len, pad_idx=self.hparams.pad_index,
        )

        adp_out = self.modules.Speech_Adapter(enc_out)

        return adp_out

    def inference(self, wav=None):
        sig = sb.dataio.dataio.read_audio(wav).to(self.device)
        wav = sig.unsqueeze(0)
        wav_len = torch.tensor([1.], device=wav.device)

        # compute features
        feats = self.hparams.compute_features(wav) # (B, T, 80)
        feats = self.modules.normalize(feats, wav_len)

        # forward modules
        src = self.modules.CNN(feats) # (B, L, 20, 32) -> (B, L, 640)

        enc_out, pred = self.modules.Transformer(
            src, tgt=None, wav_len=wav_len, pad_idx=self.hparams.pad_index,
        )

        adp_out = self.modules.Speech_Adapter(enc_out)

        prefix_prompt_feats = self.embedding_layer(self.prefix_prompt)
        endofspeech_feats = self.embedding_layer(self.endofspeech)

        batch_size, max_audio_len = adp_out.size(0), adp_out.size(1)
        actual_audio_lens = [round(wav_len.item() * max_audio_len) for wav_len in wav_len]

        for i in range(batch_size):
            audio_len = actual_audio_lens[i]
            inputs_embeds = torch.cat([
                prefix_prompt_feats[i],          
                adp_out[i, :audio_len],         
                endofspeech_feats[i],            
            ], dim=0).unsqueeze(0)  

            if isinstance(self.modules.Mamba, torch.nn.parallel.DistributedDataParallel):
                generated_token = self.modules.Mamba.module.generate(
                    inputs_embeds=inputs_embeds.detach(), do_sample=True, temperature=0.9, top_p=0.7, max_new_tokens=1
                )
            else:
                generated_token = self.modules.Mamba.generate(
                    inputs_embeds=inputs_embeds.detach(), do_sample=True, temperature=0.9, top_p=0.7, max_new_tokens=1
                )

        print("state token:", self.tokenizer.decode(generated_token.item()))
        generated_tokens_feats = self.embedding_layer(generated_token)
        assistant_prompt_feats = self.embedding_layer(self.assistant_prompt)

        for i in range(batch_size):
            audio_len = actual_audio_lens[i]
            inputs_embeds = torch.cat([
                prefix_prompt_feats[i],         
                adp_out[i, :audio_len],         
                endofspeech_feats[i],            
                generated_tokens_feats[i],       
                assistant_prompt_feats[i],      
            ], dim=0).unsqueeze(0)  

            if isinstance(self.modules.Mamba, torch.nn.parallel.DistributedDataParallel):
                generated = self.modules.Mamba.module.generate(
                    inputs_embeds=inputs_embeds.detach(), do_sample=True, temperature=0.9, top_p=0.7, max_new_tokens=1024
                )
            else:
                generated = self.modules.Mamba.generate(
                    inputs_embeds=inputs_embeds.detach(), do_sample=True, temperature=0.9, top_p=0.7, max_new_tokens=1024
                )

        return generated


def add_special_tokens_(model, tokenizer, attr_to_special_token) -> None:
    orig_num_tokens = len(tokenizer)
    num_added_tokens = tokenizer.add_special_tokens(
        attr_to_special_token  # type: ignore
    )  # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(
            new_num_tokens=orig_num_tokens + num_added_tokens
        )


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)
        
    hparams["modules"]["CNN"].requires_grad_(False)
    hparams["modules"]["Transformer"].requires_grad_(False)
    hparams["modules"]["Speech_Adapter"].requires_grad_(False)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
    )

    wav_path = hparams["wav_path"]
    out = asr_brain.inference(wav=wav_path)
    response = asr_brain.tokenizer.decode(out[0])
    print(response)

    