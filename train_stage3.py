#!/usr/bin/env python3

import os
import sys
import math
import json
import torch
import logging
import random
from pathlib import Path
import speechbrain as sb
import numpy as np
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.distributed import run_on_main, if_main_process
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

logger = logging.getLogger(__name__)

os.environ['WANDB__SERVICE_WAIT'] = '999999'

# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig # (B, N)
        prefix_prompt, prefix_prompt_lens = batch.prefix_prompt
        endofspeech, _ = batch.endofspeech
        assistant_prompt, _ = batch.assistant_prompt
        label1, _ = batch.label1
        tokens_eos, _ = batch.tokens_eos

        # compute features
        feats = self.hparams.compute_features(wavs) # (B, T, 80)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        # forward modules
        src = self.modules.CNN(feats) # (B, L, 20, 32) -> (B, L, 640)

        enc_out, pred = self.modules.Transformer(
            src, tgt=None, wav_len=wav_lens, pad_idx=self.hparams.pad_index,
        )

        adp_out = self.modules.Speech_Adapter(enc_out)

        # encode text tokens using the embedding layer of Mamba-2.8B
        if isinstance(self.modules.Mamba, torch.nn.parallel.DistributedDataParallel):
            embedding_layer = self.modules.Mamba.module.model.backbone.embeddings
        else:
            embedding_layer = self.modules.Mamba.model.backbone.embeddings

        prefix_prompt_feats = embedding_layer(prefix_prompt)
        endofspeech_feats = embedding_layer(endofspeech)
        label1_feats = embedding_layer(label1)
        tokens_eos_feats = embedding_layer(tokens_eos)

        # calculate the actual prefix length
        batch_size, max_prefix_prompt_feats_len = prefix_prompt_feats.size(0), prefix_prompt_feats.size(1)
        actual_prefix_prompt_feats_lens = [round(prefix_prompt_len.item() * max_prefix_prompt_feats_len) for prefix_prompt_len in prefix_prompt_lens]
        
        # calculate the actual audio length
        batch_size, max_audio_len = adp_out.size(0), adp_out.size(1)
        actual_audio_lens = [round(wav_len.item() * max_audio_len) for wav_len in wav_lens]

        final_feats_list = []
        for i in range(batch_size):
            prefix_prompt_feats_len = actual_prefix_prompt_feats_lens[i]
            audio_len = actual_audio_lens[i]
            
            feats = torch.cat([
                prefix_prompt_feats[i, :prefix_prompt_feats_len],    # effective prefix
                adp_out[i, :audio_len],        # effective audio features
                endofspeech_feats[i],          # <|endofspeech|>
                label1_feats[i],               # special token + <|assistant|>
                tokens_eos_feats[i],           # response
            ], dim=0)   
            final_feats_list.append(feats)

        # use pad_sequence to pad to the same length
        from torch.nn.utils.rnn import pad_sequence
        final_feats = pad_sequence(final_feats_list, batch_first=True)

        mam_out = self.modules.Mamba(inputs_embeds=final_feats).logits # .requires_grad_(True)

        # extract the model's predicted special tokens and generated response part for calculating the cross-entropy loss
        # prefix_prompt.size(1): length of the prefix
        # audio_len: actual audio length
        # label1.size(1): length of the predicted special token and <|assistant|> 
        # tokens_eos.size(1): length of the generated part (response)

        final_reply_list = []

        for i in range(batch_size):
            # calculate the start and end positions of the response
            start_idx = actual_prefix_prompt_feats_lens[i] + actual_audio_lens[i] + endofspeech.size(1) - 1
            end_idx = start_idx + label1.size(1) + tokens_eos.size(1)

            # extract the generated response
            reply = mam_out[i, start_idx:end_idx, :]  

            final_reply_list.append(reply)

        final_reply_feats = pad_sequence(final_reply_list, batch_first=True)

        # output layer for seq2seq log-probabilities
        p_seq = self.hparams.log_softmax(final_reply_feats)
        p_token = p_seq[:, :1, :]  # the first position represents the predicted special token

        # Compute outputs
        # generate greedy search sampling
        hyps = None
        current_epoch = self.hparams.epoch_counter.current
        is_test_search = stage == sb.Stage.TEST

        if any([is_test_search]):
            generated_tokens = []    # save the special tokens generated for each sample
            generated_sequences = []  # save the generated responses 

            for i in range(batch_size):
                audio_len = actual_audio_lens[i]
                inputs_embeds = torch.cat([
                    prefix_prompt_feats[i, :prefix_prompt_feats_len],    # effective prefix
                    adp_out[i, :audio_len],          # effective audio features
                    endofspeech_feats[i],            # <|endofspeech|>
                ], dim=0).unsqueeze(0)  

                # Invoke the model to generate output with a specified length of 1, 
                # predicting special tokens to determine whether the current speech input is complete or should be ignored
                if isinstance(self.modules.Mamba, torch.nn.parallel.DistributedDataParallel):
                    generated_token = self.modules.Mamba.module.generate(
                        inputs_embeds=inputs_embeds.detach(), do_sample=True, temperature=0.9, top_p=0.7, max_new_tokens=1
                    )
                else:
                    generated_token = self.modules.Mamba.generate(
                        inputs_embeds=inputs_embeds.detach(), do_sample=True, temperature=0.9, top_p=0.7, max_new_tokens=1
                    )

                generated_tokens.append(generated_token)

            tokens = [token.squeeze(0).item() for token in generated_tokens]
            self.generated_tokens_list.extend(tokens)

            # iterate over the batch size and extract the first token of each sample as the label
            label_tokens = label1[:, 0]  
            self.label_tokens_list.extend(label_tokens.tolist())  

            generated_tokens = [token.squeeze(0) for token in generated_tokens]
            # concatenate the sequentially generated results into a batch output 
            generated_tokens = torch.stack(generated_tokens, dim=0)
            generated_tokens_feats = embedding_layer(generated_tokens)
            assistant_prompt_feats = embedding_layer(assistant_prompt)

            for i in range(batch_size):
                audio_len = actual_audio_lens[i]
                inputs_embeds = torch.cat([
                    prefix_prompt_feats[i, :prefix_prompt_feats_len],    # effective prefix
                    adp_out[i, :audio_len],          # effective audio features
                    endofspeech_feats[i],            # <|endofspeech|>
                    generated_tokens_feats[i],       # the newly generated special token
                    assistant_prompt_feats[i],       # <|assistant|>
                ], dim=0).unsqueeze(0)  

                # call the model for generation
                if isinstance(self.modules.Mamba, torch.nn.parallel.DistributedDataParallel):
                    generated = self.modules.Mamba.module.generate(
                        inputs_embeds=inputs_embeds.detach(), do_sample=True, temperature=0.9, top_p=0.7
                    )
                else:
                    generated = self.modules.Mamba.generate(
                        inputs_embeds=inputs_embeds.detach(), do_sample=True, temperature=0.9, top_p=0.7
                    )
                generated_sequences.append(generated)

            generated_sequences = [seq.squeeze(0) for seq in generated_sequences]
            # concatenate the sequentially generated responses into a batch response
            hyps = pad_sequence(generated_sequences, batch_first=True)


        return final_reply_feats, hyps, p_token

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""

        (final_reply_feats, hyps, p_token) = predictions

        batch = batch.to(self.device)
        ids = batch.id
        wrds = batch.wrd
        final_label, _ = batch.final_label
        label_token = final_label[:, :1]

        loss = self.hparams.ce_loss(
            final_reply_feats.flatten(end_dim=-2), final_label.flatten()
        )

        if stage != sb.Stage.TRAIN:
            # compute the accuracy of the special token prediction
            self.acc_metric.append(p_token, label_token)

        if stage == sb.Stage.TEST:
            hyps = tokenizer.batch_decode(hyps,skip_special_tokens=True)
            batch_results = [
                {"id": id, "label": label, "hyp": hyp}
                for id, label, hyp in zip(ids, wrds, hyps)
            ]
            self.generate_output.extend(batch_results)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage == sb.Stage.TRAIN and epoch == 1:
            logger.info("Loading stage2-011_ckpt")

            pretrained_ckpts = hparams["pretrained_checkpointer"].list_checkpoints()
            pretrained_ckpt = sb.utils.checkpoints.average_checkpoints(
                pretrained_ckpts, recoverable_name="model",
            )

            # get the weights of the embedding layer and the LM head
            embedding_weight = pretrained_ckpt["3.model.backbone.embeddings.weight"]
            lm_head_weight = pretrained_ckpt["3.model.lm_head.weight"]

            # the current model's vocabulary size (with two additional tokens)
            current_vocab_size = self.hparams.model[3].model.backbone.embeddings.weight.size(0)
            logger.info(f"current_vocab_size:{current_vocab_size}")

            # expand the weights by adding two new tokens
            new_embedding_weight = torch.cat(
                [embedding_weight, torch.randn(2, embedding_weight.size(1)) * 0.01], dim=0
            )
            new_lm_head_weight = torch.cat(
                [lm_head_weight, torch.randn(2, lm_head_weight.size(1)) * 0.01], dim=0
            )

            # update the checkpoint weights
            pretrained_ckpt["3.model.backbone.embeddings.weight"] = new_embedding_weight
            pretrained_ckpt["3.model.lm_head.weight"] = new_lm_head_weight

            self.hparams.model.load_state_dict(pretrained_ckpt, strict=True)

            logger.info("Loading completed")

        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()

        if stage == sb.Stage.TEST:
            self.generate_output = []
            self.generated_tokens_list = []
            self.label_tokens_list = []
            self.score_dict = {"bleu-4": []}

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        stage_stats["PPL"] = math.exp(stage_loss)
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:

            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                meta={"PPL": stage_stats["PPL"], "epoch": epoch},
                min_keys=["PPL"],
                num_to_keep=self.hparams.avg_checkpoints,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

            # compute and save the BLEU score
            for result in self.generate_output:
                hyp = result["hyp"]
                label = result["label"]
                
                bleu_score = sentence_bleu(
                    [list(label)],  
                    list(hyp),      
                    smoothing_function=SmoothingFunction().method3
                )
                self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

            average_bleu = float(np.mean(self.score_dict["bleu-4"]))
            print(f"Average BLEU-4 score: {average_bleu}")

            if if_main_process():

                # compute the token accuracy score
                correct = 0
                total = 0

                with open(self.hparams.test_token_acc_file, "w") as file:
                    for generated, label in zip(self.generated_tokens_list, self.label_tokens_list):
                        file.write(f"Label: {tokenizer.convert_ids_to_tokens(label)}        ")
                        file.write(f"Generated: {tokenizer.convert_ids_to_tokens(generated)}\n")
                        
                        if generated == label:
                            correct += 1
                        total += 1
                    
                    accuracy = correct / total if total > 0 else 0
                    file.write(f"\nAccuracy: {accuracy:.4f}\n")

                print(f"Accuracy: {accuracy:.4f}")

                with open(self.hparams.test_wer_file, "w", encoding="utf-8") as w:
                    for k, v in self.score_dict.items():
                        w.write(json.dumps({k: float(np.mean(v))}, ensure_ascii=False)+ "\n")
                    for result in self.generate_output:
                        w.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        torch.cuda.empty_cache()
            
    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.noam_annealing(self.optimizer)

def add_special_tokens_(model, tokenizer, attr_to_special_token) -> None:
    orig_num_tokens = len(tokenizer)
    num_added_tokens = tokenizer.add_special_tokens(
        attr_to_special_token  # type: ignore
    )  # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(
            new_num_tokens=orig_num_tokens + num_added_tokens
        )

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]
    valtest_datasets = [valid_data] + [i for k, i in test_datasets.items()]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["Mamba"].tokenizer
    add_special_tokens_(
        hparams["Mamba"].model, tokenizer, {"sep_token": "<|endofuser|>", "additional_special_tokens": ["<|incomplete|>", "<|ignore|>"]}
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(valtest_datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline_train(wav):
        # Speed Perturb is done here so it is multi-threaded with the
        # workers of the dataloader (faster).
        if "speed_perturb" in hparams:
            sig = sb.dataio.dataio.read_audio(wav)

            sig = hparams["speed_perturb"](sig.unsqueeze(0)).squeeze(0)
        else:
            sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline_train)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("id", "wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_eos", "prefix_prompt", "endofspeech", "assistant_prompt", "label1", "final_label"
    )
    def text_pipeline(id, wrd):
        asr_prompts = [
            "<|user|>\nWhat does this audio say? Write it in lowercase without punctuation.\n<|beginofspeech|>",
            "<|user|>\nPlease convert this speech into text, all in lowercase and without punctuation.\n<|beginofspeech|>",
            "<|user|>\nGenerate the transcription for this audio without punctuation and keep it lowercase.\n<|beginofspeech|>",
            "<|user|>\nConvert the spoken words into lowercase text without using any punctuation.\n<|beginofspeech|>",
            "<|user|>\nWrite the words you hear in this audio in lowercase, leaving out punctuation.\n<|beginofspeech|>",
            "<|user|>\nTranscribe the speech in this audio to lowercase text with no punctuation.\n<|beginofspeech|>",
        ]
        qa_prompts = [
            "<|user|>\nPlease answer the questions in the user's input speech.\n<|beginofspeech|>",
            "<|user|>\nListen to this speech and provide an appropriate answer.\n<|beginofspeech|>",
            "<|user|>\nPlease respond to the questions asked in the audio.\n<|beginofspeech|>",
            "<|user|>\nBased on this audio, provide a clear and concise answer.\n<|beginofspeech|>",
            "<|user|>\nRespond to the query presented in this audio message.\n<|beginofspeech|>",
            "<|user|>\nPlease provide a response to the question in the speaker's voice.\n<|beginofspeech|>",
            "<|user|>\nRespond to the audio's question with the appropriate answer.\n<|beginofspeech|>",
        ]
        
        yield wrd
        tokens_list = tokenizer.encode(wrd)
        tokens_eos = torch.LongTensor(tokens_list + [tokenizer.eos_token_id])
        yield tokens_eos
        if id.startswith("asr-"):
            selected_index = random.randint(0, len(asr_prompts)-1)
            prefix_prompt = tokenizer(asr_prompts[selected_index], return_tensors="pt").input_ids
            prefix_prompt = prefix_prompt.squeeze(0)
        else:
            selected_index = random.randint(0, len(qa_prompts)-1)
            prefix_prompt = tokenizer(qa_prompts[selected_index], return_tensors="pt").input_ids
            prefix_prompt = prefix_prompt.squeeze(0)
        yield prefix_prompt
        endofspeech = tokenizer("<|endofspeech|>", return_tensors="pt").input_ids
        endofspeech = endofspeech.squeeze(0)
        yield endofspeech
        assistant_prompt = tokenizer("\n<|assistant|>\n", return_tensors="pt").input_ids
        assistant_prompt = assistant_prompt.squeeze(0)
        yield assistant_prompt
        if id.endswith("-cut"):
            label1 = tokenizer("<|incomplete|>" + "\n<|assistant|>\n", return_tensors="pt").input_ids.squeeze(0)
        elif id.endswith("-ignore"):
            label1 = tokenizer("<|ignore|>" + "\n<|assistant|>\n", return_tensors="pt").input_ids.squeeze(0)
        else:
            label1 = tokenizer(tokenizer.sep_token + "\n<|assistant|>\n", return_tensors="pt").input_ids.squeeze(0)
        yield label1
        # set all ids except the first one to -100
        label2 = label1.clone()
        label2[1:] = -100
        final_label = torch.cat([label2, tokens_eos], dim=-1)
        yield final_label

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens_eos", "prefix_prompt", "endofspeech", "assistant_prompt", "label1", "final_label"],
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams_train = hparams["dynamic_batch_sampler_train"]
        dynamic_hparams_valid = hparams["dynamic_batch_sampler_valid"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_train,
        )
        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_valid,
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # 1.  # Dataset prep (parsing Librispeech)
    from gpt_omni_prepare import prepare_gpt_omni 

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_gpt_omni,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams)

    # Init wandb
    if hparams['use_wandb']:
        hparams['train_logger'] = hparams['wandb_logger']()
        
    if hparams['no_lm']:
        print('Evaluate without LM.')
        hparams['test_search'] = hparams['valid_search']
        hparams["output_wer_folder"] = os.path.join(hparams["output_wer_folder"], 'no_lm')

    hparams["modules"]["CNN"].requires_grad_(False)
    hparams["modules"]["Transformer"].requires_grad_(False)
    # hparams["modules"]["Speech_Adapter"].requires_grad_(False)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    asr_brain.tokenizer = hparams["Mamba"].tokenizer
    add_special_tokens_(
        hparams["Mamba"].model, tokenizer, {"sep_token": "<|endofuser|>", "additional_special_tokens": ["<|incomplete|>", "<|ignore|>"]}
    )
    
    class CustomPaddedBatch(PaddedBatch):
        """PaddedBatch with custom padding values.

        See the documentation of `speechbrain.dataio.batch.PaddedBatch`.

        """

        def __init__(self, examples, *args, **kwargs):
            for k in [
                "final_label",
            ]:
                max_len = max([len(x[k]) for x in examples])
                pad_value = 0
                if k in ["final_label",]:
                    pad_value = hparams["ignore_index"]
                for example in examples:
                    x = example[k]
                    if k in ["final_label"]:
                        example[k] = torch.nn.functional.pad(
                            x, [0, max_len - len(x)], value=pad_value
                        )
            super().__init__(examples, *args, **kwargs)

    hparams["train_dataloader_opts"]["collate_fn"] = CustomPaddedBatch
    hparams["valid_dataloader_opts"]["collate_fn"] = CustomPaddedBatch
    hparams["test_dataloader_opts"]["collate_fn"] = CustomPaddedBatch

    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        collate_fn = None
        if "collate_fn" in train_dataloader_opts:
            collate_fn = train_dataloader_opts["collate_fn"]

        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }

        if collate_fn is not None:
            train_dataloader_opts["collate_fn"] = collate_fn

    if valid_bsampler is not None:
        collate_fn = None
        if "collate_fn" in valid_dataloader_opts:
            collate_fn = valid_dataloader_opts["collate_fn"]

        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

        if collate_fn is not None:
            valid_dataloader_opts["collate_fn"] = collate_fn

    if not hparams['skip_train']:
        # Training
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=train_dataloader_opts,
            valid_loader_kwargs=valid_dataloader_opts,
        )

    # Testing
    if not os.path.exists(hparams["output_wer_folder"]):
        os.makedirs(hparams["output_wer_folder"])

    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.test_token_acc_file = os.path.join(
            hparams["output_wer_folder"], f"token_acc_{k}.txt"
        )
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"generate_output_{k}.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
