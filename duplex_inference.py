import os
import sys
import copy
import logging
import speechbrain as sb
import numpy as np
from hyperpyyaml import load_hyperpyyaml
import librosa
import soundfile as sf

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

from CustomGenerator import ASR

N_ahead = 4
IDLE = "idle"
PREFILL = "prefilling"
GENERATION = "generating"

model = None
model_copy = None

chat_history = ""
chat_history_copy = ""
pre_wav = None
cur_wav = None
last_token_embed = None
last_token_embed_copy = None
temp_model_kwargs = None # The hidden state cache for token prediction during the prefilling stage

prefix_prompt_feats = None
endofspeech_feats = None
assistant_prompt_feats = None

def duplex_voice_assistant(wav=None):
    global model, model_copy
    global chat_history, chat_history_copy, pre_wav, cur_wav, last_token_embed, last_token_embed_copy, temp_model_kwargs
    global prefix_prompt_feats, endofspeech_feats, assistant_prompt_feats
    if wav is not None:
        audio_feats = model.audio_encoder(wav=wav)
    pre_wav = cur_wav
    cur_wav = wav
    print("USER: ",wav)


    # Determine the main and auxiliary model states based on the previous and current inputs
    # States remain unchanged for inputs 00 and 11
    if pre_wav is not None and cur_wav is None:
        # State 01 indicates that the previous audio has been fully input, 
        # forcing the main model into generating and the auxiliary model into idle 
        model.modules.Mamba.status = GENERATION
        model_copy.modules.Mamba.status = IDLE
        temp_model_kwargs = None  
    elif pre_wav is None and cur_wav is not None:
        # Switch between main and auxiliary models in 10 states, regardless of whether the main model has finished generating. 
        # This allows for seamless state rollback when ignoring speech.
        model_copy.modules.Mamba.status = PREFILL
        model_copy.modules.Mamba.model_kwargs = copy.deepcopy(model.modules.Mamba.model_kwargs)
        # The main and auxiliary models switch roles. 
        # The main model enters the prefilling state, while the auxiliary model remains unchanged, continuing to generate output or staying idle.
        model, model_copy = model_copy, model
        chat_history_copy = chat_history
        last_token_embed_copy = last_token_embed
        last_token_embed = None  
    
    # The main model state is either prefilling or generating. 
    # After generating, it automatically switches to idle without requiring explicit action.
    if model.modules.Mamba.status == PREFILL:  
        if model.modules.Mamba.model_kwargs == None:
            # Perform a single forward pass with parallel scanning to initialize the hidden state
            # In the prefill state, max_new_tokens controls the number of autoregressive loops. 
            # During the first prefill, model_kwargs is None, and max_new_tokens is set to 1; otherwise, it is set to 2.
            _ = model.modules.Mamba.duplex_generate(inputs_embeds=prefix_prompt_feats, max_new_tokens=1)
            chat_history += "<|user|>\nPlease answer the questions in the user's input speech.\n<|beginofspeech|>"
        elif temp_model_kwargs is None: # Indicates that the main model was in the generating state at the previous timestep
            if chat_history.endswith(model.tokenizer.eos_token):
                model.prefill(words="\n")
                chat_history += "\n"
            else:
                model.prefill(words="<|endoftext|>\n")
                chat_history += "<|endoftext|>\n"
            # Then, prefill the prefix_prompt section
            _ = model.modules.Mamba.duplex_generate(inputs_embeds=prefix_prompt_feats, max_new_tokens=2)
            chat_history += "<|user|>\nPlease answer the questions in the user's input speech.\n<|beginofspeech|>"

        # Perform token prediction; if the current audio input is complete, switch to the generating state.
        chat_history += "{" + cur_wav + "}"

        # Temporarily store the current audio slice and previous inputs as the prefill hidden state for seamless feature concatenation in the next loop.
        _ = model.modules.Mamba.duplex_generate(inputs_embeds=audio_feats, max_new_tokens=2)
        temp_model_kwargs = model.modules.Mamba.model_kwargs

        _ = model.modules.Mamba.duplex_generate(inputs_embeds=endofspeech_feats[:, :-1, :], max_new_tokens=2)
        # Temporarily set the model state to "generating" to predict a special token and determine if the audio input is complete.
        model.modules.Mamba.status = GENERATION
        # In generate mode, max_new_tokens defines the maximum length of the generated content.
        generated_token = model.modules.Mamba.duplex_generate(inputs_embeds=endofspeech_feats[:, -1:, :], do_sample=True, temperature=0.95, top_p=0.75, max_new_tokens=1)
        # Switching the model state back to prefilling
        model.modules.Mamba.status = PREFILL
        # The model's hidden state is restored to the prefilled state from the current audio slice and previous inputs
        model.modules.Mamba.model_kwargs = temp_model_kwargs

        if model.tokenizer.decode(generated_token[0]) == "<|endofuser|>":
            # When the speech input is complete, the model switches to the generating state
            model.modules.Mamba.status = GENERATION
            model_copy.modules.Mamba.status = IDLE  
            temp_model_kwargs = None  
        elif model.tokenizer.decode(generated_token[0]) == "<|ignore|>":
            # The main model determines that the current input should be ignored
            # If the auxiliary model is in the generating state, simply set the main model to idle and switch roles to make the main model ignore the input speech.
            if model_copy.modules.Mamba.status == GENERATION:
                model.modules.Mamba.status = IDLE
                model.modules.Mamba.model_kwargs = copy.deepcopy(model_copy.modules.Mamba.model_kwargs)
                model, model_copy = model_copy, model
                chat_history = chat_history_copy
                last_token_embed = last_token_embed_copy
            else: # The auxiliary model is idle in its initial state, indicating that the first input should be ignored
                # The main model and historical data are reset to their initial states
                model.modules.Mamba.model_kwargs = None
                chat_history = ""
            temp_model_kwargs = None  


    # The auxiliary model can be either idle or generating. In the idle state, it performs no operations.
    if model_copy.modules.Mamba.status == GENERATION:
        if chat_history_copy != "" and not chat_history_copy.endswith(model.tokenizer.eos_token):
            # Generate only four tokens at a time
            out_copy = model_copy.modules.Mamba.duplex_generate(inputs_embeds=last_token_embed_copy, do_sample=True, temperature=0.8, top_p=0.7, max_new_tokens=N_ahead)
            last_token_id_copy = out_copy[:, -1].unsqueeze(0)
            last_token_embed_copy = model_copy.embedding_layer(last_token_id_copy)
            response_copy = model_copy.tokenizer.decode(out_copy[0][-N_ahead:])
            print("                                 Assistant_copy: " + response_copy)
            
            chat_history_copy += response_copy


    if model.modules.Mamba.status == GENERATION:  # When the main model is in the generating state, the auxiliary model remains idle
        # Main model generation
        # Before generation, prefill with "<|endofspeech|><|endofuser|>\n<|assistant|>\n"
        # After embedding the last token, autoregressive generation begins
        if chat_history != "":
            if last_token_embed is None:
                model.modules.Mamba.status = PREFILL
                model.prefill(words="<|endofspeech|><|endofuser|>")
                chat_history += "<|endofspeech|><|endofuser|>"
                _ = model.modules.Mamba.duplex_generate(inputs_embeds=assistant_prompt_feats[:, :-1, :], max_new_tokens=2)
                chat_history += "\n<|assistant|>\n"
                model.modules.Mamba.status = GENERATION
                last_token_embed = assistant_prompt_feats[:, -1:, :]
            if not chat_history.endswith(model.tokenizer.eos_token):
                # Generate only four tokens at a time
                out = model.modules.Mamba.duplex_generate(inputs_embeds=last_token_embed, do_sample=True, temperature=0.8, top_p=0.7, max_new_tokens=N_ahead)
                last_token_id = out[:, -1].unsqueeze(0)
                last_token_embed = model.embedding_layer(last_token_id)
                response = model.tokenizer.decode(out[0][-N_ahead:])
                print("                                 Assistant: " + response)
                
                chat_history += response


def final_round_reply():
    global model
    global chat_history, chat_history_copy, pre_wav, cur_wav, last_token_embed, last_token_embed_copy

    while model.modules.Mamba.model_kwargs["cache_position"][-1:] < 1024 and not chat_history.endswith(model.tokenizer.eos_token):
        global assistant_prompt_feats
        # The main model continues generating until completion
        # If the last input is not None and the model predicts an incomplete token, the main model, still in the prefilling state, must be forced into the generating state.
        if model.modules.Mamba.status == PREFILL:
            model.prefill(words="<|endofspeech|><|endofuser|>")
            chat_history += "<|endofspeech|><|endofuser|>"
            _ = model.modules.Mamba.duplex_generate(inputs_embeds=assistant_prompt_feats[:, :-1, :], max_new_tokens=2)
            chat_history += "\n<|assistant|>\n"
            model.modules.Mamba.status = GENERATION
            last_token_embed = assistant_prompt_feats[:, -1:, :]

        out = model.modules.Mamba.duplex_generate(inputs_embeds=last_token_embed, do_sample=True, temperature=0.8, top_p=0.7, max_new_tokens=N_ahead)
        last_token_id = out[:, -1].unsqueeze(0)
        last_token_embed = model.embedding_layer(last_token_id)
        response = model.tokenizer.decode(out[0][-N_ahead:])
        print("                                 Assistant: " + response)
        
        chat_history += response


def cut_audio(input_file, output_dir, segment_duration=3):

    audio, sample_rate = librosa.load(input_file, sr=None)  
    segment_samples = int(segment_duration * sample_rate)
    
    # Segment audio data
    segments = [audio[i:i+segment_samples] for i in range(0, len(audio), segment_samples)]
    
    # If the last segment is shorter than 3 seconds, merge it with the previous one
    if len(segments) > 1 and len(segments[-1]) < segment_samples:
        segments[-2] = np.concatenate([segments[-2], segments[-1]])
        segments = segments[:-1]
    
    # Save the segmented audio
    wav_name = input_file.split("/")[-1].split(".")[0]
    output_dir =  output_dir + wav_name.split("-")[0] + "/" + wav_name.split("-")[1] 
    os.makedirs(output_dir, exist_ok=True)
    for i, segment in enumerate(segments):
        output_file = os.path.join(output_dir, f"segment_{i+1}.flac")
        sf.write(output_file, segment, sample_rate)
    
    return  output_dir

    

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
    hparams_copy = copy.deepcopy(hparams)

    model = ASR(modules=hparams["modules"], hparams=hparams, run_opts=run_opts)                 # The main model
    model_copy = ASR(modules=hparams_copy["modules"], hparams=hparams_copy, run_opts=run_opts)  # The auxiliary model

    model.modules.Mamba.status = PREFILL    # The main model
    model_copy.modules.Mamba.status = IDLE  # The auxiliary model

    prefix_prompt_feats = model.embedding_layer(model.prefix_prompt)
    endofspeech_feats = model.embedding_layer(model.endofspeech)
    assistant_prompt_feats = model.embedding_layer(model.assistant_prompt)

    wav_list = [
    ] # Your list of speech input files

    output_dir = ""  # Your output directory for temporarily storing speech slices
    for wav in wav_list:
        cut_dir = cut_audio(wav, output_dir)

        flac_files = sorted(
            [f for f in os.listdir(cut_dir) if f.endswith('.flac')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])  
        )

        for flac_file in flac_files:
            flac_path = os.path.join(cut_dir, flac_file)
            duplex_voice_assistant(wav=flac_path)
        
        # Simulate a 4-timeslice gap between two user inputs
        for i in range(4):
            duplex_voice_assistant()

    final_round_reply()
    print(chat_history)