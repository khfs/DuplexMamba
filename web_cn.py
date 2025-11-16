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


import time
import tempfile
import gradio as gr

# === 新增：解决 /tmp 权限问题 ===
custom_tmp = "/home/luxiangyu/DuplexMamba/tmp_gradio"
os.makedirs(custom_tmp, exist_ok=True)
tempfile.tempdir = custom_tmp
os.environ["GRADIO_TEMP_DIR"] = custom_tmp
print(f"📁 Gradio 临时目录: {custom_tmp}")
# =============================

# =====================
# 确认一下语音切片是否正确
# =====================
DEBUG_AUDIO_DIR = "/home/luxiangyu/DuplexMamba/debug_chunks"  # 自定义路径
os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)

# 可选：如果你要重采样，请安装 scipy: pip install scipy
try:
    from scipy.signal import resample
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️ scipy 未安装，无法自动重采样。请确保麦克风输入为 16000Hz，或运行: pip install scipy")

# =====================
# 🔹 配置参数
# =====================
TARGET_SAMPLE_RATE = 16000      # 模型期望的采样率
CHUNK_DURATION_SEC = 3.0        # 每个切片时长（秒）
CHUNK_SAMPLES = int(TARGET_SAMPLE_RATE * CHUNK_DURATION_SEC)  # 32000



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
    

def duplex_voice_assistant(wav, chunk_id):
    # import pdb
    # pdb.set_trace()
    print(f"(chunk {chunk_id}) [识别结果] {os.path.basename(wav)}")
    web_result = ""  # output on web
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

        print("*******************")
        print(model.tokenizer.decode(generated_token[0]))

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

            web_result = response_copy   # output on web
            
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

                if web_result == "":   # output on web
                    web_result = response   
                else:
                    web_result = web_result + "\n" + response
                if web_result.endswith("<|endoftext|>"):
                    web_result = web_result[:-len("<|endoftext|>")] + "\n"
                
                chat_history += response
        
    print(chat_history)
    return web_result


# =====================
# 🔹 工具函数
# =====================
def numpy_to_flac(numpy_audio, path, sr=TARGET_SAMPLE_RATE):
    pcm16 = (numpy_audio * 32767).astype(np.int16)
    sf.write(path, pcm16, sr, format="FLAC")
    return path

def get_latest_incremental_audio(temp_dir):
    """
    从 Gradio 临时目录中找出最新的 audio.wav 文件（按文件夹修改时间）
    """
    try:
        subdirs = [
            os.path.join(temp_dir, d)
            for d in os.listdir(temp_dir)
            if os.path.isdir(os.path.join(temp_dir, d))
        ]
        if not subdirs:
            return None
        # 按文件夹修改时间排序，取最新
        latest_dir = max(subdirs, key=os.path.getmtime)
        audio_path = os.path.join(latest_dir, "audio.wav")
        if os.path.exists(audio_path):
            return audio_path
        else:
            return None
    except Exception as e:
        print(f"⚠️ [ERROR] 获取最新音频失败: {e}")
        return None

# =====================
# 🔹 清空历史函数
# =====================
def clear_history(state):
    """清空网页显示的识别结果"""
    if state is None:
        state = {"history_text": ""}
    else:
        state["history_text"] = ""
    return "", state

# =====================
# 🔹 核心处理函数（带详细日志）
# =====================
def transcribe_stream(audio, state):
    """
    注意：虽然接收 audio 参数（Gradio 强制传入），但我们忽略它，
    转而从 GRADIO_TEMP_DIR 中读取真实的增量 audio.wav 文件。
    """
    print("\n🎙️ [DEBUG] transcribe_stream 被调用")

    # 从 Gradio 临时目录读取最新增量片段
    latest_audio_path = get_latest_incremental_audio(custom_tmp)
    if latest_audio_path is None:
        print("⚠️ [DEBUG] 未找到最新的 audio.wav，跳过处理")
        # 返回当前历史（避免清空）
        current_text = state.get("history_text", "") if state else ""
        return current_text, state

    print(f"🔊 [DEBUG] 读取增量音频: {latest_audio_path}")
    data, sr = sf.read(latest_audio_path, dtype='float32')
    print(f"📊 [DEBUG] 增量音频 - 采样率: {sr} Hz, 长度: {len(data)} samples ({len(data)/sr:.2f}s)")

    # 处理立体声 → 单声道
    if len(data.shape) > 1:
        print("🔊 [DEBUG] 检测到多声道，取平均转单声道")
        data = data.mean(axis=1)

    # 重采样到 TARGET_SAMPLE_RATE（如果需要）
    if sr != TARGET_SAMPLE_RATE:
        if HAS_SCIPY:
            print(f"🔄 [DEBUG] 重采样: {sr}Hz → {TARGET_SAMPLE_RATE}Hz")
            num_samples = int(len(data) * TARGET_SAMPLE_RATE / sr)
            data = resample(data, num_samples)
            sr = TARGET_SAMPLE_RATE
        else:
            error_msg = "❌ [ERROR] 采样率不匹配且无法重采样（请安装 scipy）"
            current_text = state.get("history_text", "") if state else ""
            return current_text + "\n" + error_msg, state

    # === 初始化或更新 state ===
    current_time = time.time()
    if state is None:
        print("🆕 [DEBUG] 初始化新会话状态")
        state = {
            "last_time": current_time,
            "buffer": np.zeros(0, dtype=np.float32),
            "chunk_counter": 0,
            "history_text": ""
        }
    else:
        time_gap = current_time - state["last_time"]
        # 规则：超过10秒没数据 → 新会话
        if time_gap > 10.0:
            print("🆕 [DEBUG] 超时（>10s），重置为新会话，保留历史文本")
            state = {
                "last_time": current_time,
                "buffer": np.zeros(0, dtype=np.float32),
                "chunk_counter": 0,
                "history_text": state.get("history_text", "")
            }
        else:
            state["last_time"] = current_time

    # === 新增音频就是整个 data（因为来自增量片段）===
    new_data = data
    print(f"🆕 [DEBUG] 新增音频长度: {len(new_data)} samples")

    # 累积到 buffer
    buffer = np.concatenate([state["buffer"], new_data])
    print(f"🧺 [DEBUG] buffer 总长度: {len(buffer)} samples (目标: ≥{CHUNK_SAMPLES})")

    # === 切片处理 ===
    chunk_counter = state["chunk_counter"]
    results = ""
    while len(buffer) >= CHUNK_SAMPLES:
        chunk = buffer[:CHUNK_SAMPLES]
        buffer = buffer[CHUNK_SAMPLES:]
        chunk_counter += 1
        print(f"✂️ [DEBUG] 切片 #{chunk_counter}，送入模型")

        # === 保存调试用音频文件 ===
        debug_path = os.path.join(DEBUG_AUDIO_DIR, f"chunk_{chunk_counter:03d}.flac")
        numpy_to_flac(chunk, debug_path, sr=TARGET_SAMPLE_RATE)
        print(f"💾 [DEBUG] 已保存调试音频: {debug_path}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".flac") as f:
            flac_path = numpy_to_flac(chunk, f.name, sr=TARGET_SAMPLE_RATE)
            text = duplex_voice_assistant(flac_path, chunk_counter)
            print(f"🤖 [DEBUG] 模型返回: {text}")
            results += text
            try:
                os.unlink(flac_path)
            except Exception as e:
                print(f"🗑️ [WARN] 临时文件删除失败: {e}")

    # 更新状态
    state["buffer"] = buffer
    state["chunk_counter"] = chunk_counter

    # 追加新结果到历史
    if results.strip():
        state["history_text"] += results

    print(f"📤 [DEBUG] 当前完整历史长度: {len(state['history_text'])} 字符")
    return state["history_text"], state

# =====================
# 🔹 Gradio UI
# =====================
with gr.Blocks(title="实时语音助手") as demo:
    gr.Markdown("## 🎙️ 实时语音识别助手（每3秒切片）")
    gr.Markdown("点击麦克风开始说话，系统将每2秒处理一次音频。")

    audio_input = gr.Audio(
        source="microphone",
        type="numpy",          # 保留 type="numpy"（Gradio 要求 streaming 必须传 audio）
        streaming=True,
        label="🎙️ 实时语音输入"
    )
    output_text = gr.Textbox(label="🗣️ 实时识别结果", lines=10, interactive=False)
    state = gr.State(None)

    # 流式识别
    audio_input.stream(
        fn=transcribe_stream,
        inputs=[audio_input, state],
        outputs=[output_text, state]
    )
    # 清空按钮
    clear_btn = gr.Button("🗑️ 清空文本")
    clear_btn.click(
        fn=clear_history,
        inputs=[state],
        outputs=[output_text, state]
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
    hparams_copy = copy.deepcopy(hparams)

    model = ASR(modules=hparams["modules"], hparams=hparams, run_opts=run_opts)                 # The main model
    model_copy = ASR(modules=hparams_copy["modules"], hparams=hparams_copy, run_opts=run_opts)  # The auxiliary model

    model.modules.Mamba.status = PREFILL    # The main model
    model_copy.modules.Mamba.status = IDLE  # The auxiliary model

    prefix_prompt_feats = model.embedding_layer(model.prefix_prompt)
    endofspeech_feats = model.embedding_layer(model.endofspeech)
    assistant_prompt_feats = model.embedding_layer(model.assistant_prompt)
    
    demo.queue().launch(
        server_name="0.0.0.0",  # 允许外网访问
        server_port=7860         # 自定义端口
    )