"""This lobe enables the integration of huggingface pretrained LLAMA2-chat model.

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

"""

import logging

import torch
import torch.nn as nn
from typing import Optional

from speechbrain.lobes.models.huggingface_transformers.huggingface import (
    HFTransformersInterface,
)

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  
    handlers=[logging.StreamHandler()]  
)
logger = logging.getLogger(__name__)


class Mamba(HFTransformersInterface):
    """This lobe enables the integration of HuggingFace pretrained Mamba model.
    Transformer from HuggingFace needs to be installed:
        https://huggingface.co/transformers/installation.html

    The model can be finetuned. It will download automatically the model from
    HuggingFace or use a local path.

    Notes:
    - To use this model, you need to install the extra dependencies in recipes/MultiWOZ/response_generation/llama2/extra_requirements.txt
    - transformers and peft libraries should follow the versions mentioned in the extra_requirements.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "meta-llama/Llama-2-7b-chat-hf"
    save_path : str
        Path (dir) of the downloaded model.
    freeze : bool (default: False)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    max_new_tokens: int (default: 200)
    use_4bit: bool (default: True)
    bnb_4bit_compute_dtype: str (default: "float16")
        This sets the computational type which might be different than the input time. For example, inputs might be fp32, but computation can be set to bf16 for speedups.
    bnb_4bit_quant_type: str (default:"nf4")
        This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types which are specified by fp4 or nf4.
    use_nested_quant: bool (default: False)
        You have set this to False, which means you're not using nested quantization. This seems reasonable, as nested quantization can be computationally expensive.
    min_length: int (default: 1)
        The minimum length of the sequence to be generated. Corresponds to the length of the input prompt + min_new_tokens. Its effect is overridden by min_new_tokens, if also set
    top_k: int (default: 45)
        The number of highest probability vocabulary tokens to keep for top-k-filtering.
    top_p: float (default: 0.9)
        If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    num_beams: int (default: 8)
         Number of beams for beam search. 1 means no beam search.
    early_stopping: bool (default: True)
        Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
        - True, where the generation stops as soon as there are num_beams complete candidates
        - False, where an heuristic is applied and the generation stops when is it very unlikely to find better candidates
        - "never", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).
    with_peft: bool (default:False)
        If set to True, the peft model (model + adaptors) are loaded. If set to False, the original model is loaded.

    Example
    -------
    >>> model_hub = "meta-llama/Llama-2-7b-chat-hf"
    >>> save_path = "savedir"
    >>> model = LLAMA2(model_hub, save_path)
    >>> tokens = torch.tensor([[1, 1]])
    >>> attention_mask = torch.tensor([[1, 1]])
    >>> outputs = model(tokens, attention_mask)
    """

    def __init__(
        self,
        source: str,
        save_path: str,
        freeze: bool = False,
        max_new_tokens: int = 200,
        use_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "float16",
        min_length: int = 1,
        top_k: int = 45,
        top_p: float = 0.9,
        num_beams: int = 8,
        early_stopping: bool = True,
        with_peft: bool = False,
    ) -> None:
        self.with_peft = with_peft
        self.max_new_tokens = max_new_tokens
        self.min_length = min_length
        self.top_k = top_k
        self.top_p = top_p
        self.num_beams = num_beams
        self.early_stopping = early_stopping
        self.source = source
        self.save_path = save_path
        self.is_sb = False

        self.bnb_config = None

        super().__init__(
            source=source,
            save_path=save_path,
            freeze=freeze,
            with_casual_lm=True,
            quantization_config=self.bnb_config,
        )

        self.load_tokenizer(source=source)
        # Define a custom padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Set the padding direction to the right
        self.tokenizer.padding_side = "right"

        self.print_trainable_parameters(self.model)

    def forward(self, input_ids: Optional[torch.Tensor] = None, 
                inputs_embeds: Optional[torch.FloatTensor] = None,):
        """Takes an input a history of conversation and returns its corresponding reply.

        Arguments
        ---------
        input_ids : torch.Tensor
            A batch of input-id to transform to features.

        Returns
        -------
        output : torch.Tensor
            Reply to conversation.
        """
        output = self.model.forward(input_ids = input_ids, inputs_embeds = inputs_embeds)
        return output

    def _modify_state_dict(self, path, replaceables=["base_model"]):
        """A custom loading ensures SpeechBrain compatibility for Pretrain and model
        de/serialization. Here, the scope is to remove '.wav2vec2' before loading.

        Arguments
        ---------
        path : str
            Checkpoint path, file name relative to the repo root.
        replaceables : List[str]
            State dict sub-keys that if found, shall be dropped (incl. the 'model.' parent key), elevating key structures.

        Returns
        -------
        modified_state_dict : see torch.load
            SpeechBrain-valid deserialized pretrained model.
        """

        # Set is_sb = True for the ckpt is SB's nature
        self.is_sb = True

        # Load the state_dict of the ckpt
        orig_state_dict = torch.load(path, map_location="cpu")

        # Check if the dimension of the embed_tokens layer is greater than the vocab size defined by the HF Llama config
        # If it is True, enlarge this layer
        # This happens because sometimes one wants to add a <pad> token to the vocab.
        desired_key = next(
            (key for key in orig_state_dict if "embed_tokens.weight" in key),
            None,
        )
        new_num_tokens = (
            orig_state_dict.get(desired_key).size(0)
            - self.model.config.vocab_size
        )
        if new_num_tokens > 0:
            self.model.resize_token_embeddings(new_num_tokens=32001)


        modified_state_dict = {}
        # Matching the state_dict of the ckpt with that of the HF Llama model.
        for key, params in orig_state_dict.items():
            for tag in replaceables:
                if f"{tag}" in key:
                    save_key = key.replace(f"model.{tag}", f"{tag}")
                    modified_state_dict[save_key] = params
        return modified_state_dict

    def generate(
        self,
        decoder_type="greedy",
        max_new_tokens=None,
        **kwargs,
    ):
        """Takes an input a history of conversation and returns its corresponding reply.

        Arguments
        ---------
        decoder_type : str
            It shows strategy for autoregressive decoding either beam search or greedy.

        Returns
        -------
        hyp : torch.Tensor
            Reply to conversation input.
        """

        with torch.no_grad():
            if decoder_type == "beam":
                # beam decoding based on the input_ids which are dialogue context tokens (here only history)
                hyp = self.model.generate(
                    # input_ids=input_ids,
                    # attention_mask=attention_mask,
                    do_sample=True,
                    max_new_tokens=self.max_new_tokens,
                    min_length=self.min_length,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    temperature=1.0,
                    num_beams=self.num_beams,
                    num_return_sequences=1,
                    repetition_penalty=1.0,
                    length_penalty=1,
                    early_stopping=self.early_stopping,
                )
            else:
                # greedy decoding based on the input_ids which are dialogue context tokens (here only history)
                hyp = self.model.generate(
                    max_new_tokens=max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
                    **kwargs,
                )
        return hyp

    def override_config(self, config):
        """override config to include quantization config.

        Arguments
        ---------
        config : HuggingFace config object
            The original config.

        Returns
        -------
        config : HuggingFace config object
            Overridden config.
        """
        if self.bnb_config:
            config = config.from_pretrained(
                self.source,
                cache_dir=self.save_path,
                quantization_config=self.bnb_config,
            )
        return config

    def print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        logger.info(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
