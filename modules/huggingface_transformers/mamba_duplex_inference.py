"""
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


import warnings
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from typing import TYPE_CHECKING, Callable, List, Optional, Union
from transformers.cache_utils import (
    HybridCache,
    MambaCache,
    SlidingWindowCache,
    StaticCache,
)
from transformers.utils import (
    is_torchdynamo_compiling,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

NEED_SETUP_CACHE_CLASSES_MAPPING = {
    "static": StaticCache,
    "sliding_window": SlidingWindowCache,
    "hybrid": HybridCache,
    "mamba": MambaCache,
}

# Typing shortcuts
GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]
GenerateBeamOutput = Union[GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput]
GenerateOutput = Union[GenerateNonBeamOutput, GenerateBeamOutput]

IDLE = "idle"
PREFILL = "prefilling"
GENERATION = "generating"


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

        self.model_kwargs = None  # use self.model_kwargs to replicate the model state
        self.status = None  # the status of model

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


    def generate(
        self,
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
            # greedy decoding based on the input_ids which are dialogue context tokens (here only history)
            hyp = self.model.generate(
                max_new_tokens=max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
                **kwargs,
            )
        return hyp
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=None,
        cache_params: Optional[MambaCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if use_cache:
            # `cache_position` should have been initialized in `generate`
            if cache_position is None:
                raise ValueError(
                    "`cache_position` should not be None as it should have been initialized in "
                    "`model.generate`, you are responsible for passing in a valid `cache_position` if "
                    "you are calling `prepare_inputs_for_generation` directly with `use_cache=True`"
                )
            if cache_position[0] > 0:

                if inputs_embeds is not None:
                    inputs_embeds = inputs_embeds[:, -1, :].unsqueeze(1)
                else:
                    input_ids = input_ids[:, -1].unsqueeze(-1)
            else:
                # we initialize the `cache_position` to full size of `conv_states` at prefill stage
                # considering padding will be applied when input length is shorter, and truncation
                # will be applied when it is longer, so it will be equivalent to always have it match
                # the length of `cache_params.conv_states`, which is `config.conv_kernel`
                cache_position = torch.arange(0, self.config.conv_kernel, device=input_ids.device)

        if inputs_embeds is not None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "cache_params": cache_params,
                "use_cache": use_cache,
                "cache_position": cache_position,
            }
        )
        return model_inputs
    
    
    @torch.no_grad()
    def duplex_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        r"""
        Copied and modified from /transformers/generation/utils.py in version 4.44.2.
        """
        
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self.model._validate_model_class()
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        
        generation_config, model_kwargs = self.model._prepare_generation_config(generation_config, **kwargs)
        self.model._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self.model._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )

        if self.model_kwargs is not None: 
            model_kwargs.update(self.model_kwargs)

        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self.model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.model.config.is_encoder_decoder and not is_torchdynamo_compiling():
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )
        
        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.model.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache
        
        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if not self.model.config.is_encoder_decoder:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.model.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self.model._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        self.model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode()

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if not is_torchdynamo_compiling() and self.model.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.model.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.model.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        prepared_logits_processor = self.model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
        )

        # 9. prepare stopping criteria
        prepared_stopping_criteria = self.model._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        # 10. go into different generation modes
        if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. prepare logits warper
            prepared_logits_warper = (
                self.model._get_logits_warper(generation_config, device=input_ids.device)
                if generation_config.do_sample
                else None
            )

             # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self.model._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.model.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            result = self._sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        logits_warper: Optional[LogitsProcessorList],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Copied and modified from /transformers/generation/utils.py in version 4.44.2.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        
        if 'cache_position' not in model_kwargs:
            model_kwargs = self.model._get_initial_cache_position(input_ids, model_kwargs)
        else:
            pass

        if self.model_kwargs is not None and self.status == PREFILL:
            if 'inputs_embeds' in model_kwargs:
                original_inputs_embeds = model_kwargs['inputs_embeds'].clone()
            else:
                original_input_ids = input_ids.clone()
    
        count = 0

        while self.model._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        ):
            # prepare model inputs
            if self.model_kwargs is not None and self.status == PREFILL:
                if 'inputs_embeds' in model_kwargs:
                    model_kwargs['inputs_embeds'] = original_inputs_embeds[:, count:count+1, :]  
                else:
                    input_ids = original_input_ids[:, count:count+1]  
            
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # forward pass to get next token
            outputs = self.model(**model_inputs, return_dict=True)

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone()

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            if do_sample:
                next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            next_tokens_embeds = self.model.backbone.embeddings(next_tokens[:, None])
            model_kwargs['inputs_embeds'] = torch.cat([model_kwargs['inputs_embeds'], next_tokens_embeds], dim=1)
            if self.model_kwargs is not None and self.status == PREFILL:
                if 'inputs_embeds' in model_kwargs and count + 1 < original_inputs_embeds.size(1):
                    input_ids = torch.tensor([[100]], device=self.model.device)
                elif 'inputs_embeds' not in model_kwargs:
                    if count + 1 == original_input_ids.size(1):  
                        input_ids = torch.cat([original_input_ids, next_tokens[:, None]], dim=-1) 
                    else:
                        input_ids = original_input_ids[:, 0:count+2]  
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.model.config.is_encoder_decoder,
            )
          
            if 'inputs_embeds' in model_kwargs:
                self.model_kwargs = {key: value for key, value in model_kwargs.items() if key != 'inputs_embeds'}
            else:
                self.model_kwargs = model_kwargs

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            count += 1
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            pass
        else:
            return input_ids

    

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
