import torch
import torch.nn as nn
from collections import OrderedDict
from peft import get_peft_model, TaskType, PeftModel
import random
import accelerate
from accelerate import dispatch_model, infer_auto_device_map, \
    load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory

from transformers import (OPTForCausalLM, GPT2LMHeadModel, BloomForCausalLM,
                          LlamaForCausalLM)

MODEL_UNIT = {
    LlamaForCausalLM: ['LlamaDecoderLayer'],
    BloomForCausalLM: ['BloomBlock'],
    GPT2LMHeadModel: ['GPT2Block'],
    OPTForCausalLM: ['OPTDecoderLayer']
}

import logging

logger = logging.getLogger(__name__)


def enable_adapter(model, package, adapter, **kwargs):
    adapter = adapter.lower()
    if package == 'peft':
        """
        PEFT: https://github.com/huggingface/peft
        Support methods:
            LoRA
            Prefix Tuning
            P-Tuning
            Prompt Tuning
            AdaLoRA
        """
        

        if adapter == 'lora':
            from peft import LoraConfig
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'fft':
            from peft import FourierFTConfig
            peft_config = FourierFTConfig(task_type=TaskType.CAUSAL_LM, **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'prefix':
            from peft import PrefixTuningConfig
            peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM,
                                             **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'prompt':
            from peft import PromptTuningConfig
            peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM,
                                             **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'p-tuning':
            from peft import PromptEncoderConfig
            peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM,
                                              **kwargs)
            model = get_peft_model(model, peft_config)
        else:
            raise NotImplementedError
        model.print_trainable_parameters()

    elif package == 'adapterhub':
        """
        AdapterHub: https://docs.adapterhub.ml/model_overview.html
        Support methods:
            Bottleneck Adapters
            Prefix Tuning
            LoRA
            Compacter
            Adapter Fusion
            Invertible Adapters
            Parallel block
        """
        # TODO:  After supporting adapterhub, we will move the following
        #   parameters in yaml file for users' convenient
        if adapter == 'lora':
            from transformers.adapters import LoRAConfig

            config = LoRAConfig(r=8, alpha=16)
            model.add_adapter("lora_adapter", config=config)
            model.train_adapter(['lora_adapter'])
        elif adapter == 'bottleneck':
            from transformers.adapters import AdapterConfig

            config = AdapterConfig(mh_adapter=True,
                                   output_adapter=True,
                                   reduction_factor=16,
                                   non_linearity="relu")
            model.add_adapter("bottleneck_adapter", config=config)
            model.train_adapter(['bottleneck_adapter'])
        elif adapter == 'lang':
            from transformers.adapters import PfeifferInvConfig

            config = PfeifferInvConfig()
            model.add_adapter("lang_adapter", config=config)
            model.train_adapter(['lang_adapter'])
        elif adapter == 'prefix':
            from transformers.adapters import PrefixTuningConfig

            config = PrefixTuningConfig(flat=False, prefix_length=30)
            model.add_adapter("prefix_tuning", config=config)
            model.train_adapter(['prefix_tuning'])
        elif adapter == 'compacter':
            from transformers.adapters import CompacterConfig

            config = CompacterConfig()
            model.add_adapter("dummy", config=config)
            model.train_adapter(['dummy'])
        elif adapter == 'ia_3':
            from transformers.adapters import IA3Config

            config = IA3Config()
            model.add_adapter("ia3_adapter", config=config)
            model.train_adapter(['ia3_adapter'])
        elif adapter == 'union':
            from transformers.adapters import AdapterConfig, ConfigUnion

            # TODO: configure these args in cfg
            config = ConfigUnion(
                AdapterConfig(mh_adapter=True,
                              output_adapter=False,
                              reduction_factor=16,
                              non_linearity="relu"),
                AdapterConfig(mh_adapter=False,
                              output_adapter=True,
                              reduction_factor=2,
                              non_linearity="relu"),
            )
            model.add_adapter("union_adapter", config=config)
            model.train_adapter(['union_adapter'])
        elif adapter == 'mam':
            from transformers.adapters import \
                ConfigUnion, ParallelConfig, PrefixTuningConfig

            config = ConfigUnion(
                PrefixTuningConfig(bottleneck_size=800),
                ParallelConfig(),
            )
            model.add_adapter("mam_adapter", config=config)
            model.train_adapter(['mam_adapter'])
        else:
            raise NameError(
                f"There is no adapter named {adapter} in {package}")
    else:
        raise NotImplementedError
    return model
 
class AdapterModel(nn.Module):
    def __init__(self, model, use_adapter=False, *args, **kwargs):
        super().__init__()

        self.model = None
        try:
            self.model_unit = MODEL_UNIT[type(model)]
        except:
            self.model_unit = None

        if use_adapter:
            adapter_package = kwargs.pop('adapter_package', 'peft')
            adapter_method = kwargs.pop('adapter_method', 'lora')

            self.model = enable_adapter(model, adapter_package, adapter_method,
                                        **kwargs)
        else:
            self.model = model

        # print(type(self.model))
        # merged_model = self.model.merge_and_unload()
        # print(type(merged_model))
        # print(type(self.model))
        # exit(-1)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(self, disable_adapter=False, *args, **kwargs):
        if isinstance(self.model, PeftModel) and disable_adapter:
            with self.model.disable_adapter():
                return self.model(*args, **kwargs)

        return self.model.forward(*args, **kwargs)

    def generate(self, disable_adapter=False, *args, **kwargs):
        try:
            if isinstance(self.model, PeftModel) and disable_adapter:
                with self.model.disable_adapter():
                    res = self.model.generate(*args, **kwargs)

            else:
                res = self.model.generate(*args, **kwargs)
        except RuntimeError as e:
            # When does evaluation in HELM,
            # half precision will cause RuntimeError,
            # the following solves it
            if 'do_sample' in kwargs.keys():
                del kwargs['do_sample']
                if isinstance(self.model, PeftModel) and disable_adapter:
                    with self.model.disable_adapter():
                        res = self.model.generate(*args, **kwargs)
                else:
                    res = self.model.generate(*args, **kwargs)
            else:
                raise RuntimeError(e)
        return res
    def state_dict(self, return_trainable=True, *args, **kwargs):
        if return_trainable:
            return self.get_trainable_state_dict()
        else:
            return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=False):
       return self.model.load_state_dict(state_dict, strict=False)

    def get_trainable_state_dict(self):
        grad_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad_params.append(name)
        model_state_dict = self.model.state_dict()
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            if k in grad_params:
                new_state_dict[k] = v
        if str(self.model.base_model).startswith("FourierFTModel"):#
            if self.config.model_type == 'gpt2':
                new_state_dict["posit"] = self.model.base_model.model.transformer.h[0].mlp.c_proj.indices['default']
            # this is llama solution,
            else:
                layer = self.model.base_model.model.model.layers[0]
                if hasattr(layer.self_attn.q_proj, 'indices'):
                    q_posit = layer.self_attn.q_proj.indices['default']
                    # v_posit = layer.self_attn.v_proj.indices['default']
                    new_state_dict["posit"] = q_posit
        return new_state_dict

    def save_model(self, path, state=0, merge_adapter=False):
        if merge_adapter and isinstance(self.model, PeftModel):
            merged_model = self.model.merge_and_unload()
            ckpt = {'cur_round': state, 'model': merged_model.state_dict()}
        else:
            ckpt = {'cur_round': state, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    def sharding(self):
        if hasattr(self, 'device_map') is False:
            max_memory = get_balanced_memory(
                self.model,
                max_memory=None,
                no_split_module_classes=self.model_unit,
                low_zero=False,
            )
            self.device_map = infer_auto_device_map(
                self.model,
                max_memory=max_memory,
                no_split_module_classes=self.model_unit,
            )
        self.model = dispatch_model(self.model, device_map=self.device_map)

    def print_model_map(self):
        for i in self.model.named_parameters():
            print(f"{i[0]} -> {i[1].device}")

    @property
    def config(self):
        return self.model.config

    @property
    def layers(self):
        _layers = []
        for module in self.model.modules():
            if isinstance(module, nn.ModuleList):
                # This one should be encoders/decoders
                _layers.append(module)

        if len(_layers) == 1:
            return _layers[0]
        return _layers

    def set_layers(self, layers):
        if isinstance(self.layers, nn.ModuleList) and isinstance(
                layers, nn.ModuleList):
            self.layers._modules = layers._modules

        elif type(layers) == list and type(self.layers) == list:
            # This consists of multiple ModuleLists
            assert len(self.layers) == len(layers)
            for src, tgt in zip(self.layers, layers):
                assert isinstance(tgt, nn.ModuleList)
                src._modules = tgt._modules

        else:
            raise ValueError(
                'Layers cannot be set due to the mismatched type. ')

    @property
    def trainable_param_name_pattern(self):
        if isinstance(self.model, PeftModel):
            return self.model.active_adapter
        return None

    def set_trainable_modules(self, modules=None):
        # First, set all modules to untrainable
        for module in self.model.modules():
            module.requires_grad_(False)

        # Second, search for the capable modules
        if modules is None:
            # Set the encoders/decoders to be trainable
            modules = self.layers

        if isinstance(modules, nn.ModuleList):
            # Make it to the list
            trainable_modules = [modules]

        elif type(modules) == list:
            trainable_modules = modules

        else:
            raise ValueError(f'{modules} cannot be trainable because '
                             f'{type(modules)}.')

        pattern = self.trainable_param_name_pattern
        for module in trainable_modules:
            for layer in module:
                for name, param in layer.named_parameters():
                    if pattern is None or pattern in name:
                        param.requires_grad = True

    # TODO: Fix `__getattr__`
    # def __getattr__(self, item):
    #     return getattr(self.model, item)


class LLMDataParallel(nn.DataParallel):
    def __init__(self, adap_model, device_ids=None, output_device=None, dim=0):
        assert isinstance(adap_model, AdapterModel)
        super().__init__(adap_model.model,
                         device_ids=device_ids,
                         output_device=output_device,
                         dim=dim)
        self.model = adap_model

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def state_dict(self, return_trainable=True, *args, **kwargs):
        return self.model.state_dict(return_trainable, *args, **kwargs)

    def load_state_dict(self, state_dict, strict=False):
        return self.model.load_state_dict(state_dict, strict)

    def save_model(self, path, state=0):
        self.model.save_model(path, state)
