"""
Simple implementation of baukit Trace and TraceDict
"""
import torch
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional

class Trace:
    """Capture activations from a single layer"""
    def __init__(self, module, edit_output: Optional[Callable] = None):
        self.module = module
        self.edit_output = edit_output
        self.output = None
        self.hook = None

    def __enter__(self):
        def hook(module, input, output):
            # 保存原始输出的结构信息
            was_tuple = isinstance(output, tuple)
            if was_tuple:
                original_output = output
                output_for_edit = output[0]
            else:
                output_for_edit = output

            if self.edit_output is not None:
                output_for_edit = self.edit_output(output_for_edit)

            # 如果原始输出是 tuple，需要保持 tuple 结构
            if was_tuple:
                # 用修改后的 output 替换第一个元素，保持其他元素不变
                output = (output_for_edit,) + original_output[1:]
            else:
                output = output_for_edit

            self.output = output_for_edit.detach() if isinstance(output_for_edit, torch.Tensor) else output_for_edit
            return output
        self.hook = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *args):
        if self.hook:
            self.hook.remove()


class TraceDict:
    """Capture activations from multiple layers"""
    def __init__(self, model, layers: List[str], edit_output: Optional[Callable] = None):
        self.model = model
        self.layers = set(layers)
        self.edit_output = edit_output
        self.hooks = []
        self.output = {}

    def __enter__(self):
        name_to_module = dict(self.model.named_modules())

        def make_hook(name):
            def hook(module, input, output):
                # 保存原始输出的结构信息
                was_tuple = isinstance(output, tuple)
                if was_tuple:
                    original_output = output
                    output_for_edit = output[0]
                else:
                    output_for_edit = output

                if self.edit_output is not None:
                    output_for_edit = self.edit_output(output_for_edit, name)

                # 如果原始输出是 tuple，需要保持 tuple 结构
                if was_tuple:
                    # 用修改后的 output 替换第一个元素，保持其他元素不变
                    output = (output_for_edit,) + original_output[1:]
                else:
                    output = output_for_edit

                self.output[name] = type('Output', (), {'output': output_for_edit})()
                return output
            return hook

        for name in self.layers:
            if name in name_to_module:
                hook = name_to_module[name].register_forward_hook(make_hook(name))
                self.hooks.append((name, hook))
        return self

    def __exit__(self, *args):
        for name, hook in self.hooks:
            hook.remove()

    def __getitem__(self, name):
        return self.output.get(name)
