"""Patch vLLM ovis config registration for compatibility with transformers >= 4.53"""
import importlib
import sys

def _patch_vllm_ovis():
    """Monkey-patch the ovis module to use exist_ok=True in AutoConfig.register"""
    try:
        import vllm.transformers_utils.configs.ovis as ovis_mod
        # Already loaded — patch in-place
        if hasattr(ovis_mod, 'AutoConfig'):
            ovis_mod.AutoConfig.register("aimv2", ovis_mod.AIMv2Config, exist_ok=True)
    except ValueError:
        # Already registered, that's fine
        pass
    except Exception:
        pass

_patch_vllm_ovis()