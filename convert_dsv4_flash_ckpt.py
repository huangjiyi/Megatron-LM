#!/usr/bin/env python

"""Convert DeepSeek-V4-Flash HF/inference weights to a Megatron checkpoint.

This converter is intentionally tied to the local DSv4-Flash Megatron model:
it initializes the target Megatron model with the requested PP/EP/TP layout,
loads only the parameters owned by the current rank from the HF safetensors
directory, and then lets Megatron save a model-only checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


class HFShardReader:
    def __init__(self, checkpoint_dir: str):
        self.root = Path(checkpoint_dir)
        index_path = self.root / "model.safetensors.index.json"
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        self.weight_map: Dict[str, str] = index["weight_map"]
        self._handles = {}

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        filename = self.weight_map[name]
        handle = self._handles.get(filename)
        if handle is None:
            handle = safe_open(str(self.root / filename), framework="pt", device="cpu")
            self._handles[filename] = handle
        return handle.get_tensor(name)


def parse_pipeline_layer_counts(layout: Optional[str], num_layers: int, pp_size: int) -> list[int]:
    if not layout or layout == "none":
        assert num_layers % pp_size == 0, "num_layers must be divisible by PP size without layout"
        return [num_layers // pp_size] * pp_size
    counts = []
    for stage in layout.split("|"):
        count = 0
        for match in re.finditer(r"t(?:\*(\d+))?", stage):
            count += int(match.group(1) or 1)
        counts.append(count)
    assert len(counts) == pp_size, f"layout has {len(counts)} stages, expected {pp_size}: {layout}"
    assert sum(counts) == num_layers, f"layout has {sum(counts)} layers, expected {num_layers}"
    return counts


def local_to_global_layer(local_layer: int, counts: list[int], pp_rank: int) -> int:
    return sum(counts[:pp_rank]) + local_layer


def fp8_weight_to_bf16(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    assert weight.dtype == torch.float8_e4m3fn, weight.dtype
    assert scale.dtype == torch.float8_e8m0fnu, scale.dtype
    out_blocks, in_blocks = scale.shape
    assert weight.shape[0] == out_blocks * 128
    assert weight.shape[1] == in_blocks * 128
    out = weight.float().view(out_blocks, 128, in_blocks, 128)
    out = out * scale.float()[:, None, :, None]
    return out.reshape(weight.shape).bfloat16()


def fp4_weight_to_bf16(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    assert weight.dtype == torch.int8, weight.dtype
    assert scale.dtype == torch.float8_e8m0fnu, scale.dtype
    packed = weight.view(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    values = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(1)
    assert values.shape[1] == scale.shape[1] * 32
    values = values.view(values.shape[0], scale.shape[1], 32)
    values = values * scale.float().unsqueeze(-1)
    return values.reshape(values.shape[0], -1).bfloat16()


def maybe_dequant_linear(reader: HFShardReader, name: str) -> torch.Tensor:
    weight = reader.get(name)
    if weight.dtype == torch.float8_e4m3fn:
        return fp8_weight_to_bf16(weight, reader.get(name.replace(".weight", ".scale")))
    if weight.dtype == torch.int8:
        return fp4_weight_to_bf16(weight, reader.get(name.replace(".weight", ".scale")))
    return weight


def copy_tensor(dst: torch.Tensor, src: torch.Tensor, name: str) -> None:
    if tuple(dst.shape) != tuple(src.shape):
        raise RuntimeError(f"shape mismatch for {name}: dst={tuple(dst.shape)} src={tuple(src.shape)}")
    dst.copy_(src.to(device=dst.device, dtype=dst.dtype))


def layer_hf_prefix(megatron_prefix: str, counts: list[int], pp_rank: int) -> Optional[str]:
    match = re.match(r"decoder\.layers\.(\d+)\.(.*)", megatron_prefix)
    if match:
        global_layer = local_to_global_layer(int(match.group(1)), counts, pp_rank)
        return f"layers.{global_layer}.", match.group(2)
    match = re.match(r"mtp\.layers\.0\.mtp_model_layer\.(.*)", megatron_prefix)
    if match:
        return "mtp.0.", match.group(1)
    match = re.match(r"mtp\.layers\.0\.(.*)", megatron_prefix)
    if match:
        return "mtp.0.", match.group(1)
    return None


def map_nonexpert_name(name: str, counts: list[int], pp_rank: int) -> Optional[str]:
    top_level = {
        "embedding.word_embeddings.weight": "embed.weight",
        "decoder.final_layernorm.weight": "norm.weight",
        "decoder.hc_head_fn": "hc_head_fn",
        "decoder.hc_head_base": "hc_head_base",
        "decoder.hc_head_scale": "hc_head_scale",
        "output_layer.weight": "head.weight",
        "mtp.layers.0.hc_head_fn": "mtp.0.hc_head_fn",
        "mtp.layers.0.hc_head_base": "mtp.0.hc_head_base",
        "mtp.layers.0.hc_head_scale": "mtp.0.hc_head_scale",
        "mtp.layers.0.enorm.weight": "mtp.0.enorm.weight",
        "mtp.layers.0.hnorm.weight": "mtp.0.hnorm.weight",
        "mtp.layers.0.e_proj.weight": "mtp.0.e_proj.weight",
        "mtp.layers.0.h_proj.weight": "mtp.0.h_proj.weight",
        "mtp.layers.0.final_layernorm.weight": "mtp.0.norm.weight",
    }
    if name in top_level:
        return top_level[name]

    mapped = layer_hf_prefix(name, counts, pp_rank)
    if mapped is None:
        return None
    hf_prefix, suffix = mapped

    suffix_map = {
        "input_layernorm.weight": "attn_norm.weight",
        "pre_mlp_layernorm.weight": "ffn_norm.weight",
        "self_attention.linear_o_group_proj": "attn.wo_a.weight",
        "self_attention.linear_proj.weight": "attn.wo_b.weight",
        "self_attention.linear_q_down_proj.weight": "attn.wq_a.weight",
        "self_attention.q_layernorm.weight": "attn.q_norm.weight",
        "self_attention.linear_q_up_proj.weight": "attn.wq_b.weight",
        "self_attention.linear_kv_proj.weight": "attn.wkv.weight",
        "self_attention.kv_layernorm.weight": "attn.kv_norm.weight",
        "self_attention.core_attention.attn_sink": "attn.attn_sink",
        "self_attention.core_attention.compressor.ape": "attn.compressor.ape",
        "self_attention.core_attention.compressor.linear_wkv.weight": "attn.compressor.wkv.weight",
        "self_attention.core_attention.compressor.linear_wgate.weight": "attn.compressor.wgate.weight",
        "self_attention.core_attention.compressor.norm.weight": "attn.compressor.norm.weight",
        "self_attention.core_attention.indexer.linear_wq_b.weight": "attn.indexer.wq_b.weight",
        "self_attention.core_attention.indexer.linear_weights_proj.weight": "attn.indexer.weights_proj.weight",
        "self_attention.core_attention.indexer.compressor.ape": "attn.indexer.compressor.ape",
        "self_attention.core_attention.indexer.compressor.linear_wkv.weight": "attn.indexer.compressor.wkv.weight",
        "self_attention.core_attention.indexer.compressor.linear_wgate.weight": "attn.indexer.compressor.wgate.weight",
        "self_attention.core_attention.indexer.compressor.norm.weight": "attn.indexer.compressor.norm.weight",
        "mlp.router.weight": "ffn.gate.weight",
        "self_attention_hyper_connection.mapping_proj.weight": "hc_attn_fn",
        "self_attention_hyper_connection.bias": "hc_attn_base",
        "mlp_hyper_connection.mapping_proj.weight": "hc_ffn_fn",
        "mlp_hyper_connection.bias": "hc_ffn_base",
    }
    if suffix in suffix_map:
        return hf_prefix + suffix_map[suffix]
    return None


def hyper_scale_source(name: str, counts: list[int], pp_rank: int) -> Optional[tuple[str, int]]:
    mapped = layer_hf_prefix(name, counts, pp_rank)
    if mapped is None:
        return None
    hf_prefix, suffix = mapped
    if suffix.startswith("self_attention_hyper_connection."):
        scale_name = hf_prefix + "hc_attn_scale"
    elif suffix.startswith("mlp_hyper_connection."):
        scale_name = hf_prefix + "hc_ffn_scale"
    else:
        return None
    scale_idx = {
        "alpha_pre": 0,
        "alpha_post": 1,
        "alpha_res": 2,
    }.get(suffix.rsplit(".", 1)[-1])
    if scale_idx is None:
        return None
    return scale_name, scale_idx


def expert_source(
    name: str,
    counts: list[int],
    pp_rank: int,
    ep_rank: int,
    experts_per_rank: int,
) -> Optional[tuple[str, Optional[int], str]]:
    mapped = layer_hf_prefix(name, counts, pp_rank)
    if mapped is None:
        return None
    hf_prefix, suffix = mapped

    local_match = re.match(r"mlp\.experts\.local_experts\.(\d+)\.(linear_fc[12])\.weight", suffix)
    if local_match:
        expert_id = ep_rank * experts_per_rank + int(local_match.group(1))
        return hf_prefix, expert_id, local_match.group(2)

    grouped_match = re.match(r"mlp\.experts\.(linear_fc[12])\.weight(\d+)", suffix)
    if grouped_match:
        expert_id = ep_rank * experts_per_rank + int(grouped_match.group(2))
        return hf_prefix, expert_id, grouped_match.group(1)

    shared_match = re.match(r"mlp\.shared_experts\.(linear_fc[12])\.weight", suffix)
    if shared_match:
        return hf_prefix, None, shared_match.group(1)

    return None


def load_expert_tensor(
    reader: HFShardReader,
    hf_prefix: str,
    expert_id: Optional[int],
    linear_name: str,
) -> torch.Tensor:
    if expert_id is None:
        base = hf_prefix + "ffn.shared_experts"
    else:
        base = hf_prefix + f"ffn.experts.{expert_id}"

    if linear_name == "linear_fc1":
        w1 = maybe_dequant_linear(reader, base + ".w1.weight")
        w3 = maybe_dequant_linear(reader, base + ".w3.weight")
        return torch.cat([w1, w3], dim=0)
    if linear_name == "linear_fc2":
        return maybe_dequant_linear(reader, base + ".w2.weight")
    raise AssertionError(linear_name)


def load_parameters(model, reader: HFShardReader, counts: list[int]) -> tuple[int, list[str]]:
    from megatron.core import parallel_state

    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    args = __import__("megatron.training", fromlist=["get_args"]).get_args()
    experts_per_rank = args.num_experts // args.expert_model_parallel_size

    loaded = 0
    missing = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            scale_src = hyper_scale_source(name, counts, pp_rank)
            if scale_src is not None:
                scale_name, scale_idx = scale_src
                copy_tensor(param, reader.get(scale_name)[scale_idx : scale_idx + 1], name)
                loaded += 1
                continue

            expert = expert_source(name, counts, pp_rank, ep_rank, experts_per_rank)
            if expert is not None:
                copy_tensor(param, load_expert_tensor(reader, *expert), name)
                loaded += 1
                continue

            hf_name = map_nonexpert_name(name, counts, pp_rank)
            if hf_name is not None and reader.has(hf_name):
                copy_tensor(param, maybe_dequant_linear(reader, hf_name), name)
                loaded += 1
                continue

            missing.append(name)

    return loaded, missing


def load_buffers(model, reader: HFShardReader, counts: list[int]) -> tuple[int, list[str]]:
    from megatron.core import parallel_state

    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    loaded = 0
    missing = []
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            mapped = layer_hf_prefix(name, counts, pp_rank)
            if mapped is None:
                continue
            hf_prefix, suffix = mapped
            if suffix == "mlp.router.tid2eid":
                hf_name = hf_prefix + "ffn.gate.tid2eid"
                if reader.has(hf_name):
                    copy_tensor(buffer, reader.get(hf_name), name)
                    loaded += 1
                else:
                    missing.append(name)
            elif suffix == "mlp.router.expert_bias":
                hf_name = hf_prefix + "ffn.gate.bias"
                if reader.has(hf_name):
                    copy_tensor(buffer, reader.get(hf_name), name)
                    loaded += 1
                else:
                    missing.append(name)
    return loaded, missing


def add_converter_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--hf-checkpoint-dir", required=True)
    parser.add_argument("--megatron-save-dir", required=True)
    parser.add_argument("--save-iteration", type=int, default=1)
    parser.add_argument("--no-save-checkpoint", action="store_true")
    return parser


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    add_converter_args(parser)
    converter_args, megatron_args = parser.parse_known_args()

    sys.argv = [
        sys.argv[0],
        *megatron_args,
        "--save",
        converter_args.megatron_save_dir,
        "--save-interval",
        "1",
        "--no-save-optim",
        "--no-save-rng",
        "--no-load-optim",
        "--no-load-rng",
    ]

    from gpt_builders import gpt_builder
    from megatron.core import parallel_state
    from megatron.training import get_args, print_rank_0
    from megatron.training.arguments import parse_and_validate_args
    from megatron.training.checkpointing import save_checkpoint
    from megatron.training.initialize import initialize_megatron
    from model_provider import model_provider

    parse_and_validate_args()
    initialize_megatron()
    args = get_args()

    counts = parse_pipeline_layer_counts(
        args.pipeline_model_parallel_layout, args.num_layers, args.pipeline_model_parallel_size
    )
    reader = HFShardReader(converter_args.hf_checkpoint_dir)

    model = model_provider(
        gpt_builder,
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
    )

    loaded_params, missing_params = load_parameters(model, reader, counts)
    loaded_buffers, missing_buffers = load_buffers(model, reader, counts)

    rank = torch.distributed.get_rank()
    print(
        f"[rank {rank}] loaded_params={loaded_params} missing_params={len(missing_params)} "
        f"loaded_buffers={loaded_buffers} missing_buffers={len(missing_buffers)}",
        flush=True,
    )
    if missing_params or missing_buffers:
        print(f"[rank {rank}] missing params: {missing_params[:40]}", flush=True)
        print(f"[rank {rank}] missing buffers: {missing_buffers[:40]}", flush=True)
        raise RuntimeError("HF to Megatron conversion has missing tensors")

    torch.distributed.barrier()
    if not converter_args.no_save_checkpoint:
        save_checkpoint(
            converter_args.save_iteration,
            [model],
            None,
            None,
            num_floating_point_operations_so_far=0,
        )
    torch.distributed.barrier()
    print_rank_0(f"Saved DSv4-Flash model-only Megatron checkpoint to {converter_args.megatron_save_dir}")


if __name__ == "__main__":
    main()
