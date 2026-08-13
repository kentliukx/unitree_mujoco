#!/usr/bin/env python3
"""Export the student GRU policy to ONNX and optionally build a TensorRT engine.

The exported graph keeps the GRU hidden state explicit so its state transition can
be checked against the existing PyTorch deployment before TensorRT is used.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Export the student GRU policy for TensorRT.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--onnx", type=Path, default=None, help="Defaults next to the checkpoint.")
    parser.add_argument("--engine", type=Path, default=None, help="Defaults next to the checkpoint.")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--build-engine", action="store_true", help="Build the TensorRT engine with local trtexec.")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 inference instead of the default FP32.")
    parser.add_argument("--workspace-mb", type=int, default=512)
    parser.add_argument("--verify-samples", type=int, default=32, help="Number of TensorRT/PyTorch comparisons.")
    parser.add_argument("--verify-atol", type=float, default=None, help="Override the precision-specific absolute-error limit.")
    return parser.parse_args()


def import_tensorrt():
    """Import the JetPack TensorRT binding when this runs inside a conda env."""
    system_site = Path(f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages")
    if system_site.is_dir() and str(system_site) not in sys.path:
        sys.path.append(str(system_site))
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError(
            "TensorRT Python bindings are required for engine verification. "
            "On the Go2 Orin, install python3-libnvinfer and run this script with Python 3.8."
        ) from error
    return trt


def verify_engine(engine_path, wrapper, obs_shape, hidden_shape, action_shape, samples, atol):
    """Compare the serialized TensorRT engine with the PyTorch policy."""
    import torch

    trt = import_tensorrt()
    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as engine_file, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(engine_file.read())
    if engine is None:
        raise RuntimeError(f"TensorRT could not deserialize engine: {engine_path}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT could not create an execution context.")
    binding_indices = {engine.get_binding_name(index): index for index in range(engine.num_bindings)}
    expected_bindings = {"obs", "hidden_in", "action", "hidden_out"}
    if set(binding_indices) != expected_bindings:
        raise RuntimeError(
            "Unexpected TensorRT binding names: "
            f"{sorted(binding_indices)}; expected {sorted(expected_bindings)}"
        )

    generator = torch.Generator(device="cpu").manual_seed(20260718)
    max_action_error = 0.0
    max_hidden_error = 0.0
    with torch.inference_mode():
        for sample_index in range(samples):
            # Include the all-zero initial state once; remaining samples cover normal values.
            if sample_index == 0:
                obs_cpu = torch.zeros(obs_shape, dtype=torch.float32)
                hidden_cpu = torch.zeros(hidden_shape, dtype=torch.float32)
            else:
                obs_cpu = torch.randn(obs_shape, generator=generator, dtype=torch.float32).clamp_(-3.0, 3.0)
                hidden_cpu = torch.randn(hidden_shape, generator=generator, dtype=torch.float32).mul_(0.25)

            reference_action, reference_hidden = wrapper(obs_cpu, hidden_cpu)
            obs_cuda = obs_cpu.cuda()
            hidden_cuda = hidden_cpu.cuda()
            action_cuda = torch.empty(action_shape, dtype=torch.float32, device="cuda")
            hidden_out_cuda = torch.empty(hidden_shape, dtype=torch.float32, device="cuda")
            bindings = [0] * engine.num_bindings
            bindings[binding_indices["obs"]] = obs_cuda.data_ptr()
            bindings[binding_indices["hidden_in"]] = hidden_cuda.data_ptr()
            bindings[binding_indices["action"]] = action_cuda.data_ptr()
            bindings[binding_indices["hidden_out"]] = hidden_out_cuda.data_ptr()
            if not context.execute_v2(bindings):
                raise RuntimeError(f"TensorRT inference failed on verification sample {sample_index}.")
            torch.cuda.synchronize()

            max_action_error = max(max_action_error, float((action_cuda.cpu() - reference_action).abs().max()))
            max_hidden_error = max(max_hidden_error, float((hidden_out_cuda.cpu() - reference_hidden).abs().max()))

    max_error = max(max_action_error, max_hidden_error)
    print(
        "[trt-export] TensorRT verified "
        f"samples={samples} action_max_abs={max_action_error:.3e} "
        f"hidden_max_abs={max_hidden_error:.3e} atol={atol:.3e}",
        flush=True,
    )
    if max_error > atol:
        raise RuntimeError(
            "TensorRT engine differs from PyTorch beyond the allowed tolerance: "
            f"max_abs={max_error:.3e}, atol={atol:.3e}"
        )


def main():
    args = parse_args()

    import onnx
    import torch
    from torch import nn

    from deploy_student import StudentPolicy, resolve_config

    cfg = resolve_config()
    if args.checkpoint is not None:
        cfg.checkpoint = args.checkpoint.resolve()
    if not cfg.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {cfg.checkpoint}")
    output_dir = cfg.checkpoint.parent
    checkpoint_stem = cfg.checkpoint.stem
    if args.onnx is None:
        args.onnx = output_dir / f"{checkpoint_stem}_gru.onnx"
    precision = "fp16" if args.fp16 else "fp32"
    if args.engine is None:
        args.engine = output_dir / f"{checkpoint_stem}_gru_{precision}.engine"
    args.onnx = args.onnx.resolve()
    args.engine = args.engine.resolve()
    print(
        f"[trt-export] checkpoint={cfg.checkpoint}\n"
        f"[trt-export] onnx={args.onnx}\n"
        f"[trt-export] engine={args.engine}\n"
        f"[trt-export] precision={precision}",
        flush=True,
    )
    cfg.device = "cpu"
    # The engine is being created, so load the checkpoint through PyTorch first.
    cfg.inference_backend = "pytorch"
    policy = StudentPolicy(cfg)
    module = policy.module
    module.eval()

    if not hasattr(module, "memory_a") or not isinstance(module.memory_a.rnn, nn.GRU):
        raise RuntimeError("This exporter expects the current StudentActorCritic GRU policy.")

    class ExplicitGruPolicy(nn.Module):
        def __init__(self, policy_module):
            super().__init__()
            self.policy_module = policy_module

        def forward(self, obs, hidden_in):
            # act_inference owns the feature encoders; make its GRU state an I/O tensor.
            self.policy_module.memory_a.hidden_states = hidden_in
            action = self.policy_module.act_inference(obs)
            hidden_out = self.policy_module.memory_a.hidden_states
            return action, hidden_out

    wrapper = ExplicitGruPolicy(module).eval()
    hidden_shape = (
        module.memory_a.rnn.num_layers,
        1,
        module.memory_a.rnn.hidden_size,
    )
    obs = torch.zeros((1, cfg.obs_dim), dtype=torch.float32)
    hidden_in = torch.zeros(hidden_shape, dtype=torch.float32)

    with torch.inference_mode():
        module.memory_a.hidden_states = hidden_in.clone()
        reference_action = module.act_inference(obs).clone()
        reference_hidden = module.memory_a.hidden_states.clone()
        module.memory_a.hidden_states = None

        exported_action, exported_hidden = wrapper(obs, hidden_in)
        action_error = float((reference_action - exported_action).abs().max())
        hidden_error = float((reference_hidden - exported_hidden).abs().max())
    if action_error > 1e-6 or hidden_error > 1e-6:
        raise RuntimeError(
            "Explicit GRU wrapper disagrees with act_inference: "
            f"action_max_abs={action_error:.3e}, hidden_max_abs={hidden_error:.3e}"
        )
    print(
        "[trt-export] PyTorch wrapper verified "
        f"action_max_abs={action_error:.3e} hidden_max_abs={hidden_error:.3e}",
        flush=True,
    )

    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (obs, hidden_in),
        str(args.onnx),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["obs", "hidden_in"],
        output_names=["action", "hidden_out"],
    )
    onnx_model = onnx.load(str(args.onnx))
    onnx.checker.check_model(onnx_model)
    if not args.onnx.is_file() or args.onnx.stat().st_size == 0:
        raise RuntimeError(f"ONNX export did not produce a non-empty file: {args.onnx}")
    print(
        f"[trt-export] wrote valid ONNX ({args.onnx.stat().st_size / 1024:.1f} KiB): {args.onnx} "
        f"obs={tuple(obs.shape)} hidden={hidden_shape}",
        flush=True,
    )

    if not args.build_engine:
        print(
            "[trt-export] build command:\n"
            f"  /usr/src/tensorrt/bin/trtexec --onnx={args.onnx} --saveEngine={args.engine} "
            f"{'--fp16 ' if args.fp16 else '--noTF32 '}--workspace={args.workspace_mb}",
            flush=True,
        )
        return

    trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    if not Path(trtexec).is_file():
        raise RuntimeError("trtexec was not found. Install TensorRT on the target Orin first.")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    command = [
        trtexec,
        f"--onnx={args.onnx}",
        f"--saveEngine={args.engine}",
        f"--workspace={args.workspace_mb}",
    ]
    if args.fp16:
        command.append("--fp16")
    else:
        # TensorRT enables TF32 by default on Ampere. This is faster but is not
        # full FP32 numerically, so disable it for a meaningful equivalence check.
        command.append("--noTF32")
    print("[trt-export] running: " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    if not args.engine.is_file() or args.engine.stat().st_size == 0:
        raise RuntimeError(f"TensorRT did not produce a non-empty engine: {args.engine}")
    print(
        f"[trt-export] wrote TensorRT engine ({args.engine.stat().st_size / 1024:.1f} KiB): {args.engine}",
        flush=True,
    )
    verify_atol = args.verify_atol if args.verify_atol is not None else (5e-3 if args.fp16 else 1e-5)
    verify_engine(
        args.engine,
        wrapper,
        tuple(obs.shape),
        hidden_shape,
        tuple(reference_action.shape),
        args.verify_samples,
        verify_atol,
    )


if __name__ == "__main__":
    main()
