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
    parser.add_argument("--split-engine", action="store_true", help="Export separate depth and policy-core TensorRT engines.")
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


def load_engine(engine_path):
    """Deserialize an engine and return its execution context and bindings."""
    trt = import_tensorrt()
    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as engine_file, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(engine_file.read())
    if engine is None:
        raise RuntimeError(f"TensorRT could not deserialize engine: {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"TensorRT could not create an execution context: {engine_path}")
    bindings = {engine.get_binding_name(index): index for index in range(engine.num_bindings)}
    return trt, engine, context, bindings


def verify_split_engines(depth_engine_path, core_engine_path, depth_wrapper, core_wrapper,
                         module, obs_shape, hidden_shape, samples, atol):
    """Check the two TensorRT engines together against the PyTorch model."""
    import torch

    trt, depth_engine, depth_context, depth_indices = load_engine(depth_engine_path)
    _, core_engine, core_context, core_indices = load_engine(core_engine_path)
    if set(depth_indices) != {"depth", "depth_latent"}:
        raise RuntimeError(f"Unexpected depth-engine bindings: {sorted(depth_indices)}")
    expected_core = {"curr_proprio_noisy", "goal", "proprio_history", "contact_precision", "depth_latent", "hidden_in", "action", "hidden_out"}
    if set(core_indices) != expected_core:
        raise RuntimeError(f"Unexpected core-engine bindings: {sorted(core_indices)}")

    slices = module.obs_slices
    generator = torch.Generator(device="cpu").manual_seed(20260814)
    max_action_error = 0.0
    max_hidden_error = 0.0
    with torch.inference_mode():
        for index in range(samples):
            obs_cpu = torch.zeros(obs_shape, dtype=torch.float32) if index == 0 else torch.randn(
                obs_shape, generator=generator, dtype=torch.float32
            ).clamp_(-3.0, 3.0)
            hidden_cpu = torch.zeros(hidden_shape, dtype=torch.float32) if index == 0 else torch.randn(
                hidden_shape, generator=generator, dtype=torch.float32
            ).mul_(0.25)
            depth_cpu = obs_cpu[:, slices["depth_image"]].reshape(1, 1, 36, 54)
            reference_depth = depth_wrapper(depth_cpu)
            reference_action, reference_hidden = core_wrapper(
                obs_cpu[:, slices["curr_proprio_noisy"]],
                obs_cpu[:, slices["goal"]],
                obs_cpu[:, slices["proprio_history"]],
                obs_cpu[:, slices["contact_precision"]],
                reference_depth,
                hidden_cpu,
            )

            depth_cuda = depth_cpu.cuda()
            depth_latent_cuda = torch.empty_like(reference_depth, device="cuda")
            depth_bindings = [0] * depth_engine.num_bindings
            depth_bindings[depth_indices["depth"]] = depth_cuda.data_ptr()
            depth_bindings[depth_indices["depth_latent"]] = depth_latent_cuda.data_ptr()
            if not depth_context.execute_v2(depth_bindings):
                raise RuntimeError(f"TensorRT depth inference failed on verification sample {index}.")

            core_inputs = {
                "curr_proprio_noisy": obs_cpu[:, slices["curr_proprio_noisy"]].cuda(),
                "goal": obs_cpu[:, slices["goal"]].cuda(),
                "proprio_history": obs_cpu[:, slices["proprio_history"]].cuda(),
                "contact_precision": obs_cpu[:, slices["contact_precision"]].cuda(),
                "depth_latent": depth_latent_cuda,
                "hidden_in": hidden_cpu.cuda(),
            }
            action_cuda = torch.empty_like(reference_action, device="cuda")
            hidden_out_cuda = torch.empty_like(hidden_cpu, device="cuda")
            core_outputs = {"action": action_cuda, "hidden_out": hidden_out_cuda}
            core_bindings = [0] * core_engine.num_bindings
            for name, tensor in {**core_inputs, **core_outputs}.items():
                core_bindings[core_indices[name]] = tensor.data_ptr()
            if not core_context.execute_v2(core_bindings):
                raise RuntimeError(f"TensorRT core inference failed on verification sample {index}.")
            torch.cuda.synchronize()
            max_action_error = max(max_action_error, float((action_cuda.cpu() - reference_action).abs().max()))
            max_hidden_error = max(max_hidden_error, float((hidden_out_cuda.cpu() - reference_hidden).abs().max()))

    max_error = max(max_action_error, max_hidden_error)
    print(
        "[trt-export] split TensorRT verified "
        f"samples={samples} action_max_abs={max_action_error:.3e} "
        f"hidden_max_abs={max_hidden_error:.3e} atol={atol:.3e}",
        flush=True,
    )
    if max_error > atol:
        raise RuntimeError(
            "Split TensorRT engines differ from PyTorch beyond the allowed tolerance: "
            f"max_abs={max_error:.3e}, atol={atol:.3e}"
        )


def build_engine(trtexec, onnx_path, engine_path, args):
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={args.workspace_mb}",
    ]
    command.append("--fp16" if args.fp16 else "--noTF32")
    print("[trt-export] running: " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    if not engine_path.is_file() or engine_path.stat().st_size == 0:
        raise RuntimeError(f"TensorRT did not produce a non-empty engine: {engine_path}")
    print(f"[trt-export] wrote TensorRT engine ({engine_path.stat().st_size / 1024:.1f} KiB): {engine_path}")


def export_split_engines(args, cfg, module, torch, nn, onnx, precision):
    """Export the convolutional depth encoder separately from the recurrent core.

    TensorRT 8.5 on the Orin crashes while selecting tactics for the complete
    graph. Keeping the small depth CNN out of the recurrent graph avoids that
    failing fused graph and leaves the policy state transition explicit.
    """
    if module.memory_a.rnn.num_layers != 1 or not isinstance(module.memory_a.rnn, nn.GRU):
        raise RuntimeError("Split TensorRT export currently requires a one-layer GRU policy.")
    if module.depth_height != 36 or module.depth_width != 54:
        raise RuntimeError("Split TensorRT export expects a 36x54 depth input.")

    class DepthEncoder(nn.Module):
        def __init__(self, policy_module):
            super().__init__()
            self.depth_encoder = policy_module.depth_encoder

        def forward(self, depth):
            return self.depth_encoder(depth)

    class PolicyCore(nn.Module):
        def __init__(self, policy_module):
            super().__init__()
            self.history_encoder = policy_module.history_encoder
            self.estimator = policy_module.estimator
            self.mixer = policy_module.mixer
            self.actor = policy_module.actor
            self.gru = policy_module.memory_a.rnn
            self.rnn_latent_dim = policy_module.recurrent_latent_dim
            self.ladder_obs_dim = policy_module.recurrent_output_dim - self.rnn_latent_dim
            self.estimator_dim = policy_module.estimator_dim

        def forward(self, curr_proprio_noisy, goal, proprio_history, contact_precision, depth_latent, hidden_in):
            history_latent = self.history_encoder(proprio_history)
            estimator_output = self.estimator(history_latent)
            estimated = torch.cat(
                (estimator_output[:, :3], torch.sigmoid(estimator_output[:, 3:7]), estimator_output[:, 7:15]),
                dim=-1,
            )
            mixer_latent = self.mixer(torch.cat((history_latent, depth_latent), dim=-1))
            previous = hidden_in[0]
            input_gates = torch.nn.functional.linear(mixer_latent, self.gru.weight_ih_l0, self.gru.bias_ih_l0)
            hidden_gates = torch.nn.functional.linear(previous, self.gru.weight_hh_l0, self.gru.bias_hh_l0)
            input_reset, input_update, input_new = input_gates.chunk(3, dim=-1)
            hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)
            reset = torch.sigmoid(input_reset + hidden_reset)
            update = torch.sigmoid(input_update + hidden_update)
            candidate = torch.tanh(input_new + reset * hidden_new)
            next_hidden = (1.0 - update) * candidate + update * previous
            z = next_hidden[:, :self.rnn_latent_dim]
            ladder = next_hidden[:, self.rnn_latent_dim:]
            if self.ladder_obs_dim == 13 and self.estimator_dim == 11:
                actor_input = torch.cat(
                    (curr_proprio_noisy, goal, estimator_output[:, :3], contact_precision, ladder[:, :8],
                     estimator_output[:, 3:5], estimator_output[:, 5:11], ladder[:, 8:13], z),
                    dim=-1,
                )
            elif self.ladder_obs_dim == 13:
                actor_input = torch.cat(
                    (curr_proprio_noisy, goal, estimated[:, :7], ladder[:, :8], estimated[:, 7:9],
                     estimated[:, 9:15], ladder[:, 8:13], z),
                    dim=-1,
                )
            elif self.ladder_obs_dim == 5:
                actor_input = torch.cat(
                    (curr_proprio_noisy, goal, estimated, ladder, z), dim=-1
                )
            else:
                raise RuntimeError(f"Unsupported reconstructed ladder dimension: {self.ladder_obs_dim}")
            return self.actor(actor_input), next_hidden.unsqueeze(0)

    checkpoint_stem = cfg.checkpoint.stem
    output_dir = cfg.checkpoint.parent
    depth_onnx = output_dir / f"{checkpoint_stem}_depth.onnx"
    core_onnx = output_dir / f"{checkpoint_stem}_core.onnx"
    depth_engine = output_dir / f"{checkpoint_stem}_depth_{precision}.engine"
    core_engine = output_dir / f"{checkpoint_stem}_core_{precision}.engine"
    depth_wrapper = DepthEncoder(module).eval()
    core_wrapper = PolicyCore(module).eval()
    slices = module.obs_slices
    hidden_shape = (1, 1, module.memory_a.rnn.hidden_size)
    depth = torch.zeros((1, 1, module.depth_height, module.depth_width), dtype=torch.float32)
    noisy = torch.zeros((1, slices["curr_proprio_noisy"].stop - slices["curr_proprio_noisy"].start), dtype=torch.float32)
    goal = torch.zeros((1, slices["goal"].stop - slices["goal"].start), dtype=torch.float32)
    history = torch.zeros((1, slices["proprio_history"].stop - slices["proprio_history"].start), dtype=torch.float32)
    contact_precision = torch.zeros((1, 4), dtype=torch.float32)
    hidden = torch.zeros(hidden_shape, dtype=torch.float32)

    action_error = 0.0
    hidden_error = 0.0
    generator = torch.Generator(device="cpu").manual_seed(20260814)
    with torch.inference_mode():
        for sample_index in range(8):
            sample_obs = torch.zeros((1, cfg.obs_dim), dtype=torch.float32) if sample_index == 0 else torch.randn(
                (1, cfg.obs_dim), generator=generator, dtype=torch.float32
            ).clamp_(-3.0, 3.0)
            sample_hidden = torch.zeros(hidden_shape, dtype=torch.float32) if sample_index == 0 else torch.randn(
                hidden_shape, generator=generator, dtype=torch.float32
            ).mul_(0.25)
            sample_depth = sample_obs[:, slices["depth_image"]].reshape(1, 1, 36, 54)
            split_depth = depth_wrapper(sample_depth)
            split_action, split_hidden = core_wrapper(
                sample_obs[:, slices["curr_proprio_noisy"]], sample_obs[:, slices["goal"]],
                sample_obs[:, slices["proprio_history"]], sample_obs[:, slices["contact_precision"]],
                split_depth, sample_hidden,
            )
            module.memory_a.hidden_states = sample_hidden.clone()
            reference_action = module.act_inference(sample_obs).clone()
            reference_hidden = module.memory_a.hidden_states.clone()
            action_error = max(action_error, float((reference_action - split_action).abs().max()))
            hidden_error = max(hidden_error, float((reference_hidden - split_hidden).abs().max()))
        module.memory_a.hidden_states = None
    # The explicit equations and torch.nn.GRU use a different FP32 operation
    # order. A few micro-units is expected and still far below TRT tolerance.
    if action_error > 1e-5 or hidden_error > 1e-5:
        raise RuntimeError(
            "Split PyTorch wrappers disagree with act_inference: "
            f"action_max_abs={action_error:.3e}, hidden_max_abs={hidden_error:.3e}"
        )
    print(
        "[trt-export] split PyTorch wrappers verified "
        f"action_max_abs={action_error:.3e} hidden_max_abs={hidden_error:.3e}", flush=True
    )

    for path, model, model_inputs, input_names, output_names in (
        (depth_onnx, depth_wrapper, (depth,), ["depth"], ["depth_latent"]),
        (core_onnx, core_wrapper, (noisy, goal, history, contact_precision, split_depth, hidden),
         ["curr_proprio_noisy", "goal", "proprio_history", "contact_precision", "depth_latent", "hidden_in"],
         ["action", "hidden_out"]),
    ):
        torch.onnx.export(
            model, model_inputs, str(path), export_params=True, opset_version=args.opset,
            do_constant_folding=True, input_names=input_names, output_names=output_names,
        )
        onnx.checker.check_model(onnx.load(str(path)))
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"ONNX export did not produce a non-empty file: {path}")
        print(f"[trt-export] wrote valid ONNX ({path.stat().st_size / 1024:.1f} KiB): {path}", flush=True)

    print(f"[trt-export] depth engine={depth_engine}\n[trt-export] core engine={core_engine}", flush=True)
    if not args.build_engine:
        return
    trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    if not Path(trtexec).is_file():
        raise RuntimeError("trtexec was not found. Install TensorRT on the target Orin first.")
    build_engine(trtexec, depth_onnx, depth_engine, args)
    build_engine(trtexec, core_onnx, core_engine, args)
    # TensorRT's FP32 kernels may reorder reductions relative to PyTorch. The
    # split path has two such kernels, so allow a small numerical margin while
    # still rejecting errors large enough to indicate a wiring mistake.
    verify_atol = args.verify_atol if args.verify_atol is not None else (5e-3 if args.fp16 else 2e-5)
    verify_split_engines(
        depth_engine, core_engine, depth_wrapper, core_wrapper, module,
        (1, cfg.obs_dim), hidden_shape, args.verify_samples, verify_atol,
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

    if args.split_engine:
        export_split_engines(args, cfg, module, torch, nn, onnx, precision)
        return

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
