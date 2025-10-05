# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os
import sys
from pathlib import Path
import torch
from hunyuan_image_3.hunyuan import HunyuanImage3ForCausalMM
from PE.deepseek import DeepSeekClient
from PE.system_prompt import system_prompt_universal, system_prompt_text_rendering


def print_profile_help():
    """Print detailed help for profiling options"""
    print("\n" + "="*60)
    print("PROFILING OPTIONS HELP")
    print("="*60)
    print("Basic Profiling:")
    print("  --profile                    Enable profiling")
    print("  --profile-trace-path PATH    Save trace to PATH (default: profile_trace.json)")
    print("  --profile-steps N            Number of steps to profile (default: 1)")
    print()
    print("Schedule Control:")
    print("  --profile-wait N             Wait N steps before profiling starts (default: 0)")
    print("  --profile-warmup N           Warmup N steps before profiling (default: 0)")
    print("  --profile-repeat N           Repeat profiling N times (default: 1)")
    print()
    print("Activities (what to profile):")
    print("  --profile-activities [cpu] [cuda] [kineto] [privateuse1]")
    print("    Default: cpu cuda")
    print()
    print("Output Format:")
    print("  --profile-export-format [chrome|json|both]")
    print("    chrome: Chrome trace format (for Chrome DevTools)")
    print("    json:   JSON format")
    print("    both:   Export both formats")
    print()
    print("Analysis Options:")
    print("  --profile-sort-by METRIC     Sort results by metric")
    print("    Options: cpu_time, cuda_time, cpu_time_total, cuda_time_total,")
    print("             cpu_memory_usage, cuda_memory_usage, self_cpu_time,")
    print("             self_cuda_time, count")
    print("  --profile-row-limit N        Show top N results (default: 10)")
    print("  --profile-group-by FIELD     Group results by field")
    print("    Options: none, stack")
    print()
    print("Memory Analysis:")
    print("  --profile-memory-format [alloc_self|alloc_total|self|total]")
    print("  --profile-memory-peak        Track peak memory usage")
    print()
    print("Advanced Options:")
    print("  --profile-detailed           Enable detailed profiling (FLOPS, modules)")
    print("  --profile-timing             Enable simple timing measurements")
    print("  --profile-diffusion-steps N  Profile only N diffusion steps (default: all)")
    print("  --profile-reduce-size        Reduce profile size by disabling detailed recording")
    print()
    print("Torch Compile Options:")
    print("  --compile-model              Compile entire model with torch.compile")
    print("  --compile-mode MODE          Compile mode: default, reduce-overhead, max-autotune")
    print()
    print("Example Usage:")
    print("  python run_image_gen.py --profile --prompt 'a cat'")
    print("  python run_image_gen.py --profile --profile-diffusion-steps 2 --profile-reduce-size")
    print("  python run_image_gen.py --profile --profile-detailed --profile-timing \\")
    print("    --profile-activities cpu cuda --profile-sort-by cuda_time_total \\")
    print("    --profile-export-format both --profile-trace-path my_trace")
    print("  python run_image_gen.py --profile --compile-model --profile-reduce-size")
    print("="*60)


def parse_args():
    parser = argparse.ArgumentParser("Commandline arguments for running HunyuanImage-3 locally")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to run")
    parser.add_argument("--model-id", type=str, default="./HunyuanImage-3", help="Path to the model")
    parser.add_argument("--attn-impl", type=str, default="flash_attention_2", choices=["sdpa", "flash_attention_2"],
                        help="Attention implementation. 'flash_attention_2' requires flash attention to be installed.")
    parser.add_argument("--moe-impl", type=str, default="eager", choices=["eager", "flashinfer"],
                        help="MoE implementation. 'flashinfer' requires FlashInfer to be installed.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Use None for random seed.")
    parser.add_argument("--diff-infer-steps", type=int, default=5, help="Number of inference steps.")
    parser.add_argument("--image-size", type=str, default="auto",
                        help="'auto' means image size is determined by the model. Alternatively, it can be in the "
                             "format of 'HxW' or 'H:W', which will be aligned to the set of preset sizes.")
    parser.add_argument("--use-system-prompt", type=str,
                        choices=["None", "dynamic", "en_vanilla", "en_recaption", "en_think_recaption", "custom"],
                        help="Use system prompt. 'None' means no system prompt; 'dynamic' means the system prompt is "
                             "determined by --bot-task; 'en_vanilla', 'en_recaption', 'en_think_recaption' are "
                             "three predefined system prompts; 'custom' means using the custom system prompt. When "
                             "using 'custom', --system-prompt must be provided. Default to load from the model "
                             "generation config.")
    parser.add_argument("--system-prompt", type=str, help="Custom system prompt. Used when --use-system-prompt "
                                                          "is 'custom'.")
    parser.add_argument("--bot-task", type=str, choices=["image", "auto", "think", "recaption"],
                        help="Type of task for the model. 'image' for direct image generation; 'auto' for text "
                             "generation; 'think' for think->re-write->image; 'recaption' for re-write->image."
                             "Default to load from the model generation config.")
    parser.add_argument("--save", type=str, default="image.png", help="Path to save the generated image")
    parser.add_argument("--verbose", type=int, default=0, help="Verbose level")
    parser.add_argument("--rewrite", type=int, default=1, help="Whether to rewrite the prompt with DeepSeek")
    parser.add_argument("--sys-deepseek-prompt", type=str, choices=["universal", "text_rendering"],
                        default="universal", help="System prompt for rewriting the prompt")

    parser.add_argument("--reproduce", action="store_true", help="Whether to reproduce the results")
    parser.add_argument("--profile", action="store_true", help="Enable torch profiler to capture trace")
    parser.add_argument("--profile-trace-path", type=str, default="profile_trace.json",
                        help="Path to save the profiler trace")
    parser.add_argument("--profile-steps", type=int, default=1,
                        help="Number of steps to profile (default: 1)")
    parser.add_argument("--profile-warmup", type=int, default=0,
                        help="Number of warmup steps before profiling starts (default: 0)")
    parser.add_argument("--profile-wait", type=int, default=0,
                        help="Number of wait steps before profiling starts (default: 0)")
    parser.add_argument("--profile-repeat", type=int, default=1,
                        help="Number of times to repeat profiling (default: 1)")
    parser.add_argument("--profile-activities", type=str, nargs="+", 
                        choices=["cpu", "cuda", "kineto", "privateuse1"],
                        default=["cpu", "cuda"],
                        help="Profiler activities to record (default: cpu cuda)")
    parser.add_argument("--profile-sort-by", type=str, 
                        choices=["cpu_time", "cuda_time", "cpu_time_total", "cuda_time_total", 
                                "cpu_memory_usage", "cuda_memory_usage", "self_cpu_time", 
                                "self_cuda_time", "count"],
                        default="cuda_time_total",
                        help="Sort profiling results by this metric (default: cuda_time_total)")
    parser.add_argument("--profile-row-limit", type=int, default=10,
                        help="Number of rows to show in profiling summary (default: 10)")
    parser.add_argument("--profile-export-format", type=str, 
                        choices=["chrome", "json", "both"],
                        default="chrome",
                        help="Export format for profiler trace (default: chrome)")
    parser.add_argument("--profile-memory-format", type=str,
                        choices=["alloc_self", "alloc_total", "self", "total"],
                        default="total",
                        help="Memory format for profiling (default: total)")
    parser.add_argument("--profile-group-by", type=str,
                        choices=["none", "stack"],
                        default="none",
                        help="Group profiling results by this field (default: none)")
    parser.add_argument("--profile-detailed", action="store_true",
                        help="Enable detailed profiling with additional metrics")
    parser.add_argument("--profile-timing", action="store_true",
                        help="Enable simple timing measurements alongside profiling")
    parser.add_argument("--profile-memory-peak", action="store_true",
                        help="Track peak memory usage during profiling")
    parser.add_argument("--profile-help", action="store_true",
                        help="Show detailed help for profiling options")
    parser.add_argument("--compile-model", action="store_true",
                        help="Compile the entire model with torch.compile for additional optimization")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="Torch compile mode (default: reduce-overhead)")
    parser.add_argument("--profile-diffusion-steps", type=int, default=None,
                        help="Number of diffusion steps to profile (default: same as diff-infer-steps)")
    parser.add_argument("--profile-reduce-size", action="store_true",
                        help="Reduce profile size by disabling detailed recording (shapes, stack, flops)")
    return parser.parse_args()


def validate_profile_args(args):
    """Validate profiling arguments and provide helpful warnings"""
    if args.profile:
        # Check if CUDA is available when CUDA profiling is requested
        if "cuda" in args.profile_activities and not torch.cuda.is_available():
            print("Warning: CUDA profiling requested but CUDA is not available. Removing CUDA from activities.")
            args.profile_activities = [a for a in args.profile_activities if a != "cuda"]
        
        # Check if Kineto is available when Kineto profiling is requested
        if "kineto" in args.profile_activities:
            try:
                # Check if Kineto is available
                torch.profiler.ProfilerActivity.KINETO
            except AttributeError:
                print("Warning: Kineto profiling requested but not available. Removing Kineto from activities.")
                args.profile_activities = [a for a in args.profile_activities if a != "kineto"]
        
        # Ensure at least one activity is selected
        if not args.profile_activities:
            print("Warning: No valid profiling activities selected. Defaulting to CPU profiling.")
            args.profile_activities = ["cpu"]
        
        # Validate schedule parameters
        if args.profile_wait < 0 or args.profile_warmup < 0 or args.profile_steps <= 0 or args.profile_repeat <= 0:
            raise ValueError("Profile schedule parameters must be non-negative, and steps/repeat must be positive")
        
        # Validate export format
        if args.profile_export_format not in ["chrome", "json", "both"]:
            raise ValueError("Profile export format must be 'chrome', 'json', or 'both'")
        
        # Validate sort options
        valid_sort_options = ["cpu_time", "cuda_time", "cpu_time_total", "cuda_time_total", 
                             "cpu_memory_usage", "cuda_memory_usage", "self_cpu_time", 
                             "self_cuda_time", "count"]
        if args.profile_sort_by not in valid_sort_options:
            raise ValueError(f"Profile sort by must be one of: {valid_sort_options}")
        
        # Validate memory format
        valid_memory_formats = ["alloc_self", "alloc_total", "self", "total"]
        if args.profile_memory_format not in valid_memory_formats:
            raise ValueError(f"Profile memory format must be one of: {valid_memory_formats}")
        
        # Validate group by options
        valid_group_options = ["none", "stack"]
        if args.profile_group_by not in valid_group_options:
            raise ValueError(f"Profile group by must be one of: {valid_group_options}")


def set_reproducibility(enable, global_seed=None, benchmark=None):
    import torch
    if enable:
        # Configure the seed for reproducibility
        import random
        random.seed(global_seed)
        # Seed the RNG for Numpy
        import numpy as np
        np.random.seed(global_seed)
        # Seed the RNG for all devices (both CPU and CUDA)
        torch.manual_seed(global_seed)
    # Set following debug environment variable
    # See the link for details: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    if enable:
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Cudnn benchmarking
    torch.backends.cudnn.benchmark = (not enable) if benchmark is None else benchmark
    # Use deterministic algorithms in PyTorch
    torch.backends.cudnn.deterministic = enable
    torch.use_deterministic_algorithms(enable)


def main(args):
    # Handle profile help
    if args.profile_help:
        print_profile_help()
        return
    
    # Validate profiling arguments first
    validate_profile_args(args)
    
    if args.reproduce:
        set_reproducibility(args.reproduce, global_seed=args.seed)

    if not args.prompt:
        raise ValueError("Prompt is required")
    if not Path(args.model_id).exists():
        raise ValueError(f"Model path {args.model_id} does not exist")

    kwargs = dict(
        attn_implementation=args.attn_impl,
        torch_dtype="auto",
        device_map="auto",
        moe_impl=args.moe_impl,
    )
    model = HunyuanImage3ForCausalMM.from_pretrained(args.model_id, **kwargs)
    model.load_tokenizer(args.model_id)
    
    # Apply model-level compilation if requested
    if args.compile_model:
        print(f"Compiling model with torch.compile (mode: {args.compile_mode})...")
        
        # Set environment variables to improve compilation
        import os
        os.environ["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] = "1"
        os.environ["TORCH_COMPILE_DEBUG"] = "1"
        
        # Compile the model with better settings
        model = torch.compile(
            model, 
            mode=args.compile_mode,
            fullgraph=False,  # Allow graph breaks for better compatibility
            dynamic=True,     # Enable dynamic shapes
        )
        print("Model compilation completed!")

    # Rewrite prompt with DeepSeek (or use dummy prompts if no API key)
    if args.rewrite:
        # Get request key_id and key_secret for DeepSeek
        deepseek_key_id = os.getenv("DEEPSEEK_KEY_ID")
        deepseek_key_secret = os.getenv("DEEPSEEK_KEY_SECRET")
        if not deepseek_key_id or not deepseek_key_secret:
            print("DeepSeek API key is not set. Using dummy prompt enhancement instead.")
            # Use dummy prompt enhancement based on the system prompt type
            if args.sys_deepseek_prompt == "universal":
                # Enhanced prompt for universal style
                enhanced_prompt = args.prompt #f"A brown and white dog is running on the grass. Photorealistic style, dynamic action shot from a low angle perspective. Natural outdoor lighting with warm sunlight filtering through the grass. The dog's fur is detailed and textured, with brown and white patches clearly visible. The grass is lush and green, slightly blurred in the background to create depth of field. f/2.8 aperture, 85mm lens, shallow depth of field, 8K resolution."
            elif args.sys_deepseek_prompt == "text_rendering":
                # Enhanced prompt for text rendering style
                enhanced_prompt = args.prompt # f"This is a photorealistic image of a brown and white dog running on grass. The dog is captured in mid-stride with its legs extended, showing dynamic movement. The dog has a mixed brown and white coat with natural fur texture. The background consists of lush green grass that extends to the horizon. Natural outdoor lighting illuminates the scene with warm, golden sunlight. The composition uses a low-angle perspective to emphasize the dog's movement and energy. The image has high resolution and sharp detail throughout."
            else:
                enhanced_prompt = args.prompt  # Fallback to original prompt

            print("Enhanced prompt (dummy): {}".format(enhanced_prompt))
            args.prompt = enhanced_prompt
        else:
            # Use actual DeepSeek API
            deepseek_client = DeepSeekClient(deepseek_key_id, deepseek_key_secret)

            if args.sys_deepseek_prompt == "universal":
                system_prompt = system_prompt_universal
            elif args.sys_deepseek_prompt == "text_rendering":
                system_prompt = system_prompt_text_rendering
            else:
                raise ValueError(f"Invalid system prompt: {args.sys_deepseek_prompt}")
            prompt, _ = deepseek_client.run_single_recaption(system_prompt, args.prompt)
            print("rewrite prompt: {}".format(prompt))
            args.prompt = prompt

    if args.profile:
        print(f"Profiling enabled. Will capture {args.profile_steps} step(s) and save trace to {args.profile_trace_path}")
        print(f"Profile schedule: wait={args.profile_wait}, warmup={args.profile_warmup}, active={args.profile_steps}, repeat={args.profile_repeat}")

        # Convert activity strings to ProfilerActivity enums
        activities = []
        for activity in args.profile_activities:
            if activity == "cpu":
                activities.append(torch.profiler.ProfilerActivity.CPU)
            elif activity == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            elif activity == "kineto":
                activities.append(torch.profiler.ProfilerActivity.KINETO)
            elif activity == "privateuse1":
                activities.append(torch.profiler.ProfilerActivity.PRIVATEUSE1)

        # Configure profiler with size reduction options
        record_shapes = not args.profile_reduce_size
        with_stack = not args.profile_reduce_size
        with_flops = args.profile_detailed and not args.profile_reduce_size
        with_modules = args.profile_detailed and not args.profile_reduce_size
        
        # Create a custom profiler that can be controlled at diffusion step level
        if args.profile_diffusion_steps is not None:
            # Use a more targeted profiler for specific diffusion steps
            profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    wait=0,  # Start immediately
                    warmup=0,  # No warmup needed
                    active=args.profile_diffusion_steps,  # Profile only the specified diffusion steps
                    repeat=1  # Single run
                ),
                record_shapes=record_shapes,
                profile_memory=True,
                with_stack=with_stack,
                with_flops=with_flops,
                with_modules=with_modules
            )
        else:
            # Use the original profiler configuration
            profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    wait=args.profile_wait,
                    warmup=args.profile_warmup,
                    active=args.profile_steps,
                    repeat=args.profile_repeat
                ),
                record_shapes=record_shapes,
                profile_memory=True,
                with_stack=with_stack,
                with_flops=with_flops,
                with_modules=with_modules
            )

        # Initialize timing and memory tracking
        import time
        start_time = None
        end_time = None
        peak_memory = 0
        
        if args.profile_timing:
            start_time = time.time()
        
        if args.profile_memory_peak and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        print(f"start profiling")
        profiler.start()

        # Determine number of diffusion steps for profiling
        profile_diffusion_steps = args.profile_diffusion_steps if args.profile_diffusion_steps is not None else args.diff_infer_steps
        
        if args.profile_diffusion_steps is not None and args.profile_diffusion_steps != args.diff_infer_steps:
            print(f"Profiling with {profile_diffusion_steps} diffusion steps (different from inference steps: {args.diff_infer_steps})")

        # Generate image with profiling
        image = model.generate_image(
            prompt=args.prompt,
            seed=args.seed,
            image_size=args.image_size,
            use_system_prompt=args.use_system_prompt,
            system_prompt=args.system_prompt,
            bot_task=args.bot_task,
            diff_infer_steps=profile_diffusion_steps,
            verbose=args.verbose,
            stream=True,
        )

        profiler.stop()
        
        if args.profile_timing:
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Total execution time: {total_time:.4f} seconds")
        
        if args.profile_memory_peak and torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
            print(f"Peak GPU memory usage: {peak_memory:.4f} GB")
        
        print(f"Profiling completed")

        # Export trace based on format choice
        if args.profile_export_format in ["chrome", "both"]:
            chrome_path = args.profile_trace_path
            if not chrome_path.endswith('.json'):
                chrome_path = chrome_path.replace('.json', '_chrome.json')
            profiler.export_chrome_trace(chrome_path)
            print(f"Chrome trace saved to {chrome_path}")
        
        if args.profile_export_format in ["json", "both"]:
            json_path = args.profile_trace_path
            if not json_path.endswith('.json'):
                json_path = json_path.replace('.json', '_trace.json')
            profiler.export_stacks(json_path)
            print(f"JSON trace saved to {json_path}")

        # Print profiling summary with configurable options
        print(f"\nProfiling Summary (sorted by {args.profile_sort_by}):")
        
        # Configure grouping
        group_by_input_shape = False
        if args.profile_group_by == "stack":
            group_by_input_shape = True
        
        # Get key averages with grouping
        key_averages = profiler.key_averages(group_by_input_shape=group_by_input_shape)
        
        # Print table with configurable sorting and row limit
        print(key_averages.table(
            sort_by=args.profile_sort_by, 
            row_limit=args.profile_row_limit
        ))
        
        # Print memory summary if detailed profiling is enabled
        if args.profile_detailed:
            print(f"\nMemory Summary (format: {args.profile_memory_format}):")
            memory_summary = profiler.key_averages().table(
                sort_by="cpu_memory_usage",
                row_limit=args.profile_row_limit
            )
            print(memory_summary)

    else:
        # Generate image without profiling
        image = model.generate_image(
            prompt=args.prompt,
            seed=args.seed,
            image_size=args.image_size,
            use_system_prompt=args.use_system_prompt,
            system_prompt=args.system_prompt,
            bot_task=args.bot_task,
            diff_infer_steps=args.diff_infer_steps,
            verbose=args.verbose,
            stream=True,
        )

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.save)
    print(f"Image saved to {args.save}")


if __name__ == "__main__":
    # Check for profile help before parsing all arguments
    if "--profile-help" in sys.argv:
        print_profile_help()
        sys.exit(0)
    
    args = parse_args()
    main(args)
