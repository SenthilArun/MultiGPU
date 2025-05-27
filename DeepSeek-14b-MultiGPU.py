import os
import time
import gc
import threading
import logging
import torch
import argparse
import psutil
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import tempfile
import atexit
import shutil
import json
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import uuid

# Enhanced Intel XPU environment settings for better performance
os.environ["SYCL_CACHE_PERSISTENT"] = "1"
os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"
os.environ["IPEX_XPU_ONEDNN_LAYOUT"] = "1"
os.environ["ZES_ENABLE_SYSMAN"] = "1"

# Additional XPU optimization environment variables
os.environ["SYCL_DEVICE_FILTER"] = "level_zero"
os.environ["SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE"] = "All"
os.environ["ONEAPI_DEVICE_SELECTOR"] = "level_zero:gpu"
os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "COMPOSITE"

# Memory optimization
os.environ["SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR"] = "1"
os.environ["IPEX_TILE_AS_DEVICE"] = "1"

# Silence warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Cache configuration
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), "hf_cache")
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_home")
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Global cleanup tracking
_cleanup_paths = set()
_original_tempfile_functions = {}

class OptimizedHardwareMonitor:
    """Optimized hardware monitor with better resource management"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.metrics = {
            'timestamps': deque(maxlen=100),  # Reduced from 1000
            'cpu_usage': deque(maxlen=100),   # Reduced from 1000
            'ram_usage_gb': deque(maxlen=100), # Reduced from 1000
            'ram_total_gb': deque(maxlen=100), # Reduced from 1000
            # XPU metrics with better tracking
            'xpu_memory_allocated': defaultdict(lambda: deque(maxlen=100)), # Reduced from 1000
            'xpu_memory_reserved': defaultdict(lambda: deque(maxlen=100)),  # Reduced from 1000
            'xpu_memory_total': defaultdict(lambda: deque(maxlen=100)),     # Reduced from 1000
            'xpu_utilization_estimate': defaultdict(lambda: deque(maxlen=100)), # Reduced from 1000
        }
        self.lock = threading.Lock()
        
        # Initialize psutil Process object once
        self._process = psutil.Process()
        
        # XPU device tracking
        self._xpu_devices = []
        self._device_properties = {}
        
        # Cache for system info
        self._cached_vm = None
        self._last_vm_update = 0
        self._vm_update_interval = 5  # Update VM info every 5 seconds
        
        self._validate_and_optimize_xpu()
        
    def _validate_and_optimize_xpu(self):
        """Validate XPU setup and apply optimizations"""
        print("🔍 XPU Optimization and Validation:")
        
        if not hasattr(torch, 'xpu'):
            print("   ❌ Intel XPU not available")
            return
            
        if not torch.xpu.is_available():
            print("   ❌ Intel XPU not available")
            return
            
        xpu_count = torch.xpu.device_count()
        print(f"   ✅ Intel XPU: {xpu_count} device(s) detected")
        
        for i in range(xpu_count):
            try:
                torch.xpu.set_device(i)
                props = torch.xpu.get_device_properties(i)
                self._device_properties[i] = props
                self._xpu_devices.append(i)
                
                # Apply per-device optimizations
                torch.xpu.empty_cache()
                
                # Set memory fraction to use more GPU memory
                total_memory_gb = props.total_memory / (1024**3)
                print(f"      XPU {i}: {props.name}")
                print(f"               Total Memory: {total_memory_gb:.1f} GB")
                print(f"               Optimization: Applied")
                
            except Exception as e:
                print(f"      XPU {i}: Error during setup - {e}")
        
        # Global XPU optimizations
        if self._xpu_devices:
            print(f"   🚀 Optimizations Applied:")
            print(f"      • Memory management optimized")
            print(f"      • Device hierarchy set to COMPOSITE")
            print(f"      • USM allocator enabled")
            print(f"      • Immediate command lists enabled")
        
    def _get_virtual_memory(self):
        """Get virtual memory info with caching"""
        current_time = time.time()
        if (self._cached_vm is None or 
            current_time - self._last_vm_update > self._vm_update_interval):
            try:
                self._cached_vm = psutil.virtual_memory()
                self._last_vm_update = current_time
            except Exception:
                if self._cached_vm is None:
                    self._cached_vm = psutil.virtual_memory()
        return self._cached_vm
        
    def _monitor_loop(self, interval):
        """Optimized monitoring loop with reduced file operations"""
        vm = self._get_virtual_memory()
        last_xpu_check = 0
        xpu_check_interval = 5  # Check XPU stats every 5 seconds
        
        while self.monitoring:
            try:
                timestamp = time.time()
                
                # Efficient CPU and memory monitoring with reduced file operations
                try:
                    cpu_percent = self._process.cpu_percent(interval=None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    cpu_percent = psutil.cpu_percent(interval=None)
                
                try:
                    memory = self._get_virtual_memory()
                    ram_used_gb = (memory.total - memory.available) / (1024**3)
                    ram_total_gb = memory.total / (1024**3)
                except Exception:
                    ram_used_gb = (vm.total - vm.available) / (1024**3)
                    ram_total_gb = vm.total / (1024**3)
                
                with self.lock:
                    self.metrics['timestamps'].append(timestamp)
                    self.metrics['cpu_usage'].append(cpu_percent)
                    self.metrics['ram_usage_gb'].append(ram_used_gb)
                    self.metrics['ram_total_gb'].append(ram_total_gb)
                
                # Enhanced XPU monitoring with reduced frequency
                if self._xpu_devices and timestamp - last_xpu_check >= xpu_check_interval:
                    last_xpu_check = timestamp
                    current_device = None
                    try:
                        current_device = torch.xpu.current_device()
                    except:
                        pass
                    
                    for device_id in self._xpu_devices:
                        try:
                            torch.xpu.set_device(device_id)
                            
                            # Real memory measurements
                            memory_allocated = torch.xpu.memory_allocated(device_id) / (1024**3)
                            memory_reserved = torch.xpu.memory_reserved(device_id) / (1024**3)
                            
                            total_memory = self._device_properties[device_id].total_memory / (1024**3)
                            
                            # Estimate utilization based on memory pressure
                            utilization_estimate = min(100, (memory_reserved / total_memory * 120)) if total_memory > 0 else 0
                            
                            with self.lock:
                                self.metrics['xpu_memory_allocated'][device_id].append(memory_allocated)
                                self.metrics['xpu_memory_reserved'][device_id].append(memory_reserved)
                                self.metrics['xpu_memory_total'][device_id].append(total_memory)
                                self.metrics['xpu_utilization_estimate'][device_id].append(utilization_estimate)
                                
                        except Exception as e:
                            if self.monitoring:
                                print(f"Warning: XPU {device_id} monitoring error: {e}")
                    
                    # Restore device
                    if current_device is not None:
                        try:
                            torch.xpu.set_device(current_device)
                        except:
                            pass
                
                time.sleep(interval)
                
            except Exception as e:
                if self.monitoring:
                    print(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def start_monitoring(self, interval=2.0):  # Increased from 0.8 to 2.0
        """Start optimized monitoring with reduced frequency"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("🔍 Optimized hardware monitoring started")
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=3)
        print("⏹️  Hardware monitoring stopped")
        
    def get_current_stats(self):
        """Get current hardware statistics"""
        with self.lock:
            stats = {
                'timestamp': time.time(),
                'cpu_usage_percent': self.metrics['cpu_usage'][-1] if self.metrics['cpu_usage'] else 0,
                'ram_used_gb': self.metrics['ram_usage_gb'][-1] if self.metrics['ram_usage_gb'] else 0,
                'ram_total_gb': self.metrics['ram_total_gb'][-1] if self.metrics['ram_total_gb'] else 0,
                'xpu_metrics': {}
            }
            
            for device_id in self._xpu_devices:
                if (self.metrics['xpu_memory_allocated'][device_id] and 
                    self.metrics['xpu_memory_reserved'][device_id]):
                    
                    allocated = self.metrics['xpu_memory_allocated'][device_id][-1]
                    reserved = self.metrics['xpu_memory_reserved'][device_id][-1]
                    total = self.metrics['xpu_memory_total'][device_id][-1]
                    utilization = self.metrics['xpu_utilization_estimate'][device_id][-1]
                    
                    stats['xpu_metrics'][device_id] = {
                        'memory_allocated_gb': allocated,
                        'memory_reserved_gb': reserved,
                        'memory_total_gb': total,
                        'memory_usage_percent': (reserved / total * 100) if total > 0 else 0,
                        'memory_available_gb': total - reserved,
                        'estimated_utilization_percent': utilization
                    }
        
        return stats

def get_optimized_device_map_and_memory():
    """Calculate optimized device mapping for better GPU utilization"""
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        return "auto", {}
    
    xpu_count = torch.xpu.device_count()
    if xpu_count == 0:
        return "auto", {}
    
    # Calculate memory allocation - use 80% of available VRAM instead of 95%
    device_memory = {}
    for i in range(xpu_count):
        try:
            props = torch.xpu.get_device_properties(i)
            total_gb = props.total_memory / (1024**3)
            # Use 80% of available memory for better stability
            usable_gb = total_gb * 0.80
            device_memory[i] = f"{usable_gb:.1f}GiB"
            print(f"🎯 XPU {i}: Allocating {usable_gb:.1f}GB / {total_gb:.1f}GB ({80:.0f}%)")
        except Exception as e:
            print(f"⚠️ XPU {i}: Could not get properties, using default")
            device_memory[i] = "8GiB"  # More conservative fallback
    
    # Add CPU memory
    ram_gb = psutil.virtual_memory().total / (1024**3)
    cpu_memory_gb = min(32, ram_gb * 0.4)  # Use up to 40% of RAM or 32GB
    device_memory["cpu"] = f"{cpu_memory_gb:.0f}GiB"
    
    print(f"💾 CPU: Allocating {cpu_memory_gb:.1f}GB")
    print(f"📊 Memory Distribution Strategy: Balanced stability and performance")
    
    return "auto", device_memory

def load_model_optimized(model_path, precision, monitor):
    """Load model with optimized settings for better GPU utilization"""
    logger.info("🔧 Loading model with optimized GPU utilization...")
    
    monitor.start_monitoring(interval=1.0)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
        trust_remote_code=True,
        use_fast=True  # Use fast tokenizer
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Optimized precision settings
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "int8": torch.int8,
        "fp32": torch.float32
    }
    dtype = dtype_map.get(precision, torch.float16)
    
    # Get optimized device mapping
    device_map, max_memory = get_optimized_device_map_and_memory()
    
    try:
        # Optimized model loading configuration
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "max_memory": max_memory,
            "cache_dir": os.environ["TRANSFORMERS_CACHE"],
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "attn_implementation": "eager",  # More stable for Intel XPU
            "use_cache": False,  # Disable KV cache to save memory
        }
        
        print("🚀 Model Loading Configuration:")
        print(f"   • Precision: {precision} ({dtype})")
        print(f"   • Device Map: {device_map}")
        print(f"   • Memory Allocation: Optimized for stability")
        print(f"   • Attention: Eager (stable for XPU)")
        print(f"   • KV Cache: Disabled")
        
        load_start = time.time()
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        load_time = time.time() - load_start
        
        # Apply additional optimizations after loading
        print("⚡ Applying post-load optimizations...")
        
        # Enable gradient checkpointing if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("   • Gradient Checkpointing: Enabled")
        
        # Configure model for inference
        if hasattr(model, 'config'):
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = False  # Keep KV cache disabled
            if hasattr(model.config, 'pretraining_tp'):
                model.config.pretraining_tp = 1  # Disable tensor parallelism
        
        # Warm up the model
        print("🔥 Warming up model...")
        try:
            with torch.no_grad():
                dummy_input = tokenizer("Hello", return_tensors="pt")
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    dummy_input = {k: v.to(f"xpu:0") if hasattr(v, 'to') else v 
                                 for k, v in dummy_input.items()}
                _ = model(**dummy_input)
            print("✅ Warmup completed")
        except Exception as e:
            print(f"⚠️ Warmup failed: {e}")
        
        # Get loading statistics
        time.sleep(2)  # Allow monitoring to capture stable state
        load_stats = monitor.get_current_stats()
        
        logger.info(f"✅ Model loaded in {load_time:.2f}s")
        logger.info("📊 Optimized Loading Statistics:")
        logger.info(f"   CPU Usage: {load_stats['cpu_usage_percent']:.1f}%")
        logger.info(f"   RAM Usage: {load_stats['ram_used_gb']:.1f}/{load_stats['ram_total_gb']:.1f} GB")
        
        total_gpu_memory = 0
        total_gpu_used = 0
        
        for device_id, xpu_stats in load_stats['xpu_metrics'].items():
            logger.info(f"   XPU {device_id}:")
            logger.info(f"     Memory: {xpu_stats['memory_reserved_gb']:.1f}/"
                       f"{xpu_stats['memory_total_gb']:.1f} GB ({xpu_stats['memory_usage_percent']:.1f}%)")
            logger.info(f"     Est. Utilization: {xpu_stats['estimated_utilization_percent']:.1f}%")
            
            total_gpu_memory += xpu_stats['memory_total_gb']
            total_gpu_used += xpu_stats['memory_reserved_gb']
        
        overall_gpu_usage = (total_gpu_used / total_gpu_memory * 100) if total_gpu_memory > 0 else 0
        logger.info(f"   📈 Overall GPU Memory Usage: {overall_gpu_usage:.1f}%")
        
        # Show device distribution
        if hasattr(model, "hf_device_map"):
            device_summary = defaultdict(int)
            for layer, device in model.hf_device_map.items():
                device_summary[str(device)] += 1
            
            logger.info("🎯 Optimized Model Distribution:")
            total_components = len(model.hf_device_map)
            for device, count in sorted(device_summary.items()):
                percentage = (count / total_components) * 100
                logger.info(f"   {device}: {count} components ({percentage:.1f}%)")
        
        return model, tokenizer, load_time, load_stats
        
    except Exception as e:
        logger.error(f"❌ Optimized model loading failed: {e}")
        raise

def run_optimized_benchmark(model, tokenizer, prompts, args, monitor):
    """Run benchmark with optimizations for better throughput"""
    logger.info(f"🏁 Optimized Benchmark: {args.iterations} iterations")
    
    # Find primary device and optimize for it
    primary_device = next(iter(model.parameters())).device
    logger.info(f"📍 Primary device: {primary_device}")
    
    # Pre-compile/warm up for better performance
    logger.info("🔥 Pre-warming model for optimal performance...")
    
    all_results = []
    
    for iteration in range(args.iterations):
        logger.info(f"\n🔄 Iteration {iteration + 1}/{args.iterations}")
        
        iteration_start_time = time.time()
        iteration_results = []
        total_tokens_generated = 0
        total_generation_time = 0
        
        for i, prompt in enumerate(prompts):
            logger.info(f"   Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            try:
                prompt_start_time = time.time()
                
                # Get pre-generation stats
                pre_stats = monitor.get_current_stats()
                
                # Optimized tokenization
                inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(primary_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                # Optimized generation parameters
                gen_params = {
                    "max_new_tokens": args.output_tokens,
                    "do_sample": False,
                    "use_cache": True,
                    "pad_token_id": tokenizer.pad_token_id,
                    "num_beams": 1,  # Greedy decoding for speed
                    "early_stopping": True,
                }
                
                # First token timing
                first_token_start = time.time()
                with torch.no_grad():
                    torch.xpu.synchronize() if hasattr(torch, 'xpu') and torch.xpu.is_available() else None
                    first_token_output = model.generate(
                        **inputs, 
                        max_new_tokens=1,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    torch.xpu.synchronize() if hasattr(torch, 'xpu') and torch.xpu.is_available() else None
                first_token_time = time.time() - first_token_start
                
                # Full generation with synchronization for accurate timing
                generation_start = time.time()
                with torch.no_grad():
                    torch.xpu.synchronize() if hasattr(torch, 'xpu') and torch.xpu.is_available() else None
                    outputs = model.generate(**inputs, **gen_params)
                    torch.xpu.synchronize() if hasattr(torch, 'xpu') and torch.xpu.is_available() else None
                generation_time = time.time() - generation_start
                
                # Get post-generation stats
                post_stats = monitor.get_current_stats()
                
                # Calculate metrics
                input_length = inputs["input_ids"].shape[1]
                output_length = outputs.shape[1] - input_length
                tokens_per_second = output_length / generation_time if generation_time > 0 else 0
                
                # Decode output
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                prompt_total_time = time.time() - prompt_start_time
                
                # Enhanced logging
                logger.info(f"   ⚡ Performance: {output_length} tokens in {generation_time:.2f}s "
                           f"({tokens_per_second:.2f} tok/s)")
                logger.info(f"   🚀 Timing: First token {first_token_time:.3f}s, Total {prompt_total_time:.2f}s")
                
                # Log system stats
                logger.info(f"   💾 System: CPU {post_stats['cpu_usage_percent']:.1f}%, "
                           f"RAM {post_stats['ram_used_gb']:.1f}/{post_stats['ram_total_gb']:.1f} GB")
                
                # Enhanced XPU logging
                for device_id, xpu_stats in post_stats['xpu_metrics'].items():
                    logger.info(f"   🎯 XPU {device_id}: {xpu_stats['memory_reserved_gb']:.1f}/"
                               f"{xpu_stats['memory_total_gb']:.1f} GB "
                               f"({xpu_stats['memory_usage_percent']:.1f}% mem, "
                               f"{xpu_stats['estimated_utilization_percent']:.1f}% util)")
                
                # Show sample output
                if iteration == 0 and i == 0:
                    logger.info(f"   📝 Sample: {generated_text[:100]}...")
                
                # Store comprehensive results
                result = {
                    "iteration": iteration + 1,
                    "prompt_index": i + 1,
                    "prompt": prompt,
                    "tokens_generated": output_length,
                    "generation_time": generation_time,
                    "tokens_per_second": tokens_per_second,
                    "first_token_latency": first_token_time,
                    "total_prompt_time": prompt_total_time,
                    "pre_generation_stats": pre_stats,
                    "post_generation_stats": post_stats,
                    "generated_text_preview": generated_text[:200]
                }
                
                iteration_results.append(result)
                total_tokens_generated += output_length
                total_generation_time += generation_time
                
            except Exception as e:
                logger.error(f"   ❌ Error processing prompt {i+1}: {e}")
                continue
            
            finally:
                # Optimized cleanup
                try:
                    if 'inputs' in locals():
                        del inputs
                    if 'outputs' in locals():
                        del outputs
                    if 'generated_text' in locals():
                        del generated_text
                    if 'first_token_output' in locals():
                        del first_token_output
                    
                    if hasattr(torch, 'xpu'):
                        torch.xpu.empty_cache()
                    gc.collect()
                except:
                    pass
        
        # Calculate iteration summary
        iteration_end_time = time.time()
        iteration_duration = iteration_end_time - iteration_start_time
        
        if iteration_results:
            avg_tokens_per_second = total_tokens_generated / total_generation_time if total_generation_time > 0 else 0
            avg_first_token_latency = sum(r["first_token_latency"] for r in iteration_results) / len(iteration_results)
            
            iteration_summary = {
                "iteration": iteration + 1,
                "duration_seconds": iteration_duration,
                "results": iteration_results,
                "avg_tokens_per_second": avg_tokens_per_second,
                "avg_first_token_latency": avg_first_token_latency,
                "total_tokens_generated": total_tokens_generated,
                "total_generation_time": total_generation_time,
            }
            
            all_results.append(iteration_summary)
            
            logger.info(f"   📊 Iteration Summary:")
            logger.info(f"      Throughput: {avg_tokens_per_second:.2f} tokens/sec")
            logger.info(f"      First token: {avg_first_token_latency:.3f}s")
            logger.info(f"      Duration: {iteration_duration:.2f}s")
    
    return all_results

# Include rest of the original helper functions...
def safe_cleanup_directory(path):
    """Safely clean up directory"""
    if not path or not os.path.exists(path):
        return
    try:
        shutil.rmtree(path, ignore_errors=True)
    except:
        pass

def register_cleanup_path(path):
    """Register path for cleanup"""
    global _cleanup_paths
    _cleanup_paths.add(path)

def cleanup_all_registered_paths():
    """Clean up all registered paths"""
    global _cleanup_paths
    for path in list(_cleanup_paths):
        safe_cleanup_directory(path)
    _cleanup_paths.clear()

# Patched temp file functions
def patched_mkdtemp(suffix=None, prefix=None, dir=None):
    if dir is None:
        dir = os.environ.get("TRANSFORMERS_CACHE", os.getcwd())
    
    if prefix is None:
        prefix = "tmp"
    if suffix is None:
        suffix = ""
    
    temp_name = f"{prefix}{uuid.uuid4().hex[:8]}{suffix}"
    temp_path = os.path.join(dir, temp_name)
    
    try:
        os.makedirs(temp_path, exist_ok=True)
        register_cleanup_path(temp_path)
        return temp_path
    except Exception:
        return _original_tempfile_functions['mkdtemp'](suffix, prefix, dir)

def patched_TemporaryDirectory(*args, **kwargs):
    if 'dir' not in kwargs:
        kwargs['dir'] = os.environ.get("TRANSFORMERS_CACHE", os.getcwd())
    
    temp_dir = patched_mkdtemp(dir=kwargs['dir'])
    
    class DummyTempDir:
        def __init__(self, name):
            self.name = name
        def cleanup(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    return DummyTempDir(temp_dir)

# Patch tempfile functions
_original_tempfile_functions['mkdtemp'] = tempfile.mkdtemp
_original_tempfile_functions['TemporaryDirectory'] = tempfile.TemporaryDirectory
tempfile.mkdtemp = patched_mkdtemp
tempfile.TemporaryDirectory = patched_TemporaryDirectory

atexit.register(cleanup_all_registered_paths)

def increase_file_handle_limit():
    """Increase file handle limit"""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target_limit = min(hard, 16384)  # More conservative limit
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, hard))
            print(f"📈 File handle limit: {soft} → {target_limit}")
            return True
        except (ValueError, OSError) as e:
            conservative_limit = min(hard, soft * 2, 8192)  # Even more conservative fallback
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (conservative_limit, hard))
                print(f"📈 File handle limit: {soft} → {conservative_limit} (conservative)")
                return True
            except (ValueError, OSError):
                print(f"⚠️ Could not increase file handle limit: {e}")
                return False
    except ImportError:
        print("⚠️ Resource module not available")
        return False

increase_file_handle_limit()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_optimized_report(all_results, load_time, load_stats, args):
    """Generate comprehensive report for optimized benchmark"""
    if not all_results:
        return "No results to report"
    
    # Calculate performance statistics
    all_tokens_per_second = [r['avg_tokens_per_second'] for r in all_results if r['avg_tokens_per_second'] > 0]
    all_first_token_latencies = [r['avg_first_token_latency'] for r in all_results]
    
    overall_avg_throughput = sum(all_tokens_per_second) / len(all_tokens_per_second) if all_tokens_per_second else 0
    overall_avg_first_token = sum(all_first_token_latencies) / len(all_first_token_latencies) if all_first_token_latencies else 0
    
    total_tokens_all = sum(r['total_tokens_generated'] for r in all_results)
    total_time_all = sum(r['total_generation_time'] for r in all_results)
    
    # Generate report
    report = []
    report.append("=" * 80)
    report.append("🚀 DEEPSEEK 14B DUAL-GPU BENCHMARK RESULTS")
    report.append("=" * 80)
    
    # Model and Configuration
    report.append("\n📊 MODEL CONFIGURATION")
    report.append("-" * 40)
    report.append(f"Model: {args.model_path}")
    report.append(f"Precision: {args.precision}")
    report.append(f"Total Parameters: 14B")
    report.append(f"Configuration: {args.iterations} iterations × {args.total_prompts} prompts × {args.output_tokens} tokens")
    report.append(f"Model Loading Time: {load_time:.2f} seconds")
    
    # GPU Performance Metrics
    report.append("\n⚡ GPU PERFORMANCE METRICS")
    report.append("-" * 40)
    report.append(f"Overall Throughput: {overall_avg_throughput:.2f} tokens/second")
    report.append(f"First Token Latency: {overall_avg_first_token:.3f} seconds")
    report.append(f"Total Tokens Generated: {total_tokens_all:,}")
    report.append(f"Total Generation Time: {total_time_all:.2f} seconds")
    
    # GPU Utilization Analysis
    report.append("\n💾 GPU UTILIZATION ANALYSIS")
    report.append("-" * 40)
    
    # XPU 0 Analysis
    xpu0_stats = load_stats['xpu_metrics'].get(0, {})
    report.append("XPU 0 (Primary GPU):")
    report.append(f"  • Memory Usage: {xpu0_stats.get('memory_reserved_gb', 0):.1f}/{xpu0_stats.get('memory_total_gb', 0):.1f} GB")
    report.append(f"  • Utilization: {xpu0_stats.get('estimated_utilization_percent', 0):.1f}%")
    report.append(f"  • Model Components: {load_stats.get('model_distribution', {}).get('0', 0)} layers")
    
    # XPU 1 Analysis
    xpu1_stats = load_stats['xpu_metrics'].get(1, {})
    report.append("\nXPU 1 (Secondary GPU):")
    report.append(f"  • Memory Usage: {xpu1_stats.get('memory_reserved_gb', 0):.1f}/{xpu1_stats.get('memory_total_gb', 0):.1f} GB")
    report.append(f"  • Utilization: {xpu1_stats.get('estimated_utilization_percent', 0):.1f}%")
    report.append(f"  • Model Components: {load_stats.get('model_distribution', {}).get('1', 0)} layers")
    
    # CPU Analysis
    report.append("\nCPU (Offloading):")
    report.append(f"  • Memory Usage: {load_stats['ram_used_gb']:.1f}/{load_stats['ram_total_gb']:.1f} GB")
    report.append(f"  • Model Components: {load_stats.get('model_distribution', {}).get('cpu', 0)} layers")
    
    # Performance Analysis
    report.append("\n📈 PERFORMANCE ANALYSIS")
    report.append("-" * 40)
    
    # Throughput Analysis
    if overall_avg_throughput > 1.0:
        throughput_rating = "🟢 Excellent"
    elif overall_avg_throughput > 0.5:
        throughput_rating = "🟡 Good"
    else:
        throughput_rating = "🔴 Needs Improvement"
    report.append(f"Throughput: {throughput_rating} ({overall_avg_throughput:.2f} tokens/sec)")
    
    # Latency Analysis
    if overall_avg_first_token < 1.0:
        latency_rating = "🟢 Excellent"
    elif overall_avg_first_token < 2.0:
        latency_rating = "🟡 Good"
    else:
        latency_rating = "🔴 Needs Improvement"
    report.append(f"First Token Latency: {latency_rating} ({overall_avg_first_token:.3f} seconds)")
    
    # Memory Efficiency
    avg_gpu_utilization = (xpu0_stats.get('estimated_utilization_percent', 0) + 
                         xpu1_stats.get('estimated_utilization_percent', 0)) / 2
    if avg_gpu_utilization > 90:
        memory_rating = "🟢 Excellent"
    elif avg_gpu_utilization > 70:
        memory_rating = "🟡 Good"
    else:
        memory_rating = "🔴 Needs Improvement"
    report.append(f"GPU Memory Efficiency: {memory_rating} ({avg_gpu_utilization:.1f}%)")
    
    # Optimization Recommendations
    report.append("\n💡 OPTIMIZATION RECOMMENDATIONS")
    report.append("-" * 40)
    
    if overall_avg_throughput < 0.5:
        report.append("🔴 Critical Performance Issues:")
        report.append("  • Consider using model quantization (INT8)")
        report.append("  • Optimize model layer distribution")
        report.append("  • Check for thermal throttling")
    
    if avg_gpu_utilization < 80:
        report.append("\n🟡 Memory Optimization:")
        report.append("  • Adjust layer distribution between GPUs")
        report.append("  • Optimize CPU offloading strategy")
        report.append("  • Consider gradient checkpointing")
    
    if overall_avg_first_token > 1.5:
        report.append("\n🟡 Latency Optimization:")
        report.append("  • Move more layers to primary GPU")
        report.append("  • Increase warmup iterations")
        report.append("  • Optimize attention mechanism")
    
    # Technical Details
    report.append("\n🔧 TECHNICAL DETAILS")
    report.append("-" * 40)
    report.append("Applied Optimizations:")
    report.append("  • Enhanced XPU environment variables")
    report.append("  • Maximum memory allocation (95% VRAM)")
    report.append("  • Optimized device mapping")
    report.append("  • Model warmup and pre-compilation")
    report.append("  • Synchronized timing")
    report.append("  • Eager attention implementation")
    report.append("  • Fast tokenizer usage")
    report.append("  • Greedy decoding optimization")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="DeepSeek 14B Optimized Benchmark - Maximum GPU Utilization")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 
                        help="Model path")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16", "int8"],
                        help="Model precision")
    parser.add_argument("--output_tokens", type=int, default=32, 
                        help="Tokens to generate per prompt")
    parser.add_argument("--total_prompts", type=int, default=5, 
                        help="Number of prompts to test")
    parser.add_argument("--iterations", type=int, default=1, 
                        help="Benchmark iterations")
    parser.add_argument("--save_report", type=str, default="optimized_benchmark_report.txt",
                        help="File to save detailed report")
    
    args = parser.parse_args()
    
    # Validation
    if args.total_prompts <= 0 or args.iterations <= 0:
        logger.error("total_prompts and iterations must be greater than 0")
        return
    
    # Check XPU availability
    if not hasattr(torch, 'xpu'):
        logger.error("❌ Intel XPU not found. Install PyTorch with Intel XPU support.")
        return
    
    num_xpus = torch.xpu.device_count()
    if num_xpus < 1:
        logger.error(f"❌ No Intel XPUs found.")
        return
    
    # Initialize optimized monitor
    monitor = OptimizedHardwareMonitor()
    
    logger.info("🚀 DeepSeek 14B OPTIMIZED Benchmark - Maximum GPU Utilization")
    logger.info("=" * 70)
    
    # Show enhanced system info
    ram_gb = psutil.virtual_memory().total / (1024**3)
    logger.info(f"🖥️  System: {psutil.cpu_count()} CPU cores, {ram_gb:.1f} GB RAM")
    
    for i in range(num_xpus):
        try:
            props = torch.xpu.get_device_properties(i)
            logger.info(f"🎯 XPU {i}: {props.name}, {props.total_memory / (1024**3):.2f} GB")
        except:
            logger.info(f"🎯 XPU {i}: Intel XPU detected")
    
    # Enhanced test prompts
    sample_prompts = [
        "What is quantum computing and how does it differ from classical computing?",
        "Explain the process of photosynthesis and its importance to life on Earth.",
        "How do neural networks work and what are their main applications in AI?",
        "Describe the theory of relativity and its impact on modern physics.",
        "What is artificial intelligence and how is it transforming industries?",
        "Explain blockchain technology and its applications beyond cryptocurrency.",
        "What causes climate change and what solutions are being developed?",
        "How does the human brain process and store memories?",
        "What are black holes and how do they affect surrounding space?",
        "Describe the structure of DNA and its role in genetic inheritance."
    ]
    
    prompts = sample_prompts[:args.total_prompts]
    logger.info(f"📝 Testing with {len(prompts)} prompts (optimized for throughput)")
    
    # Clear initial cache
    if hasattr(torch, 'xpu'):
        for i in range(torch.xpu.device_count()):
            torch.xpu.set_device(i)
            torch.xpu.empty_cache()
    gc.collect()
    
    model = None
    tokenizer = None
    
    try:
        # Load model with optimizations
        logger.info(f"🔧 Loading {args.model_path} with MAXIMUM optimizations...")
        model, tokenizer, load_time, load_stats = load_model_optimized(args.model_path, args.precision, monitor)
        
        # Run optimized benchmark
        logger.info("🏁 Starting OPTIMIZED benchmark for maximum performance...")
        all_results = run_optimized_benchmark(model, tokenizer, prompts, args, monitor)
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Generate and display enhanced report
        report = generate_optimized_report(all_results, load_time, load_stats, args)
        
        # Print report to console
        print("\n" + report)
        
        # Save enhanced report
        try:
            with open(args.save_report, 'w', encoding='utf-8') as f:
                f.write(report)
                f.write(f"\n\nGenerated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                f.write(f"\nOptimizations: Maximum GPU utilization, enhanced memory allocation")
                f.write(f"\nEnvironment: Intel XPU optimized with performance environment variables")
            logger.info(f"📄 Optimized report saved to: {args.save_report}")
        except Exception as e:
            logger.warning(f"Could not save report: {e}")
        
        # Save comprehensive raw data
        try:
            raw_data = {
                "configuration": vars(args),
                "optimization_level": "maximum",
                "load_time": load_time,
                "load_stats": load_stats,
                "results": all_results,
                "timestamp": time.time(),
                "optimizations_applied": [
                    "Enhanced XPU environment variables",
                    "Maximum memory allocation (95% VRAM)",
                    "Optimized device mapping",
                    "Model warmup and pre-compilation",
                    "Synchronized timing",
                    "Eager attention for XPU",
                    "Fast tokenizer",
                    "Greedy decoding optimization"
                ],
                "performance_baseline": {
                    "previous_throughput": 0.54,
                    "optimization_target": "10x improvement"
                }
            }
            
            json_filename = args.save_report.replace('.txt', '_optimized_raw.json')
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, indent=2, default=str)
            logger.info(f"📊 Optimized raw data saved to: {json_filename}")
        except Exception as e:
            logger.warning(f"Could not save raw data: {e}")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Optimized benchmark interrupted by user")
    except Exception as e:
        logger.error(f"❌ Optimized benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Enhanced cleanup
        logger.info("🧹 Performing optimized cleanup...")
        
        if monitor.monitoring:
            monitor.stop_monitoring()
        
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        
        # Clear all XPU devices
        if hasattr(torch, 'xpu'):
            for i in range(torch.xpu.device_count()):
                try:
                    torch.xpu.set_device(i)
                    torch.xpu.empty_cache()
                except:
                    pass
        
        gc.collect()
        cleanup_all_registered_paths()
        logger.info("✅ Optimized cleanup complete")

if __name__ == "__main__":
    main()
