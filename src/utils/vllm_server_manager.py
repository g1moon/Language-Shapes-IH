"""
vLLM Server Manager - Start, stop, and check vLLM server
"""

import os
import sys
import time
import signal
import subprocess
import argparse
import requests
from typing import Optional


def find_vllm_server_pid(port: int) -> Optional[int]:
    """Find the PID of a vLLM server running on the given port"""
    try:
        # Use lsof to find the process using the port
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            return int(result.stdout.strip().split()[0])
    except:
        pass
    return None


def is_server_running(port: int, timeout: int = 2) -> bool:
    """Check if vLLM server is running and responding"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False


def stop_server(port: int):
    """Stop vLLM server running on the given port"""
    pid = find_vllm_server_pid(port)
    if pid:
        print(f"Stopping vLLM server (PID: {pid}) on port {port}...")
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
            # Check if still running
            try:
                os.kill(pid, 0)
                # Still running, force kill
                print(f"Force killing server...")
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            print(f"Server stopped successfully")
        except Exception as e:
            print(f"Error stopping server: {e}")
    else:
        print(f"No server found on port {port}")


def start_server(model_path: str, port: int, max_model_len: int = 4096, 
                 tensor_parallel: int = 1, gpu_memory_utilization: float = 0.95,
                 model_family: str = "qwen"):
    """Start vLLM server"""
    
    # Check if server is already running
    if is_server_running(port):
        print(f"Server already running on port {port}")
        return True
    
    # Stop any existing server on this port
    stop_server(port)
    
    print(f"Starting vLLM server for {model_path} on port {port}...")
    
    # Build command
    cmd = [
        "vllm", "serve", model_path,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--tensor-parallel-size", str(tensor_parallel),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--trust-remote-code",
        "--dtype", "auto",
        "--exclude-tools-when-tool-choice-none",
    ]
    
    # Add model-specific options
    if model_family.lower() == "qwen":
        model_lower = model_path.lower()
        if "qwen3.5" in model_lower:
            # Qwen3.5: reasoning-parser qwen3 + tool-call-parser qwen3_coder
            cmd.extend([
                "--reasoning-parser", "qwen3",
                "--enable-auto-tool-choice",
                "--tool-call-parser", "qwen3_coder"
            ])
        else:
            # Qwen3 and older: hermes parser
            cmd.extend(["--enable-auto-tool-choice", "--tool-call-parser", "hermes"])
    elif model_family.lower() == "mistral":
        cmd.extend([
            "--tokenizer-mode", "mistral",
            "--config-format", "mistral", 
            "--load-format", "mistral",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "mistral"
        ])
    elif model_family.lower() == "llama":
        cmd.extend(["--enable-auto-tool-choice"])
        model_lower = model_path.lower()
        if "llama-3.2" in model_lower or "llama3.2" in model_lower:
            cmd.extend([
                "--tool-call-parser", "llama3_json",
                "--chat-template", "vllm_template/tool_chat_template_llama3.2_json.jinja"
            ])
        else:
            # Llama-3.1 and other Llama-3.x models
            cmd.extend([
                "--tool-call-parser", "llama3_json",
                "--chat-template", "vllm_template/tool_chat_template_llama3.1_json.jinja"
            ])
    
    # Start server in background
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/vllm_server_{port}.log"
    print(f"Command: {' '.join(cmd)}")
    print(f"Logging to: {log_file}")
    
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
    
    # Wait for server to be ready
    max_wait_time = 9000  # 150 minutes
    wait_interval = 5
    elapsed = 0
    
    print("Waiting for server to start...", end="", flush=True)
    while elapsed < max_wait_time:
        if is_server_running(port):
            print(f"\nServer is ready! (took {elapsed}s)")
            return True
        print(".", end="", flush=True)
        time.sleep(wait_interval)
        elapsed += wait_interval
    
    print(f"\nTimeout waiting for server to start. Check logs at {log_file}")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "check", "restart"])
    parser.add_argument("-model", type=str, help="Model path for start action")
    parser.add_argument("-model_family", type=str, default="qwen", choices=["qwen", "mistral", "llama"])
    parser.add_argument("-port", type=int, default=8000)
    parser.add_argument("-max_model_len", type=int, default=4096)
    parser.add_argument("-tensor_parallel", type=int, default=1)
    parser.add_argument("-gpu_memory_utilization", type=float, default=0.95)
    args = parser.parse_args()
    
    if args.action == "start":
        if not args.model:
            print("Error: -model is required for start action")
            sys.exit(1)
        success = start_server(
            args.model, 
            args.port, 
            args.max_model_len,
            args.tensor_parallel,
            args.gpu_memory_utilization,
            args.model_family
        )
        sys.exit(0 if success else 1)
    
    elif args.action == "stop":
        stop_server(args.port)
    
    elif args.action == "check":
        if is_server_running(args.port):
            print(f"Server is running on port {args.port}")
            sys.exit(0)
        else:
            print(f"Server is not running on port {args.port}")
            sys.exit(1)
    
    elif args.action == "restart":
        if not args.model:
            print("Error: -model is required for restart action")
            sys.exit(1)
        stop_server(args.port)
        time.sleep(2)
        success = start_server(
            args.model, 
            args.port, 
            args.max_model_len,
            args.tensor_parallel,
            args.gpu_memory_utilization,
            args.model_family
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
