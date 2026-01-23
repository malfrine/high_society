#!/usr/bin/env python3
"""
Run training on RunPod GPU cloud.

Usage:
    python run_cloud.py              # Start pod, sync code, train, stop pod
    python run_cloud.py --stop       # Just stop the pod
    python run_cloud.py --status     # Check pod status

Requires:
    pip install runpod

    Set RUNPOD_API_KEY environment variable or create .env file with:
    RUNPOD_API_KEY=your_api_key_here
"""

import os
import subprocess
import sys
import time
import argparse

try:
    import runpod
except ImportError:
    print("Please install runpod: pip install runpod")
    sys.exit(1)


# Configuration - Multiple pods with different GPU types for fallback
PODS = [
    {"id": "59o0bnhxq37675", "name": "RTX 4090"},
    {"id": "s8hcnzqxpbv1gk", "name": "RTX 4090"},
]
SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519")


def get_api_key() -> str:
    """Get RunPod API key from environment or .env file."""
    api_key = os.environ.get("RUNPOD_API_KEY")
    if api_key:
        return api_key

    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.startswith("RUNPOD_API_KEY="):
                    return line.strip().split("=", 1)[1]

    print("Error: RUNPOD_API_KEY not found")
    print("Set it via: export RUNPOD_API_KEY=your_key")
    print("Or create a .env file with: RUNPOD_API_KEY=your_key")
    sys.exit(1)


def get_pod_ssh_command(pod: dict) -> str | None:
    """Extract SSH command from pod info."""
    runtime = pod.get("runtime")
    if not runtime:
        return None

    ports = runtime.get("ports", [])
    for port in ports:
        if port.get("privatePort") == 22:
            ip = port.get("ip")
            public_port = port.get("publicPort")
            if ip and public_port:
                return f"ssh root@{ip} -p {public_port} -i {SSH_KEY_PATH}"
    return None


def wait_for_pod_ready(pod_id: str, timeout: int = 300) -> dict:
    """Wait for pod to be ready and return pod info."""
    print("Waiting for pod to be ready", end="", flush=True)
    start = time.time()

    while time.time() - start < timeout:
        pod = runpod.get_pod(pod_id)
        status = pod.get("desiredStatus")
        runtime = pod.get("runtime")

        if status == "RUNNING" and runtime:
            print(" Ready!")
            return pod

        print(".", end="", flush=True)
        time.sleep(5)

    print(" Timeout!")
    raise TimeoutError(f"Pod did not become ready within {timeout}s")


def start_pod() -> dict:
    """Try to resume pods in order until one starts successfully."""
    for pod_info in PODS:
        pod_id = pod_info["id"]
        name = pod_info["name"]

        pod = runpod.get_pod(pod_id)
        if pod.get("desiredStatus") == "RUNNING":
            print(f"Pod {name} ({pod_id}) is already running")
            return pod_info

        print(f"Trying to start {name} ({pod_id})...")
        try:
            runpod.resume_pod(pod_id, gpu_count=1)
            print(f"Started {name}!")
            return pod_info
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    raise RuntimeError("Could not start any pod - no GPUs available")


def stop_pod(pod_id: str, name: str = ""):
    """Stop the running pod (keeps data, stops billing for GPU)."""
    print(f"Stopping pod {name} ({pod_id})...")
    runpod.stop_pod(pod_id)
    print("Pod stopped. You're no longer being charged for GPU.")


def sync_code(ssh_cmd: str):
    """Sync local code to the pod using rsync."""
    parts = ssh_cmd.split()
    host = parts[1]  # root@ip
    port = parts[3]  # port number
    key = parts[5]   # key path

    project_dir = os.path.dirname(os.path.abspath(__file__))

    print("Syncing code to pod...")
    rsync_cmd = [
        "rsync", "-avz", "--progress",
        "-e", f"ssh -p {port} -i {key} -o StrictHostKeyChecking=no",
        "--exclude", ".venv",
        "--exclude", "__pycache__",
        "--exclude", ".git",
        "--exclude", "*.pyc",
        f"{project_dir}/",
        f"{host}:~/high_society_rl/"
    ]

    subprocess.run(rsync_cmd, check=True)
    print("Code synced!")


def run_training(ssh_cmd: str):
    """Run training on the pod."""
    parts = ssh_cmd.split()
    host = parts[1]
    port = parts[3]
    key = parts[5]

    remote_commands = """
set -e
cd ~/high_society_rl

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:$PATH"

# Sync dependencies
echo "Installing dependencies..."
uv sync

# Show GPU info
echo ""
echo "=== GPU Info ==="
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU')"

# Run training
echo ""
echo "=== Starting Training ==="
uv run python -m high_society.main

echo ""
echo "=== Training Complete ==="
"""

    print("\n" + "="*50)
    print("Running training on GPU...")
    print("="*50 + "\n")

    ssh_full_cmd = [
        "ssh", f"-p{port}", "-i", key,
        "-o", "StrictHostKeyChecking=no",
        host,
        remote_commands
    ]

    subprocess.run(ssh_full_cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run training on RunPod GPU")
    parser.add_argument("--stop", action="store_true", help="Stop the pod")
    parser.add_argument("--status", action="store_true", help="Check pod status")
    args = parser.parse_args()

    runpod.api_key = get_api_key()

    if args.status:
        for pod_info in PODS:
            pod = runpod.get_pod(pod_info["id"])
            print(f"{pod_info['name']} ({pod_info['id']}): {pod.get('desiredStatus')}")
        return

    if args.stop:
        for pod_info in PODS:
            pod = runpod.get_pod(pod_info["id"])
            if pod.get("desiredStatus") == "RUNNING":
                stop_pod(pod_info["id"], pod_info["name"])
        return

    # Try to start a pod (will try each one until one works)
    active_pod = start_pod()
    pod_id = active_pod["id"]
    pod_name = active_pod["name"]

    # Wait for pod to be ready
    pod = wait_for_pod_ready(pod_id)

    ssh_cmd = get_pod_ssh_command(pod)
    if not ssh_cmd:
        print("Error: Could not get SSH connection info")
        sys.exit(1)

    print(f"\nSSH command: {ssh_cmd}\n")

    try:
        sync_code(ssh_cmd)
        run_training(ssh_cmd)
    finally:
        print("\n" + "="*50)
        stop_pod(pod_id, pod_name)
        print("="*50)


if __name__ == "__main__":
    main()
