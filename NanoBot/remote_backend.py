"""Launch Nanobot with a private, per-run config and a scoped SSH tunnel."""

import json
import shlex
import signal
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from nanobot.config.schema import Config

from mcp_profiles import apply_profile

HOST = "bincu008@192.168.50.128"
MODEL = "Qwen/Qwen3.5-9B"
PORTS = {"llama.cpp": 18080, "OpenVINO": 18081}


def matching_listener(port, forward):
    result = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpcn"],
        capture_output=True, text=True,
    )
    if result.stderr or result.returncode not in (0, 1):
        raise SystemExit(f"Cannot inspect listeners on port {port}: {result.stderr.strip()}")
    if not result.stdout:
        return False

    fields = result.stdout.splitlines()
    pids = [line[1:] for line in fields if line.startswith("p")]
    sockets = [line[1:] for line in fields if line.startswith("n")]
    if len(pids) == 1 and sockets == [f"127.0.0.1:{port}"] and "cssh" in fields:
        process = subprocess.run(
            ["ps", "-ww", "-p", pids[0], "-o", "command="],
            capture_output=True, text=True,
        )
        args = shlex.split(process.stdout)
        if args and Path(args[0]).name == "ssh" and args[-1] == HOST:
            options = iter(args[1:-1])
            # Recognize direct SSH forwards, including the existing launcher.
            forwards = []
            for option in options:
                if option == "-L":
                    forwards.append(next(options, ""))
                elif option == "-o":
                    name, _, _ = next(options, "").partition("=")
                    if name.lower() not in {
                        "exitonforwardfailure", "serveraliveinterval",
                        "serveralivecountmax", "connecttimeout", "batchmode",
                    }:
                        break
                elif option not in {"-f", "-N", "-T", "-NT", "-TN"}:
                    break
            else:
                if forwards == [forward]:
                    return True
    raise SystemExit(
        f"Local port {port} is occupied (PID {', '.join(pids) or 'unknown'}). "
        f"Could not verify an SSH forward to {HOST} with -L {forward}. "
        "Inspect the listener and release the port manually if appropriate."
    )


def config_for(endpoint, args, context_tokens=49152):
    config_path = Path.home() / ".nanobot/config.json"
    profile = None
    if args[:1] == ["--tools-profile"]:
        if len(args) < 2:
            raise SystemExit("--tools-profile requires a profile name.")
        profile = args[1]
        args = args[2:]
    forwarded = []
    options = iter(args)
    for arg in options:
        if arg == "--":
            forwarded.extend([arg, *options])
            break
        if arg in {"--config", "-c"}:
            value = next(options, None)
            if not value:
                raise SystemExit(f"{arg} requires a config path.")
            config_path = Path(value).expanduser()
        elif arg.startswith(("--config=", "-c=")):
            config_path = Path(arg.split("=", 1)[1]).expanduser()
        else:
            forwarded.append(arg)

    config = json.loads(config_path.read_text())
    if profile is not None:
        apply_profile(profile, config)
    defaults = config.setdefault("agents", {}).setdefault("defaults", {})
    defaults.update(provider="custom", model=MODEL, modelPreset=None,
                    contextWindowTokens=context_tokens, maxTokens=1024, reasoningEffort="none")
    provider = config.setdefault("providers", {}).setdefault("custom", {})
    provider["apiBase"] = endpoint
    provider.setdefault("apiKey", "local")
    extra_body = provider.get("extraBody") or {}
    template = extra_body.get("chat_template_kwargs") or {}
    template["enable_thinking"] = False
    extra_body["chat_template_kwargs"] = template
    provider["extraBody"] = extra_body
    Config.model_validate(config)
    return config, forwarded


def check_models(backend, endpoint, provider):
    headers = {}
    if provider.get("apiKey"):
        headers["Authorization"] = f"Bearer {provider['apiKey']}"
    headers.update(provider.get("extraHeaders") or {})
    request = urllib.request.Request(endpoint + "/models", headers=headers)
    try:
        direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with direct.open(request, timeout=15) as response:
            models = {entry["id"] for entry in json.load(response)["data"]}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(
            f"{backend} unavailable at {endpoint}/models ({error}). "
            "Check the selected backend in its separate server terminal."
        ) from None
    if MODEL not in models:
        raise SystemExit(f"{backend} advertised {sorted(models)!r}; expected {MODEL!r}.")
    print(f"{backend}: {MODEL} verified at {endpoint}/models", flush=True)


def main(backend, args):
    port = PORTS[backend]
    endpoint = f"http://127.0.0.1:{port}/v1"
    forward = f"127.0.0.1:{port}:127.0.0.1:{port}"
    check_only = args[:1] == ["--check"]
    if check_only:
        args = args[1:]
    config, forwarded = config_for(endpoint, args, context_tokens=57344 if backend == "llama.cpp" else 49152)
    reuse = matching_listener(port, forward)

    with tempfile.TemporaryDirectory(prefix="nb-", dir="/tmp") as directory:
        socket = str(Path(directory) / "ssh")
        try:
            if reuse:
                print(f"Reusing verified SSH forward on port {port}.", flush=True)
            else:
                print(f"Opening SSH forward on port {port} to {HOST}.", flush=True)
                result = subprocess.run([
                    "ssh", "-f", "-N", "-T", "-M", "-S", socket,
                    "-o", "ControlPersist=no", "-o", "ExitOnForwardFailure=yes",
                    "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
                    "-o", "ConnectTimeout=10", "-L", forward, HOST,
                ])
                if result.returncode:
                    raise SystemExit(f"{backend}: SSH could not establish the forward on port {port}.")
            check_models(backend, endpoint, config["providers"]["custom"])
            if check_only:
                return 0
            path = Path(directory) / "config.json"
            with path.open("x") as stream:
                json.dump(config, stream)
            path.chmod(0o600)
            return subprocess.run([
                "./start-nanobot.command", "--new", "--config", str(path), *forwarded,
            ]).returncode
        finally:
            if Path(socket).exists():
                result = subprocess.run(
                    ["ssh", "-S", socket, "-O", "exit", HOST], capture_output=True, text=True,
                )
                if result.returncode:
                    print(f"Could not close the owned SSH tunnel: {result.stderr.strip()}", file=sys.stderr)


def terminate(signum, _frame):
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGHUP, terminate)
    try:
        sys.exit(main(sys.argv[1], sys.argv[2:]))
    except KeyboardInterrupt:
        sys.exit(130)
