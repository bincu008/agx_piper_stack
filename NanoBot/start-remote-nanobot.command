#!/bin/zsh
set -eu
export NANOBOT_STREAM_IDLE_TIMEOUT_S=1800
export NANOBOT_OPENAI_COMPAT_TIMEOUT_S=1800
cd /Users/huynhnguyentrunghieu/Documents/NanoBot

profile_name=""
if [[ "${1:-}" == "--tools-profile" ]]; then
  profile_name="${2:?Supply workspace-tasks, robot-tasks, perception, view, memory, grasp, placement, diagnostics, queue or all-calculations}"
  shift 2
fi
# Reuse a responding tunnel, or authenticate interactively to create one.
if ! curl -fs --max-time 3 http://127.0.0.1:18080/v1/models >/dev/null; then
  ssh -f -N -T \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ConnectTimeout=10 \
    -L 127.0.0.1:18080:127.0.0.1:18080 \
    bincu008@192.168.50.128
fi

.venv/bin/python - <<'PY'
import datetime, json, os, pathlib, shutil, tempfile, urllib.request
from nanobot.config.schema import Config
endpoint='http://127.0.0.1:18080/v1'
request=urllib.request.Request(endpoint+'/models', headers={'Authorization':'Bearer local'})
with urllib.request.urlopen(request, timeout=15) as response:
    model_ids={entry['id'] for entry in json.load(response)['data']}
model='Qwen/Qwen3.5-9B'
if model not in model_ids:
    raise SystemExit(f'Expected {model!r}; server advertised {sorted(model_ids)!r}. Configuration unchanged.')
path=pathlib.Path.home()/'.nanobot/config.json'
config=json.loads(path.read_text())
config.setdefault('providers',{}).setdefault('custom',{}).update(apiBase=endpoint,apiKey='local')
config['agents']['defaults'].update(provider='custom',model=model,contextWindowTokens=49152)
config['agents']['defaults'].update(maxTokens=1024,maxConcurrentSubagents=1,idleCompactAfterMinutes=0,reasoningEffort='none')
config['providers']['custom']['extraBody']=config['providers']['custom'].get('extraBody') or {}
config['providers']['custom']['extraBody'].setdefault('chat_template_kwargs',{})['enable_thinking']=False
Config.model_validate(config)
backup=path.with_name('config.json.before-remote-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
shutil.copy2(path,backup);backup.chmod(0o600)
fd,temp=tempfile.mkstemp(dir=path.parent,prefix='.remote-config-')
try:
    with os.fdopen(fd,'w') as stream:
        json.dump(config,stream,indent=2);stream.write('\n')
    os.replace(temp,path)
finally:
    if os.path.exists(temp): os.unlink(temp)
print(f'Model discovered: {model}\nConfiguration backup: {backup}')
print('Discovery passed. Generation, vision and tool execution still require functional validation.')
PY

if [[ -n "$profile_name" ]]; then
  profile_config=$(.venv/bin/python mcp_profiles.py "$profile_name")
  print "MCP tool profile: $profile_name (calculation/perception only; physical tools excluded)"
  exec ./start-nanobot.command --new --config "$profile_config" "$@"
fi
exec ./start-nanobot.command --new "$@"
