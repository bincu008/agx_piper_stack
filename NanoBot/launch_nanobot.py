"""Local launcher: initialize Nanobot's token counter before its first model turn."""
import sys,time
from nanobot.utils.token_encoding import warmup_token_encoding,get_token_encoding
warmup_token_encoding()
deadline=time.monotonic()+10
while get_token_encoding() is None and time.monotonic()<deadline:
    time.sleep(0.05)
if get_token_encoding() is None:
    raise SystemExit('Nanobot token counter did not initialize. Check its tokenizer cache/network before retrying.')
from mcp_profiles import install_task_tool_filter
from repeat_tool_guard import install
from task_checkpoint import install as install_task_checkpoints
from camera_visuals import install as install_camera_visuals
from workspace_arguments import install as install_workspace_arguments
from nanobot.cli.entry import main

install_task_tool_filter()
install_task_checkpoints()
install_camera_visuals()
install_workspace_arguments()
install()
sys.exit(main())
