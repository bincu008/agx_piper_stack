"""Authenticated discovery only; no perception, calculation, motion or gripper calls."""
import argparse
import asyncio
import json
from pathlib import Path
import ssl
import sys
import uuid

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workspace_discovery import validate_snapshot  # noqa: E402


async def discover(args):
    if not args.url.startswith('https://'):
        raise ValueError('TLS_REQUIRED')
    token = (ROOT / '.private/ros-mcp-token').read_text().strip()
    tls = ssl.create_default_context(cafile=args.ca_file)
    async with httpx.AsyncClient(verify=tls, timeout=30,
                                 headers={'Authorization': 'Bearer ' + token}) as http:
        async with streamable_http_client(args.url, http_client=http) as (r, w, _):
            async with ClientSession(r, w) as session:
                init = await session.initialize()
                tools, cursor = [], None
                while True:
                    page = await session.list_tools(cursor=cursor)
                    tools.extend(t.model_dump(mode='json') for t in page.tools)
                    cursor = page.nextCursor
                    if not cursor:
                        break
                result = await session.call_tool('list_functions', {'summary_only': True})
                identity = {'discovery_id': uuid.uuid4().hex, 'endpoint': args.url}
                native = dict(identity, tools=tools,
                              server_info=init.serverInfo.model_dump(mode='json'))
                status = dict(identity, result=result.model_dump(mode='json'))
                _, data = validate_snapshot(native, status, args.control)
                if data['input_provenance']['classification'] != args.expect_source:
                    raise ValueError('UNEXPECTED_SOURCE_MODE: snapshots not saved')
                out = ROOT / 'results'
                out.mkdir(exist_ok=True)
                prefix = 'mcp-control' if args.control else 'mcp-workspace'
                for name, payload in ((prefix + '-tools.json', native),
                                      (prefix + '-status-list_functions.json', status)):
                    path = out / name
                    temporary = path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(payload, indent=2))
                    temporary.replace(path)
                print('Workspace discovery verified:', init.serverInfo.version,
                      len(tools), 'native tools;', args.expect_source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', action='store_true',
                        help='Discover guarded control mode; requires live sources, not replay')
    parser.add_argument('--url', default='https://10.211.55.3:8765/mcp')
    parser.add_argument('--ca-file', help='Trusted CA file; certificate verification stays enabled')
    parser.add_argument('--expect-source', required=True,
                        choices=['mock_replay_publishers', 'live_driver_publishers'])
    asyncio.run(asyncio.wait_for(discover(parser.parse_args()), timeout=60))
