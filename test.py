#!/usr/bin/env python3
"""
Standalone script to verify that the API token sent in a request
is correctly reflected in the audit_llm_requests table.

Sends one non-streaming and one streaming request, each with a unique
identifiable token, then prints the SQL to verify both appear correctly.

Usage:
    API_URL=https://<your-dev-url> GATEWAY_EMAIL=your@acvauctions.com uv run verify_audit_key.py
"""

import json
import os
import uuid

import requests

BASE_URL = os.environ.get("API_URL", "https://llm-gateway.internal.internal-development.acvauctions.com").rstrip("/")
EMAIL = os.environ.get("GATEWAY_EMAIL", "rsomasundaram@acvauctions.com")

PAYLOAD = {
    "model": "EXTERNAL:google/gemini-3-flash-preview",
    "messages": [{"role": "user", "content": "What is 1+1?"}],
    "max_tokens": 10,
}


def _headers(token: str) -> dict:
    return {
        "x-acv-llm-gateway-token": token,
        "Content-Type": "application/json",
    }


def send_sync(token: str) -> str | None:
    print(f"\n[NON-STREAMING] token: {token}")
    try:
        resp = requests.post(
            f"{BASE_URL}/openai/v1/chat/completions",
            headers=_headers(token),
            json={**PAYLOAD, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        response_id = resp.json().get("id", "N/A")
        print(f"  HTTP {resp.status_code}  response_id={response_id}")
        return response_id
    except requests.HTTPError as e:
        print(f"  ERROR {e.response.status_code}: {e.response.text[:200]}")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def send_streaming(token: str) -> str | None:
    print(f"\n[STREAMING]     token: {token}")
    try:
        resp = requests.post(
            f"{BASE_URL}/openai/v1/chat/completions",
            headers=_headers(token),
            json={**PAYLOAD, "stream": True},
            timeout=60,
            stream=True,
        )
        resp.raise_for_status()
        response_id = None
        for raw in resp.iter_lines():
            if not raw or raw == b"data: [DONE]":
                continue
            if raw.startswith(b"data: "):
                chunk = json.loads(raw[6:])
                if not response_id:
                    response_id = chunk.get("id")
        print(f"  HTTP {resp.status_code}  response_id={response_id or 'N/A'}")
        return response_id
    except requests.HTTPError as e:
        print(f"  ERROR {e.response.status_code}: {e.response.text[:200]}")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


if __name__ == "__main__":
    run_id = uuid.uuid4().hex[:8]
    sync_token = f"verify-sync-{run_id}|{EMAIL}"
    stream_token = f"verify-stream-{run_id}|{EMAIL}"

    print(f"Target : {BASE_URL}")
    print(f"Run ID : {run_id}")

    sync_resp_id = send_sync(sync_token)
    stream_resp_id = send_streaming(stream_token)

    print("\n" + "=" * 60)
    print("Verification SQL — run against the internal-dev audit DB:")
    print("=" * 60)
    print(f"""
SELECT
    api_key_id,
    response_id,
    model_type,
    is_success,
    request_timestamp
FROM audit_llm_requests
WHERE api_key_id IN (
    '{sync_token}',
    '{stream_token}'
)
ORDER BY request_timestamp DESC;
""")
    print("Expected: both rows show the token exactly as above, NOT a UUID.")
    print()
    print(f"  sync   token : {sync_token}")
    print(f"  stream token : {stream_token}")
    if sync_resp_id:
        print(f"\nAlternatively search by response_id:")
        print(f"  sync   response_id : {sync_resp_id}")
    if stream_resp_id:
        print(f"  stream response_id : {stream_resp_id}")
