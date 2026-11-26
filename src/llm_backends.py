"""Điều phối dịch vụ tạo nội dung và dịch vụ chấm độc lập.

Khi có cấu hình dịch vụ, chương trình dùng API. Nếu không, các lệnh chạy cục bộ tương ứng được dùng cho từng vai trò.
Mỗi lần gọi được ghi thời gian và độ trễ vào ``logs/llm_calls.jsonl`` để phục vụ việc đối chiếu kết quả.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "llm_calls.jsonl"
GENERATION_RUNNER = Path(os.environ.get("APPDATA", "")) / "npm" / "codex.cmd"
JUDGING_RUNNER = Path(os.environ.get("LOCALAPPDATA", "")) / "agy" / "bin" / "agy.exe"

GEN_MODEL = "gpt-6.1-terra"
GEN_EFFORT = "low"
JUDGE_MODEL = "gemini-3.8-flash-medium"


def _log(rec: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def extract_json(text: str):
    """Lấy khối JSON đầu tiên trong câu trả lời (chấp nhận ```json ... ```)."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    body = m.group(1) if m else text
    start = min([i for i in (body.find("{"), body.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("Không tìm thấy JSON trong câu trả lời")
    return json.JSONDecoder().raw_decode(body[start:])[0]


def generate(prompt: str, model: str = GEN_MODEL, effort: str = GEN_EFFORT, tag: str = "", timeout: int = 600) -> str:
    """Tạo văn bản từ yêu cầu đã chuẩn bị."""
    t0 = time.time()
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        resp = client.responses.create(model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"), input=prompt)
        text, backend = resp.output_text, "openai-api"
    else:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.txt"
            cmd = [str(GENERATION_RUNNER), "exec", "-m", model, "-c", f"model_reasoning_effort={effort}",
                   "-c", 'service_tier="default"', "-s", "read-only", "--skip-git-repo-check",
                   "--ephemeral", "-C", td, "-o", str(out), "-"]
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
            if not out.exists():
                raise RuntimeError(f"Dịch vụ tạo nội dung không trả kết quả: {p.stderr[-500:] or p.stdout[-500:]}")
            text, backend = out.read_text(encoding="utf-8"), f"generation-runner:{model}"
    _log({"ts": datetime.now().isoformat(), "role": "generate", "backend": backend, "tag": tag,
          "latency_s": round(time.time() - t0, 2), "prompt_chars": len(prompt), "output_chars": len(text)})
    return text


def judge(prompt: str, model: str = JUDGE_MODEL, tag: str = "", timeout: int = 300, *, cli_print: bool = False) -> str:
    """Chấm nội dung theo tiêu chí đã xác định."""
    t0 = time.time()
    if os.environ.get("GEMINI_API_KEY"):
        from google import genai
        client = genai.Client()
        text = client.models.generate_content(model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"), contents=prompt).text
        backend = "gemini-api"
    elif cli_print:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as workspace:
            effort = model.rsplit("-", 1)[-1]
            p = subprocess.run([str(JUDGING_RUNNER), "--model", model, "--effort", effort, "--sandbox",
                                "--disable-slash-commands", "--mode", "plan", "--output-format", "text",
                                "--print-timeout", f"{timeout}s", "--print", prompt],
                               cwd=workspace, capture_output=True, text=True, encoding="utf-8", timeout=timeout + 30)
        text, backend = p.stdout, f"judging-runner:{model}"
        if p.returncode or not text.strip():
            raise RuntimeError(f"Dịch vụ chấm không trả kết quả: {p.stderr[-500:]}")
    else:
        p = subprocess.run([str(JUDGING_RUNNER), "--model", model, "--print-timeout", f"{timeout}s"], input=prompt,
                           capture_output=True, text=True, encoding="utf-8", timeout=timeout + 30)
        text, backend = p.stdout, f"judging-runner:{model}"
        if not text.strip():
            raise RuntimeError(f"Dịch vụ chấm không trả kết quả: {p.stderr[-500:]}")
    _log({"ts": datetime.now().isoformat(), "role": "judge", "backend": backend, "tag": tag,
          "latency_s": round(time.time() - t0, 2), "prompt_chars": len(prompt), "output_chars": len(text)})
    return text
