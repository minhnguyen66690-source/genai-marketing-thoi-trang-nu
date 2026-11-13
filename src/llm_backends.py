"""Gọi các mô hình ngôn ngữ lớn dùng trong thực nghiệm.

- GPT (OpenAI) qua Codex CLI, đăng nhập bằng tài khoản ChatGPT Business của tác giả: dùng để SINH nội dung.
- Gemini (Google) qua Antigravity CLI (agy): dùng làm GIÁM KHẢO (LLM-as-a-judge), khác hãng với mô hình sinh
  để giảm thiên lệch tự ưu ái (self-enhancement bias, Zheng et al., 2023).
- Nếu có biến môi trường OPENAI_API_KEY / GEMINI_API_KEY thì ưu tiên gọi API chính thức (dùng cho Colab, web demo).

Mỗi lần gọi được ghi log (thời gian, độ trễ, model) vào logs/llm_calls.jsonl để báo cáo độ trễ và tái lập.
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
CODEX = Path(os.environ.get("APPDATA", "")) / "npm" / "codex.cmd"
AGY = Path(os.environ.get("LOCALAPPDATA", "")) / "agy" / "bin" / "agy.exe"

GEN_MODEL = "gpt-6.1-terra"        # mô hình sinh (Codex CLI); effort thấp vì là tác vụ viết, không cần suy luận dài
GEN_EFFORT = "low"
JUDGE_MODEL = "gemini-3.8-flash-medium"  # giám khảo khác hãng


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
    """Sinh văn bản bằng GPT. Ưu tiên OpenAI API nếu có key, nếu không dùng Codex CLI (ChatGPT Business)."""
    t0 = time.time()
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        resp = client.responses.create(model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"), input=prompt)
        text, backend = resp.output_text, "openai-api"
    else:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.txt"
            cmd = [str(CODEX), "exec", "-m", model, "-c", f"model_reasoning_effort={effort}",
                   "-c", 'service_tier="default"', "-s", "read-only", "--skip-git-repo-check",
                   "--ephemeral", "-C", td, "-o", str(out), "-"]
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
            if not out.exists():
                raise RuntimeError(f"Codex lỗi: {p.stderr[-500:] or p.stdout[-500:]}")
            text, backend = out.read_text(encoding="utf-8"), f"codex-cli:{model}"
    _log({"ts": datetime.now().isoformat(), "role": "generate", "backend": backend, "tag": tag,
          "latency_s": round(time.time() - t0, 2), "prompt_chars": len(prompt), "output_chars": len(text)})
    return text


def judge(prompt: str, model: str = JUDGE_MODEL, tag: str = "", timeout: int = 300) -> str:
    """Chấm điểm bằng Gemini. Ưu tiên Gemini API nếu có key, nếu không dùng agy CLI."""
    t0 = time.time()
    if os.environ.get("GEMINI_API_KEY"):
        from google import genai
        client = genai.Client()
        text = client.models.generate_content(model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"), contents=prompt).text
        backend = "gemini-api"
    else:
        p = subprocess.run([str(AGY), "--model", model, "--print-timeout", f"{timeout}s"], input=prompt,
                           capture_output=True, text=True, encoding="utf-8", timeout=timeout + 30)
        text, backend = p.stdout, f"agy:{model}"
        if not text.strip():
            raise RuntimeError(f"agy lỗi: {p.stderr[-500:]}")
    _log({"ts": datetime.now().isoformat(), "role": "judge", "backend": backend, "tag": tag,
          "latency_s": round(time.time() - t0, 2), "prompt_chars": len(prompt), "output_chars": len(text)})
    return text
