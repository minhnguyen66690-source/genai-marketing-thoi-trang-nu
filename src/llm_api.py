"""Gọi API chính thức mà không lưu khóa hoặc nội dung."""
import os
import re
import time


def generate_via_api(prompt, provider, api_key, model=None):
    """Trả văn bản, tên mô hình và thời gian gọi theo giây."""
    if not api_key:
        raise ValueError("Vui lòng nhập khóa API.")
    started = time.monotonic()
    try:
        if provider == "gemini":
            from google import genai
            client = genai.Client(api_key=api_key)
            if not model:
                candidates = []
                for item in client.models.list():
                    name = item.name or ""
                    actions = item.supported_actions or []
                    if "flash" in name.lower() and "generateContent" in actions:
                        candidates.append(name)
                stable = [name for name in candidates if not re.search(r"(?:-exp|preview)", name, re.I)]
                choices = stable or candidates
                if not choices:
                    raise ValueError("Không tìm thấy mô hình Gemini Flash hỗ trợ sinh nội dung.")
                model = sorted(choices)[-1]
            response = client.models.generate_content(model=model, contents=prompt)
            text = response.text
        elif provider == "openai":
            from openai import OpenAI
            model = model or os.environ.get("OPENAI_MODEL")
            if not model:
                raise ValueError("Vui lòng nhập tên mô hình OpenAI hoặc đặt OPENAI_MODEL.")
            response = OpenAI(api_key=api_key).responses.create(model=model, input=prompt)
            text = response.output_text
        else:
            raise ValueError("Nhà cung cấp API không hợp lệ.")
        if not text:
            raise ValueError("API không trả về nội dung văn bản.")
        return text, model, time.monotonic() - started
    except ValueError:
        raise
    except Exception as exc:
        # Không hiển thị lỗi gốc: thông báo của SDK có thể chứa dữ liệu nhạy cảm.
        name = type(exc).__name__.lower()
        if "rate" in name or "quota" in name or getattr(exc, "status_code", None) == 429:
            raise RuntimeError("API đã đạt giới hạn hoặc hết hạn mức; vui lòng thử lại sau.") from None
        if "connection" in name or "timeout" in name:
            raise RuntimeError("Không kết nối được API hoặc kết nối quá thời gian; vui lòng thử lại.") from None
        raise RuntimeError("Không gọi được API. Hãy kiểm tra khóa, tên mô hình và quyền truy cập.") from None
