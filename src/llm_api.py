"""Gọi API chính thức mà không lưu khóa hoặc nội dung."""
import os
import re
import time


SKIP_OPENAI = ("audio", "realtime", "image", "tts", "transcribe", "search", "embedding", "moderation", "instruct", "codex")


def pick_openai_model(client):
    """Chọn mô hình GPT sinh văn bản mới nhất mà khóa này truy cập được, ưu tiên bản mini cho rẻ.

    Người dùng cuối không biết tên mô hình, nên khi không chỉ định thì tự chọn thay vì báo lỗi.
    """
    ids = [getattr(item, "id", "") for item in client.models.list()]
    candidates = [name for name in ids
                  if name.startswith("gpt-")
                  and not any(word in name for word in SKIP_OPENAI)
                  and not re.search(r"\d{4}-\d{2}-\d{2}", name)]
    if not candidates:
        raise ValueError("Khóa này chưa truy cập được mô hình GPT nào để sinh văn bản.")
    # Bản "mini" cân bằng giữa chất lượng và chi phí; "nano" rẻ hơn nhưng viết kém hơn rõ rệt.
    mini = [name for name in candidates if "mini" in name]
    nano = [name for name in candidates if "nano" in name]
    return sorted(mini or nano or candidates)[-1]


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
            client = OpenAI(api_key=api_key)
            model = model or os.environ.get("OPENAI_MODEL") or pick_openai_model(client)
            response = client.responses.create(model=model, input=prompt)
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
