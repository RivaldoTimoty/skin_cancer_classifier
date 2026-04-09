import requests
import json


OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"


def is_ollama_running() -> bool:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def get_available_models() -> list[str]:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    stream: bool = False,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": 2048,
        },
    }
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama tidak berjalan. Jalankan Ollama terlebih dahulu.")
    except requests.exceptions.Timeout:
        raise RuntimeError("Request ke Ollama timeout. Model mungkin sedang loading.")
    except Exception as e:
        raise RuntimeError(f"Error saat memanggil Ollama: {e}")


def generate_stream(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": 2048,
        },
    }
    with requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        stream=True,
        timeout=300,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break
