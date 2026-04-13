"""
Meme / Alt-text Generator — complete reference implementation.

Pipeline:
  1. Load image (local file -> base64 data URI, or URL passthrough).
  2. Vision call -> objective description.
  3. Text call -> accessibility alt-text (<=125 chars).
  4. Text call -> humorous meme caption.
  5. Image generation call -> stylized variant image.
  6. Save outputs (alt_text.txt, caption.txt, variant.png) + print summary.

Run:
    python solution.py samples/sample.png --out ./out
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from litellm import completion, completion_cost, image_generation

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")
TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-4o-mini")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "dall-e-3")

ALT_TEXT_MAX_CHARS = 125


def load_image(path_or_url: str) -> str:
    """Return an image reference the vision API can accept."""
    if path_or_url.startswith(("http://", "https://")):
        return path_or_url

    path = Path(path_or_url)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path_or_url}")

    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        mime = "image/png"

    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _usage_from_response(response) -> dict:
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

    try:
        cost = completion_cost(completion_response=response) or 0.0
    except Exception:
        cost = 0.0

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
    }


def describe_image(image_ref: str, model: str = VISION_MODEL) -> tuple[str, dict]:
    """Objective description of the image via the vision model."""
    response = completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in 2-3 sentences. "
                            "Be factual and objective. Mention subject, setting, "
                            "mood, and any visible text. If something is unclear, "
                            "say so instead of guessing."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_ref}},
                ],
            }
        ],
    )
    text = response.choices[0].message.content.strip()
    return text, _usage_from_response(response)


def generate_alt_text(description: str, model: str = TEXT_MODEL) -> tuple[str, dict]:
    """Accessibility alt-text: <=125 chars, factual, screen-reader friendly."""
    response = completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write accessibility alt-text for screen readers. "
                    f"Rules: under {ALT_TEXT_MAX_CHARS} characters, factual, "
                    "no subjective adjectives, no 'image of' prefix. "
                    "Output only the alt-text, no quotes, no explanation."
                ),
            },
            {
                "role": "user",
                "content": f"Image description:\n{description}\n\nWrite the alt-text.",
            },
        ],
    )
    text = response.choices[0].message.content.strip().strip('"').strip("'")
    if len(text) > ALT_TEXT_MAX_CHARS:
        text = text[: ALT_TEXT_MAX_CHARS - 1].rstrip() + "…"
    return text, _usage_from_response(response)


def generate_meme_caption(description: str, model: str = TEXT_MODEL) -> tuple[str, dict]:
    """Short punchy humorous caption derived from the description."""
    response = completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write meme captions. Rules: one line, under 80 "
                    "characters, punchy and funny, no hashtags, no emoji. "
                    "Output only the caption."
                ),
            },
            {
                "role": "user",
                "content": f"Image description:\n{description}\n\nWrite the caption.",
            },
        ],
    )
    text = response.choices[0].message.content.strip().strip('"').strip("'")
    return text, _usage_from_response(response)


def generate_variant_image(
    description: str,
    caption: str,
    out_path: Path,
    model: str = IMAGE_MODEL,
) -> Path:
    """Create a stylized variant image from description + caption."""
    prompt = (
        f"A stylized illustration inspired by this scene: {description} "
        f"Mood and tone: {caption}. "
        "Style: bold colors, clean lines, modern editorial illustration, "
        "balanced composition, no text in the image."
    )
    response = image_generation(
        prompt=prompt,
        model=model,
        size="1024x1024",
        n=1,
    )

    data = response["data"][0]
    if "url" in data and data["url"]:
        resp = requests.get(data["url"], timeout=60)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
    elif "b64_json" in data and data["b64_json"]:
        out_path.write_bytes(base64.b64decode(data["b64_json"]))
    else:
        raise RuntimeError("Image generation response contained no image data")

    return out_path


def run_pipeline(image_ref: str, out_dir: Path) -> dict:
    """Full pipeline. Writes outputs and returns a result dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded_ref = load_image(image_ref)

    description, u1 = describe_image(loaded_ref)
    print(f"[1/4] Description: {description[:120]}{'...' if len(description) > 120 else ''}")

    alt_text, u2 = generate_alt_text(description)
    print(f"[2/4] Alt-text:    {alt_text}")

    caption, u3 = generate_meme_caption(description)
    print(f"[3/4] Caption:     {caption}")

    variant_path = out_dir / "variant.png"
    try:
        generate_variant_image(description, caption, variant_path)
        print(f"[4/4] Variant:     {variant_path}")
    except Exception as e:
        variant_path = None
        print(f"[4/4] Variant:     FAILED ({e})")

    (out_dir / "alt_text.txt").write_text(alt_text, encoding="utf-8")
    (out_dir / "caption.txt").write_text(caption, encoding="utf-8")
    (out_dir / "description.txt").write_text(description, encoding="utf-8")

    total_input = u1["input_tokens"] + u2["input_tokens"] + u3["input_tokens"]
    total_output = u1["output_tokens"] + u2["output_tokens"] + u3["output_tokens"]
    total_cost = u1["cost"] + u2["cost"] + u3["cost"]

    return {
        "description": description,
        "alt_text": alt_text,
        "caption": caption,
        "variant_path": str(variant_path) if variant_path else None,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost": round(total_cost, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Meme / Alt-text Generator")
    parser.add_argument("image", help="Path or URL to the input image")
    parser.add_argument("--out", default="./out", help="Output directory")
    args = parser.parse_args()

    result = run_pipeline(args.image, Path(args.out))

    print("\n=== Summary ===")
    print(f"Alt-text: {result['alt_text']}")
    print(f"Caption:  {result['caption']}")
    print(f"Variant:  {result['variant_path']}")
    print(
        f"Tokens:   in={result['total_input_tokens']} "
        f"out={result['total_output_tokens']}"
    )
    print(f"Cost:     ${result['total_cost']}")


if __name__ == "__main__":
    main()
