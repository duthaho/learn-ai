# Project: Meme / Alt-text Generator

Build a CLI pipeline that takes any image — local file or URL — and produces an accessibility alt-text, a meme caption, and a stylized variant image, all driven by multimodal LLM calls.

## What you'll build

A five-stage pipeline that accepts an image path or URL from the command line and produces three outputs: a screen-reader-friendly alt-text (under 125 characters), a short punchy meme caption, and a new AI-generated image inspired by the original. The project demonstrates the two main vision API patterns — base64 data URIs for local files and direct URL passthrough for remote images — alongside the content-array message format required by vision models. You will also call `litellm.image_generation()` to create the variant image, download the result, and write all outputs to a local directory, giving you a complete picture of how text and image generation work together in a single pipeline.

## Prerequisites

- Completed reading the Module 10 README
- Python 3.11+ with project dependencies installed (`pip install -r requirements.txt` from the repo root)
- An OpenAI-compatible API key set in `.env` at the repo root (`OPENAI_API_KEY` or whichever provider you use)

## Setup

```bash
cd 10-image-multimodal-ai/project
```

Confirm that `.env` at the repo root contains your API key. The scripts load it automatically using `python-dotenv`. Confirm that `samples/sample.png` is present — if it is missing, regenerate it with:

```bash
python - <<'EOF'
from pathlib import Path
from PIL import Image, ImageDraw
out = Path("samples/sample.png")
out.parent.mkdir(parents=True, exist_ok=True)
img = Image.new("RGB", (512, 512), (245, 222, 179))
draw = ImageDraw.Draw(img)
draw.ellipse((350, 60, 470, 180), fill=(255, 204, 0))
draw.polygon([(0,400),(200,260),(400,380),(512,300),(512,512),(0,512)], fill=(107,142,35))
draw.rectangle((200,360,280,430), fill=(139,69,19))
draw.polygon([(195,360),(240,320),(285,360)], fill=(80,40,20))
draw.rectangle((230,390,250,430), fill=(60,30,10))
img.save(out)
EOF
```

## Step 1 — Load an image

Implement `load_image(path_or_url)`. This function must return a value the vision API can accept:

- If the argument starts with `http://` or `https://`, return it unchanged — the API fetches it directly.
- Otherwise, treat it as a local file path, read the bytes, and return a base64 data URI:

```python
b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
return f"data:{mime};base64,{b64}"
```

Use `mimetypes.guess_type(path.name)` to detect the MIME type; fall back to `"image/png"` if it returns `None`.

## Step 2 — Describe the image (vision call)

Find `describe_image(image_ref, model)`. Vision models accept a `content` list instead of a plain string. Build the message like this:

```python
messages=[{
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this image in 2-3 sentences. Be factual and objective."},
        {"type": "image_url", "image_url": {"url": image_ref}},
    ],
}]
```

Call `litellm.completion()` with that message. Return a tuple of `(description_string, usage_dict)`. Build `usage_dict` from `response.usage` (keys: `input_tokens`, `output_tokens`, `cost`). You can get the cost with `litellm.completion_cost(completion_response=response)`.

## Step 3 — Generate alt-text

Find `generate_alt_text(description, model)`. Send the description to the text model with a system prompt that enforces the accessibility rules: under 125 characters, factual, no "image of" prefix, no subjective adjectives. Return `(alt_text, usage_dict)`.

After receiving the response, trim to 125 chars if the model ignored the instruction:

```python
if len(text) > 125:
    text = text[:124].rstrip() + "…"
```

## Step 4 — Generate a meme caption

Find `generate_meme_caption(description, model)`. Use a system prompt that constrains the model to one line, under 80 characters, humorous, no hashtags or emoji. Return `(caption, usage_dict)`.

The pattern is identical to Step 3 — same structure, different system prompt and constraint.

## Step 5 — Generate a variant image

Find `generate_variant_image(description, caption, out_path, model)`. Build a prompt that combines the description and caption to convey both the scene and its mood:

```python
prompt = (
    f"A stylized illustration inspired by this scene: {description} "
    f"Mood and tone: {caption}. "
    "Style: bold colors, clean lines, modern editorial illustration, no text in the image."
)
```

Call `litellm.image_generation(prompt=prompt, model=model, size="1024x1024", n=1)`. The response contains a `data` list; each entry has either a `url` key or a `b64_json` key. Handle both:

```python
data = response["data"][0]
if data.get("url"):
    resp = requests.get(data["url"], timeout=60)
    out_path.write_bytes(resp.content)
elif data.get("b64_json"):
    out_path.write_bytes(base64.b64decode(data["b64_json"]))
```

Return `out_path`.

## Step 6 — Wire up the pipeline

Find `run_pipeline(image_ref, out_dir)`. This function calls each stage in order, accumulates token counts and costs, writes the three output files, and returns a result dict:

```python
loaded_ref = load_image(image_ref)
description, u1 = describe_image(loaded_ref)
alt_text,    u2 = generate_alt_text(description)
caption,     u3 = generate_meme_caption(description)
generate_variant_image(description, caption, out_dir / "variant.png")

(out_dir / "alt_text.txt").write_text(alt_text, encoding="utf-8")
(out_dir / "caption.txt").write_text(caption,   encoding="utf-8")
```

Sum up tokens and costs from `u1`, `u2`, `u3` and include them in the returned dict.

## Step 7 — CLI entry point

Wire up a small `main()` with `argparse`. Once `run_pipeline()` works, the full CLI works:

```python
parser.add_argument("image", help="Path or URL to the input image")
parser.add_argument("--out", default="./out", help="Output directory")
```

No changes needed here — just make sure the function you filled in returns the expected keys (`alt_text`, `caption`, `variant_path`, `total_input_tokens`, `total_output_tokens`, `total_cost`).

## Running it

```bash
python solution.py samples/sample.png --out ./out
```

Expected console output:

```
[1/4] Description: A peaceful rural landscape with rolling green hills under a warm sun...
[2/4] Alt-text:    Rolling green hills with a small wooden cabin under a bright yellow sun
[3/4] Caption:     When Monday feels like a pastoral painting nobody asked for
[4/4] Variant:     out/variant.png

=== Summary ===
Alt-text: Rolling green hills with a small wooden cabin under a bright yellow sun
Caption:  When Monday feels like a pastoral painting nobody asked for
Variant:  out/variant.png
Tokens:   in=892 out=145
Cost:     $0.003241
```

After the run, `./out/` contains:

| File | Contents |
|---|---|
| `alt_text.txt` | The accessibility alt-text |
| `caption.txt` | The meme caption |
| `description.txt` | The raw vision description |
| `variant.png` | The AI-generated variant image |

Pass a URL instead of a local path to skip the base64 step:

```bash
python solution.py "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/240px-PNG_transparency_demonstration_1.png" --out ./out
```

## What to try next

1. **Batch mode** — accept a directory of images and process each one, writing outputs into per-image subdirectories under `--out`
2. **Caption personas** — add a `--persona` flag (e.g. `sarcastic`, `inspirational`, `gen-z`) and vary the system prompt accordingly
3. **Swap the image model** — set `IMAGE_MODEL=stable-diffusion-3` in `.env` and see how the variant style changes; litellm routes automatically
4. **Streaming description** — replace the `completion()` call in `describe_image` with a streaming call (Module 05 pattern) and print the description token by token as it arrives
