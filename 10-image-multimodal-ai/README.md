# Module 10: Image & Multimodal AI

Working with images in AI applications — reading images with vision models, generating images from text, and chaining both into practical pipelines.

| Detail        | Value                                                                 |
|---------------|-----------------------------------------------------------------------|
| Level         | Intermediate                                                          |
| Time          | ~3 hours                                                              |
| Prerequisites | Module 04 (The AI API Layer), Module 08 (Structured Output), Module 09 (Conversational AI & Memory) |

## What you'll learn

- Why multimodal AI matters and what problems it unlocks beyond text
- How vision encoders convert images into tokens that LLMs can reason about
- How to send images to a vision API via URL and base64, with real code
- Prompt patterns for image description, OCR, extraction, and failure modes to avoid
- How to generate images with `litellm.image_generation()` and tune the key parameters
- Prompt engineering techniques for reliable, high-quality image generation
- How to chain vision and generation into a multi-stage pipeline
- Where multimodal fits alongside RAG, tools, structured output, and memory

---

## Table of Contents

1. [Why Multimodal Matters](#1-why-multimodal-matters)
2. [How LLMs See Images](#2-how-llms-see-images)
3. [Sending Images to a Vision API](#3-sending-images-to-a-vision-api)
4. [Prompting Vision Models](#4-prompting-vision-models)
5. [Image Generation Basics](#5-image-generation-basics)
6. [Prompt Engineering for Image Gen](#6-prompt-engineering-for-image-gen)
7. [Pipelines: Chaining Vision → Generation](#7-pipelines-chaining-vision--generation)
8. [Multimodal in the AI Stack](#8-multimodal-in-the-ai-stack)

---

## 1. Why Multimodal Matters

For most of AI's history, language models worked with a single modality: text in, text out. The shift to multimodal changes the fundamental unit of input. Vision models accept images alongside text. Generation models produce images from text descriptions. Together they expand what AI applications can perceive, understand, and create.

This is not a niche capability. A huge portion of real-world information — documents, photographs, diagrams, interfaces, product photos, medical scans — arrives as images. Text-only AI leaves all of that information on the floor. Multimodal AI can process and reason about it.

### Four concrete use cases

**Accessibility — automatic alt-text.** Screen readers require descriptive text for every image. Writing alt-text by hand is expensive and inconsistently done. A vision model can describe any image in a sentence or two, producing usable alt-text at scale for web content, product catalogs, and media archives.

**Document extraction — receipts and forms.** Receipts, invoices, insurance forms, and scanned contracts are images first, data second. A vision model paired with structured output (see [Module 08](../08-structured-output/)) can extract vendor name, line items, totals, and dates from an image of a receipt without any OCR preprocessing step.

**Content moderation.** User-generated content platforms need to detect policy violations at image upload time. Vision models can classify images for nudity, violence, hate symbols, or spam content — providing a first-pass filter before human review.

**Creative tools — mockups and memes.** Designers upload a rough sketch and receive a styled rendering. Marketing teams upload a product photo and generate caption variants. Vision plus generation enables creative loops that would require multiple specialized tools operating separately.

### Text-only limitation vs what multimodal unlocks

| Text-only limitation | What multimodal unlocks |
|---|---|
| Cannot read text in images (receipts, signs, screenshots) | Vision models extract text directly from image content |
| Cannot describe visual content for downstream processing | Images become first-class inputs alongside natural language |
| Cannot generate images — only describe them in words | Text-to-image generation produces actual pixels from descriptions |
| Cannot detect visual policy violations in user uploads | Content moderation extends to images, not just text |
| Documents must be pre-processed to extract text before LLM sees them | Vision models read PDFs, scans, and forms without an OCR preprocessing step |

### Three pillars of this module

**Vision** — sending images to a model and extracting structured meaning from them. This is the "read" half of multimodal.

**Generation** — converting a text prompt into an image. This is the "write" half of multimodal.

**Pipelines** — chaining vision and generation together, with a text reasoning step in the middle, to produce workflows that none of the pieces can accomplish alone.

---

## 2. How LLMs See Images

Before you can use vision models well, you need a working mental model of how they process images. The mechanism shapes everything: token counts, cost, resolution trade-offs, and failure modes.

### Vision encoders: patches to embeddings

When you send an image to a vision-capable LLM, the image does not go directly into the transformer. A separate component — the **vision encoder** — handles the conversion:

1. **Tiling** — the image is divided into fixed-size patches (commonly 14×14 or 16×16 pixels each). A 512×512 image at 16-pixel patches produces 1,024 patches.
2. **Encoding** — each patch passes through a convolutional or transformer-based encoder that converts the pixels into a dense vector (an embedding).
3. **Projection** — the patch embeddings are projected into the same vector space as the text token embeddings. After this step, image patches and text tokens are directly comparable.
4. **Fusion** — the projected image embeddings are concatenated with the text token embeddings and passed together into the main LLM transformer. The model reasons over both as a unified sequence.

From the LLM's perspective, an image is just a sequence of special tokens — richer than text tokens, but processed through the same attention mechanism. This is why vision models can answer questions that interleave text and image reasoning: the representations are in the same space.

### Why resolution matters for cost

Image tokens cost the same as text tokens in the context budget, and they accumulate quickly. Higher resolution means more patches, which means more tokens per image.

Providers handle resolution in different ways, but the pattern is consistent: there is a low-detail mode with a fixed, small token count, and a high-detail mode where token count scales with image dimensions.

| Model | Low-detail cost | High-detail cost (approximate) |
|---|---|---|
| GPT-4o | 85 tokens flat | 85 base + 170 × number of 512px tiles |
| GPT-4o Mini | 2,833 tokens flat | 2,833 base + 5,667 × number of 512px tiles |
| Claude 3.5 Sonnet | ~1,500 tokens flat (short edge ≤ 384px) | (width × height) / 750, roughly |

For a 1024×1024 image sent to GPT-4o in high-detail mode: the image is divided into four 512-pixel tiles, costing 85 + (170 × 4) = 765 tokens. The same image at low detail costs 85 tokens — a 9× difference for the same image.

At 10 images per request and GPT-4o's output pricing, this adds up fast. Resize images to the minimum resolution needed for the task before sending them.

### Images consume context budget like text

Image tokens count against the same context window that text tokens use. A conversation with several high-resolution images can consume thousands of tokens before a single word of text prompt has been sent. This connects directly to the context budgeting principles from [Module 09](../09-conversational-ai-memory/): track your token budget across all input types, not just text.

Practical implication: if you are building a multi-turn vision chatbot, image tokens from earlier turns stay in context even as the conversation grows. Either strip images from older messages or switch to low-detail mode for history management.

### Hallucination risk in vision models

Vision models can confidently describe things that are not present in an image. This is not random noise — it follows patterns:

- **Text in images**: models sometimes read characters that are blurry or partially occluded and fill in plausible-sounding words
- **Counting**: vision models frequently miscount objects, especially when items overlap or the image is crowded
- **Spatial relationships**: left/right and above/below relationships are frequently confused
- **Fine detail**: logos, small labels, and low-contrast text are read with higher error rates than humans achieve on the same images

Treat vision model output as a confident draft that needs validation, not a ground truth. Where accuracy matters, ask the model to express uncertainty explicitly.

---

## 3. Sending Images to a Vision API

### The content-array message format

Standard LLM messages have a `content` field that is a string. For multimodal inputs, `content` becomes a list of typed parts — each part is either text or an image. The model receives both parts and reasons over them together:

```python
messages=[{
    "role": "user",
    "content": [
        {"type": "text", "text": "What does this chart show?"},
        {"type": "image_url", "image_url": {"url": "https://..."}},
    ],
}]
```

You can mix any number of text and image parts in a single message. The model processes them in order — you can place instructions before the image, after it, or interleave them with multiple images.

### Two input modes: URL vs base64

**URL mode** — pass a publicly accessible HTTPS URL. The provider fetches the image server-side.

- Use when: the image is already hosted publicly (CDN, S3 with public read, web URL)
- Advantages: no bandwidth cost sending the image data, cleaner request payload
- Disadvantages: the URL must be publicly accessible; providers do not follow redirects or handle authentication; the image at the URL must remain stable during the request

**Base64 data URI** — encode the image bytes and embed them directly in the request.

- Use when: the image is local or private (user upload, file on disk, image in memory)
- Advantages: no hosting required, works for private images, no external dependency during the request
- Disadvantages: adds ~33% overhead to the request size (base64 encoding inflates binary data), slower for large images on slow connections

### Working code example

```python
import base64
from litellm import completion

def image_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

response = completion(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in one sentence."},
            {"type": "image_url", "image_url": {"url": image_to_data_uri("photo.png")}},
        ],
    }],
)
print(response.choices[0].message.content)
```

This uses LiteLLM's `completion()` — the same interface as all previous modules, with the content-array format for the image. Switching from `gpt-4o-mini` to `claude-3-5-sonnet-20241022` or another vision model requires only changing the `model` string.

### Multi-image inputs

Add more `image_url` parts to the content list. The model receives all images in sequence:

```python
content=[
    {"type": "text", "text": "Which of these two screenshots has a broken layout?"},
    {"type": "image_url", "image_url": {"url": image_to_data_uri("before.png")}},
    {"type": "image_url", "image_url": {"url": image_to_data_uri("after.png")}},
]
```

There is no fixed limit on the number of images per request, but token costs accumulate quickly and context window capacity is shared across all images plus text.

### Size limits and resizing

Typical provider caps:

- Maximum file size per image: ~20 MB (OpenAI), ~5 MB (Anthropic for base64)
- Maximum dimensions: 2048px on the long side before downscaling is applied automatically (varies by provider)
- Maximum images per request: 10–20 depending on provider and model

Always resize images before upload. Sending a 4032×3024 smartphone photo to get a receipt description wastes tokens and money — a 1024px version of the same photo contains all the information needed. Use Pillow or similar:

```python
from PIL import Image

def resize_image(path: str, max_side: int = 1024) -> bytes:
    img = Image.open(path)
    img.thumbnail((max_side, max_side))
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
```

### Cost note on base64

Base64 encoding increases the byte size of an image by approximately 33%. A 500 KB PNG becomes ~665 KB in the request payload. Over many requests with large images, this can meaningfully affect bandwidth costs and request latency. For production systems processing high volumes of images, hosting images and using URL mode is preferable.

### Supported image formats

Most providers accept JPEG, PNG, GIF (first frame only), and WebP. JPEG is the best default for photographs — good quality at small file sizes. PNG is best for screenshots, diagrams, and images with text where lossless quality matters. Avoid sending uncompressed TIFF or BMP; convert to PNG or JPEG first.

For the MIME type in the data URI, match the actual format:

```python
def image_to_data_uri(path: str, mime_type: str = "image/jpeg") -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"

# PNG screenshot
uri = image_to_data_uri("screenshot.png", mime_type="image/png")

# JPEG photograph
uri = image_to_data_uri("photo.jpg", mime_type="image/jpeg")
```

Sending the wrong MIME type can cause parsing errors on the provider side. When in doubt, convert to PNG and use `image/png`.

---

## 4. Prompting Vision Models

### Four prompt patterns

Vision models respond to prompt design in the same way text models do — the framing of your question shapes the form and focus of the answer. Four patterns cover most use cases:

**1. Open description**

Ask for a general description when you want the model's unprompted reading of the image. Useful for accessibility alt-text, general cataloging, or when you are not sure what to look for.

```
Describe this image in detail.
```

**2. Targeted question**

Focus the model's attention on one element. Produces shorter, more accurate responses than open description for specific facts.

```
What text appears on the label in this image?
```

**3. OCR / text extraction**

Instruct the model to return only the verbatim text from the image, without interpretation. Works for receipts, screenshots, street signs, and printed forms.

```
Extract all text visible in this image verbatim. Output as plain text, preserving line breaks where they appear.
```

**4. Structured extraction**

Combine vision with JSON output (see [Module 08](../08-structured-output/)) for extracting typed fields from a document image. Pass a schema in the prompt and use `response_format` with a Pydantic model.

```
Extract the following fields from this receipt image and return them as JSON:
- vendor_name (string)
- date (string, format YYYY-MM-DD)
- total_amount (number)
- line_items (list of objects with 'description' and 'amount')
```

### Failure modes

Vision models fail in predictable ways. Knowing the failure modes lets you design prompts that mitigate them:

| Failure mode | Description | Mitigation |
|---|---|---|
| Hallucinated text | Model reads characters that are not there or fills in plausible-sounding words for blurry text | Ask for "exact text only, character by character." If a word is unclear, ask the model to say so rather than guess. |
| Over-confidence on blurry input | Model gives a definitive answer on low-resolution or out-of-focus images without flagging uncertainty | Explicitly ask: "If any part of the image is unclear, say so rather than guessing." |
| Left/right and spatial errors | Model confuses left/right positions; misidentifies which object is above/below another | Describe a reference anchor point: "The object in the top-left corner of the image is..." |
| Counting errors | Model misses items in crowded or overlapping scenes | Ask the model to describe each item individually before giving a count, or break the image into regions |
| Poor numeric precision on charts | Model reads approximate values from bar charts and line graphs, often 10–20% off | Ask for ranges rather than exact values: "What is the approximate range of the Y axis?" |

### Requesting higher-quality output

Several prompt additions reliably improve output quality:

- Add `"If any detail is unclear, say so rather than guessing."` — surfaces uncertainty instead of hiding it
- Use high-detail mode for text-heavy images where OCR accuracy matters
- Combine vision with structured output and a validation step for production data extraction
- For charts, prefer asking about trends and comparisons rather than specific numeric values

---

## 5. Image Generation Basics

### Text-to-image at a high level

Image generation models take a text prompt and produce pixels. The dominant approach is diffusion: start with random noise and iteratively remove noise in a direction guided by the prompt, until a coherent image emerges. The prompt is encoded (often by a CLIP-style encoder) into an embedding that directs the denoising process at each step.

Unlike vision models, generation models are not LLMs — they do not use transformers over text tokens to produce image tokens. But from an API perspective, the pattern is the same: send a prompt, get a result back.

### LiteLLM's image_generation() interface

LiteLLM provides `image_generation()` as a provider-agnostic interface for text-to-image, following the same design as `completion()`. You switch between providers by changing the `model` string:

```python
from litellm import image_generation

response = image_generation(
    prompt="A minimalist logo of a fox reading a book, flat design, pastel colors",
    model="dall-e-3",
    size="1024x1024",
)
image_url = response["data"][0]["url"]
```

The returned URL is a temporary link to the generated image. Download the image immediately if you need to keep it — these URLs typically expire within one hour.

### Key parameters

| Parameter | Purpose | Typical values |
|---|---|---|
| `model` | Which model to use | `dall-e-3`, `dall-e-2`, `stability/stable-diffusion-xl-1024-v1-0` (via Replicate) |
| `size` | Output dimensions | `"1024x1024"`, `"1792x1024"` (landscape), `"1024x1792"` (portrait) |
| `quality` | Detail level and rendering effort | `"standard"` (faster, cheaper), `"hd"` (higher detail, DALL-E 3 only) |
| `style` | Rendering aesthetic | `"vivid"` (saturated, dramatic), `"natural"` (realistic, subtle) — DALL-E 3 only |
| `n` | Number of images to generate | `1`–`10` (DALL-E 2); DALL-E 3 only supports `1` |
| `response_format` | How to return the image | `"url"` (a temporary link), `"b64_json"` (base64-encoded bytes for immediate use) |

### Cost model

Image generation is priced per image, not per token. The cost multipliers are:

- **Model quality** — DALL-E 3 HD at 1024×1024 costs more than DALL-E 3 Standard; DALL-E 2 is cheaper than DALL-E 3
- **Size** — larger outputs cost more; 1792×1024 costs more than 1024×1024
- **Count** — `n=4` costs 4× `n=1`

There is no per-token charge for the prompt. Long, detailed prompts cost the same as short ones. This is different from text completion pricing and means iteration on prompts is cheap — you pay for images generated, not for prompt experimentation.

### Provider notes

- **DALL-E 3 / DALL-E 2** — OpenAI's models, accessed via LiteLLM with `model="dall-e-3"`. DALL-E 3 is significantly better at following complex prompts and handling text in images.
- **Stable Diffusion XL** — open-weight model available via Replicate, Stability AI API, and self-hosted inference. LiteLLM supports Replicate with `model="stability/stable-diffusion-xl-1024-v1-0"`.
- **Flux** — a newer open-weight family with strong prompt adherence; available via Replicate and other hosts.
- **Imagen** — Google's model, available through the Vertex AI API.

The `model` string in LiteLLM controls which provider and model you use. The calling code stays the same.

---

## 6. Prompt Engineering for Image Gen

Image generation prompts reward specificity. A vague prompt produces a generic image. A well-structured prompt controls the subject, the visual style, and the composition — giving you a much narrower distribution of outputs centered on what you actually want.

### The subject / style / composition framework

Structure your prompt around three axes:

**Subject** — who or what is in the image, and what are they doing.

**Style** — the visual language of the image: photorealistic, flat design, watercolor, oil painting, pixel art, 3D render, sketch, etc.

**Composition** — camera angle, framing, lighting, depth of field, color palette, time of day.

Combining all three gives the model enough constraints to generate something intentional rather than generic.

### Weak vs strong prompt comparison

| Weak prompt | Strong prompt |
|---|---|
| A dog | A golden retriever puppy sitting in tall grass at golden hour, shallow depth of field, warm natural lighting, photorealistic |
| A city | Aerial view of a dense Tokyo street intersection at night, neon signs reflecting on wet pavement, cinematic color grading, wide angle |
| A logo | A minimalist logo of a lighthouse, flat vector design, two-color palette of navy blue and white, geometric shapes, no text |

The strong prompts add style and composition details. The weak prompts leave every visual decision to the model — the results are unpredictable and generic.

### Iteration pattern: one axis at a time

The most efficient way to improve a generated image is to fix one axis per iteration rather than rewriting the entire prompt:

1. Generate with a complete but rough prompt
2. Inspect: which axis is wrong? (Subject is off? Style is wrong? Composition is poor?)
3. Refine only that axis in the next prompt
4. Repeat until the image is close enough to use

Changing everything at once makes it impossible to know which change improved the result. Changing one axis keeps the rest stable and lets you isolate what each part of the prompt controls.

### Negative prompts

Some models (Stable Diffusion and its derivatives) support a separate `negative_prompt` parameter that tells the model what to exclude. DALL-E 3 does not have a dedicated parameter, but you can embed exclusions in the prompt itself:

```python
# Stable Diffusion style (separate parameter)
response = image_generation(
    prompt="A portrait of a woman, studio lighting, professional",
    model="stability/stable-diffusion-xl-1024-v1-0",
    # negative_prompt="blurry, low quality, distorted, extra limbs",
)

# DALL-E style (inline)
prompt = "A portrait of a woman, studio lighting, professional. Avoid: blurry, low quality, distorted anatomy."
```

Common negative prompt inclusions: `blurry`, `low quality`, `watermark`, `text overlay`, `extra limbs`, `distorted face`, `overexposed`.

### Reference-style prompts

You can anchor the style of a generated image to a recognizable visual vocabulary without directly copying any specific work:

```
in the style of a 1950s travel poster
in the style of a children's book illustration
in the style of a technical diagram from a science textbook
flat design icon, similar to material design guidelines
```

Style anchors give you consistent aesthetics across a set of generated images — useful for creating coherent sets of illustrations for a product, documentation, or presentation.

### Seed reproducibility

Most providers support a `seed` parameter that makes generation reproducible. With the same prompt and seed, you get the same (or very similar) image each time:

```python
response = image_generation(
    prompt="A minimalist logo of a fox reading a book, flat design, pastel colors",
    model="dall-e-3",
    size="1024x1024",
    seed=42,  # supported by some providers/models
)
```

Seed is useful for: A/B testing prompt changes (hold seed constant, change the prompt), saving a working combination for later, and ensuring consistent output in staging vs production environments. Not all providers support seed — check provider documentation.

---

## 7. Pipelines: Chaining Vision → Generation

### The core pattern

Vision models are strong at perception: describing what is in an image, reading text, identifying objects, and extracting structured data. Generation models are strong at synthesis: producing new images from a description. Neither does both well in a single step.

The pipeline pattern chains them together with a text processing step in the middle:

```
Image input
    → Vision model: describe / extract
        → Text model: transform / reason
            → Image model: generate
```

Each stage does what it is best at. The vision model handles perception. The text model handles reasoning and transformation. The image model handles synthesis. The result is a pipeline that can accomplish tasks that no single model handles well alone.

### Why this works

The key insight is that the middle text step lets you apply all the prompt engineering, structured output, and reasoning capabilities from earlier modules — before handing off to image generation. You are not hoping the generation model will infer a complex instruction from an image. You are using the vision model to convert the image into a description, using the text model to transform that description into a clear generation prompt, and then using the generation model for what it does well: turning a clear text prompt into pixels.

### The pipeline for this module's project

The project builds a pipeline that takes any input image and produces three outputs: an accessibility alt-text, a meme caption, and a stylized variant image. The pipeline runs four stages:

**Stage 1 — Vision: objective description**

Send the input image to a vision model with a prompt that asks for a detailed, objective description with no interpretation. This gives the rest of the pipeline a reliable text representation to work from.

```python
description = vision_describe(image_path)
# → "A golden retriever sitting in tall grass, facing the camera, with a blue sky behind it."
```

**Stage 2 — Text: derive alt-text**

Pass the description to a text model with a prompt asking for accessible, concise alt-text following WCAG guidelines. The alt-text describes what the image shows and what purpose it serves.

```python
alt_text = generate_alt_text(description)
# → "A golden retriever sits in tall grass under a blue sky, looking directly at the camera."
```

**Stage 3 — Text: derive meme caption**

Pass the same description to the text model with a different prompt: generate a short, funny meme caption appropriate for the image.

```python
caption = generate_caption(description)
# → "When someone says 'fetch' in a meeting and you take it literally."
```

**Stage 4 — Image gen: stylized variant**

Combine the description and caption into an image generation prompt. The generation model produces a stylized variant of the original scene.

```python
gen_prompt = f"{description}. Stylized as a vintage comic strip panel. Caption text at the bottom: '{caption}'."
variant_url = generate_image(gen_prompt)
```

### Orchestration concerns

**Per-stage error handling.** Each stage is an independent API call that can fail. Do not catch all exceptions at the pipeline level — catch them at the stage level, log which stage failed, and decide whether to continue with a fallback or abort. A failed alt-text generation should not block meme caption generation.

**Aggregate cost and token tracking.** A four-stage pipeline makes four API calls. Track tokens and costs separately per stage so you can identify which step is expensive and optimize it. Vision calls and generation calls have very different cost profiles — vision is charged per token like text, generation is charged per image.

**Latency stacks.** Four sequential API calls means latency compounds. A 2-second vision call + 1-second text call + 1-second text call + 5-second generation call = ~9 seconds minimum. Stages 2 and 3 (both text-only) can run in parallel since they both read from Stage 1's output.

### A minimal pipeline implementation

The four-stage pipeline can be wired up with straightforward function calls. Keeping each stage as a separate function makes it easier to test stages independently and to swap in different models:

```python
from litellm import completion, image_generation

def vision_describe(image_path: str, model: str = "gpt-4o-mini") -> str:
    """Stage 1: objective description of the input image."""
    uri = image_to_data_uri(image_path, mime_type="image/jpeg")
    response = completion(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image objectively and in detail. Focus on what is visible — do not interpret or editorialize."},
                {"type": "image_url", "image_url": {"url": uri}},
            ],
        }],
    )
    return response.choices[0].message.content

def derive_alt_text(description: str, model: str = "gpt-4o-mini") -> str:
    """Stage 2: generate WCAG-compliant alt-text from the description."""
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": "You write concise, accurate alt-text for web images following WCAG guidelines. Keep alt-text under 125 characters."},
            {"role": "user", "content": f"Write alt-text for an image described as:\n\n{description}"},
        ],
    )
    return response.choices[0].message.content.strip()

def derive_meme_caption(description: str, model: str = "gpt-4o-mini") -> str:
    """Stage 3: generate a short meme caption from the description."""
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": "You write short, funny meme captions. Keep captions under 15 words."},
            {"role": "user", "content": f"Write a meme caption for an image described as:\n\n{description}"},
        ],
    )
    return response.choices[0].message.content.strip()

def generate_stylized_variant(description: str, caption: str) -> str:
    """Stage 4: generate a stylized image variant using description + caption."""
    gen_prompt = (
        f"{description}. "
        f"Rendered as a vintage comic strip panel with bold outlines and flat colors. "
        f"Caption text at the bottom reads: '{caption}'."
    )
    resp = image_generation(prompt=gen_prompt, model="dall-e-3", size="1024x1024")
    return resp["data"][0]["url"]
```

Run Stages 2 and 3 in parallel (they both depend only on Stage 1's output) using `concurrent.futures.ThreadPoolExecutor` to cut wall-clock time by almost half.

### When NOT to chain

Pipelines add latency, cost, and complexity. Before chaining, ask: can one call handle this?

- If the vision model can return structured JSON directly (with `response_format`), you do not need a separate text reasoning step.
- If the generation prompt is simple and does not require deriving from an input image, skip vision entirely.
- If the task can be done in two stages instead of four, use two stages.

Add stages when each stage genuinely specializes — not as a pattern to apply by default. A pipeline that does the same thing a single call would do is just overhead.

---

## 8. Multimodal in the AI Stack

Multimodal does not replace the techniques in prior modules — it extends them. Each prior module describes a pattern that still applies; adding images is a new dimension within the same architecture.

### RAG with image documents (Module 07)

[Module 07](../07-rag/) built retrieval pipelines over text documents. Many real-world document corpuses contain images: PDFs with diagrams, scanned forms, slides, product manuals with photographs. Text-only embedding cannot index image content.

Vision models close this gap: run each image through a vision model at index time to produce a text description, then embed and store that description. At query time, the retriever finds the text description; the generator can optionally re-examine the original image. The retrieval mechanism stays the same — you are adding a vision preprocessing step before the embedding step.

### Vision as a tool (Module 06)

[Module 06](../06-tool-use-function-calling/) covered tool use: the LLM decides which function to call and with what arguments. Vision can be one of those tools — `describe_image(url)` — that an agent calls when it needs to process an image as part of a larger task.

This lets a text-only reasoning agent delegate vision tasks without the entire agent context being multimodal. The agent calls the tool, receives a text description back, and continues reasoning. Practical benefit: text-only models are faster and cheaper; reserve the vision call for when an image actually needs to be processed.

### Structured extraction from images (Module 08)

[Module 08](../08-structured-output/) covered schema-constrained output. Combining vision with structured output gives you typed document extraction: send an image of a receipt and get back a validated Pydantic object with `vendor_name`, `date`, `line_items`, and `total_amount` fields — all type-checked and schema-validated.

The pattern is the same as text extraction: define a Pydantic schema, pass `response_format=YourModel`, validate the output. The only difference is that the input `content` field uses the array format with an image part instead of plain text.

### Multimodal context and memory (Module 09)

[Module 09](../09-conversational-ai-memory/) established the principle that the context window is a shared budget across everything in the messages list. Multimodal inputs expand the cost of that budget significantly.

A single high-resolution image in high-detail mode can cost 1,000–3,000 tokens — comparable to several paragraphs of text. In a multi-turn vision conversation, images from earlier turns remain in context and continue consuming budget. Apply the same budgeting discipline: track image tokens explicitly, strip old images from history when they are no longer needed, and account for image costs in your soft and hard token limits.

### Latency and cost profile

| Operation | Typical latency | Relative cost |
|---|---|---|
| Text completion (short) | 0.5–2 seconds | Low |
| Vision call (image description) | 2–5 seconds | Medium |
| Image generation (standard) | 5–15 seconds | High |
| Image generation (HD) | 10–30 seconds | Very high |

Vision calls are slower than text-only calls because the image must be fetched, encoded, and processed before generation begins. Image generation is the slowest and most expensive operation in the stack. Structure your pipelines to run text-only steps in parallel where possible, and consider showing partial results to users while generation completes.

Budget rule of thumb: vision is 2–5× the cost of a comparable text call for the same output. Image generation is 10–100× the cost of a text completion, depending on quality settings. Do not default to HD generation in a prototype — standard quality is sufficient for most development and testing.

### When NOT to use vision

Vision models are powerful but not always the right tool:

- **When OCR would do** — if you need to extract text from a high-quality, machine-generated PDF (not a scan), a dedicated OCR tool will be faster, cheaper, and more accurate than a vision model.
- **When image hashing would do** — if you need to detect duplicate images or find visually similar images in a corpus, perceptual hashing (pHash, dHash) is orders of magnitude faster and cheaper than vision embeddings.
- **When metadata is available** — if the image already has alt-text, EXIF data, or file metadata that answers your question, use that first before paying for a vision API call.
- **When accuracy is critical and stakes are high** — vision models hallucinate. For medical imaging, legal documents, or financial records where errors have serious consequences, treat vision model output as a draft that requires human review, not a final answer.

### Forward pointer: audio as a modality

Module 22 (Voice & Audio) extends the multimodal pattern to audio: speech-to-text for input, text-to-speech for output, and audio models for tasks like transcription, speaker diarization, and music generation. The architectural patterns are the same — audio is converted to an intermediate representation, passed through the model, and the output is converted back to audio. The same budgeting, pipeline chaining, and integration patterns from this module apply directly to audio workflows.

---
