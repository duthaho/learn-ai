# Module 10 Quiz: Image & Multimodal AI

Test your understanding. Try answering before revealing the answer.

---

### Q1: Why can't LLMs read raw pixels the way humans see images?

<details>
<summary>Answer</summary>
Vision-capable LLMs never process individual pixels directly. A separate component — the vision encoder — first divides the image into fixed-size patches (e.g., 16×16 pixels), converts each patch into a dense embedding vector, and then projects those embeddings into the same vector space as text token embeddings. After projection, image patches and text tokens are directly comparable and are passed together into the LLM transformer as a unified sequence. The language model reasons over these patch embeddings, not raw pixel values. This is why you cannot simply pass a raw image array to an LLM — the encoder is a trained preprocessing step that converts visual information into a representation the transformer can consume.
</details>

---

### Q2: When should you send an image as a URL vs a base64 data URI?

<details>
<summary>Answer</summary>
Use a URL when the image is already publicly hosted — on a CDN, a public S3 bucket, or any public web address. The provider fetches it server-side; your request payload stays small and there is no encoding overhead. Use a base64 data URI when the image is local or private: a file on disk, a user upload that has not been stored publicly, or an in-memory image. The trade-off is that base64 encoding inflates the binary data by approximately 33%, so a 500 KB PNG becomes roughly 665 KB in the request body. For high-volume production systems processing large images, hosting images and using URL mode reduces bandwidth costs and request latency.
</details>

---

### Q3: How do images affect context budget and cost?

<details>
<summary>Answer</summary>
Vision models tokenize images, and those image tokens count against the same context window as text tokens. The token count depends on the model and the detail mode. For example, a 1024×1024 image sent to GPT-4o in high-detail mode costs 85 base tokens plus 170 tokens per 512-pixel tile — four tiles for that image, totalling 765 tokens. The same image in low-detail mode costs 85 tokens flat. This connects directly to the context budget formula from Module 09: available_for_history = context_window - system_prompt_tokens - output_reserve. Image tokens consume that budget just like text tokens. In a multi-turn vision conversation, images from earlier turns stay in context and continue accumulating cost. Resize images to the minimum resolution the task requires before sending — this is the most direct lever for reducing image token cost.
</details>

---

### Q4: Name three common vision-model failure modes and one mitigation for each.

<details>
<summary>Answer</summary>

- **Hallucinated text** — the model reads characters that are not present, or fills in plausible-sounding words for blurry or partially occluded text. Mitigation: instruct the model to return exact text only, character by character, and to explicitly say so when any word is illegible rather than guessing.
- **Counting errors** — the model miscounts objects in crowded or overlapping scenes, often by a significant margin. Mitigation: ask the model to describe each item individually before producing a count, or request structured output with an explicit count field that forces item-by-item enumeration.
- **Low-resolution and fine-detail confusion** — logos, small labels, and low-contrast text are read with higher error rates. Mitigation: upsize the image to the minimum resolution where the detail becomes legible, or switch to high-detail mode so the model receives more patches from the image region in question.

</details>

---

### Q5: What does the `size` parameter trade off in image generation?

<details>
<summary>Answer</summary>
Larger sizes produce higher-resolution output but cost more per image and take longer to generate. Image generation is priced per image, not per token, so size is one of the primary cost multipliers — a 1792×1024 output costs more than a 1024×1024 output from the same model. Start with the smallest size that serves the use case: 512×512 or 1024×1024 for thumbnails and most development work, larger only when the image will actually be displayed at that resolution. Do not default to the largest size in a prototype — standard quality at 1024×1024 is sufficient for almost all development and testing work, and switching to HD or larger sizes when you go to production is straightforward.
</details>

---

### Q6: When should you chain vision then generation vs use only one?

<details>
<summary>Answer</summary>
Chain vision then generation when the task transforms an existing image into a different representation — a photo into a stylized variant, a diagram into a marketing visual, a receipt into a summary poster. The vision model handles perception (extracting what is in the image), a text reasoning step transforms that description, and the generation model synthesizes a new image from the result. Use vision alone when the task is analysis: "what does this show," OCR, structured data extraction from a document, or content moderation — tasks where the output is meaning, not a new image. Use generation alone when the task is "make this" from a text description with no source image involved. Do not add pipeline stages to look sophisticated — each extra call adds latency and cost. Add a stage only when it genuinely specializes the work and a single call cannot handle the task.
</details>

---

### Q7: How does multimodal relate to structured output (Module 08)?

<details>
<summary>Answer</summary>
Vision models can return JSON conforming to a Pydantic schema just as text models can. This combination is the standard pattern for real-world document extraction: receipts, invoices, insurance forms, charts, screenshots of tables. Define a Pydantic schema for the fields you need, pass `response_format=YourModel` (or use a prompt-based JSON instruction), and validate the output. The only difference from a text extraction call is that the `content` field in the message uses the content-array format with an image part instead of plain text. All the schema definition, validation, and error-handling patterns from Module 08 apply directly. The practical result: send an image of a receipt, receive a type-checked object with `vendor_name`, `date`, `line_items`, and `total_amount` fields — no OCR preprocessing required.
</details>

---

### Q8: Where does multimodal fit in an agent pipeline?

<details>
<summary>Answer</summary>
Three natural integration points:

1. **As a specialized tool the agent calls (Module 06 pattern)** — define a `describe_image(url)` tool that the agent invokes when its task involves an image. A text-only reasoning agent can stay fast and cheap for text steps, and delegate to the vision model only when an image actually needs processing. The agent receives a text description back and continues reasoning in its normal context.
2. **As a preprocessing step before RAG indexing (Module 07 pattern)** — at document index time, run each image through a vision model to produce a text description, embed that description, and store it in the vector database. At query time, retrieval works over the text representation. This extends a text-only RAG pipeline to handle image-bearing documents without changing the retrieval mechanism.
3. **As an output stage where the agent generates an image after reasoning** — the agent uses text reasoning to decide what to create, constructs a detailed generation prompt, and calls the image generation model as a final step. This is the most expensive stage in any pipeline; budget latency (5–30 seconds per image) and cost (10–100× a text completion) accordingly.

Each integration point has a different latency and cost profile. Use the tool pattern for on-demand vision in a mixed-input agent, the preprocessing pattern for document corpora, and the output pattern when image creation is the explicit goal.
</details>
