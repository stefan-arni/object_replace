> This is the living spec. If I tell you something in chat that contradicts
> this file, ask me whether to update the file or ignore it for this turn.


# Project: Real-image object replacement via DDIM inversion + attention control

I'm building one module of a 3-person project for CS 5788 (generative models, Cornell).
My module does **text-guided object replacement on real photographs** using Stable
Diffusion v1.5. The other teammates handle repositioning and style; I'm standalone.

## What this is NOT
- Not a `diffusers` wrapper. I cannot call `StableDiffusionPipeline(...)` or any
  `*Pipeline` that does the denoising for me — the assignment explicitly forbids it.
  I load the components (`UNet2DConditionModel`, `AutoencoderKL`, `CLIPTextModel`,
  tokenizer, `DDIMScheduler`) individually and write the denoising loop myself.
- Not a repo clone. I can read the Prompt-to-Prompt and null-text inversion papers
  and reference implementations for understanding, but the code here is mine.
- Not over-engineered. No pydantic schemas, no dependency injection framework, no
  100-line docstrings. I like code that reads like someone thinking out loud.

## Code style I want
- Flat module structure. One file per concept.
- Comments explain *why*, never *what*. If the code is obvious, no comment.
- No defensive programming against bugs that don't exist. Assert shapes at
  boundaries, not inside every function.
- Type hints on public functions. Skip them on tiny internal helpers.
- No try/except around things that should just crash loudly if they fail.
- Variable names that match the math in the paper (`eps_pred`, `alpha_bar`, `z_t`)
  rather than Java-esque (`predictedNoiseResidual`).

## Target layout
```
.
├── README.md
├── requirements.txt
├── src/
│   ├── sd_components.py      # load UNet, VAE, text encoder, scheduler
│   ├── ddim.py               # forward + reverse DDIM, no inversion magic here
│   ├── inversion.py          # DDIM inversion + null-text optimization
│   ├── attention_store.py    # forward hooks to capture & inject cross-attn maps
│   ├── editor.py             # the main Editor class: invert -> edit -> decode
│   ├── schedules.py          # my novel contribution: time/token-dependent swap schedules
│   ├── masks.py              # derive localized blending masks from attention
│   └── metrics.py            # CLIP directional sim, LPIPS, reconstruction error
├── scripts/
│   ├── reconstruct.py        # invert and re-decode, measure LPIPS (sanity check)
│   ├── edit_single.py        # CLI: one image, source prompt, target prompt -> edit
│   └── run_ablation.py       # sweep schedules across a small benchmark set
├── notebooks/
│   └── walkthrough.ipynb     # the figure-generating notebook for the report
├── data/
│   ├── real/                 # ~20 photos I'll add later
│   └── prompts.json          # (source_prompt, target_prompt, swap_token) triples
└── outputs/                  # all runs dump here, gitignored
```

## Environment
- Dev on Mac (MPS, 24GB unified). Real runs on Colab A100.
- `torch`, `diffusers` (for the model *weights/components only*, not pipelines),
  `transformers`, `Pillow`, `einops`, `lpips`, `open_clip_torch`, `tqdm`, `matplotlib`.
- Float16 on Colab, float32 on MPS (MPS is flaky with fp16 on attention ops).
- Device helper that returns `cuda` > `mps` > `cpu` and warns on MPS that runs
  will be ~10x slower than A100 — Mac is for editing code, not benchmarks.

## Build order (do these in order, don't skip ahead)

### Step 1: sd_components.py
Load SD v1.5 components from the HF hub. Return a dataclass with unet, vae,
tokenizer, text_encoder, scheduler. Set them all to eval(). One helper to
encode a prompt to CLIP embeddings. One helper to encode/decode latents via VAE
(remember the 0.18215 scale factor). That's it. ~60 lines.

### Step 2: ddim.py
Implement DDIM sampling from scratch. Given latents at timestep T and a text
embedding, denoise to t=0. Use classifier-free guidance (concat uncond + cond,
run UNet once on the batch of 2). Reference the DDIM paper's deterministic
sampler equation — x_{t-1} as a function of x_t, predicted eps, and the alpha
schedule. Do not wrap `scheduler.step`; compute it explicitly so I can see the
math. This file should read like the paper.

**Verify:** generate an image from a random seed with a prompt, compare to what
`diffusers` pipeline would produce (run pipeline once just for the comparison,
then delete that test). They should match to within floating point slop.

### Step 3: Sanity checkpoint — DDIM inversion on *generated* images
Before touching real photos: generate an image, invert it back to noise using
the DDIM inversion procedure (run the sampler in reverse, using the *predicted*
epsilon at each step to back out x_t from x_{t-1}), then re-denoise. You should
recover the image nearly exactly. If not, fix it now. This is the most common
failure point and it's much easier to debug here than on real photos.

### Step 4: inversion.py — null-text inversion for real images
Plain DDIM inversion drifts on real images under classifier-free guidance with
guidance scale > 1. Null-text inversion fixes this: during inversion, optimize
the *unconditional* embedding (not the text embedding, not the latents) at each
timestep so that the CFG-guided trajectory matches the DDIM-inverted trajectory.
~10 inner optimization steps per timestep, Adam, learning rate ~1e-2, loss is
MSE between predicted x_{t-1} and target x_{t-1}. Store the per-timestep
optimized null embeddings.

This step is slow (~1-2 min per image on A100). That's fine. Cache aggressively:
any image I invert gets its (latents trajectory, null embeddings) pickled to
disk keyed by image hash + hyperparams.

### Step 5: attention_store.py
Register forward hooks on every cross-attention module in the UNet
(`unet.named_modules()`, filter for `CrossAttention` / `Attention` with
`is_cross_attention=True` depending on diffusers version).

Two modes:
- **Store mode:** hook captures `attention_probs` at each call, organized by
  (timestep, layer_name, head). Attention is B×H×(HW)×L where L is token count.
- **Inject mode:** hook *replaces* the stored maps with precomputed ones before
  they hit the value projection.

The hook function swaps based on a `controller` object I pass in. Controller has
the current timestep and decides per-layer, per-token whether to inject, scale,
or leave alone. This is where my novel contribution lives.

### Step 6: editor.py
The `Editor` class. Constructor takes SD components. Main method:

```python
def edit(self, image, source_prompt, target_prompt, swap_tokens, schedule, mask_mode="attention"):
    # returns edited PIL image
```

Algorithm:
1. Invert `image` with `source_prompt` (cached if seen before).
2. Run two denoising passes in parallel (batch of 2: source + target prompts),
   with the attention controller active.
3. At each timestep, the controller looks at `schedule(t, token)` and decides
   how to blend source and target attention maps for each token.
4. After denoising, if `mask_mode="attention"`, derive a mask from the
   source-token attention at mid-timesteps, threshold + dilate, and composite
   the edited latents with original latents outside the mask.
5. Decode to pixels. Return.

### Step 7: schedules.py — THE NOVEL BIT
This is what makes the project mine and not a P2P reproduction.

Prompt-to-Prompt's original scheme: replace cross-attn for the first τ·T steps,
then use target attention. Single knob τ, same for every token.

My scheme: `schedule(t, token_role) -> swap_weight ∈ [0, 1]` where `token_role`
is one of {preserved, replaced, context}. Preserved tokens (background, scene)
use high swap weight throughout. Replaced tokens (the actual object swap) use a
schedule I can shape — try `constant`, `linear_decay`, `cosine`, `step`, and a
learned one where I sweep 3-4 inflection points.

Empirical claim I want to test: the optimal schedule for the replaced token is
*not* the same as for preserved tokens, and not the same across edit types.
Structural edits (dog → cat, same rough shape) want the replaced-token swap
weight to decay fast — let the target token drive texture early. Geometric
edits (apple → banana, different shape) want slow decay so layout adapts.

Output of this step: a config-driven system where a schedule is a small Python
object with `(t_fraction, token_role) -> weight`, plus 4-5 built-in schedule
shapes, plus the ability to stack different schedules per token role.

### Step 8: masks.py
Derive a binary/soft mask from attention maps. For the source token being
replaced, average attention across mid-timesteps (t ∈ [0.3T, 0.7T]) and across
the 16×16 and 32×32 UNet layers (these are the ones that encode layout).
Upsample to 64×64 (latent resolution), threshold at the 80th percentile,
morphological dilate by a few pixels. That's the "where the object is" mask.

Two mask modes:
- `"attention"`: derive from attention as above
- `"none"`: no blending, pure attention-controlled generation (for ablation)

Composite: `z_final = mask · z_edited + (1 - mask) · z_source` at each
timestep, not just at the end — continuous blending gives smoother results than
one-shot compositing.

### Step 9: metrics.py
- `clip_directional_similarity(src_img, edit_img, src_prompt, tgt_prompt)`:
  cosine between (CLIP_image(edit) - CLIP_image(src)) and
  (CLIP_text(tgt) - CLIP_text(src)). This measures whether the edit went in
  the right semantic direction.
- `background_lpips(src_img, edit_img, mask)`: LPIPS computed only outside the
  mask. Measures how much we didn't disturb.
- `reconstruction_lpips(src_img, reconstructed_img)`: sanity metric for the
  inversion step alone.

### Step 10: scripts and ablation
`scripts/reconstruct.py` takes an image path, runs inversion + re-denoising
with no edit, reports LPIPS. Use this to verify the pipeline on every new
photo I add.

`scripts/edit_single.py` is the "does it work" CLI.

`scripts/run_ablation.py` runs my 5 schedules × ~15 (image, source, target)
triples, dumps a grid of images to `outputs/ablation/` plus a CSV of metrics.
This generates the main experimental figure for the report.

## Data
- 15-20 photos I take myself (varied scenes: indoor, outdoor, single subject,
  multiple subjects).
- ~20 COCO val2017 images with clear objects.
- `data/prompts.json` has entries like:
  `{"image": "cat_couch.jpg", "source": "a cat sitting on a couch",
    "target": "a dog sitting on a couch", "swap_token": "cat->dog"}`

## What "done" looks like for this module alone
1. Reconstruction LPIPS < 0.05 on the test set (proposal target).
2. CLIP directional similarity > 0.20 with background LPIPS < 0.10
   (proposal target).
3. Ablation table showing my time-dependent schedule beats vanilla P2P (fixed τ)
   on at least one metric, with qualitative examples.
4. Clean `Editor` class that teammates can import and call in 3 lines.
5. Walkthrough notebook that generates all the figures I need for the report.

## Rules for Claude Code
- If a step is unclear, ask before writing. Don't invent requirements.
- Run each file's smoke test before moving to the next step.
- If MPS throws on an op, fall back to CPU for that op with a one-line comment,
  don't spend 30 minutes debugging MPS — A100 is the real target.
- When you hit the novel-schedule step, *pause* and let me sanity-check the
  design before you implement all 5 schedule variants.
- Don't add tests for trivial getters. Do add a test that inversion +
  re-denoising recovers the image — that one has caught real bugs for everyone
  who's ever built this.
- No emojis in code or commits.

Start with Step 1. Write `sd_components.py` and a tiny script that loads the
model, encodes a prompt, and prints the embedding shape. Stop there and wait
for me.

