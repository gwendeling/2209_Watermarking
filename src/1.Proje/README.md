## Hybrid DWT + Chaotic Scrambling + DCT Watermarking

This project implements a robust, blind image watermarking scheme that combines 1-level DWT, block-wise chaotic scrambling, and DCT-domain AC-coefficient embedding.

### Core Concept
- Perform 1-level DWT on the cover image; work in the LL sub-band only.
- Divide LL into non-overlapping blocks. Before DCT, scramble each block's pixels using a chaotic map (Logistic or Tent). This disperses spatial structure and adds a security key.
- Apply 2D-DCT to the scrambled block.
- The watermark (AI metadata image) can also be scrambled with the same chaotic map using a different key.
- Embed watermark bits by enforcing an inequality between a mid-frequency and a high-frequency AC coefficient, e.g., D(2,4) vs D(4,4). This favors imperceptibility and robustness to compression over DC-based schemes.
- Reconstruct with inverse DCT per block and inverse DWT to obtain the stego image.

Why this is robust and novel:
- The “scrambled-DCT” embedding domain spreads watermark energy, improving resistance to cropping and noise.
- AC-coefficient embedding offers a good trade-off between invisibility and robustness to JPEG and common attacks.

### What this implementation does
- Uses 1-level DWT with `haar` (configurable) and embeds only in `LL`.
- Splits `LL` into `BLOCK_SIZE x BLOCK_SIZE` blocks (default 8×8).
- Scrambles each block with a chaotic permutation derived from the chosen map and keys.
- Applies DCT to each scrambled block and embeds one bit by enforcing the relationship between coefficients D(2,4) and D(4,4) with threshold `EMBED_STRENGTH`.
- Leaves the spatial block in its scrambled state after inverse DCT, then performs inverse DWT to reconstruct.
- Extraction repeats DWT → block scramble → DCT → reads bit via the sign of D(2,4)−D(4,4), and reassembles the watermark.

Note on watermark scrambling:
- The config includes separate keys for watermark scrambling. The current embedding path uses the watermark bits directly (binarized and tiled). The code structure leaves room to add explicit watermark scrambling with different chaotic parameters if desired.

### Project Structure
- `src/1.Proje/run_embed.py`: Generates a sample cover if missing, ensures a synthetic AI-metadata watermark, runs embedding, prints PSNR.
- `src/1.Proje/run_extract.py`: Extracts the watermark from the stego image and saves it.
- `src/1.Proje/config.py`: Central configuration for paths, DWT, DCT block size, embedding strength, and chaotic map parameters.
- `src/1.Proje/embedder.py`: Implements DWT→scramble→DCT→AC-coefficient embedding.
- `src/1.Proje/extractor.py`: Implements the corresponding blind extraction.
- `src/1.Proje/chaos.py`, `src/1.Proje/utils.py`, `src/1.Proje/watermark_generator.py`: Utilities, chaotic permutations, and synthetic watermark generation.

### Installation
- Python 3.10+ recommended.
- Install dependencies:
```bash
pip install -r src/1.Proje/requirements.txt
```

### Usage
1) Configure paths and parameters in `src/1.Proje/config.py` as needed:
- `COVER_IMAGE_PATH`, `WATERMARK_IMAGE_PATH`, `STEGO_IMAGE_PATH`, `EXTRACTED_WATERMARK_PATH`
- `BLOCK_SIZE` (default 8), `EMBED_STRENGTH` (inequality threshold)
- `DWT_WAVELET` (e.g., "haar")
- Chaotic map and keys: `CHAOS_MAP` in {"logistic", "tent"}, seeds and map parameters for cover/watermark

2) Embed the watermark:
```bash
python src/1.Proje/run_embed.py
```
- If no cover exists at `COVER_IMAGE_PATH`, a synthetic 512×512 grayscale gradient is created.
- If no watermark exists, a synthetic AI metadata watermark is generated.
- Produces the stego image at `STEGO_IMAGE_PATH` and prints PSNR vs. cover.

3) Extract the watermark:
```bash
python src/1.Proje/run_extract.py
```
- Saves the extracted watermark to `EXTRACTED_WATERMARK_PATH` using `WATERMARK_WIDTH`×`WATERMARK_HEIGHT` from `config.py`.

### Algorithm Steps (detailed)
Embedding (cover `I`, watermark `W`):
1. Convert `I` to grayscale double; binarize `W` to bits in {0,1}.
2. 1-level DWT: `I → (LL, LH, HL, HH)` with wavelet `DWT_WAVELET`.
3. Partition `LL` into `BLOCK_SIZE×BLOCK_SIZE` blocks.
4. Generate a chaotic permutation `P` from `CHAOS_MAP`, `(seed=r0, r)`.
5. For each block `B` in `LL`:
   - Scramble pixels: `B' = scramble(B, P)`.
   - DCT: `D = DCT2(B')`.
   - Embed bit `b` by enforcing:
     - If `b=1`: ensure `D(2,4) ≥ D(4,4) + T`.
     - If `b=0`: ensure `D(4,4) ≥ D(2,4) + T`.
   - Inverse DCT to get `B*`; place back into `LL` at the same block location.
6. Inverse DWT with the original `(LH, HL, HH)` to reconstruct the stego image.

Extraction (blind):
1. Grayscale stego → 1-level DWT: `(LL_s, LH, HL, HH)`.
2. Partition `LL_s` into blocks; regenerate the same permutation `P` from keys and map.
3. For each block:
   - Scramble pixels, `B' = scramble(B, P)`.
   - DCT: `D`.
   - Read bit as `b̂ = 1` if `D(2,4)−D(4,4) ≥ 0`, else `0`.
4. Collect bits and reshape to `WATERMARK_HEIGHT × WATERMARK_WIDTH` (tile/truncate as needed) and save.

### Tips
- Increase `EMBED_STRENGTH` for more robustness (potentially lower PSNR), decrease for higher imperceptibility.
- `BLOCK_SIZE=8` is a common sweet spot; larger blocks may reduce localization robustness.
- Keep logistic map parameter `r` in a chaotic regime (≈3.57–4.0) and seeds in (0,1).

### Extending: Explicit Watermark Scrambling
- To add explicit watermark-bit scrambling, derive an independent permutation from `CHAOS_SEED_WM`/`CHAOS_R_WM` and permute the flattened watermark bitstream before embedding; invert that permutation after extraction when reconstructing the watermark image.

### License
- For academic and research use. Please cite or reference this repository if you build upon it.
