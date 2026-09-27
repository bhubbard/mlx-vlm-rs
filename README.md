# mlx-vlm-rs 👁️⚡

[![Crates.io](https://img.shields.io/badge/crates.io-v0.0.1-orange.svg)](https://crates.io/crates/mlx-vlm-rs)
[![Documentation](https://img.shields.io/badge/docs-GitHub_Pages-blue.svg)](http://code.brandonhubbard.com/mlx-vlm-rs/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MLX_Accelerated-black.svg?logo=apple)](https://github.com/ml-explore/mlx)

Native Apple Silicon Rust engine for Vision-Language Models (**Qwen2-VL**, **PaliGemma**, **SmolVLM**, **LLaVA**) using MLX.

A 100% pure Rust port of [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm), eliminating all Python runtime dependencies while delivering blazing-fast multi-modal prefill and autoregressive token decoding directly in Apple unified memory.

---

## 🚀 Features

- **Multi-Modal Vision Backbones**: SigLIP and ViT patch extraction, projection layers, and multi-modal token interleaving.
- **Autoregressive Causal LM**: Rotary/RoPE & Causal Attention with persistent multi-layer `KVCache`.
- **Flexible Token Fusion**: Token replacement (in-place `<image>` substitution) and prefix fusion.
- **High-Performance Sampling**: Greedy, temperature scaling, top-$p$ nucleus, top-$k$, and repetition penalties.
- **Preconfigured Presets**:
  - `PaliGemma-3B-pt-224`
  - `Qwen2-VL-2B-Instruct`
  - `SmolVLM-500M-Instruct`
  - `LLaVA-1.5-7B`
  - `Tiny-VLM` (ideal for instant unit tests & micro-agents)
- **Interactive Documentation**: Explore model architectures and test token streaming at [code.brandonhubbard.com/mlx-vlm-rs](http://code.brandonhubbard.com/mlx-vlm-rs/).

---

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mlx-vlm-rs = "0.0.1"
```

Or install the standalone CLI:

```bash
cargo install mlx-vlm-rs
```

---

## 🛠️ CLI Usage

```bash
# Display model specifications
mlx-vlm info --arch paligemma

# Run performance benchmark
mlx-vlm bench --arch tiny --num-tokens 32

# Multi-modal inference on an image
mlx-vlm generate \
  --image ./diagram.png \
  --prompt "Describe the key architectural blocks." \
  --arch paligemma \
  --temperature 0.7
```

---

## 💻 Rust SDK Example

```rust
use mlx_vlm_rs::{VlmPipeline, VlmConfig, VlmArchitecture, GenerationConfig};

fn main() -> anyhow::Result<()> {
    // 1. Initialize pipeline with PaliGemma architecture
    let config = VlmConfig::preset(VlmArchitecture::PaliGemma);
    let mut pipeline = VlmPipeline::new(config)?;

    // 2. Load input image
    let img = pipeline.load_image("photo.jpg")?;

    // 3. Define prompt tokens (including <image> token placeholder)
    let prompt_tokens = vec![257152, 101, 102, 103];

    // 4. Autoregressively stream tokens
    let gen_config = GenerationConfig {
        max_new_tokens: 64,
        temperature: 0.7,
        ..Default::default()
    };

    let generated = pipeline.generate(
        Some(&img),
        &prompt_tokens,
        &gen_config,
        |token_id| {
            print!("[{}] ", token_id);
            true // Continue generation
        },
    )?;

    println!("\nGenerated {} tokens successfully!", generated.len());
    Ok(())
}
```

---

## 🧪 Testing

Run all unit and integration tests:

```bash
cargo test
```

---

## 📄 License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))
