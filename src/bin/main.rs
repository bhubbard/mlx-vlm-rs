use clap::{Parser, Subcommand};
use mlx_rs::Array;
use mlx_vlm_rs::config::{GenerationConfig, VlmArchitecture, VlmConfig};
use mlx_vlm_rs::pipeline::VlmPipeline;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "mlx-vlm",
    about = "Native Apple Silicon Vision-Language Model Engine using MLX",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a description or answer questions about an image
    Generate {
        /// Path to the input image (JPEG or PNG)
        #[arg(short, long)]
        image: Option<String>,

        /// Text prompt for the vision-language model
        #[arg(short, long, default_value = "Describe this image in detail.")]
        prompt: String,

        /// Model architecture (paligemma, qwen2vl, smolvlm, llava, tiny)
        #[arg(short, long, default_value = "paligemma")]
        arch: String,

        /// Maximum new tokens to generate
        #[arg(short = 'm', long, default_value_t = 64)]
        max_tokens: usize,

        /// Sampling temperature (0.0 = greedy)
        #[arg(short, long, default_value_t = 0.7)]
        temperature: f32,

        /// Top-p (nucleus) sampling cutoff
        #[arg(long, default_value_t = 0.9)]
        top_p: f32,
    },

    /// Run benchmark measuring vision encoding, prefill, and decode tokens/sec
    Bench {
        /// Model architecture (paligemma, qwen2vl, smolvlm, llava, tiny)
        #[arg(short, long, default_value = "tiny")]
        arch: String,

        /// Number of tokens to decode in benchmark
        #[arg(short = 'n', long, default_value_t = 32)]
        num_tokens: usize,
    },

    /// Display architectural specs and parameters for a VLM config
    Info {
        /// Model architecture (paligemma, qwen2vl, smolvlm, llava, tiny)
        #[arg(short, long, default_value = "paligemma")]
        arch: String,
    },
}

fn get_config(arch: &str) -> VlmConfig {
    match arch.to_lowercase().as_str() {
        "qwen2vl" | "qwen" => VlmConfig::preset(VlmArchitecture::Qwen2VL),
        "smolvlm" | "smol" => VlmConfig::preset(VlmArchitecture::SmolVLM),
        "llava" => VlmConfig::preset(VlmArchitecture::Llava),
        "tiny" => VlmConfig::preset(VlmArchitecture::Tiny),
        _ => VlmConfig::preset(VlmArchitecture::PaliGemma),
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            image,
            prompt,
            arch,
            max_tokens,
            temperature,
            top_p,
        } => {
            println!("==> Loading MLX Vision-Language Model ({})", arch);
            let config = get_config(&arch);
            let mut pipeline = VlmPipeline::new(config)?;

            let image_arr = if let Some(ref img_path) = image {
                println!("==> Processing image: {}", img_path);
                Some(pipeline.load_image(img_path)?)
            } else {
                println!("==> No image provided, running text-only mode");
                None
            };

            let gen_config = GenerationConfig {
                max_new_tokens: max_tokens,
                temperature,
                top_p,
                ..Default::default()
            };

            println!("==> Prompt: \"{}\"", prompt);
            println!("==> Generating response:\n");

            // Mock tokenize prompt tokens: [1, 2, 3, image_token, 4, 5]
            let mut prompt_tokens = vec![101, 102, 103];
            if image.is_some() {
                prompt_tokens.insert(0, pipeline.config.image_token_id as i32);
            }

            let start = Instant::now();
            let mut token_count = 0;

            let tokens = pipeline.generate(
                image_arr.as_ref(),
                &prompt_tokens,
                &gen_config,
                |tok| {
                    token_count += 1;
                    print!("[tok:{}] ", tok);
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                    true
                },
            )?;

            let elapsed = start.elapsed();
            println!("\n\n==> Done! Generated {} tokens in {:.2}s ({:.1} tok/s)",
                tokens.len(),
                elapsed.as_secs_f32(),
                tokens.len() as f32 / elapsed.as_secs_f32().max(0.001)
            );
        }

        Commands::Bench { arch, num_tokens } => {
            println!("==> Benchmarking MLX Vision-Language Model ({})", arch);
            let config = get_config(&arch);
            let mut pipeline = VlmPipeline::new(config.clone())?;

            // Generate synthetic image
            let dummy_img = Array::zeros::<f32>(&[
                1,
                config.vision_image_size as i32,
                config.vision_image_size as i32,
                3,
            ])?;

            println!("1. Benchmarking Vision Encoder + MultiModal Projector...");
            let v_start = Instant::now();
            let vision_embeds = pipeline.encode_image(&dummy_img)?;
            let v_time = v_start.elapsed();
            println!("   Vision encoding time: {:.2} ms (embed shape: {:?})",
                v_time.as_secs_f32() * 1000.0,
                vision_embeds.shape()
            );

            println!("2. Benchmarking Multi-Modal Prefill + Autoregressive Decode ({} tokens)...", num_tokens);
            let gen_config = GenerationConfig {
                max_new_tokens: num_tokens,
                temperature: 0.0,
                ..Default::default()
            };

            let prompt_tokens = vec![config.image_token_id as i32, 1, 2, 3, 4, 5];
            let gen_start = Instant::now();
            let out_tokens = pipeline.generate(
                Some(&dummy_img),
                &prompt_tokens,
                &gen_config,
                |_| true,
            )?;
            let gen_time = gen_start.elapsed();

            let tok_per_sec = out_tokens.len() as f32 / gen_time.as_secs_f32().max(0.001);
            println!("   Generated {} tokens in {:.2} ms ({:.2} tok/s)",
                out_tokens.len(),
                gen_time.as_secs_f32() * 1000.0,
                tok_per_sec
            );
        }

        Commands::Info { arch } => {
            let config = get_config(&arch);
            println!("============================================================");
            println!(" MLX-VLM Architecture Specifications: {:?}", config.arch);
            println!("============================================================");
            println!(" Vision Backbone:");
            println!("   Image Size:         {}x{}", config.vision_image_size, config.vision_image_size);
            println!("   Patch Size:         {}x{}", config.vision_patch_size, config.vision_patch_size);
            println!("   Patches Count:      {}", config.num_patches());
            println!("   Vision Dim:         {}", config.vision_hidden_size);
            println!("   Vision Layers:      {}", config.vision_num_layers);
            println!("   Vision Heads:       {}", config.vision_num_heads);
            println!(" Language Backbone:");
            println!("   Hidden Size:        {}", config.text_hidden_size);
            println!("   Layers:             {}", config.text_num_layers);
            println!("   Attention Heads:    {}", config.text_num_heads);
            println!("   Intermediate Dim:   {}", config.text_intermediate_size);
            println!("   Vocab Size:         {}", config.text_vocab_size);
            println!("   Image Token ID:     {}", config.image_token_id);
            println!("============================================================");
        }
    }

    Ok(())
}
