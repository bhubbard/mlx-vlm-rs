use mlx_rs::Array;
use mlx_vlm_rs::config::{GenerationConfig, VlmArchitecture, VlmConfig};
use mlx_vlm_rs::pipeline::VlmPipeline;

#[test]
fn test_pipeline_generate_multimodal() -> anyhow::Result<()> {
    let config = VlmConfig::preset(VlmArchitecture::Tiny);
    let mut pipeline = VlmPipeline::new(config.clone())?;

    // Create dummy 32x32 image
    let dummy_img = Array::zeros::<f32>(&[
        1,
        config.vision_image_size as i32,
        config.vision_image_size as i32,
        3,
    ])?;

    let prompt = vec![config.image_token_id as i32, 10, 20];
    let gen_config = GenerationConfig {
        max_new_tokens: 5,
        temperature: 0.0,
        stop_token_ids: vec![],
        ..Default::default()
    };

    let mut streamed_tokens = Vec::new();
    let tokens = pipeline.generate(
        Some(&dummy_img),
        &prompt,
        &gen_config,
        |t| {
            streamed_tokens.push(t);
            true
        },
    )?;

    assert_eq!(tokens.len(), 5);
    assert_eq!(streamed_tokens, tokens);
    Ok(())
}

#[test]
fn test_pipeline_generate_text_only() -> anyhow::Result<()> {
    let config = VlmConfig::preset(VlmArchitecture::Tiny);
    let mut pipeline = VlmPipeline::new(config)?;

    let prompt = vec![5, 6, 7];
    let gen_config = GenerationConfig {
        max_new_tokens: 4,
        temperature: 0.0,
        stop_token_ids: vec![],
        ..Default::default()
    };

    let tokens = pipeline.generate(None, &prompt, &gen_config, |_| true)?;
    assert_eq!(tokens.len(), 4);
    Ok(())
}
