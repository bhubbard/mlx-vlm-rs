use mlx_rs::Array;
use mlx_vlm_rs::fusion::MultiModalFusion;

#[test]
fn test_fusion_replacement() -> anyhow::Result<()> {
    let mut fusion = MultiModalFusion::new(1000, 64, 999)?;

    // Text tokens: [10, 999, 20] (999 is image placeholder)
    let token_ids = vec![10, 999, 20];
    let token_ids_arr = Array::from_slice(&token_ids, &[1, 3]);
    let text_embeds = fusion.embed_tokens(&token_ids_arr)?;
    assert_eq!(text_embeds.shape(), &[1, 3, 64]);

    // Vision embeds: 4 patches [1, 4, 64]
    let vision_embeds = Array::zeros::<f32>(&[1, 4, 64])?;

    let fused = fusion.fuse(&token_ids, &text_embeds, Some(&vision_embeds))?;

    // Expected length: 1 (prefix: 10) + 4 (vision) + 1 (suffix: 20) = 6 tokens
    assert_eq!(fused.shape(), &[1, 6, 64]);
    Ok(())
}

#[test]
fn test_prefix_fusion_without_placeholder() -> anyhow::Result<()> {
    let mut fusion = MultiModalFusion::new(1000, 64, 999)?;

    // Prompt tokens without 999
    let token_ids = vec![10, 11, 12];
    let token_ids_arr = Array::from_slice(&token_ids, &[1, 3]);
    let text_embeds = fusion.embed_tokens(&token_ids_arr)?;

    let vision_embeds = Array::zeros::<f32>(&[1, 4, 64])?;
    let fused = fusion.fuse(&token_ids, &text_embeds, Some(&vision_embeds))?;

    // Vision is prepended: 4 + 3 = 7 tokens
    assert_eq!(fused.shape(), &[1, 7, 64]);
    Ok(())
}
