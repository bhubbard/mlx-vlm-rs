use mlx_rs::Array;
use mlx_vlm_rs::config::{VlmArchitecture, VlmConfig};
use mlx_vlm_rs::vision::{MultiModalProjector, SigLipVisionEncoder};

#[test]
fn test_vision_encoder_tiny_forward() -> anyhow::Result<()> {
    let config = VlmConfig::preset(VlmArchitecture::Tiny);
    let mut encoder = SigLipVisionEncoder::new(config.clone())?;

    // Create synthetic 4D image [1, 32, 32, 3]
    let img = Array::zeros::<f32>(&[
        1,
        config.vision_image_size as i32,
        config.vision_image_size as i32,
        3,
    ])?;

    let out = encoder.forward(&img)?;
    let expected_patches = config.num_patches();
    assert_eq!(out.shape(), &[1, expected_patches as i32, config.vision_hidden_size as i32]);
    Ok(())
}

#[test]
fn test_multimodal_projector() -> anyhow::Result<()> {
    let mut proj = MultiModalProjector::new(64, 128)?;
    let x = Array::zeros::<f32>(&[1, 4, 64])?;
    let out = proj.forward(&x)?;
    assert_eq!(out.shape(), &[1, 4, 128]);
    Ok(())
}
