use mlx_rs::Array;
use mlx_vlm_rs::config::{VlmArchitecture, VlmConfig};
use mlx_vlm_rs::language::{rms_norm, CausalDecoderLayer, CausalLanguageModel, KVCache};

#[test]
fn test_rms_norm() -> anyhow::Result<()> {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let arr = Array::from_slice(&data, &[1, 4]);
    let normed = rms_norm(&arr, 1e-6)?;
    assert_eq!(normed.shape(), &[1, 4]);
    Ok(())
}

#[test]
fn test_kv_cache() -> anyhow::Result<()> {
    let mut cache = KVCache::new(2);
    let k1 = Array::zeros::<f32>(&[1, 2, 4, 16])?;
    let v1 = Array::zeros::<f32>(&[1, 2, 4, 16])?;

    let (full_k, full_v) = cache.update(0, &k1, &v1)?;
    assert_eq!(full_k.shape(), &[1, 2, 4, 16]);
    assert_eq!(full_v.shape(), &[1, 2, 4, 16]);

    // Append single new token
    let k2 = Array::zeros::<f32>(&[1, 2, 1, 16])?;
    let v2 = Array::zeros::<f32>(&[1, 2, 1, 16])?;
    let (full_k2, full_v2) = cache.update(0, &k2, &v2)?;
    assert_eq!(full_k2.shape(), &[1, 2, 5, 16]);
    assert_eq!(full_v2.shape(), &[1, 2, 5, 16]);

    cache.reset();
    assert!(cache.key_cache[0].is_none());
    Ok(())
}

#[test]
fn test_causal_decoder_layer() -> anyhow::Result<()> {
    let mut layer = CausalDecoderLayer::new(64, 2, 128, 0)?;
    let mut cache = KVCache::new(1);
    let x = Array::zeros::<f32>(&[1, 6, 64])?;

    let out = layer.forward(&x, Some(&mut cache))?;
    assert_eq!(out.shape(), &[1, 6, 64]);
    Ok(())
}

#[test]
fn test_causal_language_model_forward() -> anyhow::Result<()> {
    let config = VlmConfig::preset(VlmArchitecture::Tiny);
    let mut lm = CausalLanguageModel::new(config.clone())?;
    let mut cache = KVCache::new(config.text_num_layers);

    let x = Array::zeros::<f32>(&[1, 8, config.text_hidden_size as i32])?;
    let logits = lm.forward(&x, Some(&mut cache))?;

    // Logits should be [1, vocab_size]
    assert_eq!(logits.shape(), &[1, 1, config.text_vocab_size as i32]);
    Ok(())
}
