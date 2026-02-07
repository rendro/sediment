use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};
use tracing::info;

use crate::error::{Result, SedimentError};

/// Default model to use for embeddings
pub const DEFAULT_MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// Embedding dimension for the default model
pub const EMBEDDING_DIM: usize = 384;

/// Embedder for converting text to vectors.
///
/// # Thread Safety
/// `Embedder` wraps `BertModel` and `Tokenizer` which are `Send + Sync`.
/// It is shared via `Arc<Embedder>` across the server. All inference runs
/// synchronously on the calling thread (via `rt.block_on`), so there are
/// no cross-thread mutation concerns.
pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    /// Create a new embedder, downloading the model if necessary
    pub fn new() -> Result<Self> {
        Self::with_model(DEFAULT_MODEL_ID)
    }

    /// Create an embedder with a specific model
    pub fn with_model(model_id: &str) -> Result<Self> {
        info!("Loading embedding model: {}", model_id);

        let device = Device::Cpu;
        let (model_path, tokenizer_path, config_path) = download_model(model_id)?;

        // Load config
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| SedimentError::ModelLoading(format!("Failed to read config: {}", e)))?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| SedimentError::ModelLoading(format!("Failed to parse config: {}", e)))?;

        // Load tokenizer
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| SedimentError::Tokenizer(format!("Failed to load tokenizer: {}", e)))?;

        // Configure tokenizer for batch processing
        let padding = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        let truncation = TruncationParams {
            max_length: 512,
            ..Default::default()
        };
        tokenizer.with_padding(Some(padding));
        tokenizer
            .with_truncation(Some(truncation))
            .map_err(|e| SedimentError::Tokenizer(format!("Failed to set truncation: {}", e)))?;

        // Load model weights into memory and verify integrity.
        // Uses from_buffered_safetensors instead of unsafe from_mmaped_safetensors
        // to eliminate the TOCTOU window between hash verification and file use.
        // The same bytes that pass SHA-256 verification are the ones parsed.
        let model_bytes = std::fs::read(&model_path).map_err(|e| {
            SedimentError::ModelLoading(format!("Failed to read model weights: {}", e))
        })?;
        verify_bytes_hash(&model_bytes, MODEL_SHA256, "model.safetensors")?;
        let vb = VarBuilder::from_buffered_safetensors(model_bytes, DTYPE, &device)
            .map_err(|e| SedimentError::ModelLoading(format!("Failed to load weights: {}", e)))?;

        let model = BertModel::load(vb, &config)
            .map_err(|e| SedimentError::ModelLoading(format!("Failed to load model: {}", e)))?;

        info!("Embedding model loaded successfully");

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Embed a single text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text])?;
        embeddings.into_iter().next().ok_or_else(|| {
            SedimentError::Embedding("embed_batch returned empty result for non-empty input".into())
        })
    }

    /// Embed multiple texts at once
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| SedimentError::Tokenizer(format!("Tokenization failed: {}", e)))?;

        let token_ids: Vec<Vec<u32>> = encodings.iter().map(|e| e.get_ids().to_vec()).collect();

        let attention_masks: Vec<Vec<u32>> = encodings
            .iter()
            .map(|e| e.get_attention_mask().to_vec())
            .collect();

        let token_type_ids: Vec<Vec<u32>> = encodings
            .iter()
            .map(|e| e.get_type_ids().to_vec())
            .collect();

        // Convert to tensors
        let batch_size = texts.len();
        let seq_len = token_ids[0].len();

        let token_ids_flat: Vec<u32> = token_ids.into_iter().flatten().collect();
        let attention_mask_flat: Vec<u32> = attention_masks.into_iter().flatten().collect();
        let token_type_ids_flat: Vec<u32> = token_type_ids.into_iter().flatten().collect();

        let token_ids_tensor =
            Tensor::from_vec(token_ids_flat, (batch_size, seq_len), &self.device).map_err(|e| {
                SedimentError::Embedding(format!("Failed to create token tensor: {}", e))
            })?;

        let attention_mask_tensor =
            Tensor::from_vec(attention_mask_flat, (batch_size, seq_len), &self.device).map_err(
                |e| SedimentError::Embedding(format!("Failed to create mask tensor: {}", e)),
            )?;

        let token_type_ids_tensor =
            Tensor::from_vec(token_type_ids_flat, (batch_size, seq_len), &self.device).map_err(
                |e| SedimentError::Embedding(format!("Failed to create type tensor: {}", e)),
            )?;

        // Run model
        let embeddings = self
            .model
            .forward(
                &token_ids_tensor,
                &token_type_ids_tensor,
                Some(&attention_mask_tensor),
            )
            .map_err(|e| SedimentError::Embedding(format!("Model forward failed: {}", e)))?;

        // Mean pooling with attention mask
        let attention_mask_f32 = attention_mask_tensor
            .to_dtype(DType::F32)
            .map_err(|e| SedimentError::Embedding(format!("Mask conversion failed: {}", e)))?
            .unsqueeze(2)
            .map_err(|e| SedimentError::Embedding(format!("Unsqueeze failed: {}", e)))?;

        let masked_embeddings = embeddings
            .broadcast_mul(&attention_mask_f32)
            .map_err(|e| SedimentError::Embedding(format!("Broadcast mul failed: {}", e)))?;

        let sum_embeddings = masked_embeddings
            .sum(1)
            .map_err(|e| SedimentError::Embedding(format!("Sum failed: {}", e)))?;

        let sum_mask = attention_mask_f32
            .sum(1)
            .map_err(|e| SedimentError::Embedding(format!("Mask sum failed: {}", e)))?;

        let mean_embeddings = sum_embeddings
            .broadcast_div(&sum_mask)
            .map_err(|e| SedimentError::Embedding(format!("Division failed: {}", e)))?;

        // L2 normalize embeddings
        let final_embeddings = normalize_l2(&mean_embeddings)?;

        // Convert to Vec<Vec<f32>>
        let embeddings_vec: Vec<Vec<f32>> = final_embeddings
            .to_vec2()
            .map_err(|e| SedimentError::Embedding(format!("Tensor to vec failed: {}", e)))?;

        Ok(embeddings_vec)
    }

    /// Get the embedding dimension
    pub fn dimension(&self) -> usize {
        EMBEDDING_DIM
    }
}

/// Download model files from Hugging Face Hub
fn download_model(model_id: &str) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let api = ApiBuilder::from_env()
        .with_progress(true)
        .build()
        .map_err(|e| SedimentError::ModelLoading(format!("Failed to create HF API: {}", e)))?;

    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        "e4ce9877abf3edfe10b0d82785e83bdcb973e22e".to_string(),
    ));

    let model_path = repo
        .get("model.safetensors")
        .map_err(|e| SedimentError::ModelLoading(format!("Failed to download model: {}", e)))?;

    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| SedimentError::ModelLoading(format!("Failed to download tokenizer: {}", e)))?;

    let config_path = repo
        .get("config.json")
        .map_err(|e| SedimentError::ModelLoading(format!("Failed to download config: {}", e)))?;

    // Verify integrity of tokenizer and config files using hardcoded SHA-256 hashes.
    // The model weights file is verified later via verify_bytes_hash on the actual
    // bytes passed to from_buffered_safetensors, eliminating any TOCTOU window.
    verify_file_hash(&tokenizer_path, TOKENIZER_SHA256, "tokenizer.json")?;
    verify_file_hash(&config_path, CONFIG_SHA256, "config.json")?;
    info!("Tokenizer and config integrity verified (SHA-256)");

    Ok((model_path, tokenizer_path, config_path))
}

/// Expected SHA-256 hashes for the pinned revision.
const MODEL_SHA256: &str = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db";
const TOKENIZER_SHA256: &str = "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037";
const CONFIG_SHA256: &str = "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41";

/// Verify the SHA-256 hash of a file against an expected value.
fn verify_file_hash(path: &std::path::Path, expected: &str, file_label: &str) -> Result<()> {
    use sha2::{Digest, Sha256};

    let file_bytes = std::fs::read(path).map_err(|e| {
        SedimentError::ModelLoading(format!(
            "Failed to read {} for hash verification: {}",
            file_label, e
        ))
    })?;

    let hash = Sha256::digest(&file_bytes);
    let hex_hash = format!("{:x}", hash);

    if hex_hash != expected {
        return Err(SedimentError::ModelLoading(format!(
            "{} integrity check failed: expected SHA-256 {}, got {}",
            file_label, expected, hex_hash
        )));
    }

    Ok(())
}

/// Verify the SHA-256 hash of in-memory bytes against an expected value.
///
/// This is used for model weights to eliminate the TOCTOU window: the same bytes
/// that are hash-verified are the ones passed to the safetensors parser.
fn verify_bytes_hash(data: &[u8], expected: &str, file_label: &str) -> Result<()> {
    use sha2::{Digest, Sha256};

    let hash = Sha256::digest(data);
    let hex_hash = format!("{:x}", hash);

    if hex_hash != expected {
        return Err(SedimentError::ModelLoading(format!(
            "{} integrity check failed: expected SHA-256 {}, got {}",
            file_label, expected, hex_hash
        )));
    }

    Ok(())
}

/// L2 normalize a tensor
fn normalize_l2(tensor: &Tensor) -> Result<Tensor> {
    let norm = tensor
        .sqr()
        .map_err(|e| SedimentError::Embedding(format!("Sqr failed: {}", e)))?
        .sum_keepdim(1)
        .map_err(|e| SedimentError::Embedding(format!("Sum keepdim failed: {}", e)))?
        .sqrt()
        .map_err(|e| SedimentError::Embedding(format!("Sqrt failed: {}", e)))?;

    tensor
        .broadcast_div(&norm)
        .map_err(|e| SedimentError::Embedding(format!("Normalize div failed: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires model download
    fn test_embedder() -> Result<()> {
        let embedder = Embedder::new()?;

        let text = "Hello, world!";
        let embedding = embedder.embed(text)?;

        assert_eq!(embedding.len(), EMBEDDING_DIM);

        // Check normalization (L2 norm should be ~1.0)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);

        Ok(())
    }

    #[test]
    #[ignore] // Requires model download
    fn test_batch_embedding() -> Result<()> {
        let embedder = Embedder::new()?;

        let texts = vec!["Hello", "World", "Test sentence"];
        let embeddings = embedder.embed_batch(&texts)?;

        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), EMBEDDING_DIM);
        }

        Ok(())
    }

    #[test]
    fn test_verify_bytes_hash_correct() {
        let data = b"hello world";
        let expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        assert!(verify_bytes_hash(data, expected, "test").is_ok());
    }

    #[test]
    fn test_verify_bytes_hash_incorrect() {
        let data = b"hello world";
        let wrong = "0000000000000000000000000000000000000000000000000000000000000000";
        let err = verify_bytes_hash(data, wrong, "test").unwrap_err();
        assert!(err.to_string().contains("integrity check failed"));
    }

    #[test]
    fn test_verify_bytes_hash_empty() {
        let data = b"";
        // SHA-256 of empty input
        let expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert!(verify_bytes_hash(data, expected, "empty").is_ok());
    }
}
