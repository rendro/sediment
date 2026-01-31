use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use sha2::{Digest, Sha256};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};
use tracing::info;

use crate::error::{Result, SedimentError};

/// Default model to use for embeddings
pub const DEFAULT_MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// Embedding dimension for the default model
pub const EMBEDDING_DIM: usize = 384;

/// Embedder for converting text to vectors
pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    normalize: bool,
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

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], DTYPE, &device).map_err(|e| {
                SedimentError::ModelLoading(format!("Failed to load weights: {}", e))
            })?
        };

        let model = BertModel::load(vb, &config)
            .map_err(|e| SedimentError::ModelLoading(format!("Failed to load model: {}", e)))?;

        info!("Embedding model loaded successfully");

        Ok(Self {
            model,
            tokenizer,
            device,
            normalize: true,
        })
    }

    /// Embed a single text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text])?;
        Ok(embeddings.into_iter().next().unwrap())
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

        // Normalize if requested
        let final_embeddings = if self.normalize {
            normalize_l2(&mean_embeddings)?
        } else {
            mean_embeddings
        };

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

    // TOFU: verify model integrity
    verify_tofu_hash(&model_path, "model.safetensors")?;
    verify_tofu_hash(&tokenizer_path, "tokenizer.json")?;
    verify_tofu_hash(&config_path, "config.json")?;

    Ok((model_path, tokenizer_path, config_path))
}

/// Compute SHA256 hash of a file.
fn sha256_file(path: &Path) -> Result<String> {
    let data = std::fs::read(path)
        .map_err(|e| SedimentError::ModelLoading(format!("Failed to read file for hash: {}", e)))?;
    let hash = Sha256::digest(&data);
    Ok(format!("{:x}", hash))
}

/// Trust-on-first-use hash verification for model files.
///
/// On first download, stores the SHA256 hash in a `.sha256` sidecar file next to the model.
/// On subsequent loads, verifies the file still matches the stored hash.
fn verify_tofu_hash(file_path: &Path, label: &str) -> Result<()> {
    let hash_path = file_path.with_extension("sha256");
    let current_hash = sha256_file(file_path)?;

    if hash_path.exists() {
        let stored_hash = std::fs::read_to_string(&hash_path)
            .map_err(|e| SedimentError::ModelLoading(format!("Failed to read hash file: {}", e)))?;
        let stored_hash = stored_hash.trim();
        if stored_hash != current_hash {
            return Err(SedimentError::ModelLoading(format!(
                "TOFU integrity check failed for {}: expected {}, got {}",
                label, stored_hash, current_hash
            )));
        }
        info!("TOFU hash verified for {}", label);
    } else {
        std::fs::write(&hash_path, &current_hash).map_err(|e| {
            SedimentError::ModelLoading(format!("Failed to write hash file: {}", e))
        })?;
        info!("TOFU hash recorded for {}: {}", label, current_hash);
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
}
