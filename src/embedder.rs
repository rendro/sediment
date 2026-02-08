use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};
use tracing::info;

use crate::error::{Result, SedimentError};

/// Default embedding dimension (384-dim for small models).
/// Kept as a pub const for backward compatibility; prefer `Embedder::dimension()`.
pub const EMBEDDING_DIM: usize = 384;

/// Supported embedding models.
///
/// Each variant carries model metadata: HF repo ID, pinned revision,
/// SHA-256 hashes for integrity verification, and prefix functions
/// for asymmetric query/document embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbeddingModel {
    /// sentence-transformers/all-MiniLM-L6-v2 (default, no prefixes, 384-dim)
    #[default]
    AllMiniLmL6V2,
    /// intfloat/e5-small-v2 (query: "query: {text}", document: "passage: {text}", 384-dim)
    E5SmallV2,
    /// BAAI/bge-small-en-v1.5 (query prefix, no doc prefix, 384-dim)
    BgeSmallEnV15,
    /// BAAI/bge-base-en-v1.5 (query prefix, no doc prefix, 768-dim)
    BgeBaseEnV15,
}

impl EmbeddingModel {
    /// Embedding dimension for this model
    pub fn embedding_dim(&self) -> usize {
        match self {
            Self::AllMiniLmL6V2 | Self::E5SmallV2 | Self::BgeSmallEnV15 => 384,
            Self::BgeBaseEnV15 => 768,
        }
    }

    /// Hugging Face model repository ID
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::AllMiniLmL6V2 => "sentence-transformers/all-MiniLM-L6-v2",
            Self::E5SmallV2 => "intfloat/e5-small-v2",
            Self::BgeSmallEnV15 => "BAAI/bge-small-en-v1.5",
            Self::BgeBaseEnV15 => "BAAI/bge-base-en-v1.5",
        }
    }

    /// Pinned git revision for reproducible downloads
    pub fn revision(&self) -> &'static str {
        match self {
            Self::AllMiniLmL6V2 => "e4ce9877abf3edfe10b0d82785e83bdcb973e22e",
            Self::E5SmallV2 => "ffb93f3bd4047442299a41ebb6fa998a38507c52",
            Self::BgeSmallEnV15 => "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            Self::BgeBaseEnV15 => "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        }
    }

    /// Expected SHA-256 hash of model.safetensors
    pub fn model_sha256(&self) -> &'static str {
        match self {
            Self::AllMiniLmL6V2 => {
                "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
            }
            Self::E5SmallV2 => "45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1",
            Self::BgeSmallEnV15 => {
                "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad"
            }
            Self::BgeBaseEnV15 => {
                "c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7"
            }
        }
    }

    /// Expected SHA-256 hash of tokenizer.json
    pub fn tokenizer_sha256(&self) -> &'static str {
        match self {
            Self::AllMiniLmL6V2 => {
                "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"
            }
            Self::E5SmallV2 => "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
            Self::BgeSmallEnV15 => {
                "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
            }
            Self::BgeBaseEnV15 => {
                "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
            }
        }
    }

    /// Expected SHA-256 hash of config.json
    pub fn config_sha256(&self) -> &'static str {
        match self {
            Self::AllMiniLmL6V2 => {
                "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41"
            }
            Self::E5SmallV2 => "5dfb0363cd0243be179c03bcaafd1542d0fbb95e8cbcf575fff3e229342adc2f",
            Self::BgeSmallEnV15 => {
                "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750"
            }
            Self::BgeBaseEnV15 => {
                "bc00af31a4a31b74040d73370aa83b62da34c90b75eb77bfa7db039d90abd591"
            }
        }
    }

    /// Apply query prefix for asymmetric search
    pub fn prefix_query<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        match self {
            Self::AllMiniLmL6V2 => std::borrow::Cow::Borrowed(text),
            Self::E5SmallV2 => std::borrow::Cow::Owned(format!("query: {text}")),
            Self::BgeSmallEnV15 | Self::BgeBaseEnV15 => std::borrow::Cow::Owned(format!(
                "Represent this sentence for searching relevant passages: {text}"
            )),
        }
    }

    /// Apply document prefix for asymmetric search
    pub fn prefix_document<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        match self {
            Self::AllMiniLmL6V2 => std::borrow::Cow::Borrowed(text),
            Self::E5SmallV2 => std::borrow::Cow::Owned(format!("passage: {text}")),
            Self::BgeSmallEnV15 | Self::BgeBaseEnV15 => std::borrow::Cow::Borrowed(text),
        }
    }

    /// Parse from env var value (e.g. "e5-small-v2", "bge-small-en-v1.5")
    pub fn from_env_str(s: &str) -> Option<Self> {
        match s {
            "all-MiniLM-L6-v2" | "all-minilm-l6-v2" => Some(Self::AllMiniLmL6V2),
            "e5-small-v2" => Some(Self::E5SmallV2),
            "bge-small-en-v1.5" => Some(Self::BgeSmallEnV15),
            "bge-base-en-v1.5" => Some(Self::BgeBaseEnV15),
            _ => None,
        }
    }
}

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
    embedding_model: EmbeddingModel,
}

impl Embedder {
    /// Create a new embedder, downloading the model if necessary.
    ///
    /// Reads `SEDIMENT_EMBEDDING_MODEL` env var to select the model.
    /// Falls back to AllMiniLmL6V2 if unset or unrecognized.
    pub fn new() -> Result<Self> {
        let embedding_model = std::env::var("SEDIMENT_EMBEDDING_MODEL")
            .ok()
            .and_then(|s| EmbeddingModel::from_env_str(&s))
            .unwrap_or_default();
        Self::with_embedding_model(embedding_model)
    }

    /// Create an embedder with a specific embedding model variant
    pub fn with_embedding_model(embedding_model: EmbeddingModel) -> Result<Self> {
        let model_id = embedding_model.model_id();
        info!("Loading embedding model: {}", model_id);

        let device = Device::Cpu;
        let (model_path, tokenizer_path, config_path) =
            download_model(model_id, embedding_model.revision())?;

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

        // Verify integrity of tokenizer and config files using hardcoded SHA-256 hashes.
        // Skip verification for models with empty (placeholder) hashes.
        let tokenizer_hash = embedding_model.tokenizer_sha256();
        if !tokenizer_hash.is_empty() {
            verify_file_hash(&tokenizer_path, tokenizer_hash, "tokenizer.json")?;
        }
        let config_hash = embedding_model.config_sha256();
        if !config_hash.is_empty() {
            verify_file_hash(&config_path, config_hash, "config.json")?;
        }
        if !tokenizer_hash.is_empty() || !config_hash.is_empty() {
            info!("Tokenizer and config integrity verified (SHA-256)");
        }

        // Load model weights into memory and verify integrity.
        // Uses from_buffered_safetensors instead of unsafe from_mmaped_safetensors
        // to eliminate the TOCTOU window between hash verification and file use.
        // The same bytes that pass SHA-256 verification are the ones parsed.
        let model_bytes = std::fs::read(&model_path).map_err(|e| {
            SedimentError::ModelLoading(format!("Failed to read model weights: {}", e))
        })?;
        let model_hash = embedding_model.model_sha256();
        if !model_hash.is_empty() {
            verify_bytes_hash(&model_bytes, model_hash, "model.safetensors")?;
        }
        let vb = VarBuilder::from_buffered_safetensors(model_bytes, DTYPE, &device)
            .map_err(|e| SedimentError::ModelLoading(format!("Failed to load weights: {}", e)))?;

        let model = BertModel::load(vb, &config)
            .map_err(|e| SedimentError::ModelLoading(format!("Failed to load model: {}", e)))?;

        info!("Embedding model loaded successfully");

        Ok(Self {
            model,
            tokenizer,
            device,
            embedding_model,
        })
    }

    /// Embed a single text (raw, no prefix applied)
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text])?;
        embeddings.into_iter().next().ok_or_else(|| {
            SedimentError::Embedding("embed_batch returned empty result for non-empty input".into())
        })
    }

    /// Embed multiple texts at once (raw, no prefix applied)
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

    /// Embed a single query text with model-specific query prefix
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let prefixed = self.embedding_model.prefix_query(text);
        self.embed(&prefixed)
    }

    /// Embed a single document text with model-specific document prefix
    pub fn embed_document(&self, text: &str) -> Result<Vec<f32>> {
        let prefixed = self.embedding_model.prefix_document(text);
        self.embed(&prefixed)
    }

    /// Embed multiple document texts with model-specific document prefix
    pub fn embed_document_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| self.embedding_model.prefix_document(t).into_owned())
            .collect();
        let refs: Vec<&str> = prefixed.iter().map(|s| s.as_str()).collect();
        self.embed_batch(&refs)
    }

    /// Get the embedding dimension for the active model
    pub fn dimension(&self) -> usize {
        self.embedding_model.embedding_dim()
    }

    /// Get the active embedding model
    pub fn embedding_model(&self) -> EmbeddingModel {
        self.embedding_model
    }
}

/// Download model files from Hugging Face Hub
fn download_model(model_id: &str, revision: &str) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let api = ApiBuilder::from_env()
        .with_progress(true)
        .build()
        .map_err(|e| SedimentError::ModelLoading(format!("Failed to create HF API: {}", e)))?;

    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
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

    Ok((model_path, tokenizer_path, config_path))
}

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
    #[ignore] // Requires model download
    fn test_embed_query_and_document() -> Result<()> {
        let embedder = Embedder::new()?;

        let query_emb = embedder.embed_query("What database do we use?")?;
        let doc_emb = embedder.embed_document("We use Postgres for the main database")?;

        assert_eq!(query_emb.len(), EMBEDDING_DIM);
        assert_eq!(doc_emb.len(), EMBEDDING_DIM);

        // For AllMiniLmL6V2 (no prefixes), embed_query and embed should be identical
        let raw_emb = embedder.embed("What database do we use?")?;
        assert_eq!(query_emb, raw_emb);

        Ok(())
    }

    #[test]
    #[ignore] // Requires model download
    fn test_e5_small_v2_embedder() -> Result<()> {
        let embedder = Embedder::with_embedding_model(EmbeddingModel::E5SmallV2)?;

        // Verify dimension
        let emb = embedder.embed("test")?;
        assert_eq!(emb.len(), EMBEDDING_DIM);

        // Verify L2 normalization
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);

        // Query and document prefixes should produce different vectors
        let query_emb = embedder.embed_query("What is the capital of France?")?;
        let doc_emb = embedder.embed_document("What is the capital of France?")?;
        assert_ne!(
            query_emb, doc_emb,
            "E5 query and document embeddings should differ due to prefixes"
        );

        Ok(())
    }

    #[test]
    #[ignore] // Requires model download
    fn test_bge_small_en_v15_embedder() -> Result<()> {
        let embedder = Embedder::with_embedding_model(EmbeddingModel::BgeSmallEnV15)?;

        // Verify dimension
        let emb = embedder.embed("test")?;
        assert_eq!(emb.len(), EMBEDDING_DIM);

        // Verify L2 normalization
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);

        // Query prefix should produce different vector; document has no prefix
        let query_emb = embedder.embed_query("What is the capital of France?")?;
        let doc_emb = embedder.embed_document("What is the capital of France?")?;
        let raw_emb = embedder.embed("What is the capital of France?")?;

        assert_ne!(
            query_emb, doc_emb,
            "BGE query and document embeddings should differ due to query prefix"
        );
        // Document embedding should be identical to raw embedding (no prefix)
        assert_eq!(
            doc_emb, raw_emb,
            "BGE document embedding should equal raw embedding (no prefix)"
        );

        Ok(())
    }

    #[test]
    #[ignore] // Requires model download
    fn test_bge_base_en_v15_embedder() -> Result<()> {
        let embedder = Embedder::with_embedding_model(EmbeddingModel::BgeBaseEnV15)?;

        // Verify 768-dim output
        let emb = embedder.embed("test")?;
        assert_eq!(emb.len(), 768);
        assert_eq!(embedder.dimension(), 768);

        // Verify L2 normalization
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);

        // Query prefix should produce different vector; document has no prefix
        let query_emb = embedder.embed_query("What is the capital of France?")?;
        let doc_emb = embedder.embed_document("What is the capital of France?")?;
        let raw_emb = embedder.embed("What is the capital of France?")?;

        assert_eq!(query_emb.len(), 768);
        assert_eq!(doc_emb.len(), 768);

        assert_ne!(
            query_emb, doc_emb,
            "BGE-base query and document embeddings should differ due to query prefix"
        );
        assert_eq!(
            doc_emb, raw_emb,
            "BGE-base document embedding should equal raw embedding (no prefix)"
        );

        Ok(())
    }

    #[test]
    fn test_embedding_model_from_env_str() {
        assert_eq!(
            EmbeddingModel::from_env_str("e5-small-v2"),
            Some(EmbeddingModel::E5SmallV2)
        );
        assert_eq!(
            EmbeddingModel::from_env_str("bge-small-en-v1.5"),
            Some(EmbeddingModel::BgeSmallEnV15)
        );
        assert_eq!(
            EmbeddingModel::from_env_str("bge-base-en-v1.5"),
            Some(EmbeddingModel::BgeBaseEnV15)
        );
        assert_eq!(
            EmbeddingModel::from_env_str("all-MiniLM-L6-v2"),
            Some(EmbeddingModel::AllMiniLmL6V2)
        );
        assert_eq!(EmbeddingModel::from_env_str("unknown-model"), None);
    }

    #[test]
    fn test_embedding_model_dimensions() {
        assert_eq!(EmbeddingModel::AllMiniLmL6V2.embedding_dim(), 384);
        assert_eq!(EmbeddingModel::E5SmallV2.embedding_dim(), 384);
        assert_eq!(EmbeddingModel::BgeSmallEnV15.embedding_dim(), 384);
        assert_eq!(EmbeddingModel::BgeBaseEnV15.embedding_dim(), 768);
    }

    #[test]
    fn test_embedding_model_prefixes() {
        let text = "hello world";

        // AllMiniLmL6V2: no prefixes
        let m = EmbeddingModel::AllMiniLmL6V2;
        assert_eq!(m.prefix_query(text).as_ref(), "hello world");
        assert_eq!(m.prefix_document(text).as_ref(), "hello world");

        // E5SmallV2: query and document prefixes
        let m = EmbeddingModel::E5SmallV2;
        assert_eq!(m.prefix_query(text).as_ref(), "query: hello world");
        assert_eq!(m.prefix_document(text).as_ref(), "passage: hello world");

        // BgeSmallEnV15: query prefix only
        let m = EmbeddingModel::BgeSmallEnV15;
        assert_eq!(
            m.prefix_query(text).as_ref(),
            "Represent this sentence for searching relevant passages: hello world"
        );
        assert_eq!(m.prefix_document(text).as_ref(), "hello world");

        // BgeBaseEnV15: same prefixes as BgeSmallEnV15
        let m = EmbeddingModel::BgeBaseEnV15;
        assert_eq!(
            m.prefix_query(text).as_ref(),
            "Represent this sentence for searching relevant passages: hello world"
        );
        assert_eq!(m.prefix_document(text).as_ref(), "hello world");
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
