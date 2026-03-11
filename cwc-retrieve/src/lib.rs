pub mod cross_encoder;
pub mod fallback;
pub mod hybrid;
pub mod mmr;
pub mod normalize;
pub mod router;
pub mod rrf;

pub use cross_encoder::{
    calibrate_scores, combined_score, CrossEncoderReranker, RerankConfig,
};
pub use fallback::{FallbackBehavior, ResilientRetriever};
pub use hybrid::HybridRetriever;
pub use mmr::mmr_select;
pub use normalize::{min_max_normalize, ScoreField};
pub use router::{classify_retrieval_need, RetrievalDecision};
pub use rrf::reciprocal_rank_fusion;
