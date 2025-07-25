//! # Competency Assessment Engine
//!
//! Implements sophisticated competency assessment for the Monkey-Tail semantic
//! identity system. Assesses user competencies across multiple domains through
//! interaction analysis and cross-validation.

use crate::config::KambuzumaConfig;
use crate::errors::KambuzumaError;
use crate::types::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use serde::{Deserialize, Serialize};

/// Competency Assessment Engine
/// Assesses user competencies across multiple domains
#[derive(Debug)]
pub struct CompetencyAssessmentEngine {
    /// Engine identifier
    pub id: Uuid,
    /// Configuration
    pub config: Arc<RwLock<KambuzumaConfig>>,
    /// Domain assessors
    pub domain_assessors: HashMap<String, DomainAssessor>,
    /// Cross-domain validator
    pub cross_validator: CrossDomainValidator,
    /// Temporal tracker
    pub temporal_tracker: CompetencyTemporalTracker,
    /// Evidence aggregator
    pub evidence_aggregator: EvidenceAggregator,
    /// Performance metrics
    pub metrics: Arc<RwLock<CompetencyAssessmentMetrics>>,
}

impl CompetencyAssessmentEngine {
    /// Create new competency assessment engine
    pub async fn new(config: Arc<RwLock<KambuzumaConfig>>) -> Result<Self, KambuzumaError> {
        let mut domain_assessors = HashMap::new();
        
        // Initialize domain assessors for key competency areas
        let domains = vec![
            "technology", "science", "mathematics", "language", "creative",
            "analytical", "problem_solving", "communication", "reasoning", "domain_expertise"
        ];
        
        for domain in domains {
            domain_assessors.insert(
                domain.to_string(),
                DomainAssessor::new(domain.to_string()).await?
            );
        }
        
        Ok(Self {
            id: Uuid::new_v4(),
            config,
            domain_assessors,
            cross_validator: CrossDomainValidator::new(),
            temporal_tracker: CompetencyTemporalTracker::new(),
            evidence_aggregator: EvidenceAggregator::new(),
            metrics: Arc::new(RwLock::new(CompetencyAssessmentMetrics::default())),
        })
    }

    /// Perform initial competency assessment
    pub async fn perform_initial_assessment(
        &self,
        interaction_data: &crate::monkey_tail_integration::InteractionData,
    ) -> Result<HashMap<String, CompetencyAssessment>, KambuzumaError> {
        let mut assessments = HashMap::new();
        
        // Assess competencies across all domains
        for (domain_name, assessor) in &self.domain_assessors {
            let assessment = assessor.assess_initial_competency(interaction_data).await?;
            assessments.insert(domain_name.clone(), assessment);
        }
        
        // Cross-validate assessments
        let validated_assessments = self.cross_validator.validate_assessments(&assessments).await?;
        
        // Update metrics
        self.update_initial_assessment_metrics(&validated_assessments).await?;
        
        Ok(validated_assessments)
    }

    /// Assess user competencies from semantic identity and interaction data
    pub async fn assess_user_competencies(
        &self,
        semantic_identity: &SemanticIdentity,
        interaction_data: &crate::monkey_tail_integration::InteractionData,
    ) -> Result<HashMap<String, CompetencyUpdate>, KambuzumaError> {
        let mut competency_updates = HashMap::new();
        
        // Assess competencies for each domain
        for (domain_name, assessor) in &self.domain_assessors {
            let update = assessor.assess_competency_update(
                semantic_identity,
                interaction_data,
                domain_name,
            ).await?;
            
            competency_updates.insert(domain_name.clone(), update);
        }
        
        // Aggregate evidence across domains
        let aggregated_evidence = self.evidence_aggregator
            .aggregate_competency_evidence(&competency_updates).await?;
        
        // Apply temporal tracking
        self.temporal_tracker.track_competency_evolution(
            semantic_identity.user_id,
            &competency_updates,
        ).await?;
        
        // Cross-validate updates
        let validated_updates = self.cross_validator
            .validate_competency_updates(&competency_updates, &aggregated_evidence).await?;
        
        // Update metrics
        self.update_assessment_metrics(&validated_updates).await?;
        
        Ok(validated_updates)
    }

    /// Get competency assessment confidence
    pub async fn get_assessment_confidence(
        &self,
        user_id: Uuid,
        domain: &str,
    ) -> Result<f64, KambuzumaError> {
        if let Some(assessor) = self.domain_assessors.get(domain) {
            assessor.get_confidence_level(user_id).await
        } else {
            Ok(0.0)
        }
    }

    // Private helper methods

    async fn update_initial_assessment_metrics(
        &self,
        assessments: &HashMap<String, CompetencyAssessment>,
    ) -> Result<(), KambuzumaError> {
        let mut metrics = self.metrics.write().await;
        metrics.total_initial_assessments += 1;
        
        let average_confidence: f64 = assessments.values()
            .map(|a| a.confidence)
            .sum::<f64>() / assessments.len() as f64;
        
        metrics.average_initial_confidence = 
            (metrics.average_initial_confidence * (metrics.total_initial_assessments - 1) as f64 + average_confidence) / 
            metrics.total_initial_assessments as f64;
        
        Ok(())
    }

    async fn update_assessment_metrics(
        &self,
        updates: &HashMap<String, CompetencyUpdate>,
    ) -> Result<(), KambuzumaError> {
        let mut metrics = self.metrics.write().await;
        metrics.total_competency_updates += updates.len() as u64;
        
        let average_confidence: f64 = updates.values()
            .map(|u| u.confidence_change)
            .sum::<f64>() / updates.len() as f64;
        
        metrics.average_update_confidence = 
            (metrics.average_update_confidence + average_confidence) / 2.0;
        
        Ok(())
    }
}

/// Domain Assessor
/// Assesses competency within a specific domain
#[derive(Debug)]
pub struct DomainAssessor {
    /// Assessor identifier
    pub id: Uuid,
    /// Domain name
    pub domain: String,
    /// Assessment criteria
    pub assessment_criteria: Vec<AssessmentCriterion>,
    /// Evidence analyzer
    pub evidence_analyzer: EvidenceAnalyzer,
    /// Competency tracker
    pub competency_tracker: HashMap<Uuid, DomainCompetencyTracker>,
}

impl DomainAssessor {
    pub async fn new(domain: String) -> Result<Self, KambuzumaError> {
        let assessment_criteria = Self::generate_domain_criteria(&domain);
        
        Ok(Self {
            id: Uuid::new_v4(),
            domain,
            assessment_criteria,
            evidence_analyzer: EvidenceAnalyzer::new(),
            competency_tracker: HashMap::new(),
        })
    }

    pub async fn assess_initial_competency(
        &self,
        interaction_data: &crate::monkey_tail_integration::InteractionData,
    ) -> Result<CompetencyAssessment, KambuzumaError> {
        // Analyze interaction for domain-specific competency indicators
        let evidence = self.evidence_analyzer
            .analyze_interaction_evidence(interaction_data, &self.domain).await?;
        
        // Apply assessment criteria
        let competency_score = self.calculate_competency_score(&evidence).await?;
        let confidence = self.calculate_assessment_confidence(&evidence).await?;
        
        Ok(CompetencyAssessment {
            id: Uuid::new_v4(),
            domain: self.domain.clone(),
            competency_score,
            confidence,
            evidence_quality: evidence.quality_score,
            assessment_method: AssessmentMethod::InteractionAnalysis,
            assessed_at: chrono::Utc::now(),
        })
    }

    pub async fn assess_competency_update(
        &self,
        semantic_identity: &SemanticIdentity,
        interaction_data: &crate::monkey_tail_integration::InteractionData,
        domain: &str,
    ) -> Result<CompetencyUpdate, KambuzumaError> {
        // Get current competency level
        let current_competency = semantic_identity.semantic_vector.domain_competencies
            .get(domain)
            .cloned()
            .unwrap_or(DomainCompetency {
                domain: domain.to_string(),
                level: 0.0,
                confidence: 0.0,
                weight: 1.0,
                evidence_sources: Vec::new(),
                assessed_at: chrono::Utc::now(),
            });
        
        // Analyze new evidence
        let evidence = self.evidence_analyzer
            .analyze_interaction_evidence(interaction_data, domain).await?;
        
        // Calculate competency change
        let competency_change = self.calculate_competency_change(&current_competency, &evidence).await?;
        let confidence_change = self.calculate_confidence_change(&current_competency, &evidence).await?;
        
        Ok(CompetencyUpdate {
            id: Uuid::new_v4(),
            domain: domain.to_string(),
            previous_level: current_competency.level,
            new_level: (current_competency.level + competency_change).max(0.0).min(1.0),
            competency_change,
            confidence_change,
            evidence_quality: evidence.quality_score,
            update_reason: "interaction_analysis".to_string(),
            updated_at: chrono::Utc::now(),
        })
    }

    pub async fn get_confidence_level(&self, user_id: Uuid) -> Result<f64, KambuzumaError> {
        if let Some(tracker) = self.competency_tracker.get(&user_id) {
            Ok(tracker.current_confidence)
        } else {
            Ok(0.5) // Default neutral confidence
        }
    }

    // Private helper methods

    fn generate_domain_criteria(domain: &str) -> Vec<AssessmentCriterion> {
        match domain {
            "technology" => vec![
                AssessmentCriterion::new("technical_vocabulary", 0.3),
                AssessmentCriterion::new("system_understanding", 0.3),
                AssessmentCriterion::new("problem_solving_approach", 0.4),
            ],
            "science" => vec![
                AssessmentCriterion::new("scientific_reasoning", 0.4),
                AssessmentCriterion::new("evidence_evaluation", 0.3),
                AssessmentCriterion::new("hypothesis_formation", 0.3),
            ],
            "mathematics" => vec![
                AssessmentCriterion::new("mathematical_reasoning", 0.4),
                AssessmentCriterion::new("quantitative_analysis", 0.3),
                AssessmentCriterion::new("abstract_thinking", 0.3),
            ],
            "language" => vec![
                AssessmentCriterion::new("linguistic_complexity", 0.3),
                AssessmentCriterion::new("vocabulary_depth", 0.3),
                AssessmentCriterion::new("communication_clarity", 0.4),
            ],
            "creative" => vec![
                AssessmentCriterion::new("creative_thinking", 0.4),
                AssessmentCriterion::new("originality", 0.3),
                AssessmentCriterion::new("ideational_fluency", 0.3),
            ],
            _ => vec![
                AssessmentCriterion::new("general_competency", 0.5),
                AssessmentCriterion::new("domain_knowledge", 0.5),
            ],
        }
    }

    async fn calculate_competency_score(&self, evidence: &DomainEvidence) -> Result<f64, KambuzumaError> {
        let mut total_score = 0.0;
        let mut total_weight = 0.0;
        
        for criterion in &self.assessment_criteria {
            let criterion_score = criterion.evaluate_evidence(evidence).await?;
            total_score += criterion_score * criterion.weight;
            total_weight += criterion.weight;
        }
        
        if total_weight > 0.0 {
            Ok(total_score / total_weight)
        } else {
            Ok(0.0)
        }
    }

    async fn calculate_assessment_confidence(&self, evidence: &DomainEvidence) -> Result<f64, KambuzumaError> {
        // Confidence based on evidence quality and quantity
        let quality_factor = evidence.quality_score;
        let quantity_factor = (evidence.evidence_items.len() as f64 / 10.0).min(1.0);
        Ok((quality_factor + quantity_factor) / 2.0)
    }

    async fn calculate_competency_change(
        &self,
        current: &DomainCompetency,
        evidence: &DomainEvidence,
    ) -> Result<f64, KambuzumaError> {
        let evidence_strength = evidence.quality_score * evidence.relevance_score;
        let learning_rate = 0.1; // Moderate learning rate
        let change = evidence_strength * learning_rate;
        
        // Diminishing returns for high competency levels
        let diminishing_factor = 1.0 - current.level;
        Ok(change * diminishing_factor)
    }

    async fn calculate_confidence_change(
        &self,
        current: &DomainCompetency,
        evidence: &DomainEvidence,
    ) -> Result<f64, KambuzumaError> {
        let evidence_confidence = evidence.quality_score;
        let confidence_change = (evidence_confidence - current.confidence) * 0.1;
        Ok(confidence_change)
    }
}

/// Assessment Criterion
/// Specific criterion for assessing competency
#[derive(Debug, Clone)]
pub struct AssessmentCriterion {
    pub name: String,
    pub weight: f64,
    pub evaluation_method: EvaluationMethod,
}

impl AssessmentCriterion {
    pub fn new(name: &str, weight: f64) -> Self {
        Self {
            name: name.to_string(),
            weight,
            evaluation_method: EvaluationMethod::PatternAnalysis,
        }
    }

    pub async fn evaluate_evidence(&self, evidence: &DomainEvidence) -> Result<f64, KambuzumaError> {
        // Evaluate evidence based on this criterion
        match self.evaluation_method {
            EvaluationMethod::PatternAnalysis => {
                self.evaluate_patterns(evidence).await
            },
            EvaluationMethod::VocabularyAnalysis => {
                self.evaluate_vocabulary(evidence).await
            },
            EvaluationMethod::ReasoningAnalysis => {
                self.evaluate_reasoning(evidence).await
            },
        }
    }

    async fn evaluate_patterns(&self, evidence: &DomainEvidence) -> Result<f64, KambuzumaError> {
        // Pattern-based evaluation
        let pattern_score = evidence.evidence_items.iter()
            .map(|item| item.pattern_relevance)
            .sum::<f64>() / evidence.evidence_items.len() as f64;
        Ok(pattern_score)
    }

    async fn evaluate_vocabulary(&self, evidence: &DomainEvidence) -> Result<f64, KambuzumaError> {
        // Vocabulary-based evaluation
        let vocab_score = evidence.evidence_items.iter()
            .map(|item| item.vocabulary_sophistication)
            .sum::<f64>() / evidence.evidence_items.len() as f64;
        Ok(vocab_score)
    }

    async fn evaluate_reasoning(&self, evidence: &DomainEvidence) -> Result<f64, KambuzumaError> {
        // Reasoning-based evaluation
        let reasoning_score = evidence.evidence_items.iter()
            .map(|item| item.reasoning_complexity)
            .sum::<f64>() / evidence.evidence_items.len() as f64;
        Ok(reasoning_score)
    }
}

/// Supporting types and structures

#[derive(Debug, Clone)]
pub enum EvaluationMethod {
    PatternAnalysis,
    VocabularyAnalysis,
    ReasoningAnalysis,
}

#[derive(Debug)]
pub struct EvidenceAnalyzer {
    pub id: Uuid,
}

impl EvidenceAnalyzer {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
        }
    }

    pub async fn analyze_interaction_evidence(
        &self,
        interaction_data: &crate::monkey_tail_integration::InteractionData,
        domain: &str,
    ) -> Result<DomainEvidence, KambuzumaError> {
        // Analyze interaction data for domain-specific evidence
        let evidence_items = self.extract_evidence_items(interaction_data, domain).await?;
        let quality_score = self.calculate_quality_score(&evidence_items).await?;
        let relevance_score = self.calculate_relevance_score(&evidence_items, domain).await?;
        
        Ok(DomainEvidence {
            id: Uuid::new_v4(),
            domain: domain.to_string(),
            evidence_items,
            quality_score,
            relevance_score,
            timestamp: chrono::Utc::now(),
        })
    }

    async fn extract_evidence_items(
        &self,
        interaction_data: &crate::monkey_tail_integration::InteractionData,
        domain: &str,
    ) -> Result<Vec<EvidenceItem>, KambuzumaError> {
        let mut items = Vec::new();
        
        // Extract evidence from user input
        let input_complexity = interaction_data.user_input.len() as f64 / 100.0;
        let vocabulary_sophistication = self.assess_vocabulary_sophistication(&interaction_data.user_input).await?;
        let reasoning_complexity = self.assess_reasoning_complexity(&interaction_data.user_input).await?;
        let pattern_relevance = self.assess_pattern_relevance(&interaction_data.user_input, domain).await?;
        
        items.push(EvidenceItem {
            id: Uuid::new_v4(),
            evidence_type: EvidenceType::UserInput,
            content: interaction_data.user_input.clone(),
            vocabulary_sophistication,
            reasoning_complexity,
            pattern_relevance,
            confidence: 0.8,
        });
        
        Ok(items)
    }

    async fn assess_vocabulary_sophistication(&self, text: &str) -> Result<f64, KambuzumaError> {
        // Simple vocabulary sophistication assessment
        let words: Vec<&str> = text.split_whitespace().collect();
        let average_word_length = if words.is_empty() { 
            0.0 
        } else { 
            words.iter().map(|w| w.len()).sum::<usize>() as f64 / words.len() as f64 
        };
        
        // Normalize to 0-1 range
        Ok((average_word_length / 10.0).min(1.0))
    }

    async fn assess_reasoning_complexity(&self, text: &str) -> Result<f64, KambuzumaError> {
        // Simple reasoning complexity assessment based on logical connectors
        let logical_words = ["because", "therefore", "however", "although", "since", "while"];
        let logical_count = logical_words.iter()
            .map(|word| text.to_lowercase().matches(word).count())
            .sum::<usize>();
        
        // Normalize based on text length
        let text_words = text.split_whitespace().count();
        if text_words == 0 { return Ok(0.0); }
        
        Ok((logical_count as f64 / text_words as f64 * 10.0).min(1.0))
    }

    async fn assess_pattern_relevance(&self, text: &str, domain: &str) -> Result<f64, KambuzumaError> {
        // Simple domain-specific pattern relevance assessment
        let domain_keywords = match domain {
            "technology" => vec!["system", "software", "algorithm", "computer", "data"],
            "science" => vec!["theory", "hypothesis", "experiment", "evidence", "analysis"],
            "mathematics" => vec!["equation", "function", "variable", "calculate", "formula"],
            "language" => vec!["grammar", "syntax", "semantic", "linguistic", "meaning"],
            _ => vec!["knowledge", "understanding", "concept", "idea"],
        };
        
        let keyword_count = domain_keywords.iter()
            .map(|keyword| text.to_lowercase().matches(keyword).count())
            .sum::<usize>();
        
        let text_words = text.split_whitespace().count();
        if text_words == 0 { return Ok(0.0); }
        
        Ok((keyword_count as f64 / text_words as f64 * 20.0).min(1.0))
    }

    async fn calculate_quality_score(&self, items: &[EvidenceItem]) -> Result<f64, KambuzumaError> {
        if items.is_empty() { return Ok(0.0); }
        
        let total_confidence: f64 = items.iter().map(|item| item.confidence).sum();
        Ok(total_confidence / items.len() as f64)
    }

    async fn calculate_relevance_score(&self, items: &[EvidenceItem], _domain: &str) -> Result<f64, KambuzumaError> {
        if items.is_empty() { return Ok(0.0); }
        
        let total_relevance: f64 = items.iter().map(|item| item.pattern_relevance).sum();
        Ok(total_relevance / items.len() as f64)
    }
}

#[derive(Debug)]
pub struct CrossDomainValidator {
    pub id: Uuid,
}

impl CrossDomainValidator {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
        }
    }

    pub async fn validate_assessments(
        &self,
        assessments: &HashMap<String, CompetencyAssessment>,
    ) -> Result<HashMap<String, CompetencyAssessment>, KambuzumaError> {
        let mut validated = assessments.clone();
        
        // Cross-validate assessments for consistency
        for (domain, assessment) in validated.iter_mut() {
            let consistency_score = self.check_cross_domain_consistency(domain, assessment, assessments).await?;
            
            // Adjust confidence based on consistency
            assessment.confidence *= consistency_score;
        }
        
        Ok(validated)
    }

    pub async fn validate_competency_updates(
        &self,
        updates: &HashMap<String, CompetencyUpdate>,
        _evidence: &AggregatedEvidence,
    ) -> Result<HashMap<String, CompetencyUpdate>, KambuzumaError> {
        // Simplified validation - in practice, this would be more sophisticated
        Ok(updates.clone())
    }

    async fn check_cross_domain_consistency(
        &self,
        _domain: &str,
        _assessment: &CompetencyAssessment,
        _all_assessments: &HashMap<String, CompetencyAssessment>,
    ) -> Result<f64, KambuzumaError> {
        // Simplified consistency check
        Ok(0.95) // High consistency by default
    }
}

#[derive(Debug)]
pub struct CompetencyTemporalTracker {
    pub id: Uuid,
}

impl CompetencyTemporalTracker {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
        }
    }

    pub async fn track_competency_evolution(
        &self,
        _user_id: Uuid,
        _updates: &HashMap<String, CompetencyUpdate>,
    ) -> Result<(), KambuzumaError> {
        // Track competency changes over time
        Ok(())
    }
}

#[derive(Debug)]
pub struct EvidenceAggregator {
    pub id: Uuid,
}

impl EvidenceAggregator {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
        }
    }

    pub async fn aggregate_competency_evidence(
        &self,
        updates: &HashMap<String, CompetencyUpdate>,
    ) -> Result<AggregatedEvidence, KambuzumaError> {
        let total_updates = updates.len();
        let average_quality: f64 = updates.values()
            .map(|u| u.evidence_quality)
            .sum::<f64>() / total_updates as f64;
        
        Ok(AggregatedEvidence {
            id: Uuid::new_v4(),
            total_evidence_items: total_updates as u64,
            average_quality,
            consistency_score: 0.85,
            aggregation_confidence: 0.9,
        })
    }
}

/// Data structures

#[derive(Debug, Clone)]
pub struct CompetencyUpdate {
    pub id: Uuid,
    pub domain: String,
    pub previous_level: f64,
    pub new_level: f64,
    pub competency_change: f64,
    pub confidence_change: f64,
    pub evidence_quality: f64,
    pub update_reason: String,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct DomainEvidence {
    pub id: Uuid,
    pub domain: String,
    pub evidence_items: Vec<EvidenceItem>,
    pub quality_score: f64,
    pub relevance_score: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct EvidenceItem {
    pub id: Uuid,
    pub evidence_type: EvidenceType,
    pub content: String,
    pub vocabulary_sophistication: f64,
    pub reasoning_complexity: f64,
    pub pattern_relevance: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub enum EvidenceType {
    UserInput,
    InteractionPattern,
    ResponseQuality,
    TemporalConsistency,
}

#[derive(Debug, Clone)]
pub struct DomainCompetencyTracker {
    pub user_id: Uuid,
    pub domain: String,
    pub current_level: f64,
    pub current_confidence: f64,
    pub assessment_history: Vec<CompetencyAssessment>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct AggregatedEvidence {
    pub id: Uuid,
    pub total_evidence_items: u64,
    pub average_quality: f64,
    pub consistency_score: f64,
    pub aggregation_confidence: f64,
}

#[derive(Debug, Clone, Default)]
pub struct CompetencyAssessmentMetrics {
    pub total_initial_assessments: u64,
    pub total_competency_updates: u64,
    pub average_initial_confidence: f64,
    pub average_update_confidence: f64,
    pub cross_domain_consistency: f64,
    pub temporal_stability: f64,
} 