"""
Evidence Analyst Agent
Specialized in evaluating evidence strength, admissibility, and legal value
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from .base_agent import BaseAgent, AgentResult

class EvidenceAnalystAgent(BaseAgent):
    """Agent specialized in evidence evaluation and legal analysis"""
    
    def __init__(self, ai_service=None, config: Dict[str, Any] = None):
        super().__init__("EvidenceAnalyst", ai_service, config)
        
        # Evidence evaluation criteria
        self.evaluation_criteria = {
            "relevance": {
                "weight": 0.3,
                "factors": ["case_theory_alignment", "material_fact_support", "legal_issue_connection"]
            },
            "probative_value": {
                "weight": 0.3,
                "factors": ["fact_proving_strength", "uniqueness", "directness"]
            },
            "admissibility": {
                "weight": 0.25,
                "factors": ["authentication", "hearsay_concerns", "privilege_issues", "relevance_403"]
            },
            "credibility": {
                "weight": 0.15,
                "factors": ["source_reliability", "chain_of_custody", "consistency"]
            }
        }
        
    def get_capabilities(self) -> List[str]:
        return [
            "evidence_strength_assessment",
            "admissibility_analysis",
            "probative_value_calculation",
            "rule_403_balancing",
            "authentication_requirements",
            "hearsay_analysis",
            "privilege_screening"
        ]
    
    async def analyze(self, data: Any, context: Dict[str, Any] = None) -> AgentResult:
        """Analyze evidence for legal value and admissibility"""
        start_time = datetime.now()
        
        if not await self.validate_input(data):
            return self._create_error_result("Invalid input data", start_time)
        
        try:
            # Core evidence analysis
            relevance_analysis = await self._analyze_relevance(data, context)
            probative_analysis = await self._analyze_probative_value(data, context)
            admissibility_analysis = await self._analyze_admissibility(data, context)
            credibility_analysis = await self._analyze_credibility(data, context)
            
            # Rule 403 balancing test
            rule_403_analysis = await self._perform_rule_403_balancing(
                probative_analysis, admissibility_analysis, data
            )
            
            # Overall evidence scoring
            overall_score = self._calculate_overall_evidence_score({
                "relevance": relevance_analysis,
                "probative_value": probative_analysis,
                "admissibility": admissibility_analysis,
                "credibility": credibility_analysis
            })
            
            # AI-enhanced analysis
            ai_analysis = {}
            if self.ai_service:
                ai_analysis = await self._ai_evidence_analysis(data, context)
            
            findings = {
                "relevance_analysis": relevance_analysis,
                "probative_analysis": probative_analysis,
                "admissibility_analysis": admissibility_analysis,
                "credibility_analysis": credibility_analysis,
                "rule_403_analysis": rule_403_analysis,
                "overall_evidence_score": overall_score,
                "ai_analysis": ai_analysis
            }
            
            confidence = self.calculate_confidence(findings)
            evidence_strength = overall_score.get("composite_score", 0.5)
            legal_significance = self.extract_legal_significance(findings)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                agent_name=self.name,
                analysis_type="evidence_analysis",
                confidence=confidence,
                findings=findings,
                recommendations=self._generate_recommendations(findings),
                evidence_strength=evidence_strength,
                legal_significance=legal_significance,
                processing_time=processing_time,
                timestamp=datetime.now(),
                metadata={"file_path": data.get('file_path', ''), "evidence_type": data.get('document_type', 'unknown')}
            )
            
        except Exception as e:
            self.logger.error(f"Evidence analysis failed: {e}")
            return self._create_error_result(str(e), start_time)
    
    async def _analyze_relevance(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze evidence relevance to the case"""
        content = data.get('content', '')
        case_theory = context.get('case_theory', {}) if context else {}
        
        relevance_score = 0.5  # Base relevance
        relevance_factors = []
        
        # Check alignment with case theory
        case_keywords = case_theory.get('key_terms', [])
        if case_keywords:
            keyword_matches = sum(1 for keyword in case_keywords if keyword.lower() in content.lower())
            keyword_relevance = min(keyword_matches / len(case_keywords), 1.0)
            relevance_score += keyword_relevance * 0.3
            relevance_factors.append(f"Case theory keyword alignment: {keyword_relevance:.2f}")
        
        # Check for legal concepts
        legal_concepts = [
            "fraud", "breach", "negligence", "damages", "contract", "agreement",
            "violation", "misconduct", "abuse", "harassment", "discrimination"
        ]
        
        concept_matches = sum(1 for concept in legal_concepts if concept in content.lower())
        if concept_matches > 0:
            concept_relevance = min(concept_matches / 5, 0.3)
            relevance_score += concept_relevance
            relevance_factors.append(f"Legal concept presence: {concept_relevance:.2f}")
        
        # Check for factual elements
        factual_indicators = ["date", "time", "amount", "location", "witness", "document"]
        factual_matches = sum(1 for indicator in factual_indicators if indicator in content.lower())
        if factual_matches > 0:
            factual_relevance = min(factual_matches / len(factual_indicators), 0.2)
            relevance_score += factual_relevance
            relevance_factors.append(f"Factual content richness: {factual_relevance:.2f}")
        
        relevance_score = min(relevance_score, 1.0)
        
        return {
            "relevance_score": relevance_score,
            "relevance_factors": relevance_factors,
            "case_theory_alignment": keyword_relevance if case_keywords else 0.0,
            "legal_concept_density": concept_matches,
            "factual_content_score": factual_matches
        }
    
    async def _analyze_probative_value(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the probative value of the evidence"""
        content = data.get('content', '')
        doc_type = data.get('document_type', {})
        
        probative_score = 0.4  # Base probative value
        probative_factors = []
        
        # Document type probative value
        type_probative_values = {
            "email": 0.8,
            "text_message": 0.7,
            "court_document": 0.9,
            "financial_document": 0.8,
            "contract": 0.9,
            "receipt": 0.6,
            "photo": 0.5,
            "unknown": 0.3
        }
        
        primary_type = doc_type.get('primary_type', 'unknown')
        type_value = type_probative_values.get(primary_type, 0.3)
        probative_score += type_value * 0.3
        probative_factors.append(f"Document type value ({primary_type}): {type_value:.2f}")
        
        # Content specificity
        specific_indicators = [
            r'\$[\d,]+\.?\d*',  # Specific amounts
            r'\d{1,2}/\d{1,2}/\d{4}',  # Specific dates
            r'\d{1,2}:\d{2}',  # Specific times
            r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # Proper names
        ]
        
        specificity_score = 0
        for pattern in specific_indicators:
            import re
            if re.search(pattern, content):
                specificity_score += 0.1
        
        probative_score += min(specificity_score, 0.3)
        probative_factors.append(f"Content specificity: {specificity_score:.2f}")
        
        # Directness (first-hand vs. hearsay indicators)
        direct_indicators = ["I saw", "I heard", "I did", "I was", "personally"]
        indirect_indicators = ["told me", "said that", "heard from", "according to"]
        
        direct_count = sum(1 for indicator in direct_indicators if indicator.lower() in content.lower())
        indirect_count = sum(1 for indicator in indirect_indicators if indicator.lower() in content.lower())
        
        if direct_count > indirect_count:
            directness_bonus = 0.2
            probative_factors.append("Evidence appears to be direct/first-hand")
        elif indirect_count > direct_count:
            directness_bonus = -0.1
            probative_factors.append("Evidence appears to be indirect/hearsay")
        else:
            directness_bonus = 0.0
        
        probative_score += directness_bonus
        probative_score = max(0.0, min(probative_score, 1.0))
        
        return {
            "probative_score": probative_score,
            "probative_factors": probative_factors,
            "document_type_value": type_value,
            "content_specificity": specificity_score,
            "directness_assessment": directness_bonus
        }
    
    async def _analyze_admissibility(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential admissibility issues"""
        content = data.get('content', '')
        doc_type = data.get('document_type', {})
        
        admissibility_score = 0.7  # Start optimistic
        admissibility_issues = []
        admissibility_strengths = []
        
        # Authentication concerns
        auth_indicators = ["signature", "signed", "notarized", "certified", "original"]
        auth_score = sum(1 for indicator in auth_indicators if indicator.lower() in content.lower())
        
        if auth_score > 0:
            admissibility_strengths.append(f"Authentication indicators present: {auth_score}")
        else:
            admissibility_issues.append("May require additional authentication")
            admissibility_score -= 0.1
        
        # Hearsay analysis
        hearsay_indicators = ["told me", "said that", "heard from", "according to", "someone said"]
        hearsay_count = sum(1 for indicator in hearsay_indicators if indicator.lower() in content.lower())
        
        if hearsay_count > 2:
            admissibility_issues.append("Potential hearsay concerns")
            admissibility_score -= 0.2
        elif hearsay_count > 0:
            admissibility_issues.append("Minor hearsay elements present")
            admissibility_score -= 0.1
        
        # Privilege screening
        privilege_indicators = [
            "attorney", "lawyer", "counsel", "privileged", "confidential",
            "doctor", "physician", "patient", "therapy", "counseling"
        ]
        
        privilege_count = sum(1 for indicator in privilege_indicators if indicator.lower() in content.lower())
        if privilege_count > 0:
            admissibility_issues.append("Potential privilege issues")
            admissibility_score -= 0.15
        
        # Best evidence rule (for copies)
        if "copy" in content.lower() or "duplicate" in content.lower():
            admissibility_issues.append("Best evidence rule considerations")
            admissibility_score -= 0.05
        
        # Character evidence concerns
        character_indicators = ["always", "never", "typical", "usually", "character", "reputation"]
        character_count = sum(1 for indicator in character_indicators if indicator.lower() in content.lower())
        
        if character_count > 2:
            admissibility_issues.append("Potential character evidence issues")
            admissibility_score -= 0.1
        
        admissibility_score = max(0.0, min(admissibility_score, 1.0))
        
        return {
            "admissibility_score": admissibility_score,
            "admissibility_issues": admissibility_issues,
            "admissibility_strengths": admissibility_strengths,
            "authentication_indicators": auth_score,
            "hearsay_concerns": hearsay_count,
            "privilege_concerns": privilege_count
        }
    
    async def _analyze_credibility(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze credibility factors"""
        content = data.get('content', '')
        file_info = data.get('file_info', {})
        
        credibility_score = 0.6  # Base credibility
        credibility_factors = []
        
        # Source reliability indicators
        reliable_sources = ["court", "bank", "hospital", "police", "government", "official"]
        source_reliability = sum(1 for source in reliable_sources if source in content.lower())
        
        if source_reliability > 0:
            credibility_bonus = min(source_reliability * 0.1, 0.2)
            credibility_score += credibility_bonus
            credibility_factors.append(f"Reliable source indicators: {source_reliability}")
        
        # Consistency indicators
        consistency_indicators = ["consistent", "confirms", "corroborates", "supports"]
        inconsistency_indicators = ["contradicts", "conflicts", "disputes", "denies"]
        
        consistency_count = sum(1 for indicator in consistency_indicators if indicator.lower() in content.lower())
        inconsistency_count = sum(1 for indicator in inconsistency_indicators if indicator.lower() in content.lower())
        
        if consistency_count > inconsistency_count:
            credibility_score += 0.1
            credibility_factors.append("Content suggests consistency")
        elif inconsistency_count > consistency_count:
            credibility_score -= 0.1
            credibility_factors.append("Content suggests inconsistencies")
        
        # Temporal consistency (file dates vs. content dates)
        if file_info.get('created') and file_info.get('modified'):
            # Simple check - more sophisticated analysis could be added
            credibility_factors.append("File metadata available for verification")
        
        credibility_score = max(0.0, min(credibility_score, 1.0))
        
        return {
            "credibility_score": credibility_score,
            "credibility_factors": credibility_factors,
            "source_reliability_score": source_reliability,
            "consistency_assessment": consistency_count - inconsistency_count
        }
    
    async def _perform_rule_403_balancing(self, probative_analysis: Dict[str, Any], 
                                        admissibility_analysis: Dict[str, Any], 
                                        data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Federal Rule of Evidence 403 balancing test"""
        probative_value = probative_analysis.get("probative_score", 0.5)
        
        # Assess prejudicial impact
        content = data.get('content', '')
        prejudicial_indicators = [
            "graphic", "disturbing", "shocking", "inflammatory", "prejudicial",
            "bias", "unfair", "misleading", "confusing"
        ]
        
        prejudicial_count = sum(1 for indicator in prejudicial_indicators if indicator.lower() in content.lower())
        prejudicial_impact = min(prejudicial_count * 0.2, 0.8)
        
        # Rule 403 balancing
        if probative_value > prejudicial_impact:
            rule_403_result = "LIKELY ADMISSIBLE"
            balancing_ratio = probative_value / max(prejudicial_impact, 0.1)
        else:
            rule_403_result = "POTENTIAL 403 EXCLUSION"
            balancing_ratio = probative_value / max(prejudicial_impact, 0.1)
        
        return {
            "rule_403_result": rule_403_result,
            "probative_value": probative_value,
            "prejudicial_impact": prejudicial_impact,
            "balancing_ratio": balancing_ratio,
            "recommendation": f"Probative value {'substantially outweighs' if balancing_ratio > 2 else 'may not outweigh'} prejudicial impact"
        }
    
    def _calculate_overall_evidence_score(self, analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall evidence score using weighted criteria"""
        weighted_score = 0.0
        component_scores = {}
        
        for criterion, analysis in analyses.items():
            if criterion in self.evaluation_criteria:
                weight = self.evaluation_criteria[criterion]["weight"]
                score_key = f"{criterion.split('_')[0]}_score"
                score = analysis.get(score_key, 0.5)
                
                weighted_score += score * weight
                component_scores[criterion] = score
        
        # Evidence strength categories
        if weighted_score >= 0.8:
            strength_category = "STRONG"
        elif weighted_score >= 0.6:
            strength_category = "MODERATE"
        elif weighted_score >= 0.4:
            strength_category = "WEAK"
        else:
            strength_category = "VERY WEAK"
        
        return {
            "composite_score": weighted_score,
            "strength_category": strength_category,
            "component_scores": component_scores,
            "evaluation_criteria": self.evaluation_criteria
        }
    
    async def _ai_evidence_analysis(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI for advanced evidence analysis"""
        if not self.ai_service:
            return {}
        
        try:
            content = data.get('content', '')
            case_context = context.get('case_theory', {}) if context else {}
            
            prompt = f"""
Analyze this evidence for a legal case:

Case Context: {case_context}
Evidence Content: {content[:2000]}

Provide detailed legal analysis in JSON format:
{{
    "evidence_strength": "strong|moderate|weak",
    "admissibility_likelihood": "high|medium|low",
    "key_legal_issues": ["list", "of", "issues"],
    "probative_value_assessment": "detailed assessment",
    "potential_objections": ["list", "of", "likely", "objections"],
    "strategic_value": "high|medium|low",
    "authentication_requirements": ["what", "is", "needed"],
    "foundation_elements": ["required", "foundation", "elements"],
    "recommended_use": "how to best use this evidence",
    "corroborating_evidence_needed": ["what", "would", "strengthen", "this"]
}}
"""
            
            response = await self.ai_service.provider.generate_completion(
                prompt,
                "You are an expert evidence analyst and trial attorney."
            )
            
            if response.success:
                try:
                    return json.loads(response.content)
                except json.JSONDecodeError:
                    return {"ai_summary": response.content}
            
        except Exception as e:
            self.logger.error(f"AI evidence analysis failed: {e}")
        
        return {}
    
    def calculate_confidence(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate confidence in the evidence analysis"""
        confidence = 0.6  # Base confidence
        
        # Increase confidence based on analysis completeness
        if analysis_data.get("ai_analysis"):
            confidence += 0.2
        
        overall_score = analysis_data.get("overall_evidence_score", {})
        if overall_score.get("composite_score", 0) > 0.7:
            confidence += 0.1
        
        admissibility = analysis_data.get("admissibility_analysis", {})
        if admissibility.get("admissibility_score", 0) > 0.7:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def extract_legal_significance(self, findings: Dict[str, Any]) -> str:
        """Extract legal significance from evidence analysis"""
        overall_score = findings.get("overall_evidence_score", {})
        strength_category = overall_score.get("strength_category", "UNKNOWN")
        
        rule_403 = findings.get("rule_403_analysis", {})
        rule_403_result = rule_403.get("rule_403_result", "UNKNOWN")
        
        ai_analysis = findings.get("ai_analysis", {})
        strategic_value = ai_analysis.get("strategic_value", "unknown")
        
        return f"This evidence has {strength_category.lower()} evidentiary value with {rule_403_result.lower()} admissibility and {strategic_value} strategic importance."
    
    def _generate_recommendations(self, findings: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evidence analysis"""
        recommendations = []
        
        overall_score = findings.get("overall_evidence_score", {})
        admissibility = findings.get("admissibility_analysis", {})
        ai_analysis = findings.get("ai_analysis", {})
        
        # Score-based recommendations
        if overall_score.get("composite_score", 0) < 0.5:
            recommendations.append("Consider seeking stronger corroborating evidence")
        
        # Admissibility recommendations
        for issue in admissibility.get("admissibility_issues", []):
            if "authentication" in issue.lower():
                recommendations.append("Prepare authentication foundation before trial")
            elif "hearsay" in issue.lower():
                recommendations.append("Research hearsay exceptions or seek declarant testimony")
            elif "privilege" in issue.lower():
                recommendations.append("Review privilege claims and potential waivers")
        
        # AI recommendations
        ai_recommendations = ai_analysis.get("recommended_use", "")
        if ai_recommendations:
            recommendations.append(f"Strategic use: {ai_recommendations}")
        
        corroborating_needed = ai_analysis.get("corroborating_evidence_needed", [])
        if corroborating_needed:
            recommendations.append(f"Seek corroborating evidence: {', '.join(corroborating_needed[:3])}")
        
        return recommendations
    
    def _create_error_result(self, error_message: str, start_time: datetime) -> AgentResult:
        """Create an error result"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AgentResult(
            agent_name=self.name,
            analysis_type="evidence_analysis",
            confidence=0.0,
            findings={"error": error_message},
            recommendations=["Review input data and try again"],
            evidence_strength=0.0,
            legal_significance="Analysis failed",
            processing_time=processing_time,
            timestamp=datetime.now(),
            metadata={"error": True}
        )