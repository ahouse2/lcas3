"""
Case Strategist Agent
Specialized in overall case strategy, argument development, and tactical recommendations
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from .base_agent import BaseAgent, AgentResult

class CaseStrategistAgent(BaseAgent):
    """Agent specialized in case strategy and tactical analysis"""
    
    def __init__(self, ai_service=None, config: Dict[str, Any] = None):
        super().__init__("CaseStrategist", ai_service, config)
        
    def get_capabilities(self) -> List[str]:
        return [
            "case_strategy_development",
            "argument_prioritization",
            "evidence_portfolio_analysis",
            "tactical_recommendations",
            "settlement_analysis",
            "trial_preparation_guidance",
            "risk_assessment",
            "counter_strategy_analysis"
        ]
    
    async def analyze(self, data: Any, context: Dict[str, Any] = None) -> AgentResult:
        """Develop case strategy based on all available evidence and analysis"""
        start_time = datetime.now()
        
        if not await self.validate_input(data):
            return self._create_error_result("Invalid input data", start_time)
        
        try:
            # Get comprehensive analysis from previous agents
            previous_results = context.get('previous_results', {}) if context else {}
            
            # Analyze evidence portfolio
            portfolio_analysis = await self._analyze_evidence_portfolio(data, previous_results)
            
            # Develop case strategy
            case_strategy = await self._develop_case_strategy(data, previous_results, context)
            
            # Prioritize arguments
            argument_analysis = await self._prioritize_arguments(data, previous_results)
            
            # Assess risks and opportunities
            risk_assessment = await self._assess_risks_and_opportunities(data, previous_results)
            
            # Settlement analysis
            settlement_analysis = await self._analyze_settlement_position(data, previous_results)
            
            # Trial preparation recommendations
            trial_prep = await self._develop_trial_strategy(data, previous_results)
            
            # AI-enhanced strategic analysis
            ai_analysis = {}
            if self.ai_service:
                ai_analysis = await self._ai_strategic_analysis(data, previous_results, context)
            
            findings = {
                "portfolio_analysis": portfolio_analysis,
                "case_strategy": case_strategy,
                "argument_analysis": argument_analysis,
                "risk_assessment": risk_assessment,
                "settlement_analysis": settlement_analysis,
                "trial_preparation": trial_prep,
                "ai_analysis": ai_analysis
            }
            
            confidence = self.calculate_confidence(findings)
            evidence_strength = self._calculate_strategic_strength(findings)
            legal_significance = self.extract_legal_significance(findings)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                agent_name=self.name,
                analysis_type="case_strategy",
                confidence=confidence,
                findings=findings,
                recommendations=self._generate_recommendations(findings),
                evidence_strength=evidence_strength,
                legal_significance=legal_significance,
                processing_time=processing_time,
                timestamp=datetime.now(),
                metadata={"file_path": data.get('file_path', ''), "strategy_type": case_strategy.get("primary_strategy", "unknown")}
            )
            
        except Exception as e:
            self.logger.error(f"Case strategy analysis failed: {e}")
            return self._create_error_result(str(e), start_time)
    
    async def _analyze_evidence_portfolio(self, data: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the overall evidence portfolio"""
        portfolio = {
            "evidence_strength_distribution": {},
            "evidence_types": {},
            "coverage_analysis": {},
            "portfolio_score": 0.0
        }
        
        # Analyze evidence from previous agent results
        evidence_strengths = []
        evidence_types = {}
        
        for agent_name, result in previous_results.items():
            if isinstance(result, dict) and not result.get("error"):
                strength = result.get("evidence_strength", 0.0)
                evidence_strengths.append(strength)
                
                # Categorize evidence types
                analysis_type = result.get("analysis_type", "unknown")
                evidence_types[analysis_type] = evidence_types.get(analysis_type, 0) + 1
        
        if evidence_strengths:
            portfolio["evidence_strength_distribution"] = {
                "average": sum(evidence_strengths) / len(evidence_strengths),
                "maximum": max(evidence_strengths),
                "minimum": min(evidence_strengths),
                "strong_evidence_count": len([s for s in evidence_strengths if s > 0.7]),
                "weak_evidence_count": len([s for s in evidence_strengths if s < 0.4])
            }
            
            portfolio["portfolio_score"] = sum(evidence_strengths) / len(evidence_strengths)
        
        portfolio["evidence_types"] = evidence_types
        
        # Analyze coverage of legal elements
        portfolio["coverage_analysis"] = self._analyze_legal_coverage(previous_results)
        
        return portfolio
    
    def _analyze_legal_coverage(self, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coverage of legal elements across evidence"""
        coverage = {
            "covered_elements": [],
            "missing_elements": [],
            "weak_elements": [],
            "coverage_score": 0.0
        }
        
        # Extract legal analysis if available
        legal_analysis = previous_results.get("LegalSpecialist", {})
        if isinstance(legal_analysis, dict):
            findings = legal_analysis.get("findings", {})
            element_analysis = findings.get("element_analysis", {})
            
            for theory_name, theory_data in element_analysis.items():
                elements = theory_data.get("elements", {})
                strong_elements = theory_data.get("strong_elements", [])
                missing_elements = theory_data.get("missing_elements", [])
                
                coverage["covered_elements"].extend(strong_elements)
                coverage["missing_elements"].extend(missing_elements)
                
                # Identify weak elements
                for element, score in elements.items():
                    if 0.3 <= score <= 0.6:
                        coverage["weak_elements"].append(element)
            
            # Calculate coverage score
            total_elements = len(coverage["covered_elements"]) + len(coverage["missing_elements"]) + len(coverage["weak_elements"])
            if total_elements > 0:
                coverage["coverage_score"] = len(coverage["covered_elements"]) / total_elements
        
        return coverage
    
    async def _develop_case_strategy(self, data: Dict[str, Any], previous_results: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Develop overall case strategy"""
        strategy = {
            "primary_strategy": "unknown",
            "strategic_themes": [],
            "evidence_sequencing": [],
            "narrative_approach": "",
            "strategic_priorities": []
        }
        
        # Analyze legal theories from previous results
        legal_analysis = previous_results.get("LegalSpecialist", {})
        if isinstance(legal_analysis, dict):
            findings = legal_analysis.get("findings", {})
            legal_theories = findings.get("legal_theories", [])
            
            if legal_theories:
                # Use strongest legal theory as primary strategy
                strongest_theory = max(legal_theories, key=lambda x: x.get("confidence", 0))
                strategy["primary_strategy"] = strongest_theory.get("theory", "unknown")
                
                # Develop strategic themes
                strategy["strategic_themes"] = self._develop_strategic_themes(legal_theories, previous_results)
        
        # Analyze timeline for narrative approach
        timeline_analysis = previous_results.get("Timeline", {})
        if isinstance(timeline_analysis, dict):
            timeline_findings = timeline_analysis.get("findings", {})
            ai_timeline = timeline_findings.get("ai_analysis", {})
            narrative = ai_timeline.get("timeline_narrative", "")
            
            if narrative:
                strategy["narrative_approach"] = narrative
        
        # Develop evidence sequencing strategy
        strategy["evidence_sequencing"] = self._develop_evidence_sequencing(previous_results)
        
        # Set strategic priorities
        strategy["strategic_priorities"] = self._identify_strategic_priorities(previous_results)
        
        return strategy
    
    def _develop_strategic_themes(self, legal_theories: List[Dict[str, Any]], previous_results: Dict[str, Any]) -> List[str]:
        """Develop strategic themes for the case"""
        themes = []
        
        # Theme based on legal theories
        for theory in legal_theories:
            theory_name = theory.get("theory", "")
            if "fraud" in theory_name:
                themes.append("Pattern of Deceptive Conduct")
            elif "breach" in theory_name:
                themes.append("Violation of Trust and Agreement")
            elif "negligence" in theory_name:
                themes.append("Failure to Meet Standard of Care")
            elif "discrimination" in theory_name:
                themes.append("Systematic Unfair Treatment")
        
        # Theme based on patterns
        pattern_analysis = previous_results.get("PatternDiscovery", {})
        if isinstance(pattern_analysis, dict):
            pattern_findings = pattern_analysis.get("findings", {})
            behavioral_patterns = pattern_findings.get("behavioral_patterns", {})
            
            if behavioral_patterns.get("escalation_patterns"):
                themes.append("Escalating Pattern of Misconduct")
            
            if behavioral_patterns.get("control_patterns"):
                themes.append("Systematic Control and Manipulation")
        
        return list(set(themes))  # Remove duplicates
    
    def _develop_evidence_sequencing(self, previous_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Develop evidence presentation sequence"""
        sequencing = []
        
        # Get evidence strength data
        evidence_items = []
        for agent_name, result in previous_results.items():
            if isinstance(result, dict) and not result.get("error"):
                strength = result.get("evidence_strength", 0.0)
                significance = result.get("legal_significance", "")
                
                evidence_items.append({
                    "source": agent_name,
                    "strength": strength,
                    "significance": significance,
                    "type": result.get("analysis_type", "unknown")
                })
        
        # Sort by strength (strongest first for opening, or build up to strongest)
        evidence_items.sort(key=lambda x: x["strength"], reverse=True)
        
        # Develop sequencing strategy
        if len(evidence_items) >= 3:
            sequencing = [
                {
                    "phase": "opening_impact",
                    "evidence": evidence_items[0],
                    "purpose": "Establish strong first impression"
                },
                {
                    "phase": "foundation_building", 
                    "evidence": evidence_items[1:-1],
                    "purpose": "Build comprehensive case foundation"
                },
                {
                    "phase": "closing_strength",
                    "evidence": evidence_items[-1],
                    "purpose": "End with memorable impact"
                }
            ]
        
        return sequencing
    
    def _identify_strategic_priorities(self, previous_results: Dict[str, Any]) -> List[str]:
        """Identify strategic priorities for case development"""
        priorities = []
        
        # Priority based on evidence gaps
        legal_analysis = previous_results.get("LegalSpecialist", {})
        if isinstance(legal_analysis, dict):
            findings = legal_analysis.get("findings", {})
            element_analysis = findings.get("element_analysis", {})
            
            for theory_name, theory_data in element_analysis.items():
                missing_elements = theory_data.get("missing_elements", [])
                if missing_elements:
                    priorities.append(f"Strengthen evidence for {theory_name}: {', '.join(missing_elements[:2])}")
        
        # Priority based on admissibility issues
        evidence_analysis = previous_results.get("EvidenceAnalyst", {})
        if isinstance(evidence_analysis, dict):
            findings = evidence_analysis.get("findings", {})
            admissibility = findings.get("admissibility_analysis", {})
            issues = admissibility.get("admissibility_issues", [])
            
            if issues:
                priorities.append(f"Address admissibility concerns: {', '.join(issues[:2])}")
        
        # Priority based on timeline gaps
        timeline_analysis = previous_results.get("Timeline", {})
        if isinstance(timeline_analysis, dict):
            findings = timeline_analysis.get("findings", {})
            gap_analysis = findings.get("gap_analysis", {})
            significant_gaps = gap_analysis.get("significant_gaps", [])
            
            if significant_gaps:
                priorities.append("Fill significant timeline gaps with additional evidence")
        
        return priorities
    
    async def _prioritize_arguments(self, data: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prioritize legal arguments based on strength and strategic value"""
        argument_priority = {
            "primary_arguments": [],
            "secondary_arguments": [],
            "defensive_arguments": [],
            "argument_ranking": []
        }
        
        # Get legal theories and their strengths
        legal_analysis = previous_results.get("LegalSpecialist", {})
        if isinstance(legal_analysis, dict):
            findings = legal_analysis.get("findings", {})
            legal_theories = findings.get("legal_theories", [])
            element_analysis = findings.get("element_analysis", {})
            
            # Score and rank arguments
            argument_scores = []
            for theory in legal_theories:
                theory_name = theory.get("theory", "")
                theory_confidence = theory.get("confidence", 0.0)
                
                # Get element strength for this theory
                element_strength = 0.0
                if theory_name in element_analysis:
                    element_strength = element_analysis[theory_name].get("overall_strength", 0.0)
                
                # Calculate composite score
                composite_score = (theory_confidence * 0.6) + (element_strength * 0.4)
                
                argument_scores.append({
                    "theory": theory_name,
                    "score": composite_score,
                    "confidence": theory_confidence,
                    "element_strength": element_strength,
                    "supporting_evidence": theory.get("supporting_evidence", [])
                })
            
            # Sort by score
            argument_scores.sort(key=lambda x: x["score"], reverse=True)
            
            # Categorize arguments
            for i, arg in enumerate(argument_scores):
                if i == 0 and arg["score"] > 0.7:
                    argument_priority["primary_arguments"].append(arg)
                elif arg["score"] > 0.5:
                    argument_priority["secondary_arguments"].append(arg)
                else:
                    argument_priority["defensive_arguments"].append(arg)
            
            argument_priority["argument_ranking"] = argument_scores
        
        return argument_priority
    
    async def _assess_risks_and_opportunities(self, data: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks and opportunities in the case"""
        assessment = {
            "high_risks": [],
            "medium_risks": [],
            "low_risks": [],
            "opportunities": [],
            "risk_mitigation": {},
            "overall_risk_level": "medium"
        }
        
        # Analyze admissibility risks
        evidence_analysis = previous_results.get("EvidenceAnalyst", {})
        if isinstance(evidence_analysis, dict):
            findings = evidence_analysis.get("findings", {})
            admissibility = findings.get("admissibility_analysis", {})
            
            issues = admissibility.get("admissibility_issues", [])
            for issue in issues:
                if "privilege" in issue.lower():
                    assessment["high_risks"].append(f"Privilege issue: {issue}")
                elif "hearsay" in issue.lower():
                    assessment["medium_risks"].append(f"Hearsay concern: {issue}")
                else:
                    assessment["low_risks"].append(issue)
        
        # Analyze pattern-based opportunities
        pattern_analysis = previous_results.get("PatternDiscovery", {})
        if isinstance(pattern_analysis, dict):
            findings = pattern_analysis.get("findings", {})
            cross_doc_patterns = findings.get("cross_document_patterns", [])
            
            if len(cross_doc_patterns) > 2:
                assessment["opportunities"].append("Strong cross-document patterns for compelling narrative")
            
            behavioral_patterns = findings.get("behavioral_patterns", {})
            if behavioral_patterns.get("escalation_patterns"):
                assessment["opportunities"].append("Clear escalation pattern demonstrates progression")
        
        # Analyze timeline risks/opportunities
        timeline_analysis = previous_results.get("Timeline", {})
        if isinstance(timeline_analysis, dict):
            findings = timeline_analysis.get("findings", {})
            gap_analysis = findings.get("gap_analysis", {})
            
            significant_gaps = gap_analysis.get("significant_gaps", [])
            if len(significant_gaps) > 2:
                assessment["medium_risks"].append("Multiple significant timeline gaps may weaken narrative")
            
            pattern_analysis_timeline = findings.get("pattern_analysis", {})
            patterns = pattern_analysis_timeline.get("patterns", [])
            if patterns:
                assessment["opportunities"].append("Timeline patterns support case theory")
        
        # Calculate overall risk level
        high_risk_count = len(assessment["high_risks"])
        medium_risk_count = len(assessment["medium_risks"])
        opportunity_count = len(assessment["opportunities"])
        
        if high_risk_count > 2:
            assessment["overall_risk_level"] = "high"
        elif opportunity_count > high_risk_count + medium_risk_count:
            assessment["overall_risk_level"] = "low"
        else:
            assessment["overall_risk_level"] = "medium"
        
        # Develop risk mitigation strategies
        assessment["risk_mitigation"] = self._develop_risk_mitigation(assessment)
        
        return assessment
    
    def _develop_risk_mitigation(self, assessment: Dict[str, Any]) -> Dict[str, str]:
        """Develop risk mitigation strategies"""
        mitigation = {}
        
        for risk in assessment["high_risks"]:
            if "privilege" in risk.lower():
                mitigation[risk] = "Research privilege exceptions and potential waivers"
            elif "authentication" in risk.lower():
                mitigation[risk] = "Prepare comprehensive authentication foundation"
        
        for risk in assessment["medium_risks"]:
            if "hearsay" in risk.lower():
                mitigation[risk] = "Research applicable hearsay exceptions"
            elif "timeline" in risk.lower():
                mitigation[risk] = "Seek additional evidence to fill gaps"
        
        return mitigation
    
    async def _analyze_settlement_position(self, data: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze settlement position and leverage"""
        settlement = {
            "settlement_leverage": "medium",
            "leverage_factors": [],
            "settlement_risks": [],
            "negotiation_strengths": [],
            "recommended_approach": ""
        }
        
        # Analyze evidence strength for leverage
        evidence_strengths = []
        for agent_name, result in previous_results.items():
            if isinstance(result, dict) and not result.get("error"):
                strength = result.get("evidence_strength", 0.0)
                evidence_strengths.append(strength)
        
        if evidence_strengths:
            avg_strength = sum(evidence_strengths) / len(evidence_strengths)
            max_strength = max(evidence_strengths)
            
            if avg_strength > 0.7 and max_strength > 0.8:
                settlement["settlement_leverage"] = "high"
                settlement["leverage_factors"].append("Strong evidence portfolio")
            elif avg_strength < 0.4:
                settlement["settlement_leverage"] = "low"
                settlement["settlement_risks"].append("Weak evidence may pressure settlement")
        
        # Analyze legal theory strength
        legal_analysis = previous_results.get("LegalSpecialist", {})
        if isinstance(legal_analysis, dict):
            findings = legal_analysis.get("findings", {})
            legal_theories = findings.get("legal_theories", [])
            
            if legal_theories:
                strongest_theory = max(legal_theories, key=lambda x: x.get("confidence", 0))
                if strongest_theory.get("confidence", 0) > 0.8:
                    settlement["leverage_factors"].append("Strong legal theory")
                    settlement["negotiation_strengths"].append(f"Clear {strongest_theory['theory']} claim")
        
        # Analyze admissibility risks for settlement
        evidence_analysis = previous_results.get("EvidenceAnalyst", {})
        if isinstance(evidence_analysis, dict):
            findings = evidence_analysis.get("findings", {})
            admissibility = findings.get("admissibility_analysis", {})
            
            issues = admissibility.get("admissibility_issues", [])
            if len(issues) > 3:
                settlement["settlement_risks"].append("Multiple admissibility issues increase trial risk")
        
        # Develop recommended approach
        if settlement["settlement_leverage"] == "high":
            settlement["recommended_approach"] = "Aggressive settlement posture with high demands"
        elif settlement["settlement_leverage"] == "low":
            settlement["recommended_approach"] = "Consider early settlement to avoid trial risks"
        else:
            settlement["recommended_approach"] = "Balanced approach with room for negotiation"
        
        return settlement
    
    async def _develop_trial_strategy(self, data: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Develop trial preparation strategy"""
        trial_strategy = {
            "opening_strategy": "",
            "evidence_presentation_order": [],
            "witness_strategy": [],
            "closing_themes": [],
            "jury_considerations": [],
            "trial_timeline": ""
        }
        
        # Develop opening strategy
        legal_analysis = previous_results.get("LegalSpecialist", {})
        if isinstance(legal_analysis, dict):
            findings = legal_analysis.get("findings", {})
            legal_theories = findings.get("legal_theories", [])
            
            if legal_theories:
                strongest_theory = max(legal_theories, key=lambda x: x.get("confidence", 0))
                trial_strategy["opening_strategy"] = f"Lead with {strongest_theory['theory']} theory, emphasizing strongest evidence"
        
        # Evidence presentation order
        evidence_items = []
        for agent_name, result in previous_results.items():
            if isinstance(result, dict) and not result.get("error"):
                evidence_items.append({
                    "source": agent_name,
                    "strength": result.get("evidence_strength", 0.0),
                    "type": result.get("analysis_type", "unknown")
                })
        
        # Sort for trial presentation (build momentum)
        evidence_items.sort(key=lambda x: x["strength"])
        trial_strategy["evidence_presentation_order"] = evidence_items
        
        # Develop closing themes
        pattern_analysis = previous_results.get("PatternDiscovery", {})
        if isinstance(pattern_analysis, dict):
            findings = pattern_analysis.get("findings", {})
            ai_analysis = findings.get("ai_analysis", {})
            
            behavioral_patterns = ai_analysis.get("behavioral_patterns", [])
            trial_strategy["closing_themes"].extend(behavioral_patterns[:3])
        
        return trial_strategy
    
    async def _ai_strategic_analysis(self, data: Dict[str, Any], previous_results: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI for advanced strategic analysis"""
        if not self.ai_service:
            return {}
        
        try:
            # Prepare comprehensive case summary
            case_summary = self._prepare_case_summary(previous_results)
            case_context = context.get('case_theory', {}) if context else {}
            
            prompt = f"""
Develop comprehensive case strategy based on this legal analysis:

Case Summary: {case_summary}
Case Context: {case_context}

Provide strategic analysis in JSON format:
{{
    "overall_case_strength": "strong|moderate|weak",
    "primary_strategy_recommendation": "detailed strategy approach",
    "key_strategic_advantages": ["advantage1", "advantage2"],
    "major_vulnerabilities": ["vulnerability1", "vulnerability2"],
    "settlement_recommendation": {{
        "approach": "aggressive|balanced|defensive",
        "leverage_assessment": "high|medium|low",
        "timing_recommendation": "early|mid_discovery|pre_trial"
    }},
    "trial_strategy": {{
        "opening_approach": "strategy for opening statement",
        "evidence_sequencing": "how to present evidence",
        "closing_themes": ["theme1", "theme2", "theme3"]
    }},
    "immediate_priorities": ["priority1", "priority2", "priority3"],
    "discovery_strategy": ["what", "to", "seek"],
    "expert_witness_needs": ["type1", "type2"],
    "motion_practice_recommendations": ["motion1", "motion2"],
    "timeline_to_trial": "estimated preparation time needed"
}}
"""
            
            response = await self.ai_service.provider.generate_completion(
                prompt,
                "You are an expert trial attorney and case strategist with 20+ years of experience in complex litigation."
            )
            
            if response.success:
                try:
                    return json.loads(response.content)
                except json.JSONDecodeError:
                    return {"ai_summary": response.content}
            
        except Exception as e:
            self.logger.error(f"AI strategic analysis failed: {e}")
        
        return {}
    
    def _prepare_case_summary(self, previous_results: Dict[str, Any]) -> str:
        """Prepare comprehensive case summary for AI analysis"""
        summary_parts = []
        
        # Legal analysis summary
        legal_analysis = previous_results.get("LegalSpecialist", {})
        if isinstance(legal_analysis, dict):
            findings = legal_analysis.get("findings", {})
            legal_theories = findings.get("legal_theories", [])
            if legal_theories:
                theories_summary = [f"{t['theory']} (confidence: {t['confidence']:.2f})" for t in legal_theories]
                summary_parts.append(f"Legal Theories: {', '.join(theories_summary)}")
        
        # Evidence analysis summary
        evidence_analysis = previous_results.get("EvidenceAnalyst", {})
        if isinstance(evidence_analysis, dict):
            strength = evidence_analysis.get("evidence_strength", 0.0)
            summary_parts.append(f"Evidence Strength: {strength:.2f}")
        
        # Pattern analysis summary
        pattern_analysis = previous_results.get("PatternDiscovery", {})
        if isinstance(pattern_analysis, dict):
            findings = pattern_analysis.get("findings", {})
            cross_patterns = findings.get("cross_document_patterns", [])
            summary_parts.append(f"Cross-document Patterns: {len(cross_patterns)}")
        
        # Timeline summary
        timeline_analysis = previous_results.get("Timeline", {})
        if isinstance(timeline_analysis, dict):
            findings = timeline_analysis.get("findings", {})
            events = findings.get("temporal_events", [])
            summary_parts.append(f"Timeline Events: {len(events)}")
        
        return "; ".join(summary_parts)
    
    def calculate_confidence(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate confidence in strategic analysis"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on portfolio strength
        portfolio = analysis_data.get("portfolio_analysis", {})
        portfolio_score = portfolio.get("portfolio_score", 0.0)
        confidence += portfolio_score * 0.3
        
        # Increase confidence based on argument strength
        arguments = analysis_data.get("argument_analysis", {})
        primary_args = arguments.get("primary_arguments", [])
        if primary_args:
            avg_primary_score = sum(arg.get("score", 0.0) for arg in primary_args) / len(primary_args)
            confidence += avg_primary_score * 0.2
        
        # Increase confidence if AI analysis is available
        ai_analysis = analysis_data.get("ai_analysis", {})
        if ai_analysis:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_strategic_strength(self, findings: Dict[str, Any]) -> float:
        """Calculate strategic strength of the case"""
        strength = 0.3  # Base strength
        
        # Portfolio contribution
        portfolio = findings.get("portfolio_analysis", {})
        portfolio_score = portfolio.get("portfolio_score", 0.0)
        strength += portfolio_score * 0.4
        
        # Argument strength contribution
        arguments = findings.get("argument_analysis", {})
        primary_args = arguments.get("primary_arguments", [])
        if primary_args:
            strength += 0.2
        
        # Risk assessment contribution
        risk_assessment = findings.get("risk_assessment", {})
        risk_level = risk_assessment.get("overall_risk_level", "medium")
        if risk_level == "low":
            strength += 0.2
        elif risk_level == "high":
            strength -= 0.1
        
        # AI strategic assessment
        ai_analysis = findings.get("ai_analysis", {})
        case_strength = ai_analysis.get("overall_case_strength", "moderate")
        if case_strength == "strong":
            strength += 0.2
        elif case_strength == "weak":
            strength -= 0.1
        
        return max(0.0, min(strength, 1.0))
    
    def extract_legal_significance(self, findings: Dict[str, Any]) -> str:
        """Extract legal significance from strategic analysis"""
        portfolio = findings.get("portfolio_analysis", {})
        arguments = findings.get("argument_analysis", {})
        ai_analysis = findings.get("ai_analysis", {})
        
        portfolio_score = portfolio.get("portfolio_score", 0.0)
        primary_args = len(arguments.get("primary_arguments", []))
        case_strength = ai_analysis.get("overall_case_strength", "unknown")
        
        significance = f"Strategic analysis shows {case_strength} case with portfolio score of {portfolio_score:.2f}"
        
        if primary_args > 0:
            significance += f" and {primary_args} primary legal argument{'s' if primary_args > 1 else ''}"
        
        settlement_rec = ai_analysis.get("settlement_recommendation", {})
        leverage = settlement_rec.get("leverage_assessment", "unknown")
        if leverage != "unknown":
            significance += f" with {leverage} settlement leverage"
        
        return significance + "."
    
    def _generate_recommendations(self, findings: Dict[str, Any]) -> List[str]:
        """Generate strategic recommendations"""
        recommendations = []
        
        # Portfolio-based recommendations
        portfolio = findings.get("portfolio_analysis", {})
        coverage = portfolio.get("coverage_analysis", {})
        missing_elements = coverage.get("missing_elements", [])
        if missing_elements:
            recommendations.append(f"Strengthen case by addressing missing elements: {', '.join(missing_elements[:3])}")
        
        # Risk-based recommendations
        risk_assessment = findings.get("risk_assessment", {})
        high_risks = risk_assessment.get("high_risks", [])
        if high_risks:
            recommendations.append(f"Immediately address high-risk issues: {len(high_risks)} identified")
        
        # Settlement recommendations
        settlement = findings.get("settlement_analysis", {})
        recommended_approach = settlement.get("recommended_approach", "")
        if recommended_approach:
            recommendations.append(f"Settlement strategy: {recommended_approach}")
        
        # AI recommendations
        ai_analysis = findings.get("ai_analysis", {})
        immediate_priorities = ai_analysis.get("immediate_priorities", [])
        recommendations.extend([f"Priority: {priority}" for priority in immediate_priorities[:3]])
        
        discovery_strategy = ai_analysis.get("discovery_strategy", [])
        if discovery_strategy:
            recommendations.append(f"Discovery focus: {', '.join(discovery_strategy[:3])}")
        
        return recommendations
    
    def _create_error_result(self, error_message: str, start_time: datetime) -> AgentResult:
        """Create an error result"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AgentResult(
            agent_name=self.name,
            analysis_type="case_strategy",
            confidence=0.0,
            findings={"error": error_message},
            recommendations=["Review input data and try again"],
            evidence_strength=0.0,
            legal_significance="Analysis failed",
            processing_time=processing_time,
            timestamp=datetime.now(),
            metadata={"error": True}
        )