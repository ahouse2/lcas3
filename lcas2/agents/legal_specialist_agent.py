"""
Legal Specialist Agent
Specialized in legal theory analysis, case law research, and legal argument development
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from .base_agent import BaseAgent, AgentResult

class LegalSpecialistAgent(BaseAgent):
    """Agent specialized in legal analysis and case theory development"""
    
    def __init__(self, ai_service=None, config: Dict[str, Any] = None):
        super().__init__("LegalSpecialist", ai_service, config)
        
        # Legal analysis frameworks
        self.legal_frameworks = {
            "contract_law": {
                "elements": ["offer", "acceptance", "consideration", "capacity", "legality"],
                "defenses": ["duress", "undue_influence", "mistake", "fraud", "unconscionability"]
            },
            "tort_law": {
                "elements": ["duty", "breach", "causation", "damages"],
                "defenses": ["comparative_negligence", "assumption_of_risk", "statute_of_limitations"]
            },
            "family_law": {
                "elements": ["jurisdiction", "grounds", "property_division", "custody", "support"],
                "factors": ["best_interests", "financial_capacity", "parental_fitness"]
            },
            "employment_law": {
                "elements": ["employment_relationship", "protected_class", "adverse_action", "causation"],
                "defenses": ["legitimate_business_reason", "mixed_motive", "after_acquired_evidence"]
            }
        }
        
    def get_capabilities(self) -> List[str]:
        return [
            "legal_theory_analysis",
            "case_law_research_guidance",
            "element_analysis",
            "defense_identification",
            "legal_argument_development",
            "jurisdiction_analysis",
            "statute_of_limitations_analysis",
            "damages_assessment"
        ]
    
    async def analyze(self, data: Any, context: Dict[str, Any] = None) -> AgentResult:
        """Analyze evidence for legal theories and arguments"""
        start_time = datetime.now()
        
        if not await self.validate_input(data):
            return self._create_error_result("Invalid input data", start_time)
        
        try:
            # Identify applicable legal theories
            legal_theories = await self._identify_legal_theories(data, context)
            
            # Analyze legal elements
            element_analysis = await self._analyze_legal_elements(data, legal_theories, context)
            
            # Identify potential defenses
            defense_analysis = await self._analyze_potential_defenses(data, legal_theories, context)
            
            # Assess legal arguments
            argument_analysis = await self._develop_legal_arguments(data, legal_theories, context)
            
            # Jurisdiction and procedural analysis
            procedural_analysis = await self._analyze_procedural_issues(data, context)
            
            # AI-enhanced legal analysis
            ai_analysis = {}
            if self.ai_service:
                ai_analysis = await self._ai_legal_analysis(data, legal_theories, context)
            
            findings = {
                "legal_theories": legal_theories,
                "element_analysis": element_analysis,
                "defense_analysis": defense_analysis,
                "argument_analysis": argument_analysis,
                "procedural_analysis": procedural_analysis,
                "ai_analysis": ai_analysis
            }
            
            confidence = self.calculate_confidence(findings)
            evidence_strength = self._calculate_legal_strength(findings)
            legal_significance = self.extract_legal_significance(findings)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                agent_name=self.name,
                analysis_type="legal_analysis",
                confidence=confidence,
                findings=findings,
                recommendations=self._generate_recommendations(findings),
                evidence_strength=evidence_strength,
                legal_significance=legal_significance,
                processing_time=processing_time,
                timestamp=datetime.now(),
                metadata={"file_path": data.get('file_path', ''), "legal_theories": [t["theory"] for t in legal_theories]}
            )
            
        except Exception as e:
            self.logger.error(f"Legal analysis failed: {e}")
            return self._create_error_result(str(e), start_time)
    
    async def _identify_legal_theories(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify applicable legal theories based on evidence"""
        content = data.get('content', '')
        doc_type = data.get('document_type', {}).get('primary_type', 'unknown')
        
        theories = []
        
        # Contract-related theories
        contract_indicators = ["agreement", "contract", "breach", "performance", "consideration", "offer", "acceptance"]
        if any(indicator in content.lower() for indicator in contract_indicators):
            theories.append({
                "theory": "breach_of_contract",
                "confidence": 0.7,
                "supporting_evidence": [indicator for indicator in contract_indicators if indicator in content.lower()],
                "framework": "contract_law"
            })
        
        # Tort theories
        tort_indicators = ["negligence", "duty", "damages", "injury", "harm", "fault", "liability"]
        if any(indicator in content.lower() for indicator in tort_indicators):
            theories.append({
                "theory": "negligence",
                "confidence": 0.6,
                "supporting_evidence": [indicator for indicator in tort_indicators if indicator in content.lower()],
                "framework": "tort_law"
            })
        
        # Fraud theories
        fraud_indicators = ["fraud", "misrepresentation", "deception", "false", "misleading", "concealment"]
        if any(indicator in content.lower() for indicator in fraud_indicators):
            theories.append({
                "theory": "fraud",
                "confidence": 0.8,
                "supporting_evidence": [indicator for indicator in fraud_indicators if indicator in content.lower()],
                "framework": "tort_law"
            })
        
        # Employment theories
        employment_indicators = ["discrimination", "harassment", "wrongful termination", "retaliation", "hostile work environment"]
        if any(indicator in content.lower() for indicator in employment_indicators):
            theories.append({
                "theory": "employment_discrimination",
                "confidence": 0.7,
                "supporting_evidence": [indicator for indicator in employment_indicators if indicator in content.lower()],
                "framework": "employment_law"
            })
        
        # Family law theories
        family_indicators = ["divorce", "custody", "support", "alimony", "property division", "domestic violence"]
        if any(indicator in content.lower() for indicator in family_indicators):
            theories.append({
                "theory": "family_law_matter",
                "confidence": 0.8,
                "supporting_evidence": [indicator for indicator in family_indicators if indicator in content.lower()],
                "framework": "family_law"
            })
        
        # Sort by confidence
        theories.sort(key=lambda x: x["confidence"], reverse=True)
        
        return theories
    
    async def _analyze_legal_elements(self, data: Dict[str, Any], 
                                    legal_theories: List[Dict[str, Any]], 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze legal elements for identified theories"""
        content = data.get('content', '')
        element_analysis = {}
        
        for theory in legal_theories:
            framework = theory.get("framework")
            theory_name = theory.get("theory")
            
            if framework in self.legal_frameworks:
                elements = self.legal_frameworks[framework].get("elements", [])
                
                theory_elements = {}
                for element in elements:
                    # Simple keyword-based element detection
                    element_score = self._assess_element_presence(content, element, theory_name)
                    theory_elements[element] = element_score
                
                element_analysis[theory_name] = {
                    "elements": theory_elements,
                    "overall_strength": sum(theory_elements.values()) / len(theory_elements) if theory_elements else 0.0,
                    "missing_elements": [elem for elem, score in theory_elements.items() if score < 0.3],
                    "strong_elements": [elem for elem, score in theory_elements.items() if score > 0.7]
                }
        
        return element_analysis
    
    def _assess_element_presence(self, content: str, element: str, theory: str) -> float:
        """Assess the presence and strength of a legal element in content"""
        content_lower = content.lower()
        
        # Element-specific keyword mappings
        element_keywords = {
            "offer": ["offer", "proposal", "quote", "bid", "invitation"],
            "acceptance": ["accept", "agree", "consent", "approve", "confirm"],
            "consideration": ["consideration", "payment", "exchange", "value", "benefit"],
            "duty": ["duty", "obligation", "responsibility", "owe", "required"],
            "breach": ["breach", "violation", "break", "fail", "default"],
            "causation": ["cause", "result", "because", "due to", "led to"],
            "damages": ["damages", "loss", "harm", "injury", "cost"],
            "jurisdiction": ["court", "jurisdiction", "venue", "state", "federal"],
            "grounds": ["grounds", "basis", "reason", "cause", "fault"]
        }
        
        keywords = element_keywords.get(element, [element])
        
        # Count keyword occurrences
        keyword_count = sum(1 for keyword in keywords if keyword in content_lower)
        
        # Base score from keyword presence
        base_score = min(keyword_count * 0.2, 0.6)
        
        # Context-specific adjustments
        if element == "damages" and any(indicator in content_lower for indicator in ["$", "amount", "cost", "expense"]):
            base_score += 0.3
        
        if element == "breach" and any(indicator in content_lower for indicator in ["failed to", "did not", "refused"]):
            base_score += 0.2
        
        if element == "duty" and any(indicator in content_lower for indicator in ["contract", "agreement", "law", "statute"]):
            base_score += 0.2
        
        return min(base_score, 1.0)
    
    async def _analyze_potential_defenses(self, data: Dict[str, Any],
                                        legal_theories: List[Dict[str, Any]],
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential defenses for each legal theory"""
        content = data.get('content', '')
        defense_analysis = {}
        
        for theory in legal_theories:
            framework = theory.get("framework")
            theory_name = theory.get("theory")
            
            if framework in self.legal_frameworks:
                defenses = self.legal_frameworks[framework].get("defenses", [])
                
                theory_defenses = {}
                for defense in defenses:
                    defense_score = self._assess_defense_presence(content, defense, theory_name)
                    if defense_score > 0.2:  # Only include if there's some indication
                        theory_defenses[defense] = defense_score
                
                if theory_defenses:
                    defense_analysis[theory_name] = {
                        "potential_defenses": theory_defenses,
                        "strongest_defense": max(theory_defenses.keys(), key=lambda k: theory_defenses[k]) if theory_defenses else None,
                        "defense_strength": max(theory_defenses.values()) if theory_defenses else 0.0
                    }
        
        return defense_analysis
    
    def _assess_defense_presence(self, content: str, defense: str, theory: str) -> float:
        """Assess the presence of potential defenses"""
        content_lower = content.lower()
        
        defense_keywords = {
            "duress": ["duress", "coercion", "threat", "force", "pressure"],
            "fraud": ["fraud", "misrepresentation", "deception", "false"],
            "mistake": ["mistake", "error", "misunderstanding", "confusion"],
            "comparative_negligence": ["contributory", "comparative", "fault", "negligence"],
            "assumption_of_risk": ["assumed", "risk", "voluntary", "aware"],
            "statute_of_limitations": ["statute of limitations", "time limit", "expired", "untimely"],
            "legitimate_business_reason": ["business", "legitimate", "reason", "justification"]
        }
        
        keywords = defense_keywords.get(defense, [defense])
        keyword_count = sum(1 for keyword in keywords if keyword in content_lower)
        
        return min(keyword_count * 0.3, 1.0)
    
    async def _develop_legal_arguments(self, data: Dict[str, Any],
                                     legal_theories: List[Dict[str, Any]],
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Develop legal arguments based on evidence and theories"""
        argument_analysis = {}
        
        for theory in legal_theories:
            theory_name = theory.get("theory")
            confidence = theory.get("confidence", 0.5)
            
            # Develop argument structure
            argument_structure = {
                "primary_argument": f"Evidence supports {theory_name.replace('_', ' ')} claim",
                "supporting_points": theory.get("supporting_evidence", []),
                "argument_strength": confidence,
                "evidence_gaps": [],
                "counterarguments": []
            }
            
            # Identify evidence gaps
            if confidence < 0.7:
                argument_structure["evidence_gaps"].append("Additional evidence needed to strengthen theory")
            
            if confidence < 0.5:
                argument_structure["evidence_gaps"].append("Core elements may be difficult to prove")
            
            argument_analysis[theory_name] = argument_structure
        
        return argument_analysis
    
    async def _analyze_procedural_issues(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze procedural and jurisdictional issues"""
        content = data.get('content', '')
        
        procedural_analysis = {
            "jurisdiction_indicators": [],
            "statute_of_limitations_concerns": False,
            "venue_considerations": [],
            "procedural_requirements": []
        }
        
        # Jurisdiction analysis
        jurisdiction_indicators = ["federal court", "state court", "district court", "superior court", "municipal court"]
        for indicator in jurisdiction_indicators:
            if indicator in content.lower():
                procedural_analysis["jurisdiction_indicators"].append(indicator)
        
        # Statute of limitations
        sol_indicators = ["statute of limitations", "time limit", "deadline", "expired", "untimely"]
        if any(indicator in content.lower() for indicator in sol_indicators):
            procedural_analysis["statute_of_limitations_concerns"] = True
        
        # Venue considerations
        venue_indicators = ["venue", "forum", "jurisdiction", "where filed", "proper court"]
        for indicator in venue_indicators:
            if indicator in content.lower():
                procedural_analysis["venue_considerations"].append(indicator)
        
        return procedural_analysis
    
    async def _ai_legal_analysis(self, data: Dict[str, Any], 
                               legal_theories: List[Dict[str, Any]], 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI for advanced legal analysis"""
        if not self.ai_service:
            return {}
        
        try:
            content = data.get('content', '')
            theories_summary = [f"{t['theory']} (confidence: {t['confidence']:.2f})" for t in legal_theories]
            
            prompt = f"""
Analyze this legal evidence for case strategy:

Identified Legal Theories: {theories_summary}
Evidence Content: {content[:2000]}

Provide comprehensive legal analysis in JSON format:
{{
    "strongest_legal_theory": "theory name and explanation",
    "element_analysis": {{
        "proven_elements": ["list", "of", "elements"],
        "disputed_elements": ["list", "of", "elements"],
        "missing_elements": ["list", "of", "elements"]
    }},
    "case_law_research_priorities": ["area1", "area2", "area3"],
    "legal_argument_strategy": "detailed strategy recommendation",
    "potential_motions": ["motion1", "motion2"],
    "discovery_recommendations": ["what", "to", "seek"],
    "settlement_leverage": "high|medium|low",
    "trial_strategy": "recommended approach",
    "legal_risks": ["risk1", "risk2"],
    "precedent_research_needed": ["area1", "area2"]
}}
"""
            
            response = await self.ai_service.provider.generate_completion(
                prompt,
                "You are an expert trial attorney and legal strategist with deep knowledge of case law and legal procedure."
            )
            
            if response.success:
                try:
                    return json.loads(response.content)
                except json.JSONDecodeError:
                    return {"ai_summary": response.content}
            
        except Exception as e:
            self.logger.error(f"AI legal analysis failed: {e}")
        
        return {}
    
    def calculate_confidence(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate confidence in legal analysis"""
        confidence = 0.5  # Base confidence
        
        legal_theories = analysis_data.get("legal_theories", [])
        if legal_theories:
            # Use highest theory confidence
            max_theory_confidence = max(theory.get("confidence", 0) for theory in legal_theories)
            confidence += max_theory_confidence * 0.3
        
        element_analysis = analysis_data.get("element_analysis", {})
        if element_analysis:
            # Average element strength across theories
            element_strengths = [theory_data.get("overall_strength", 0) 
                               for theory_data in element_analysis.values()]
            if element_strengths:
                avg_element_strength = sum(element_strengths) / len(element_strengths)
                confidence += avg_element_strength * 0.2
        
        ai_analysis = analysis_data.get("ai_analysis", {})
        if ai_analysis:
            confidence += 0.1  # Bonus for AI analysis
        
        return min(confidence, 1.0)
    
    def _calculate_legal_strength(self, findings: Dict[str, Any]) -> float:
        """Calculate overall legal strength"""
        strength = 0.3  # Base strength
        
        legal_theories = findings.get("legal_theories", [])
        if legal_theories:
            # Use strongest theory
            max_theory_confidence = max(theory.get("confidence", 0) for theory in legal_theories)
            strength += max_theory_confidence * 0.4
        
        element_analysis = findings.get("element_analysis", {})
        if element_analysis:
            # Consider element completeness
            for theory_name, theory_data in element_analysis.items():
                element_strength = theory_data.get("overall_strength", 0)
                missing_elements = len(theory_data.get("missing_elements", []))
                
                if missing_elements == 0:
                    strength += 0.2
                elif missing_elements <= 2:
                    strength += 0.1
        
        ai_analysis = findings.get("ai_analysis", {})
        settlement_leverage = ai_analysis.get("settlement_leverage", "low")
        if settlement_leverage == "high":
            strength += 0.1
        elif settlement_leverage == "medium":
            strength += 0.05
        
        return min(strength, 1.0)
    
    def extract_legal_significance(self, findings: Dict[str, Any]) -> str:
        """Extract legal significance from analysis"""
        legal_theories = findings.get("legal_theories", [])
        
        if not legal_theories:
            return "No clear legal theories identified"
        
        strongest_theory = max(legal_theories, key=lambda x: x.get("confidence", 0))
        theory_name = strongest_theory.get("theory", "unknown").replace("_", " ")
        confidence = strongest_theory.get("confidence", 0)
        
        ai_analysis = findings.get("ai_analysis", {})
        settlement_leverage = ai_analysis.get("settlement_leverage", "unknown")
        
        return f"Evidence supports {theory_name} with {confidence:.1%} confidence and {settlement_leverage} settlement leverage."
    
    def _generate_recommendations(self, findings: Dict[str, Any]) -> List[str]:
        """Generate legal recommendations"""
        recommendations = []
        
        legal_theories = findings.get("legal_theories", [])
        element_analysis = findings.get("element_analysis", {})
        ai_analysis = findings.get("ai_analysis", {})
        
        # Theory-based recommendations
        if legal_theories:
            strongest_theory = max(legal_theories, key=lambda x: x.get("confidence", 0))
            if strongest_theory.get("confidence", 0) > 0.7:
                recommendations.append(f"Focus case strategy on {strongest_theory['theory'].replace('_', ' ')}")
            else:
                recommendations.append("Seek additional evidence to strengthen legal theories")
        
        # Element-based recommendations
        for theory_name, theory_data in element_analysis.items():
            missing_elements = theory_data.get("missing_elements", [])
            if missing_elements:
                recommendations.append(f"Gather evidence for missing {theory_name} elements: {', '.join(missing_elements[:3])}")
        
        # AI recommendations
        discovery_recs = ai_analysis.get("discovery_recommendations", [])
        if discovery_recs:
            recommendations.append(f"Discovery focus: {', '.join(discovery_recs[:3])}")
        
        case_law_priorities = ai_analysis.get("case_law_research_priorities", [])
        if case_law_priorities:
            recommendations.append(f"Research case law on: {', '.join(case_law_priorities[:2])}")
        
        potential_motions = ai_analysis.get("potential_motions", [])
        if potential_motions:
            recommendations.append(f"Consider filing: {', '.join(potential_motions[:2])}")
        
        return recommendations
    
    def _create_error_result(self, error_message: str, start_time: datetime) -> AgentResult:
        """Create an error result"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AgentResult(
            agent_name=self.name,
            analysis_type="legal_analysis",
            confidence=0.0,
            findings={"error": error_message},
            recommendations=["Review input data and try again"],
            evidence_strength=0.0,
            legal_significance="Analysis failed",
            processing_time=processing_time,
            timestamp=datetime.now(),
            metadata={"error": True}
        )