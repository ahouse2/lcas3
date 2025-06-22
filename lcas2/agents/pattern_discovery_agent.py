"""
Pattern Discovery Agent
Specialized in identifying patterns, relationships, and connections across evidence
"""

import json
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict, Counter
from .base_agent import BaseAgent, AgentResult

class PatternDiscoveryAgent(BaseAgent):
    """Agent specialized in discovering patterns and relationships in evidence"""
    
    def __init__(self, ai_service=None, config: Dict[str, Any] = None):
        super().__init__("PatternDiscovery", ai_service, config)
        
    def get_capabilities(self) -> List[str]:
        return [
            "cross_document_pattern_analysis",
            "entity_relationship_mapping",
            "behavioral_pattern_detection",
            "communication_pattern_analysis",
            "financial_pattern_detection",
            "temporal_pattern_correlation",
            "anomaly_detection"
        ]
    
    async def analyze(self, data: Any, context: Dict[str, Any] = None) -> AgentResult:
        """Analyze patterns across evidence"""
        start_time = datetime.now()
        
        if not await self.validate_input(data):
            return self._create_error_result("Invalid input data", start_time)
        
        try:
            # Get previous results for cross-document analysis
            previous_results = context.get('previous_results', {}) if context else {}
            
            # Extract entities and relationships from current document
            entity_analysis = await self._extract_entities_and_relationships(data)
            
            # Analyze communication patterns
            communication_patterns = await self._analyze_communication_patterns(data, previous_results)
            
            # Analyze behavioral patterns
            behavioral_patterns = await self._analyze_behavioral_patterns(data, previous_results)
            
            # Analyze financial patterns
            financial_patterns = await self._analyze_financial_patterns(data, previous_results)
            
            # Cross-document pattern analysis
            cross_doc_patterns = await self._analyze_cross_document_patterns(data, previous_results)
            
            # Anomaly detection
            anomaly_analysis = await self._detect_anomalies(data, previous_results)
            
            # AI-enhanced pattern analysis
            ai_analysis = {}
            if self.ai_service:
                ai_analysis = await self._ai_pattern_analysis(data, previous_results, context)
            
            findings = {
                "entity_analysis": entity_analysis,
                "communication_patterns": communication_patterns,
                "behavioral_patterns": behavioral_patterns,
                "financial_patterns": financial_patterns,
                "cross_document_patterns": cross_doc_patterns,
                "anomaly_analysis": anomaly_analysis,
                "ai_analysis": ai_analysis
            }
            
            confidence = self.calculate_confidence(findings)
            evidence_strength = self._calculate_pattern_strength(findings)
            legal_significance = self.extract_legal_significance(findings)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                agent_name=self.name,
                analysis_type="pattern_analysis",
                confidence=confidence,
                findings=findings,
                recommendations=self._generate_recommendations(findings),
                evidence_strength=evidence_strength,
                legal_significance=legal_significance,
                processing_time=processing_time,
                timestamp=datetime.now(),
                metadata={"file_path": data.get('file_path', ''), "patterns_found": len(cross_doc_patterns)}
            )
            
        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}")
            return self._create_error_result(str(e), start_time)
    
    async def _extract_entities_and_relationships(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities and their relationships from the document"""
        content = data.get('content', '')
        
        entities = {
            "people": self._extract_people(content),
            "organizations": self._extract_organizations(content),
            "locations": self._extract_locations(content),
            "financial_entities": self._extract_financial_entities(content),
            "dates": self._extract_dates(content),
            "communications": self._extract_communication_entities(content)
        }
        
        # Analyze relationships between entities
        relationships = self._analyze_entity_relationships(content, entities)
        
        return {
            "entities": entities,
            "relationships": relationships,
            "entity_density": self._calculate_entity_density(entities, content)
        }
    
    def _extract_people(self, content: str) -> List[Dict[str, Any]]:
        """Extract people mentioned in content"""
        people = []
        
        # Name patterns
        name_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
            r'\b(?:Mr|Mrs|Ms|Dr)\.?\s+[A-Z][a-z]+\b',  # Title Name
        ]
        
        for pattern in name_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                name = match.group().strip()
                
                # Filter out common false positives
                if name not in ["United States", "New York", "Supreme Court"]:
                    people.append({
                        "name": name,
                        "position": match.start(),
                        "context": content[max(0, match.start()-50):match.end()+50]
                    })
        
        # Remove duplicates
        unique_people = []
        seen_names = set()
        for person in people:
            if person["name"] not in seen_names:
                unique_people.append(person)
                seen_names.add(person["name"])
        
        return unique_people
    
    def _extract_organizations(self, content: str) -> List[Dict[str, Any]]:
        """Extract organizations mentioned in content"""
        organizations = []
        
        # Organization patterns
        org_patterns = [
            r'\b[A-Z][a-z]+\s+(?:Inc|Corp|LLC|Ltd|Company|Bank|Hospital|School|University)\b',
            r'\b(?:Bank of|Wells Fargo|Chase|Citibank|American Express)\b',
            r'\b[A-Z][A-Z]+\b'  # Acronyms
        ]
        
        for pattern in org_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                org = match.group().strip()
                organizations.append({
                    "name": org,
                    "position": match.start(),
                    "context": content[max(0, match.start()-50):match.end()+50]
                })
        
        return organizations
    
    def _extract_locations(self, content: str) -> List[Dict[str, Any]]:
        """Extract locations mentioned in content"""
        locations = []
        
        # Location patterns
        location_patterns = [
            r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b',  # City, State
            r'\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd)\b'  # Addresses
        ]
        
        for pattern in location_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                location = match.group().strip()
                locations.append({
                    "location": location,
                    "position": match.start(),
                    "context": content[max(0, match.start()-50):match.end()+50]
                })
        
        return locations
    
    def _extract_financial_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract financial entities and amounts"""
        financial_entities = []
        
        # Financial patterns
        patterns = [
            (r'\$[\d,]+\.?\d*', 'amount'),
            (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', 'account_number'),
            (r'\b(?:checking|savings|investment|retirement)\s+account\b', 'account_type'),
            (r'\b(?:deposit|withdrawal|transfer|payment)\b', 'transaction_type')
        ]
        
        for pattern, entity_type in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                financial_entities.append({
                    "type": entity_type,
                    "value": match.group().strip(),
                    "position": match.start(),
                    "context": content[max(0, match.start()-50):match.end()+50]
                })
        
        return financial_entities
    
    def _extract_dates(self, content: str) -> List[Dict[str, Any]]:
        """Extract dates mentioned in content"""
        dates = []
        
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                dates.append({
                    "date": match.group().strip(),
                    "position": match.start(),
                    "context": content[max(0, match.start()-50):match.end()+50]
                })
        
        return dates
    
    def _extract_communication_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract communication-related entities"""
        communications = []
        
        comm_patterns = [
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email'),
            (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 'phone'),
            (r'\b(?:text|email|call|message|voicemail)\b', 'communication_type')
        ]
        
        for pattern, comm_type in comm_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                communications.append({
                    "type": comm_type,
                    "value": match.group().strip(),
                    "position": match.start(),
                    "context": content[max(0, match.start()-50):match.end()+50]
                })
        
        return communications
    
    def _analyze_entity_relationships(self, content: str, entities: Dict[str, List]) -> List[Dict[str, Any]]:
        """Analyze relationships between entities"""
        relationships = []
        
        # Find co-occurrences of entities
        all_entities = []
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                all_entities.append({
                    "type": entity_type,
                    "value": entity.get("name") or entity.get("value") or entity.get("location") or entity.get("date"),
                    "position": entity["position"]
                })
        
        # Find entities that appear close together
        for i, entity1 in enumerate(all_entities):
            for entity2 in all_entities[i+1:]:
                distance = abs(entity1["position"] - entity2["position"])
                if distance < 200:  # Within 200 characters
                    relationships.append({
                        "entity1": entity1,
                        "entity2": entity2,
                        "relationship_type": "co_occurrence",
                        "distance": distance,
                        "strength": max(0, 1 - distance/200)
                    })
        
        return relationships
    
    def _calculate_entity_density(self, entities: Dict[str, List], content: str) -> Dict[str, float]:
        """Calculate entity density in content"""
        content_length = len(content)
        if content_length == 0:
            return {}
        
        density = {}
        for entity_type, entity_list in entities.items():
            density[entity_type] = len(entity_list) / (content_length / 1000)  # Per 1000 characters
        
        return density
    
    async def _analyze_communication_patterns(self, data: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze communication patterns"""
        content = data.get('content', '')
        doc_type = data.get('document_type', {}).get('primary_type', 'unknown')
        
        patterns = {
            "communication_frequency": {},
            "communication_timing": {},
            "communication_participants": {},
            "communication_escalation": []
        }
        
        # Analyze if this is a communication document
        if doc_type in ['email', 'text_message']:
            # Extract communication metadata
            comm_data = self._extract_communication_metadata(content, doc_type)
            
            # Analyze frequency patterns across previous results
            if previous_results:
                patterns["communication_frequency"] = self._analyze_comm_frequency(comm_data, previous_results)
            
            # Analyze timing patterns
            patterns["communication_timing"] = self._analyze_comm_timing(comm_data)
            
            # Analyze participants
            patterns["communication_participants"] = self._analyze_comm_participants(comm_data)
        
        return patterns
    
    def _extract_communication_metadata(self, content: str, doc_type: str) -> Dict[str, Any]:
        """Extract metadata from communication documents"""
        metadata = {"type": doc_type}
        
        if doc_type == 'email':
            # Extract email headers
            headers = ["from:", "to:", "cc:", "bcc:", "subject:", "date:", "sent:"]
            for header in headers:
                pattern = rf'{header}\s*(.+?)(?:\n|$)'
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    metadata[header.replace(':', '')] = match.group(1).strip()
        
        elif doc_type == 'text_message':
            # Extract text message metadata
            # This would depend on the format of text message exports
            pass
        
        return metadata
    
    def _analyze_comm_frequency(self, comm_data: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze communication frequency patterns"""
        # This would analyze frequency across multiple documents
        # For now, return basic structure
        return {
            "daily_frequency": 0,
            "weekly_frequency": 0,
            "frequency_trend": "unknown"
        }
    
    def _analyze_comm_timing(self, comm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze communication timing patterns"""
        timing_analysis = {
            "time_of_day": "unknown",
            "day_of_week": "unknown",
            "urgency_indicators": []
        }
        
        # Look for urgency indicators
        urgency_keywords = ["urgent", "asap", "immediately", "emergency", "now"]
        content = str(comm_data)
        
        for keyword in urgency_keywords:
            if keyword in content.lower():
                timing_analysis["urgency_indicators"].append(keyword)
        
        return timing_analysis
    
    def _analyze_comm_participants(self, comm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze communication participants"""
        participants = {
            "sender": comm_data.get("from", "unknown"),
            "recipients": [],
            "participant_roles": {}
        }
        
        # Extract recipients
        to_field = comm_data.get("to", "")
        cc_field = comm_data.get("cc", "")
        
        if to_field:
            participants["recipients"].extend([r.strip() for r in to_field.split(",")])
        if cc_field:
            participants["recipients"].extend([r.strip() for r in cc_field.split(",")])
        
        return participants
    
    async def _analyze_behavioral_patterns(self, data: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze behavioral patterns in content"""
        content = data.get('content', '')
        
        patterns = {
            "emotional_indicators": self._analyze_emotional_patterns(content),
            "deception_indicators": self._analyze_deception_patterns(content),
            "control_patterns": self._analyze_control_patterns(content),
            "escalation_patterns": self._analyze_escalation_patterns(content)
        }
        
        return patterns
    
    def _analyze_emotional_patterns(self, content: str) -> Dict[str, Any]:
        """Analyze emotional patterns in content"""
        emotional_indicators = {
            "anger": ["angry", "furious", "mad", "rage", "hate"],
            "fear": ["scared", "afraid", "terrified", "worried", "anxious"],
            "sadness": ["sad", "depressed", "crying", "tears", "hurt"],
            "joy": ["happy", "excited", "thrilled", "pleased", "glad"],
            "frustration": ["frustrated", "annoyed", "irritated", "fed up"]
        }
        
        emotion_scores = {}
        content_lower = content.lower()
        
        for emotion, keywords in emotional_indicators.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        return {
            "emotion_scores": emotion_scores,
            "dominant_emotion": max(emotion_scores.keys(), key=lambda k: emotion_scores[k]) if emotion_scores else None,
            "emotional_intensity": sum(emotion_scores.values())
        }
    
    def _analyze_deception_patterns(self, content: str) -> Dict[str, Any]:
        """Analyze potential deception indicators"""
        deception_indicators = [
            "honestly", "to be honest", "believe me", "trust me", "I swear",
            "never", "always", "everyone knows", "obviously", "clearly"
        ]
        
        content_lower = content.lower()
        found_indicators = [indicator for indicator in deception_indicators if indicator in content_lower]
        
        return {
            "deception_indicators": found_indicators,
            "deception_score": len(found_indicators),
            "certainty_language": len([word for word in ["never", "always", "everyone", "obviously"] if word in content_lower])
        }
    
    def _analyze_control_patterns(self, content: str) -> Dict[str, Any]:
        """Analyze control and manipulation patterns"""
        control_indicators = [
            "you must", "you have to", "you need to", "you should",
            "if you don't", "or else", "you better", "don't tell",
            "keep quiet", "secret", "between us"
        ]
        
        content_lower = content.lower()
        found_indicators = [indicator for indicator in control_indicators if indicator in content_lower]
        
        return {
            "control_indicators": found_indicators,
            "control_score": len(found_indicators),
            "secrecy_language": len([word for word in ["secret", "don't tell", "keep quiet"] if word in content_lower])
        }
    
    def _analyze_escalation_patterns(self, content: str) -> Dict[str, Any]:
        """Analyze escalation patterns"""
        escalation_indicators = [
            "getting worse", "escalating", "more serious", "final warning",
            "last chance", "enough", "fed up", "can't take it"
        ]
        
        content_lower = content.lower()
        found_indicators = [indicator for indicator in escalation_indicators if indicator in content_lower]
        
        return {
            "escalation_indicators": found_indicators,
            "escalation_score": len(found_indicators)
        }
    
    async def _analyze_financial_patterns(self, data: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial patterns"""
        content = data.get('content', '')
        
        patterns = {
            "financial_amounts": self._extract_financial_amounts(content),
            "financial_behaviors": self._analyze_financial_behaviors(content),
            "financial_relationships": self._analyze_financial_relationships(content)
        }
        
        return patterns
    
    def _extract_financial_amounts(self, content: str) -> List[Dict[str, Any]]:
        """Extract financial amounts with context"""
        amounts = []
        
        amount_pattern = r'\$[\d,]+\.?\d*'
        matches = re.finditer(amount_pattern, content)
        
        for match in matches:
            amount_str = match.group()
            context = content[max(0, match.start()-100):match.end()+100]
            
            # Classify the type of amount
            amount_type = self._classify_financial_amount(context)
            
            amounts.append({
                "amount": amount_str,
                "amount_type": amount_type,
                "context": context,
                "position": match.start()
            })
        
        return amounts
    
    def _classify_financial_amount(self, context: str) -> str:
        """Classify the type of financial amount"""
        context_lower = context.lower()
        
        if any(word in context_lower for word in ["salary", "income", "wage", "pay"]):
            return "income"
        elif any(word in context_lower for word in ["deposit", "deposited"]):
            return "deposit"
        elif any(word in context_lower for word in ["withdrawal", "withdrew", "cash"]):
            return "withdrawal"
        elif any(word in context_lower for word in ["transfer", "transferred"]):
            return "transfer"
        elif any(word in context_lower for word in ["payment", "paid", "bill"]):
            return "payment"
        elif any(word in context_lower for word in ["balance", "account"]):
            return "balance"
        else:
            return "unknown"
    
    def _analyze_financial_behaviors(self, content: str) -> Dict[str, Any]:
        """Analyze financial behaviors mentioned in content"""
        behaviors = {
            "hiding_assets": [],
            "unusual_transactions": [],
            "financial_control": []
        }
        
        content_lower = content.lower()
        
        # Asset hiding indicators
        hiding_indicators = ["hide", "conceal", "secret account", "offshore", "cash only"]
        for indicator in hiding_indicators:
            if indicator in content_lower:
                behaviors["hiding_assets"].append(indicator)
        
        # Unusual transaction indicators
        unusual_indicators = ["large withdrawal", "unusual transfer", "suspicious", "unexplained"]
        for indicator in unusual_indicators:
            if indicator in content_lower:
                behaviors["unusual_transactions"].append(indicator)
        
        # Financial control indicators
        control_indicators = ["control money", "access denied", "blocked account", "financial abuse"]
        for indicator in control_indicators:
            if indicator in content_lower:
                behaviors["financial_control"].append(indicator)
        
        return behaviors
    
    def _analyze_financial_relationships(self, content: str) -> Dict[str, Any]:
        """Analyze financial relationships mentioned"""
        relationships = {
            "joint_accounts": [],
            "financial_dependencies": [],
            "financial_disputes": []
        }
        
        content_lower = content.lower()
        
        # Joint account indicators
        if any(phrase in content_lower for phrase in ["joint account", "shared account", "our account"]):
            relationships["joint_accounts"].append("joint_account_mentioned")
        
        # Financial dependency indicators
        if any(phrase in content_lower for phrase in ["financially dependent", "support", "alimony", "maintenance"]):
            relationships["financial_dependencies"].append("dependency_mentioned")
        
        # Financial dispute indicators
        if any(phrase in content_lower for phrase in ["financial dispute", "money fight", "asset division"]):
            relationships["financial_disputes"].append("dispute_mentioned")
        
        return relationships
    
    async def _analyze_cross_document_patterns(self, data: Dict[str, Any], previous_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze patterns across multiple documents"""
        patterns = []
        
        if not previous_results:
            return patterns
        
        current_entities = data.get('entities', {})
        
        # Analyze entity consistency across documents
        entity_patterns = self._analyze_entity_consistency(current_entities, previous_results)
        patterns.extend(entity_patterns)
        
        # Analyze timeline consistency
        timeline_patterns = self._analyze_timeline_consistency(data, previous_results)
        patterns.extend(timeline_patterns)
        
        # Analyze narrative consistency
        narrative_patterns = self._analyze_narrative_consistency(data, previous_results)
        patterns.extend(narrative_patterns)
        
        return patterns
    
    def _analyze_entity_consistency(self, current_entities: Dict[str, Any], previous_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze consistency of entities across documents"""
        patterns = []
        
        # This would compare entities across documents
        # For now, return basic structure
        patterns.append({
            "type": "entity_consistency",
            "description": "Entity consistency analysis across documents",
            "consistency_score": 0.5,
            "inconsistencies": []
        })
        
        return patterns
    
    def _analyze_timeline_consistency(self, data: Dict[str, Any], previous_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze timeline consistency across documents"""
        patterns = []
        
        # This would compare timelines across documents
        patterns.append({
            "type": "timeline_consistency",
            "description": "Timeline consistency analysis across documents",
            "consistency_score": 0.5,
            "conflicts": []
        })
        
        return patterns
    
    def _analyze_narrative_consistency(self, data: Dict[str, Any], previous_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze narrative consistency across documents"""
        patterns = []
        
        # This would compare narratives across documents
        patterns.append({
            "type": "narrative_consistency",
            "description": "Narrative consistency analysis across documents",
            "consistency_score": 0.5,
            "contradictions": []
        })
        
        return patterns
    
    async def _detect_anomalies(self, data: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in the current document"""
        anomalies = {
            "content_anomalies": [],
            "metadata_anomalies": [],
            "pattern_anomalies": []
        }
        
        content = data.get('content', '')
        
        # Content anomalies
        if len(content) < 50:
            anomalies["content_anomalies"].append("unusually_short_content")
        
        if len(content) > 50000:
            anomalies["content_anomalies"].append("unusually_long_content")
        
        # Check for unusual character patterns
        if re.search(r'[^\x00-\x7F]', content):
            anomalies["content_anomalies"].append("non_ascii_characters")
        
        # Check for potential OCR artifacts
        if re.search(r'\b[a-z]\s[a-z]\s[a-z]\b', content):
            anomalies["content_anomalies"].append("potential_ocr_artifacts")
        
        return anomalies
    
    async def _ai_pattern_analysis(self, data: Dict[str, Any], previous_results: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI for advanced pattern analysis"""
        if not self.ai_service:
            return {}
        
        try:
            content = data.get('content', '')
            
            # Prepare context from previous results
            previous_summary = self._summarize_previous_results(previous_results)
            
            prompt = f"""
Analyze this document for patterns and relationships in a legal case:

Current Document: {content[:2000]}
Previous Analysis Summary: {previous_summary}

Identify patterns in JSON format:
{{
    "behavioral_patterns": ["pattern1", "pattern2"],
    "relationship_patterns": ["relationship1", "relationship2"],
    "communication_patterns": ["comm_pattern1", "comm_pattern2"],
    "financial_patterns": ["fin_pattern1", "fin_pattern2"],
    "deception_indicators": ["indicator1", "indicator2"],
    "manipulation_tactics": ["tactic1", "tactic2"],
    "escalation_evidence": ["evidence1", "evidence2"],
    "pattern_significance": "high|medium|low",
    "cross_document_connections": ["connection1", "connection2"],
    "anomalies_detected": ["anomaly1", "anomaly2"],
    "pattern_based_predictions": ["prediction1", "prediction2"]
}}
"""
            
            response = await self.ai_service.provider.generate_completion(
                prompt,
                "You are an expert pattern analyst specializing in legal evidence and behavioral analysis."
            )
            
            if response.success:
                try:
                    return json.loads(response.content)
                except json.JSONDecodeError:
                    return {"ai_summary": response.content}
            
        except Exception as e:
            self.logger.error(f"AI pattern analysis failed: {e}")
        
        return {}
    
    def _summarize_previous_results(self, previous_results: Dict[str, Any]) -> str:
        """Summarize previous analysis results for AI context"""
        if not previous_results:
            return "No previous analysis available"
        
        summary_parts = []
        
        for agent_name, result in previous_results.items():
            if isinstance(result, dict) and not result.get("error"):
                findings = result.get("findings", {})
                if findings:
                    summary_parts.append(f"{agent_name}: {str(findings)[:200]}...")
        
        return "; ".join(summary_parts[:3])  # Limit to prevent token overflow
    
    def calculate_confidence(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate confidence in pattern analysis"""
        confidence = 0.4  # Base confidence
        
        # Increase confidence based on number of patterns found
        cross_doc_patterns = analysis_data.get("cross_document_patterns", [])
        if cross_doc_patterns:
            confidence += min(len(cross_doc_patterns) * 0.1, 0.3)
        
        # Increase confidence based on entity analysis
        entity_analysis = analysis_data.get("entity_analysis", {})
        entities = entity_analysis.get("entities", {})
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        if total_entities > 5:
            confidence += 0.2
        
        # Increase confidence if AI analysis is available
        ai_analysis = analysis_data.get("ai_analysis", {})
        if ai_analysis:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_pattern_strength(self, findings: Dict[str, Any]) -> float:
        """Calculate pattern strength"""
        strength = 0.3  # Base strength
        
        # Behavioral patterns contribute to strength
        behavioral_patterns = findings.get("behavioral_patterns", {})
        if behavioral_patterns:
            emotional_intensity = behavioral_patterns.get("emotional_indicators", {}).get("emotional_intensity", 0)
            strength += min(emotional_intensity * 0.05, 0.2)
        
        # Cross-document patterns are strong indicators
        cross_doc_patterns = findings.get("cross_document_patterns", [])
        if cross_doc_patterns:
            strength += min(len(cross_doc_patterns) * 0.1, 0.3)
        
        # AI pattern significance
        ai_analysis = findings.get("ai_analysis", {})
        pattern_significance = ai_analysis.get("pattern_significance", "low")
        if pattern_significance == "high":
            strength += 0.2
        elif pattern_significance == "medium":
            strength += 0.1
        
        return min(strength, 1.0)
    
    def extract_legal_significance(self, findings: Dict[str, Any]) -> str:
        """Extract legal significance from pattern analysis"""
        cross_doc_patterns = findings.get("cross_document_patterns", [])
        behavioral_patterns = findings.get("behavioral_patterns", {})
        ai_analysis = findings.get("ai_analysis", {})
        
        if not cross_doc_patterns and not behavioral_patterns:
            return "No significant patterns identified"
        
        pattern_count = len(cross_doc_patterns)
        emotional_intensity = behavioral_patterns.get("emotional_indicators", {}).get("emotional_intensity", 0)
        
        significance = f"Analysis identified {pattern_count} cross-document patterns"
        
        if emotional_intensity > 3:
            significance += " with high emotional intensity indicators"
        
        pattern_significance = ai_analysis.get("pattern_significance", "unknown")
        if pattern_significance != "unknown":
            significance += f" showing {pattern_significance} legal significance"
        
        return significance + "."
    
    def _generate_recommendations(self, findings: Dict[str, Any]) -> List[str]:
        """Generate pattern-based recommendations"""
        recommendations = []
        
        cross_doc_patterns = findings.get("cross_document_patterns", [])
        behavioral_patterns = findings.get("behavioral_patterns", {})
        ai_analysis = findings.get("ai_analysis", {})
        
        # Pattern-based recommendations
        if len(cross_doc_patterns) > 2:
            recommendations.append("Strong cross-document patterns identified - prioritize for case presentation")
        
        # Behavioral recommendations
        emotional_indicators = behavioral_patterns.get("emotional_indicators", {})
        if emotional_indicators.get("emotional_intensity", 0) > 3:
            recommendations.append("High emotional content - consider psychological expert testimony")
        
        deception_score = behavioral_patterns.get("deception_indicators", {}).get("deception_score", 0)
        if deception_score > 2:
            recommendations.append("Potential deception indicators found - investigate further")
        
        # AI recommendations
        manipulation_tactics = ai_analysis.get("manipulation_tactics", [])
        if manipulation_tactics:
            recommendations.append(f"Manipulation tactics identified: {', '.join(manipulation_tactics[:2])}")
        
        cross_connections = ai_analysis.get("cross_document_connections", [])
        if cross_connections:
            recommendations.append(f"Investigate cross-document connections: {', '.join(cross_connections[:2])}")
        
        return recommendations
    
    def _create_error_result(self, error_message: str, start_time: datetime) -> AgentResult:
        """Create an error result"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AgentResult(
            agent_name=self.name,
            analysis_type="pattern_analysis",
            confidence=0.0,
            findings={"error": error_message},
            recommendations=["Review input data and try again"],
            evidence_strength=0.0,
            legal_significance="Analysis failed",
            processing_time=processing_time,
            timestamp=datetime.now(),
            metadata={"error": True}
        )