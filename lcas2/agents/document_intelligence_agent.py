"""
Document Intelligence Agent
Specialized in understanding document types, extracting key information, and assessing document quality
"""

import re
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from .base_agent import BaseAgent, AgentResult

class DocumentIntelligenceAgent(BaseAgent):
    """Agent specialized in document analysis and intelligence extraction"""
    
    def __init__(self, ai_service=None, config: Dict[str, Any] = None):
        super().__init__("DocumentIntelligence", ai_service, config)
        
    def get_capabilities(self) -> List[str]:
        return [
            "document_type_classification",
            "key_information_extraction", 
            "document_quality_assessment",
            "metadata_analysis",
            "authenticity_indicators"
        ]
    
    async def analyze(self, data: Any, context: Dict[str, Any] = None) -> AgentResult:
        """Analyze document for intelligence and key information"""
        start_time = datetime.now()
        
        if not await self.validate_input(data):
            return self._create_error_result("Invalid input data", start_time)
        
        content = data.get('content', '')
        file_path = data.get('file_path', '')
        
        try:
            # Extract basic document intelligence
            doc_type = await self._classify_document_type(content, file_path)
            key_info = await self._extract_key_information(content, doc_type)
            quality_assessment = await self._assess_document_quality(content, data)
            authenticity_indicators = await self._analyze_authenticity(content, data)
            
            # Use AI for advanced analysis if available
            ai_analysis = {}
            if self.ai_service:
                ai_analysis = await self._ai_document_analysis(content, doc_type, context)
            
            findings = {
                "document_type": doc_type,
                "key_information": key_info,
                "quality_assessment": quality_assessment,
                "authenticity_indicators": authenticity_indicators,
                "ai_analysis": ai_analysis
            }
            
            confidence = self.calculate_confidence(findings)
            evidence_strength = self._calculate_evidence_strength(findings)
            legal_significance = self.extract_legal_significance(findings)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                agent_name=self.name,
                analysis_type="document_intelligence",
                confidence=confidence,
                findings=findings,
                recommendations=self._generate_recommendations(findings),
                evidence_strength=evidence_strength,
                legal_significance=legal_significance,
                processing_time=processing_time,
                timestamp=datetime.now(),
                metadata={"file_path": file_path, "doc_type": doc_type}
            )
            
        except Exception as e:
            self.logger.error(f"Document intelligence analysis failed: {e}")
            return self._create_error_result(str(e), start_time)
    
    async def _classify_document_type(self, content: str, file_path: str) -> Dict[str, Any]:
        """Classify the type of document"""
        filename = file_path.split('/')[-1].lower()
        content_lower = content.lower()
        
        # Pattern-based classification
        doc_types = {
            "email": ["from:", "to:", "subject:", "sent:", "@"],
            "text_message": ["message", "text", "sms", "imessage"],
            "court_document": ["court", "case no", "plaintiff", "defendant", "motion", "order"],
            "financial_document": ["account", "balance", "transaction", "deposit", "withdrawal"],
            "legal_contract": ["agreement", "contract", "whereas", "party", "consideration"],
            "medical_record": ["patient", "diagnosis", "treatment", "doctor", "medical"],
            "police_report": ["incident", "officer", "report", "police", "citation"],
            "bank_statement": ["statement", "account number", "balance", "transaction"],
            "invoice": ["invoice", "bill", "amount due", "payment"],
            "receipt": ["receipt", "paid", "total", "purchase"]
        }
        
        detected_types = []
        confidence_scores = {}
        
        for doc_type, keywords in doc_types.items():
            score = 0
            for keyword in keywords:
                if keyword in content_lower or keyword in filename:
                    score += 1
            
            if score > 0:
                confidence = min(score / len(keywords), 1.0)
                detected_types.append(doc_type)
                confidence_scores[doc_type] = confidence
        
        # Determine primary type
        if confidence_scores:
            primary_type = max(confidence_scores.keys(), key=lambda k: confidence_scores[k])
            primary_confidence = confidence_scores[primary_type]
        else:
            primary_type = "unknown"
            primary_confidence = 0.0
        
        return {
            "primary_type": primary_type,
            "confidence": primary_confidence,
            "all_detected_types": confidence_scores,
            "file_extension": file_path.split('.')[-1] if '.' in file_path else ""
        }
    
    async def _extract_key_information(self, content: str, doc_type: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key information based on document type"""
        key_info = {}
        
        # Extract dates
        dates = self._extract_dates(content)
        key_info["dates"] = dates
        
        # Extract names
        names = self._extract_names(content)
        key_info["names"] = names
        
        # Extract financial amounts
        amounts = self._extract_financial_amounts(content)
        key_info["financial_amounts"] = amounts
        
        # Extract contact information
        contacts = self._extract_contact_info(content)
        key_info["contact_info"] = contacts
        
        # Type-specific extraction
        primary_type = doc_type.get("primary_type", "unknown")
        
        if primary_type == "email":
            key_info.update(self._extract_email_info(content))
        elif primary_type == "court_document":
            key_info.update(self._extract_court_info(content))
        elif primary_type == "financial_document":
            key_info.update(self._extract_financial_info(content))
        
        return key_info
    
    def _extract_dates(self, content: str) -> List[Dict[str, Any]]:
        """Extract dates from content"""
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',
            r'\b\d{4}-\d{1,2}-\d{1,2}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                dates.append({
                    "date_string": match.group(),
                    "position": match.start(),
                    "context": content[max(0, match.start()-50):match.end()+50]
                })
        
        return dates
    
    def _extract_names(self, content: str) -> List[str]:
        """Extract potential names from content"""
        # Simple name pattern - can be enhanced
        name_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        names = re.findall(name_pattern, content)
        
        # Filter out common false positives
        false_positives = {"United States", "New York", "Los Angeles", "San Francisco"}
        names = [name for name in names if name not in false_positives]
        
        return list(set(names))
    
    def _extract_financial_amounts(self, content: str) -> List[Dict[str, Any]]:
        """Extract financial amounts from content"""
        amount_patterns = [
            r'\$[\d,]+\.?\d*',
            r'\b\d+\.\d{2}\s*(?:dollars?|USD)\b',
            r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD)\b'
        ]
        
        amounts = []
        for pattern in amount_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                amounts.append({
                    "amount_string": match.group(),
                    "position": match.start(),
                    "context": content[max(0, match.start()-30):match.end()+30]
                })
        
        return amounts
    
    def _extract_contact_info(self, content: str) -> Dict[str, List[str]]:
        """Extract contact information"""
        # Email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
        
        # Phone numbers
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', content)
        
        return {
            "emails": list(set(emails)),
            "phone_numbers": list(set(phones))
        }
    
    def _extract_email_info(self, content: str) -> Dict[str, Any]:
        """Extract email-specific information"""
        email_info = {}
        
        # Extract email headers
        headers = ["from:", "to:", "cc:", "bcc:", "subject:", "date:", "sent:"]
        for header in headers:
            pattern = rf'{header}\s*(.+?)(?:\n|$)'
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                email_info[header.replace(':', '')] = match.group(1).strip()
        
        return email_info
    
    def _extract_court_info(self, content: str) -> Dict[str, Any]:
        """Extract court document specific information"""
        court_info = {}
        
        # Case number
        case_patterns = [
            r'case\s+no\.?\s*:?\s*([A-Z0-9-]+)',
            r'case\s+number\s*:?\s*([A-Z0-9-]+)'
        ]
        
        for pattern in case_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                court_info["case_number"] = match.group(1)
                break
        
        # Court name
        court_pattern = r'(?:in\s+the\s+)?(.+?court.+?)(?:\n|case|plaintiff)'
        match = re.search(court_pattern, content, re.IGNORECASE)
        if match:
            court_info["court_name"] = match.group(1).strip()
        
        return court_info
    
    def _extract_financial_info(self, content: str) -> Dict[str, Any]:
        """Extract financial document specific information"""
        financial_info = {}
        
        # Account numbers
        account_pattern = r'account\s+(?:number|no\.?)\s*:?\s*([0-9X*-]+)'
        match = re.search(account_pattern, content, re.IGNORECASE)
        if match:
            financial_info["account_number"] = match.group(1)
        
        # Balance
        balance_pattern = r'balance\s*:?\s*\$?([\d,]+\.?\d*)'
        match = re.search(balance_pattern, content, re.IGNORECASE)
        if match:
            financial_info["balance"] = match.group(1)
        
        return financial_info
    
    async def _assess_document_quality(self, content: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality and completeness of the document"""
        quality_metrics = {}
        
        # Content length and completeness
        quality_metrics["content_length"] = len(content)
        quality_metrics["word_count"] = len(content.split())
        quality_metrics["has_substantial_content"] = len(content.split()) > 10
        
        # Readability indicators
        quality_metrics["has_clear_structure"] = bool(re.search(r'\n\s*\n', content))
        quality_metrics["has_headers"] = bool(re.search(r'^[A-Z\s]+:?$', content, re.MULTILINE))
        
        # Completeness indicators
        quality_metrics["appears_complete"] = not bool(re.search(r'\[.*?\]|\.\.\.|\(continued\)', content, re.IGNORECASE))
        quality_metrics["has_signature_block"] = bool(re.search(r'signature|signed|/s/', content, re.IGNORECASE))
        
        # Technical quality
        quality_metrics["encoding_issues"] = bool(re.search(r'[^\x00-\x7F]', content))
        quality_metrics["ocr_artifacts"] = bool(re.search(r'\b[a-z]\s[a-z]\s[a-z]\b', content))
        
        # Overall quality score
        positive_indicators = sum([
            quality_metrics["has_substantial_content"],
            quality_metrics["has_clear_structure"],
            quality_metrics["appears_complete"],
            quality_metrics["has_signature_block"]
        ])
        
        negative_indicators = sum([
            quality_metrics["encoding_issues"],
            quality_metrics["ocr_artifacts"]
        ])
        
        quality_metrics["overall_quality_score"] = max(0, (positive_indicators - negative_indicators) / 4)
        
        return quality_metrics
    
    async def _analyze_authenticity(self, content: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze indicators of document authenticity"""
        authenticity_indicators = {}
        
        # Metadata consistency
        file_info = data.get('file_info', {})
        authenticity_indicators["has_creation_date"] = bool(file_info.get('created'))
        authenticity_indicators["has_modification_date"] = bool(file_info.get('modified'))
        
        # Content authenticity indicators
        authenticity_indicators["has_timestamps"] = bool(re.search(r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?', content))
        authenticity_indicators["has_formal_language"] = bool(re.search(r'\b(?:hereby|whereas|therefore|pursuant)\b', content, re.IGNORECASE))
        authenticity_indicators["has_legal_formatting"] = bool(re.search(r'^\s*\d+\.|\([a-z]\)|\b[IVX]+\.', content, re.MULTILINE))
        
        # Suspicious indicators
        authenticity_indicators["has_copy_paste_artifacts"] = bool(re.search(r'copied from|pasted|screenshot', content, re.IGNORECASE))
        authenticity_indicators["has_editing_marks"] = bool(re.search(r'\[edited\]|\[deleted\]|<.*?>', content, re.IGNORECASE))
        
        return authenticity_indicators
    
    async def _ai_document_analysis(self, content: str, doc_type: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI for advanced document analysis"""
        if not self.ai_service:
            return {}
        
        try:
            prompt = f"""
Analyze this legal document for key insights:

Document Type: {doc_type.get('primary_type', 'unknown')}
Content: {content[:3000]}

Provide analysis in JSON format:
{{
    "document_summary": "Brief summary of document purpose and content",
    "key_legal_points": ["list", "of", "important", "legal", "points"],
    "potential_evidence_value": "high|medium|low",
    "admissibility_concerns": ["list", "of", "potential", "issues"],
    "strategic_importance": "explanation of strategic value",
    "recommended_actions": ["list", "of", "recommended", "next", "steps"]
}}
"""
            
            response = await self.ai_service.provider.generate_completion(
                prompt,
                "You are an expert legal document analyst."
            )
            
            if response.success:
                try:
                    return json.loads(response.content)
                except json.JSONDecodeError:
                    return {"ai_summary": response.content}
            
        except Exception as e:
            self.logger.error(f"AI document analysis failed: {e}")
        
        return {}
    
    def calculate_confidence(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate confidence score for document intelligence analysis"""
        confidence = 0.5  # Base confidence
        
        doc_type = analysis_data.get("document_type", {})
        quality = analysis_data.get("quality_assessment", {})
        
        # Increase confidence for clear document type
        if doc_type.get("confidence", 0) > 0.7:
            confidence += 0.2
        
        # Increase confidence for high quality documents
        if quality.get("overall_quality_score", 0) > 0.7:
            confidence += 0.2
        
        # Increase confidence for substantial content
        if quality.get("has_substantial_content", False):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_evidence_strength(self, findings: Dict[str, Any]) -> float:
        """Calculate evidence strength based on document analysis"""
        strength = 0.3  # Base strength
        
        quality = findings.get("quality_assessment", {})
        authenticity = findings.get("authenticity_indicators", {})
        key_info = findings.get("key_information", {})
        
        # Quality factors
        if quality.get("overall_quality_score", 0) > 0.8:
            strength += 0.3
        elif quality.get("overall_quality_score", 0) > 0.5:
            strength += 0.1
        
        # Authenticity factors
        if authenticity.get("has_timestamps", False):
            strength += 0.1
        if authenticity.get("has_formal_language", False):
            strength += 0.1
        
        # Content richness
        if len(key_info.get("dates", [])) > 0:
            strength += 0.1
        if len(key_info.get("names", [])) > 0:
            strength += 0.1
        if len(key_info.get("financial_amounts", [])) > 0:
            strength += 0.1
        
        return min(strength, 1.0)
    
    def extract_legal_significance(self, findings: Dict[str, Any]) -> str:
        """Extract legal significance from document intelligence findings"""
        doc_type = findings.get("document_type", {}).get("primary_type", "unknown")
        quality_score = findings.get("quality_assessment", {}).get("overall_quality_score", 0)
        
        if quality_score > 0.8:
            quality_desc = "high-quality"
        elif quality_score > 0.5:
            quality_desc = "moderate-quality"
        else:
            quality_desc = "low-quality"
        
        ai_analysis = findings.get("ai_analysis", {})
        evidence_value = ai_analysis.get("potential_evidence_value", "unknown")
        
        return f"This {quality_desc} {doc_type} document has {evidence_value} potential evidence value for the case."
    
    def _generate_recommendations(self, findings: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on document intelligence analysis"""
        recommendations = []
        
        quality = findings.get("quality_assessment", {})
        authenticity = findings.get("authenticity_indicators", {})
        ai_analysis = findings.get("ai_analysis", {})
        
        # Quality-based recommendations
        if quality.get("overall_quality_score", 0) < 0.5:
            recommendations.append("Consider obtaining higher quality version of this document")
        
        if quality.get("ocr_artifacts", False):
            recommendations.append("Review OCR accuracy and consider manual verification")
        
        # Authenticity recommendations
        if authenticity.get("has_editing_marks", False):
            recommendations.append("Investigate potential document modifications")
        
        if not authenticity.get("has_timestamps", False):
            recommendations.append("Seek additional metadata or timestamp verification")
        
        # AI recommendations
        ai_recommendations = ai_analysis.get("recommended_actions", [])
        recommendations.extend(ai_recommendations)
        
        return recommendations
    
    def _create_error_result(self, error_message: str, start_time: datetime) -> AgentResult:
        """Create an error result"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AgentResult(
            agent_name=self.name,
            analysis_type="document_intelligence",
            confidence=0.0,
            findings={"error": error_message},
            recommendations=["Review input data and try again"],
            evidence_strength=0.0,
            legal_significance="Analysis failed",
            processing_time=processing_time,
            timestamp=datetime.now(),
            metadata={"error": True}
        )