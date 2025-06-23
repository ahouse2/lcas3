"""
Document Intelligence Agent
Specialized in understanding document types, extracting key information, and assessing document quality
"""

import re
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from .base_agent import BaseAgent, AnalysisResult # Changed AgentResult to AnalysisResult

class DocumentIntelligenceAgent(BaseAgent):
    """Agent specialized in document analysis and intelligence extraction"""

    def __init__(self, ai_service=None, config: Dict[str, Any] = None):
        super().__init__(ai_service, config)

    @property
    def agent_name(self) -> str:
        return "DocumentIntelligenceAgent"

    @property
    def specialization(self) -> str:
        return "Document type classification, key information extraction, and quality assessment"

    def get_capabilities(self) -> List[str]:
        return [
            "document_type_classification",
            "key_information_extraction",
            "document_quality_assessment",
            "metadata_analysis",
            "authenticity_indicators"
        ]

    async def validate_input(self, data: Any) -> bool:
        """Validate input data for analysis."""
        if not isinstance(data, dict):
            self.logger.warning(f"Input data for {self.agent_name} is not a dictionary.")
            return False
        # Check for content or file_path; data could be a dict from FileAnalysisData
        if 'content' not in data and 'file_path' not in data.get('metadata', {}) and 'file_path' not in data :
            self.logger.warning(f"Input data for {self.agent_name} missing 'content' or 'file_path'. Input keys: {list(data.keys())}")
            return False
        return True

    async def analyze(self, data: Any, context: Dict[str, Any] = None) -> AnalysisResult:
        """Analyze document for intelligence and key information"""
        start_time = datetime.now()

        # Try to get file_path from various possible locations in data
        file_path = data.get('file_path', data.get('metadata', {}).get('file_path', 'unknown_file'))

        if not await self.validate_input(data):
            return self._create_error_result("Invalid input data", start_time, file_path=file_path)

        content = data.get('content', '')
        # Re-assign file_path here to ensure it's the one used throughout if validation passed
        # This prioritizes direct file_path in data, then metadata, then defaults to empty if not found (though validate_input should catch this)
        file_path = data.get('file_path') or data.get('metadata', {}).get('file_path', '')


        try:
            doc_type_analysis = await self._classify_document_type(content, file_path)
            key_info = await self._extract_key_information(content, doc_type_analysis)
            quality_assessment = await self._assess_document_quality(content, data)
            authenticity_indicators = await self._analyze_authenticity(content, data)

            ai_analysis_results = {}
            if self.ai_service:
                ai_analysis_results = await self._ai_document_analysis(content, doc_type_analysis, context)

            current_findings = {
                "document_type_analysis": doc_type_analysis,
                "key_information": key_info,
                "quality_assessment": quality_assessment,
                "authenticity_indicators": authenticity_indicators,
                "ai_analysis": ai_analysis_results,
                "analysis_specific_metadata": { # Consolidating metadata here
                    "source_file_path": file_path,
                    "doc_primary_type": doc_type_analysis.get("primary_type", "unknown"),
                    "original_context": context # Include context if provided
                }
            }

            confidence = self.calculate_confidence(current_findings)
            # These methods are expected to be part of this class or BaseAgent
            evidence_strength = self._calculate_evidence_strength(current_findings)
            legal_significance = self.extract_legal_significance(current_findings)

            processing_time_seconds = (datetime.now() - start_time).total_seconds()

            return AnalysisResult(
                agent_name=self.agent_name,
                success=True,
                confidence=confidence,
                findings=current_findings,
                recommendations=self._generate_recommendations(current_findings),
                evidence_strength=evidence_strength,
                legal_significance=legal_significance,
                processing_time=processing_time_seconds,
                timestamp=datetime.now().isoformat(),
                error_message=None
            )

        except Exception as e:
            self.logger.error(f"Document intelligence analysis failed for {file_path}: {e}", exc_info=True)
            return self._create_error_result(str(e), start_time, file_path=file_path)

    async def _classify_document_type(self, content: str, file_path: str) -> Dict[str, Any]:
        """Classify the type of document"""
        filename = file_path.split('/')[-1].lower() if file_path else ""
        content_lower = content.lower()

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

        for doc_type_key, keywords in doc_types.items(): # Renamed doc_type to doc_type_key
            score = 0
            for keyword in keywords:
                if keyword in content_lower or (filename and keyword in filename):
                    score += 1

            if score > 0:
                keyword_count = len(keywords)
                confidence_val = min(score / keyword_count, 1.0) if keyword_count > 0 else 0.0 # Renamed confidence
                detected_types.append(doc_type_key)
                confidence_scores[doc_type_key] = confidence_val

        # Determine primary type
        if confidence_scores:
            primary_type = max(confidence_scores, key=confidence_scores.get) # Use .get for max key
            primary_confidence = confidence_scores[primary_type]
        else:
            primary_type = "unknown"
            primary_confidence = 0.0

        return {
            "primary_type": primary_type,
            "confidence": primary_confidence,
            "all_detected_types": confidence_scores,
            "file_extension": file_path.split('.')[-1].lower() if file_path and '.' in file_path else ""
        }

    async def _extract_key_information(self, content: str, doc_type_analysis: Dict[str, Any]) -> Dict[str, Any]: # Renamed doc_type to doc_type_analysis
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
        primary_type = doc_type_analysis.get("primary_type", "unknown")

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
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', r'\b\d{1,2}-\d{1,2}-\d{2,4}\b', r'\b\d{4}-\d{1,2}-\d{1,2}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b'
        ]
        dates = []
        for pattern in date_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                dates.append({
                    "date_string": match.group(), "position": match.start(),
                    "context": content[max(0, match.start()-50):match.end()+50]
                })
        return dates

    def _extract_names(self, content: str) -> List[str]:
        """Extract potential names from content"""
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:\s+[A-Z][a-z]+)?\b'
        names = re.findall(name_pattern, content)
        false_positives = {"United States", "New York", "Los Angeles", "San Francisco", "Case Number", "Social Security",
                           "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"}
        # Filter names that are likely just capitalized words or too short
        return list(set(n for n in names if n not in false_positives and len(n.split()) >= 2 and len(n) > 3))


    def _extract_financial_amounts(self, content: str) -> List[Dict[str, Any]]:
        """Extract financial amounts from content"""
        amount_patterns = [r'\$[\d,]+\.?\d*', r'\b\d+\.\d{2}\s*(?:dollars?|USD)\b', r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD)\b']
        amounts = []
        for pattern in amount_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                amounts.append({"amount_string": match.group(), "position": match.start(), "context": content[max(0, match.start()-30):match.end()+30]})
        return amounts

    def _extract_contact_info(self, content: str) -> Dict[str, List[str]]:
        """Extract contact information"""
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', content) # Max TLD length
        phones = re.findall(r'(?:\+?1[-.\s]?)?\(?\b\d{3}\b\)?[-.\s]?\b\d{3}\b[-.\s]?\b\d{4}\b', content)
        return {"emails": list(set(emails)), "phone_numbers": list(set(phones))}

    def _extract_email_info(self, content: str) -> Dict[str, Any]:
        """Extract email-specific information"""
        email_info = {}
        headers = ["from:", "to:", "cc:", "bcc:", "subject:", "date:", "sent:"]
        for header in headers: # Use re.escape for header
            match = re.search(rf'^{re.escape(header)}\s*(.+?)(?:\r?\n|$)', content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match: email_info[header.replace(':', '')] = match.group(1).strip()
        return email_info

    def _extract_court_info(self, content: str) -> Dict[str, Any]:
        """Extract court document specific information"""
        court_info = {}
        patterns = { # More specific regexes
            "case_number": [r'Case\s+No\.?\s*[:\-]?\s*([A-Z0-9\s\-/._]+)', r'Docket\s+No\.?\s*[:\-]?\s*([A-Z0-9\s\-/._]+)'],
            "court_name": [r'(?:In\s+the\s+)?(Superior\s+Court\s+of\s+\w[\w\s]+|United\s+States\s+District\s+Court\s+for\s+the\s+[\w\s]+District\s+of\s+\w[\w\s]+|[\w\s]+County\s+Court)(?:,|\s+at\s+\w+)?']
        }
        for key, pats in patterns.items():
            for pat in pats:
                match = re.search(pat, content, re.IGNORECASE)
                if match: court_info[key] = match.group(1).strip().replace('\n', ' '); break
        return court_info

    def _extract_financial_info(self, content: str) -> Dict[str, Any]:
        """Extract financial document specific information"""
        financial_info = {}
        patterns = { # Made patterns more specific
            "account_number": [r'Account\s+(?:Number|No\.?|#)\s*[:\-]?\s*(\b[0-9X*-]{4,}\b)'], # Min 4 digits
            "balance": [r'(?:Total\s+|Current\s+|Ending\s+)?Balance\s*[:\-]?\s*\$?([\d,]+\.\d{2})'] # Require cents
        }
        for key, pats in patterns.items():
            for pat in pats:
                match = re.search(pat, content, re.IGNORECASE)
                if match: financial_info[key] = match.group(1).strip(); break
        return financial_info

    async def _assess_document_quality(self, content: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality and completeness of the document"""
        quality_metrics = {}
        wc = len(content.split())
        quality_metrics["content_length"] = len(content)
        quality_metrics["word_count"] = wc
        quality_metrics["has_substantial_content"] = wc > 20
        quality_metrics["has_clear_structure"] = bool(re.search(r'\n\s*\n{1,}', content))
        quality_metrics["has_headings_or_titles"] = bool(re.search(r'^(?:#{1,4}\s+.+?|[A-Z][A-Za-z\s()]{5,}(?::|\r?\n\r?\n))', content, re.MULTILINE | re.IGNORECASE))
        quality_metrics["appears_complete"] = not bool(re.search(r'\[\s*(?:omitted|incomplete|rest\s+of\s+document\s+redacted)\s*\]|\(\s*continued\s*\)|page\s+\d+\s+of\s+(?!\s*1\b)', content, re.IGNORECASE))
        quality_metrics["has_signature_block"] = bool(re.search(r'signature|signed\s+by|/s/|respectfully\s+submitted|attorney\s+for', content, re.IGNORECASE))
        quality_metrics["encoding_issues_detected"] = bool(re.search(r'[^\x00-\x7F\u2018\u2019\u201c\u201d\u2022]', content)) # Allow common smart quotes/bullets
        quality_metrics["potential_ocr_artifacts"] = bool(re.search(r'\b[a-z]\s[a-z]\s[a-z]\b|\b\w{3,}[Iiltf]{2,}\w{1,}\b|\b[rn]{2,}(?!\w)|(?<=\w)[m]{2,}(?=\w)', content)) # Refined OCR artifact regex

        positive = sum([
            quality_metrics["has_substantial_content"], quality_metrics["has_clear_structure"],
            quality_metrics["appears_complete"], quality_metrics["has_signature_block"]
        ])
        negative = sum([quality_metrics["encoding_issues_detected"], quality_metrics["potential_ocr_artifacts"]])
        quality_metrics["overall_quality_score"] = max(0.0, min(1.0, (positive * 0.25) - (negative * 0.2) + 0.2)) # Adjusted scoring
        return quality_metrics

    async def _analyze_authenticity(self, content: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze indicators of document authenticity"""
        auth = {}
        file_info = data.get('file_info', {})
        auth["has_creation_date_metadata"] = bool(file_info.get('created'))
        auth["has_modification_date_metadata"] = bool(file_info.get('modified'))
        auth["has_embedded_timestamps"] = bool(re.search(r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap]\.?[Mm]\.?)?', content))
        auth["uses_formal_language"] = bool(re.search(r'\b(?:hereby|whereas|therefore|pursuant\s+to|in\s+accordance\s+with|notwithstanding)\b', content, re.IGNORECASE))
        auth["contains_legal_formatting_elements"] = bool(re.search(r'^\s*(?:\d+\.|\([a-zA-Z0-9]+\)|\b[IVXLCDM]+\.)', content, re.MULTILINE)) # Allow numbers in parens
        auth["mentions_copy_paste_or_screenshots"] = bool(re.search(r'copied\s+from|pasted\s+from|screenshot\s+of|image\s+capture', content, re.IGNORECASE))
        auth["shows_editing_marks_or_comments"] = bool(re.search(r'\[\s*(?:edited|deleted|comment|note|inserted)\s*\]|<\s*(?:insert|del|comment|annotation)[^>]*>', content, re.IGNORECASE))
        return auth

    async def _ai_document_analysis(self, content: str, doc_type_analysis: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Use AI for advanced document analysis"""
        if not self.ai_service: return {"status": "AI service not available"}

        try:
            prompt = f"""Analyze the following legal document content:
Document Type (initial guess): {doc_type_analysis.get('primary_type', 'unknown')}
Content (first 3000 chars): {content[:3000]}

Provide your analysis strictly in JSON format with these exact keys:
"document_summary": "A concise (2-3 sentences) summary of the document's purpose and key content.",
"key_legal_points": ["A list of 3-5 most important legal points, facts, or arguments mentioned."],
"potential_evidence_value": "Rate as 'High', 'Medium', 'Low', or 'Minimal' based on its likely utility in a legal case.",
"admissibility_concerns": ["List any potential admissibility issues (e.g., hearsay, relevance, authenticity). If none apparent, use an empty list."],
"strategic_importance": "Briefly explain its strategic value or potential impact on a case (1-2 sentences).",
"recommended_actions": ["Suggest 1-2 concrete next steps for a legal professional handling this document (e.g., 'Verify dates with calendar entries', 'Cross-reference names with witness list')."]
"""
            ai_response_obj = None
            # Check for ai_service.provider.generate_completion (AIFoundationPlugin structure)
            if hasattr(self.ai_service, 'provider') and hasattr(self.ai_service.provider, 'generate_completion'):
                 ai_response_obj = await self.ai_service.provider.generate_completion(prompt, system_prompt="You are an expert AI legal document analyst. Respond only in the requested JSON format.")
            # Check for ai_service.generate_completion (simpler direct service)
            elif hasattr(self.ai_service, 'generate_completion'):
                 ai_response_obj = await self.ai_service.generate_completion(prompt, system_prompt="You are an expert AI legal document analyst. Respond only in the requested JSON format.")
            # Check for ai_service.analyze_content (another possible interface from AIFoundationPlugin)
            elif hasattr(self.ai_service, 'analyze_content'):
                # This analyze_content might expect different params, adapt if needed or log error
                # For now, assume it takes content and prompt.
                # This path might need more specific handling if analyze_content has a different signature
                # than generate_completion for the AI provider.
                # The current AIFoundationPlugin's AIAgent uses provider.analyze_content, which maps to the provider's methods.
                # This DocumentIntelligenceAgent is a BaseAgent, its _ai_analyze method is what we should align with if it were used.
                # However, this method is calling ai_service.provider.generate_completion directly.
                # Let's assume for now the above two checks are sufficient.
                # If ai_service is an instance of AIFoundationPlugin itself, its analyze_content is for orchestrating agents.
                self.logger.error("AI service passed to DocumentIntelligenceAgent has an 'analyze_content' method, but direct provider access is preferred for this agent's AI call.")
                return {"error": "AI service interface mismatch, direct provider access preferred."}

            else:
                self.logger.error("AI service is misconfigured or lacks a compatible 'generate_completion' method.")
                return {"error": "AI service misconfiguration"}

            if hasattr(ai_response_obj, 'success') and ai_response_obj.success and hasattr(ai_response_obj, 'content'):
                try:
                    response_text = ai_response_obj.content.strip()
                    if response_text.startswith("```json"): response_text = response_text[7:]
                    if response_text.endswith("```"): response_text = response_text[:-3]
                    return json.loads(response_text.strip())
                except json.JSONDecodeError as jde:
                    self.logger.warning(f"AI response content was not valid JSON: {ai_response_obj.content}. Error: {jde}")
                    return {"error": "AI response parsing error", "raw_response": ai_response_obj.content}
            elif isinstance(ai_response_obj, str):
                 return {"ai_raw_summary": ai_response_obj, "status": "AI returned raw string, not structured object."}
            else: # Handle cases where response object might not have 'success' or 'content'
                err_detail = str(ai_response_obj) if ai_response_obj else "No response from AI"
                self.logger.warning(f"AI analysis failed or returned unexpected response structure: {err_detail}")
                return {"error": "AI analysis failed or unexpected response", "details": err_detail}
        except Exception as e:
            self.logger.error(f"AI document analysis execution error: {e}", exc_info=True)
            return {"error": f"AI analysis execution exception: {str(e)}"}

    def calculate_confidence(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate confidence score for document intelligence analysis"""
        confidence = 0.5
        doc_type_analysis = analysis_data.get("document_type_analysis", {})
        quality = analysis_data.get("quality_assessment", {})
        if doc_type_analysis.get("confidence", 0) > 0.7: confidence += 0.15
        if quality.get("overall_quality_score", 0) > 0.7: confidence += 0.15
        if quality.get("has_substantial_content", False): confidence += 0.1
        # Check if AI analysis was successful (no error field or error is None/empty)
        ai_findings = analysis_data.get("ai_analysis", {})
        if ai_findings and not ai_findings.get("error"): confidence += 0.1
        return min(max(confidence, 0.1), 0.95)

    def _calculate_evidence_strength(self, findings_data: Dict[str, Any]) -> float:
        """Calculate evidence strength based on document analysis"""
        strength = 0.2
        quality = findings_data.get("quality_assessment", {})
        authenticity = findings_data.get("authenticity_indicators", {})
        key_info = findings_data.get("key_information", {})
        ai_analysis = findings_data.get("ai_analysis", {})

        if quality.get("overall_quality_score", 0) > 0.8: strength += 0.2
        elif quality.get("overall_quality_score", 0) > 0.5: strength += 0.1

        if authenticity.get("has_embedded_timestamps", False): strength += 0.05
        if authenticity.get("uses_formal_language", False) and authenticity.get("contains_legal_formatting_elements", False): strength += 0.1

        # More weight if key info is present
        if key_info.get("dates") and len(key_info["dates"]) > 0: strength += 0.05
        if key_info.get("names") and len(key_info["names"]) > 0: strength += 0.05
        if key_info.get("financial_amounts") and len(key_info["financial_amounts"]) > 0: strength += 0.1

        ai_value = str(ai_analysis.get("potential_evidence_value", "")).lower()
        if "high" in ai_value: strength += 0.25
        elif "medium" in ai_value: strength += 0.15
        elif "low" in ai_value: strength += 0.05

        return min(max(strength, 0.05), 0.95)

    def extract_legal_significance(self, findings_data: Dict[str, Any]) -> str:
        """Extract legal significance from document intelligence findings"""
        doc_primary_type = findings_data.get("document_type_analysis", {}).get("primary_type", "document")
        quality_score = findings_data.get("quality_assessment", {}).get("overall_quality_score", 0)
        ai_analysis = findings_data.get("ai_analysis", {})
        ai_significance = ai_analysis.get("strategic_importance", "")
        ai_points = ai_analysis.get("key_legal_points", [])

        quality_desc = "low-quality"
        if quality_score > 0.8: quality_desc = "high-quality"
        elif quality_score > 0.5: quality_desc = "moderate-quality"

        base_sig = f"A {quality_desc} {doc_primary_type}."
        if ai_significance and isinstance(ai_significance, str) and ai_significance.strip():
            return f"{base_sig} AI suggests: {ai_significance}"
        if ai_points and isinstance(ai_points, list) and len(ai_points) > 0:
            # Filter out any non-string elements just in case
            valid_points = [str(p) for p in ai_points if isinstance(p, str) and p.strip()]
            if valid_points:
                return f"{base_sig} Key points identified by AI: {'; '.join(valid_points[:2])}."
        return f"{base_sig} General relevance to be determined by further review and contextual analysis."

    def _generate_recommendations(self, findings_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on document intelligence analysis"""
        recommendations = []
        quality = findings_data.get("quality_assessment", {})
        authenticity = findings_data.get("authenticity_indicators", {})
        ai_analysis = findings_data.get("ai_analysis", {})

        if quality.get("overall_quality_score", 0) < 0.4: recommendations.append("Document quality is low; consider seeking a clearer copy or corroborating evidence.")
        if quality.get("potential_ocr_artifacts", False): recommendations.append("Potential OCR artifacts detected; verify critical text sections manually.")
        if authenticity.get("shows_editing_marks_or_comments", False): recommendations.append("Document may contain editing marks; investigate provenance if authenticity is critical.")
        if not authenticity.get("has_embedded_timestamps", False) and not authenticity.get("has_creation_date_metadata", False):
             recommendations.append("Document lacks clear timestamping; seek external verification of timing if important.")

        ai_recs = ai_analysis.get("recommended_actions", [])
        if isinstance(ai_recs, list):
            for rec in ai_recs: # Ensure recommendations are strings
                if isinstance(rec, str) and rec.strip():
                    recommendations.append(rec.strip())

        if not recommendations: recommendations.append("Review document in context of overall case strategy.")
        # Return unique, non-empty recommendations
        return list(dict.fromkeys(r for r in recommendations if r))

    def _create_error_result(self, error_message: str, start_time: datetime, file_path: Optional[str] = None) -> AnalysisResult:
        """Create an error result"""
        processing_time_seconds = (datetime.now() - start_time).total_seconds()

        error_findings = {
            "error_details": error_message,
            "input_file_path": file_path or "unknown",
            "agent_type": self.agent_name # Added for context in error
        }

        return AnalysisResult(
            agent_name=self.agent_name,
            success=False,
            confidence=0.0,
            findings=error_findings,
            recommendations=["Review error message and input data. Check agent logs for more details."],
            evidence_strength=0.0,
            legal_significance="Analysis failed due to an internal error.",
            processing_time=processing_time_seconds,
            timestamp=datetime.now().isoformat(),
            error_message=error_message
        )
