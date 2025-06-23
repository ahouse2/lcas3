"""Adding the AgentResult dataclass as requested."""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgentResult:
    """Result from an agent's analysis"""
    agent_name: str
    success: bool
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    error: Optional[str] = None

"""
Base Agent Class
Provides common functionality for all specialized agents
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from abc import ABC, abstractmethod

@dataclass
class AnalysisResult:
    """Standard result format for all agent analyses"""
    agent_name: str
    success: bool
    confidence: float
    evidence_strength: float
    legal_significance: str
    findings: Dict[str, Any]
    recommendations: List[str]
    processing_time: float
    timestamp: str
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

class BaseAgent(ABC):
    """Base class for all legal analysis agents"""

    def __init__(self, ai_service=None, config: Dict[str, Any] = None):
        self.ai_service = ai_service
        self.config = config or {}
        self.logger = logging.getLogger(f"LCAS.Agent.{self.__class__.__name__}")

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Return the agent's name"""
        pass

    @property
    @abstractmethod
    def specialization(self) -> str:
        """Return the agent's area of specialization"""
        pass

    @abstractmethod
    async def analyze(self, document_data: Dict[str, Any], context: Dict[str, Any] = None) -> AnalysisResult:
        """Perform analysis on the provided document"""
        pass

    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        return [
            f"Specialized in {self.specialization}",
            "Document content analysis", 
            "Evidence evaluation",
            "Legal significance assessment"
        ]

    async def _ai_analyze(self, content: str, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """Use AI service for analysis if available"""
        if not self.ai_service:
            return {"error": "No AI service available", "confidence": 0.0}

        try:
            # Try different AI service interfaces
            if hasattr(self.ai_service, 'analyze_content'):
                result = await self.ai_service.analyze_content(content, prompt, system_prompt)
                return result
            elif hasattr(self.ai_service, 'provider'):
                response = await self.ai_service.provider.generate_completion(
                    prompt + f"\n\nContent to analyze:\n{content}",
                    system_prompt or f"You are a {self.specialization} expert."
                )
                if hasattr(response, 'success') and response.success:
                    return {"analysis": response.content, "confidence": 0.8}
                else:
                    return {"error": "AI analysis failed", "confidence": 0.0}
            else:
                return {"error": "Incompatible AI service interface", "confidence": 0.0}
        except Exception as e:
            self.logger.error(f"AI analysis failed: {e}")
            return {"error": str(e), "confidence": 0.0}

    def _calculate_evidence_strength(self, findings: Dict[str, Any]) -> float:
        """Calculate evidence strength based on findings"""
        # Default implementation - subclasses can override
        if not findings or "error" in findings:
            return 0.0

        # Simple scoring based on number of positive findings
        positive_indicators = 0
        total_indicators = 0

        for key, value in findings.items():
            if key in ["error", "confidence"]:
                continue
            total_indicators += 1
            if value and str(value).lower() not in ["false", "no", "none", "0"]:
                positive_indicators += 1

        return positive_indicators / max(total_indicators, 1)

    def _determine_legal_significance(self, evidence_strength: float, findings: Dict[str, Any]) -> str:
        """Determine legal significance based on analysis"""
        if evidence_strength > 0.8:
            return "High - Strong evidence supporting legal arguments"
        elif evidence_strength > 0.5:
            return "Medium - Moderate probative value"
        elif evidence_strength > 0.2:
            return "Low - Limited legal relevance"
        else:
            return "Minimal - Weak or unclear legal significance"

    async def _fallback_analysis(self, document_data: Dict[str, Any], context: Dict[str, Any] = None) -> AnalysisResult:
        """Fallback analysis when AI is not available"""
        start_time = datetime.now()

        file_path = document_data.get("file_path", "unknown")
        content = document_data.get("content", "")
        file_type = document_data.get("file_type", "unknown")

        # Basic analysis without AI
        findings = {
            "file_analyzed": file_path,
            "file_type": file_type,
            "content_length": len(content),
            "has_content": len(content) > 0,
            "analysis_method": "rule-based"
        }

        evidence_strength = self._calculate_evidence_strength(findings)
        legal_significance = self._determine_legal_significance(evidence_strength, findings)

        processing_time = (datetime.now() - start_time).total_seconds()

        return AnalysisResult(
            agent_name=self.agent_name,
            success=True,
            confidence=0.3,  # Low confidence for rule-based analysis
            evidence_strength=evidence_strength,
            legal_significance=legal_significance,
            findings=findings,
            recommendations=[f"Consider AI-powered analysis for {self.specialization}"],
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )