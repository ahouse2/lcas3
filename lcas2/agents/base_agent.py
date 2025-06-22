"""
Base Agent Class for LCAS Multi-Agent System
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class AgentResult:
    """Standardized result from an agent"""
    agent_name: str
    analysis_type: str
    confidence: float
    findings: Dict[str, Any]
    recommendations: List[str]
    evidence_strength: float
    legal_significance: str
    processing_time: float
    timestamp: datetime
    metadata: Dict[str, Any]

class BaseAgent(ABC):
    """Base class for all LCAS agents"""
    
    def __init__(self, name: str, ai_service=None, config: Dict[str, Any] = None):
        self.name = name
        self.ai_service = ai_service
        self.config = config or {}
        self.logger = logging.getLogger(f"LCAS.Agent.{name}")
        self.results_cache = {}
        
    @abstractmethod
    async def analyze(self, data: Any, context: Dict[str, Any] = None) -> AgentResult:
        """Perform analysis on the provided data"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this agent provides"""
        pass
    
    async def validate_input(self, data: Any) -> bool:
        """Validate input data before analysis"""
        return data is not None
    
    def calculate_confidence(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate confidence score for analysis"""
        # Base implementation - override in subclasses
        return 0.5
    
    def extract_legal_significance(self, findings: Dict[str, Any]) -> str:
        """Extract legal significance from findings"""
        # Base implementation - override in subclasses
        return "Analysis completed"