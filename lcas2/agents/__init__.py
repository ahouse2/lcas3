
"""
LCAS Agents Module
Multi-agent system for legal case analysis
"""

from .base_agent import BaseAgent, AnalysisResult
from .agent_coordinator import AgentCoordinator, AnalysisWorkflow
from .document_intelligence_agent import DocumentIntelligenceAgent
from .evidence_analyst_agent import EvidenceAnalystAgent
from .legal_specialist_agent import LegalSpecialistAgent
from .timeline_agent import TimelineAgent
from .pattern_discovery_agent import PatternDiscoveryAgent
from .case_strategist_agent import CaseStrategistAgent

__all__ = [
    'BaseAgent',
    'AnalysisResult', 
    'AgentCoordinator',
    'AnalysisWorkflow',
    'DocumentIntelligenceAgent',
    'EvidenceAnalystAgent',
    'LegalSpecialistAgent',
    'TimelineAgent',
    'PatternDiscoveryAgent',
    'CaseStrategistAgent'
]
