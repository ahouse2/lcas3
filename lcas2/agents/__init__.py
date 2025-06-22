"""
LCAS Multi-Agent System
Specialized AI agents for legal discovery and analysis
"""

from .legal_specialist_agent import LegalSpecialistAgent
from .evidence_analyst_agent import EvidenceAnalystAgent
from .timeline_agent import TimelineAgent
from .pattern_discovery_agent import PatternDiscoveryAgent
from .case_strategist_agent import CaseStrategistAgent
from .document_intelligence_agent import DocumentIntelligenceAgent
from .agent_coordinator import AgentCoordinator

__all__ = [
    'LegalSpecialistAgent',
    'EvidenceAnalystAgent', 
    'TimelineAgent',
    'PatternDiscoveryAgent',
    'CaseStrategistAgent',
    'DocumentIntelligenceAgent',
    'AgentCoordinator'
]