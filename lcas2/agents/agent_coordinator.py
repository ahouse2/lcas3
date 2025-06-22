"""
Agent Coordinator
Orchestrates the multi-agent analysis workflow
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from .document_intelligence_agent import DocumentIntelligenceAgent
from .evidence_analyst_agent import EvidenceAnalystAgent
from .legal_specialist_agent import LegalSpecialistAgent
from .timeline_agent import TimelineAgent
from .pattern_discovery_agent import PatternDiscoveryAgent
from .case_strategist_agent import CaseStrategistAgent

logger = logging.getLogger(__name__)

@dataclass
class AnalysisWorkflow:
    """Defines the analysis workflow configuration"""
    workflow_name: str
    agents: List[str]
    parallel_execution: bool = False
    dependencies: Dict[str, List[str]] = None  # agent -> list of required agents
    
class AgentCoordinator:
    """Coordinates multi-agent analysis workflows"""
    
    def __init__(self, ai_service=None, config: Dict[str, Any] = None):
        self.ai_service = ai_service
        self.config = config or {}
        self.logger = logging.getLogger("LCAS.AgentCoordinator")
        
        # Initialize agents
        self.agents = {
            "DocumentIntelligence": DocumentIntelligenceAgent(ai_service, config),
            "EvidenceAnalyst": EvidenceAnalystAgent(ai_service, config),
            "LegalSpecialist": LegalSpecialistAgent(ai_service, config),
            "Timeline": TimelineAgent(ai_service, config),
            "PatternDiscovery": PatternDiscoveryAgent(ai_service, config),
            "CaseStrategist": CaseStrategistAgent(ai_service, config)
        }
        
        # Define standard workflows
        self.workflows = {
            "comprehensive": AnalysisWorkflow(
                workflow_name="comprehensive",
                agents=["DocumentIntelligence", "EvidenceAnalyst", "LegalSpecialist", 
                       "Timeline", "PatternDiscovery", "CaseStrategist"],
                parallel_execution=False,
                dependencies={
                    "EvidenceAnalyst": ["DocumentIntelligence"],
                    "LegalSpecialist": ["DocumentIntelligence", "EvidenceAnalyst"],
                    "Timeline": ["DocumentIntelligence"],
                    "PatternDiscovery": ["DocumentIntelligence", "EvidenceAnalyst"],
                    "CaseStrategist": ["DocumentIntelligence", "EvidenceAnalyst", "LegalSpecialist", "PatternDiscovery"]
                }
            ),
            "quick": AnalysisWorkflow(
                workflow_name="quick",
                agents=["DocumentIntelligence", "EvidenceAnalyst"],
                parallel_execution=True
            ),
            "strategic": AnalysisWorkflow(
                workflow_name="strategic",
                agents=["DocumentIntelligence", "EvidenceAnalyst", "LegalSpecialist", "CaseStrategist"],
                parallel_execution=False,
                dependencies={
                    "EvidenceAnalyst": ["DocumentIntelligence"],
                    "LegalSpecialist": ["DocumentIntelligence", "EvidenceAnalyst"],
                    "CaseStrategist": ["DocumentIntelligence", "EvidenceAnalyst", "LegalSpecialist"]
                }
            )
        }
        
        self.analysis_results = {}
        
    async def analyze_document(self, document_data: Dict[str, Any], 
                             workflow_name: str = "comprehensive",
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze a single document using the specified workflow"""
        
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        workflow = self.workflows[workflow_name]
        self.logger.info(f"Starting {workflow_name} analysis for document: {document_data.get('file_path', 'unknown')}")
        
        start_time = datetime.now()
        results = {}
        
        try:
            if workflow.parallel_execution and not workflow.dependencies:
                # Run agents in parallel
                results = await self._run_parallel_analysis(document_data, workflow, context)
            else:
                # Run agents sequentially with dependency management
                results = await self._run_sequential_analysis(document_data, workflow, context)
            
            # Generate consolidated analysis
            consolidated_result = await self._consolidate_results(results, document_data, context)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "document_path": document_data.get('file_path', 'unknown'),
                "workflow_used": workflow_name,
                "processing_time": processing_time,
                "agent_results": results,
                "consolidated_analysis": consolidated_result,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Document analysis failed: {e}")
            return {
                "document_path": document_data.get('file_path', 'unknown'),
                "workflow_used": workflow_name,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    async def analyze_case_batch(self, documents: List[Dict[str, Any]], 
                               workflow_name: str = "comprehensive",
                               context: Dict[str, Any] = None,
                               max_concurrent: int = 3) -> Dict[str, Any]:
        """Analyze multiple documents for a case"""
        
        self.logger.info(f"Starting batch analysis of {len(documents)} documents")
        
        start_time = datetime.now()
        document_results = []
        
        # Process documents in batches to avoid overwhelming the system
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_single_doc(doc_data):
            async with semaphore:
                return await self.analyze_document(doc_data, workflow_name, context)
        
        # Run document analyses
        tasks = [analyze_single_doc(doc) for doc in documents]
        document_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        successful_results = []
        failed_results = []
        
        for i, result in enumerate(document_results):
            if isinstance(result, Exception):
                failed_results.append({
                    "document_index": i,
                    "document_path": documents[i].get('file_path', 'unknown'),
                    "error": str(result)
                })
            else:
                successful_results.append(result)
        
        # Generate case-level analysis
        case_analysis = await self._generate_case_level_analysis(successful_results, context)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "case_analysis": case_analysis,
            "document_results": successful_results,
            "failed_documents": failed_results,
            "total_documents": len(documents),
            "successful_documents": len(successful_results),
            "failed_documents_count": len(failed_results),
            "processing_time": processing_time,
            "workflow_used": workflow_name,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _run_parallel_analysis(self, document_data: Dict[str, Any], 
                                   workflow: AnalysisWorkflow,
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Run agents in parallel"""
        tasks = {}
        
        for agent_name in workflow.agents:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                tasks[agent_name] = agent.analyze(document_data, context)
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        # Map results back to agent names
        agent_results = {}
        for i, agent_name in enumerate(tasks.keys()):
            if isinstance(results[i], Exception):
                agent_results[agent_name] = {
                    "error": str(results[i]),
                    "success": False
                }
            else:
                agent_results[agent_name] = asdict(results[i])
        
        return agent_results
    
    async def _run_sequential_analysis(self, document_data: Dict[str, Any],
                                     workflow: AnalysisWorkflow,
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Run agents sequentially with dependency management"""
        results = {}
        completed_agents = set()
        
        # Build execution order based on dependencies
        execution_order = self._build_execution_order(workflow)
        
        for agent_name in execution_order:
            if agent_name not in self.agents:
                self.logger.warning(f"Agent {agent_name} not found, skipping")
                continue
            
            # Check if dependencies are met
            dependencies = workflow.dependencies.get(agent_name, []) if workflow.dependencies else []
            if not all(dep in completed_agents for dep in dependencies):
                self.logger.warning(f"Dependencies not met for {agent_name}, skipping")
                continue
            
            try:
                agent = self.agents[agent_name]
                
                # Enhance context with previous results
                enhanced_context = context.copy() if context else {}
                enhanced_context["previous_results"] = results
                
                result = await agent.analyze(document_data, enhanced_context)
                results[agent_name] = asdict(result)
                completed_agents.add(agent_name)
                
                self.logger.debug(f"Completed analysis with {agent_name}")
                
            except Exception as e:
                self.logger.error(f"Agent {agent_name} failed: {e}")
                results[agent_name] = {
                    "error": str(e),
                    "success": False
                }
        
        return results
    
    def _build_execution_order(self, workflow: AnalysisWorkflow) -> List[str]:
        """Build execution order based on dependencies"""
        if not workflow.dependencies:
            return workflow.agents
        
        # Topological sort for dependency resolution
        order = []
        remaining = set(workflow.agents)
        
        while remaining:
            # Find agents with no unmet dependencies
            ready = []
            for agent in remaining:
                deps = workflow.dependencies.get(agent, [])
                if all(dep in order for dep in deps):
                    ready.append(agent)
            
            if not ready:
                # Circular dependency or missing agent - add remaining in original order
                ready = list(remaining)
            
            # Add ready agents to order
            for agent in ready:
                order.append(agent)
                remaining.remove(agent)
        
        return order
    
    async def _consolidate_results(self, agent_results: Dict[str, Any],
                                 document_data: Dict[str, Any],
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate results from multiple agents"""
        
        consolidated = {
            "overall_confidence": 0.0,
            "evidence_strength": 0.0,
            "legal_significance": "",
            "key_findings": [],
            "recommendations": [],
            "risk_factors": [],
            "strategic_value": "unknown"
        }
        
        successful_results = {k: v for k, v in agent_results.items() 
                            if isinstance(v, dict) and not v.get("error")}
        
        if not successful_results:
            return consolidated
        
        # Calculate overall metrics
        confidences = [result.get("confidence", 0.0) for result in successful_results.values()]
        evidence_strengths = [result.get("evidence_strength", 0.0) for result in successful_results.values()]
        
        consolidated["overall_confidence"] = sum(confidences) / len(confidences)
        consolidated["evidence_strength"] = sum(evidence_strengths) / len(evidence_strengths)
        
        # Collect key findings and recommendations
        for agent_name, result in successful_results.items():
            findings = result.get("findings", {})
            recommendations = result.get("recommendations", [])
            
            # Extract key findings
            if isinstance(findings, dict):
                for key, value in findings.items():
                    if key not in ["error"] and value:
                        consolidated["key_findings"].append(f"{agent_name}: {key}")
            
            # Collect recommendations
            consolidated["recommendations"].extend([
                f"{agent_name}: {rec}" for rec in recommendations
            ])
        
        # Determine overall legal significance
        legal_significances = [result.get("legal_significance", "") 
                             for result in successful_results.values()]
        
        if legal_significances:
            # Use the most detailed significance
            consolidated["legal_significance"] = max(legal_significances, key=len)
        
        # Determine strategic value
        if consolidated["evidence_strength"] > 0.7:
            consolidated["strategic_value"] = "high"
        elif consolidated["evidence_strength"] > 0.4:
            consolidated["strategic_value"] = "medium"
        else:
            consolidated["strategic_value"] = "low"
        
        return consolidated
    
    async def _generate_case_level_analysis(self, document_results: List[Dict[str, Any]],
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate case-level analysis from document results"""
        
        if not document_results:
            return {"error": "No successful document analyses"}
        
        case_analysis = {
            "case_strength_assessment": {},
            "evidence_portfolio": {},
            "strategic_recommendations": [],
            "timeline_analysis": {},
            "pattern_analysis": {},
            "risk_assessment": {}
        }
        
        # Aggregate evidence strengths
        evidence_strengths = []
        strategic_values = {"high": 0, "medium": 0, "low": 0}
        
        for doc_result in document_results:
            consolidated = doc_result.get("consolidated_analysis", {})
            strength = consolidated.get("evidence_strength", 0.0)
            evidence_strengths.append(strength)
            
            strategic_value = consolidated.get("strategic_value", "low")
            if strategic_value in strategic_values:
                strategic_values[strategic_value] += 1
        
        # Case strength assessment
        if evidence_strengths:
            avg_strength = sum(evidence_strengths) / len(evidence_strengths)
            max_strength = max(evidence_strengths)
            
            case_analysis["case_strength_assessment"] = {
                "average_evidence_strength": avg_strength,
                "strongest_evidence_score": max_strength,
                "total_documents": len(document_results),
                "high_value_documents": strategic_values["high"],
                "medium_value_documents": strategic_values["medium"],
                "low_value_documents": strategic_values["low"]
            }
        
        # Evidence portfolio analysis
        case_analysis["evidence_portfolio"] = {
            "document_distribution": strategic_values,
            "portfolio_strength": "strong" if strategic_values["high"] > 3 else 
                                 "moderate" if strategic_values["medium"] > 2 else "weak"
        }
        
        # Generate strategic recommendations
        if strategic_values["high"] > 0:
            case_analysis["strategic_recommendations"].append(
                f"Leverage {strategic_values['high']} high-value documents as primary evidence"
            )
        
        if strategic_values["low"] > strategic_values["high"] + strategic_values["medium"]:
            case_analysis["strategic_recommendations"].append(
                "Consider seeking additional stronger evidence to support case theory"
            )
        
        # Use AI for advanced case analysis if available
        if self.ai_service:
            ai_case_analysis = await self._ai_case_analysis(document_results, context)
            case_analysis["ai_insights"] = ai_case_analysis
        
        return case_analysis
    
    async def _ai_case_analysis(self, document_results: List[Dict[str, Any]],
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI for case-level strategic analysis"""
        try:
            # Prepare case summary for AI
            case_summary = {
                "total_documents": len(document_results),
                "high_strength_docs": len([d for d in document_results 
                                         if d.get("consolidated_analysis", {}).get("evidence_strength", 0) > 0.7]),
                "key_findings": []
            }
            
            # Extract key findings from top documents
            sorted_docs = sorted(document_results, 
                               key=lambda x: x.get("consolidated_analysis", {}).get("evidence_strength", 0),
                               reverse=True)
            
            for doc in sorted_docs[:5]:  # Top 5 documents
                findings = doc.get("consolidated_analysis", {}).get("key_findings", [])
                case_summary["key_findings"].extend(findings[:3])  # Top 3 findings per doc
            
            prompt = f"""
Analyze this legal case based on the evidence portfolio:

Case Summary: {case_summary}
Case Context: {context.get('case_theory', {}) if context else {}}

Provide strategic case analysis in JSON format:
{{
    "case_strength_rating": "strong|moderate|weak",
    "primary_legal_theories": ["theory1", "theory2"],
    "evidence_gaps": ["gap1", "gap2"],
    "strategic_priorities": ["priority1", "priority2"],
    "trial_readiness": "ready|needs_work|not_ready",
    "settlement_leverage": "high|medium|low",
    "key_vulnerabilities": ["vulnerability1", "vulnerability2"],
    "recommended_next_steps": ["step1", "step2", "step3"]
}}
"""
            
            response = await self.ai_service.provider.generate_completion(
                prompt,
                "You are an expert trial attorney and case strategist."
            )
            
            if response.success:
                try:
                    import json
                    return json.loads(response.content)
                except json.JSONDecodeError:
                    return {"ai_summary": response.content}
            
        except Exception as e:
            self.logger.error(f"AI case analysis failed: {e}")
        
        return {}
    
    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all agents"""
        capabilities = {}
        for agent_name, agent in self.agents.items():
            capabilities[agent_name] = agent.get_capabilities()
        return capabilities
    
    def get_available_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Get available analysis workflows"""
        workflows = {}
        for name, workflow in self.workflows.items():
            workflows[name] = {
                "agents": workflow.agents,
                "parallel_execution": workflow.parallel_execution,
                "dependencies": workflow.dependencies
            }
        return workflows