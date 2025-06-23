"""
Multi-Agent Analysis Plugin
Integrates the multi-agent system into the LCAS plugin architecture
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from lcas2.core import AnalysisPlugin, LCASCore
from lcas2.agents import AgentCoordinator

logger = logging.getLogger(__name__)

class MultiAgentAnalysisPlugin(AnalysisPlugin):
    """Plugin that orchestrates multi-agent analysis of legal documents"""
    
    def __init__(self):
        self.agent_coordinator: Optional[AgentCoordinator] = None
        self.lcas_core: Optional[LCASCore] = None
        
    @property
    def name(self) -> str:
        return "Multi-Agent Analysis"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Orchestrates specialized AI agents for comprehensive legal document analysis"
    
    @property
    def dependencies(self) -> List[str]:
        return ["openai", "anthropic", "httpx"]  # AI service dependencies
    
    async def initialize(self, core_app: LCASCore) -> bool:
        """Initialize the multi-agent system"""
        self.lcas_core = core_app
        logger.info(f"{self.name}: Initializing multi-agent system...")
        
        try:
            # Load AI service from existing AI plugin if available
            ai_service = None
            
            # Check if AI integration plugin is loaded
            ai_plugin_names = [
                "lcas_ai_wrapper_plugin", 
                "AI Integration",
                "Enhanced AI Foundation",
                "AI Integration Services"
            ]

            for plugin_name in ai_plugin_names:
                if plugin_name in core_app.plugin_manager.loaded_plugins:
                    ai_plugin = core_app.plugin_manager.loaded_plugins[plugin_name]
                    if hasattr(ai_plugin, 'ai_service'):
                        ai_service = ai_plugin.ai_service
                        logger.info(f"{self.name}: Using AI service from {plugin_name}")
                        break
                    elif hasattr(ai_plugin, 'ai_foundation'):
                        ai_service = ai_plugin.ai_foundation
                        logger.info(f"{self.name}: Using AI foundation from {plugin_name}")
                        break
                    elif hasattr(ai_plugin, 'ai_orchestrator'):
                        ai_service = ai_plugin.ai_orchestrator
                        logger.info(f"{self.name}: Using AI orchestrator from {plugin_name}")
                        break
            
            if not ai_service:
                logger.warning(f"{self.name}: No AI service found - multi-agent analysis will use basic capabilities only")
            
            # Initialize agent coordinator
            config = {
                "max_concurrent_agents": core_app.config.max_concurrent_files,
                "case_theory": core_app.config.case_theory,
                "analysis_depth": getattr(core_app.config, 'ai_analysis_depth', 'standard')
            }
            
            self.agent_coordinator = AgentCoordinator(ai_service, config)
            
            logger.info(f"{self.name}: Initialized with {len(self.agent_coordinator.agents)} agents")
            
            # Log available agents and their capabilities
            for agent_name, agent in self.agent_coordinator.agents.items():
                capabilities = agent.get_capabilities()
                logger.info(f"  - {agent_name}: {len(capabilities)} capabilities")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Initialization failed: {e}", exc_info=True)
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        logger.info(f"{self.name}: Cleaning up multi-agent system")
        self.agent_coordinator = None
        self.lcas_core = None
    
    async def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Perform multi-agent analysis on the provided data
        
        Expected data format:
        {
            "processed_files": Dict[str, FileAnalysisData],
            "source_directory": str,
            "target_directory": str,
            "case_name": str,
            "config": LCASConfig
        }
        """
        if not self.agent_coordinator:
            return {
                "error": "Multi-agent system not initialized",
                "success": False
            }
        
        try:
            processed_files = data.get("processed_files", {})
            config = data.get("config")
            
            if not processed_files:
                logger.warning(f"{self.name}: No processed files provided for analysis")
                return {
                    "message": "No files to analyze",
                    "success": True,
                    "processed_files_output": {}
                }
            
            logger.info(f"{self.name}: Starting multi-agent analysis of {len(processed_files)} files")
            
            # Prepare context for agents
            context = {
                "case_theory": config.case_theory if config else {},
                "case_name": data.get("case_name", "Unknown Case"),
                "analysis_depth": getattr(config, 'ai_analysis_depth', 'standard') if config else 'standard'
            }
            
            # Convert processed files to document format expected by agents
            documents_for_analysis = []
            file_path_mapping = {}  # Map agent format back to LCAS format
            
            for file_path, file_analysis_data in processed_files.items():
                # Convert FileAnalysisData to format expected by agents
                document_data = {
                    "file_path": file_path,
                    "content": getattr(file_analysis_data, 'content', '') or getattr(file_analysis_data, 'extracted_text_content', ''),
                    "file_info": {
                        "size": getattr(file_analysis_data, 'size_bytes', 0),
                        "created": getattr(file_analysis_data, 'created_timestamp', None),
                        "modified": getattr(file_analysis_data, 'modified_timestamp', None)
                    }
                }
                
                # Add document type if available from previous analysis
                if hasattr(file_analysis_data, 'ai_analysis_raw') and file_analysis_data.ai_analysis_raw:
                    ai_data = file_analysis_data.ai_analysis_raw
                    if isinstance(ai_data, dict):
                        document_data["document_type"] = ai_data.get("document_type", {})
                
                documents_for_analysis.append(document_data)
                file_path_mapping[file_path] = file_analysis_data
            
            # Run multi-agent analysis
            workflow = "comprehensive"  # Could be configurable
            max_concurrent = getattr(config, 'max_concurrent_files', 3) if config else 3
            
            batch_result = await self.agent_coordinator.analyze_case_batch(
                documents_for_analysis,
                workflow_name=workflow,
                context=context,
                max_concurrent=max_concurrent
            )
            
            # Process results and update FileAnalysisData objects
            updated_files = {}
            
            for doc_result in batch_result.get("document_results", []):
                doc_path = doc_result.get("document_path", "")
                
                if doc_path in file_path_mapping:
                    file_analysis_data = file_path_mapping[doc_path]
                    
                    # Update FileAnalysisData with multi-agent results
                    consolidated_analysis = doc_result.get("consolidated_analysis", {})
                    agent_results = doc_result.get("agent_results", {})
                    
                    # Store multi-agent results
                    if hasattr(file_analysis_data, 'custom_metadata'):
                        file_analysis_data.custom_metadata["multi_agent_analysis"] = {
                            "consolidated_analysis": consolidated_analysis,
                            "agent_results": agent_results,
                            "workflow_used": workflow,
                            "processing_time": doc_result.get("processing_time", 0)
                        }
                    
                    # Update evidence scores if available
                    evidence_strength = consolidated_analysis.get("evidence_strength", 0.0)
                    if hasattr(file_analysis_data, 'evidence_scores'):
                        file_analysis_data.evidence_scores["multi_agent_strength"] = evidence_strength
                    
                    # Update AI summary with consolidated findings
                    if hasattr(file_analysis_data, 'ai_summary'):
                        key_findings = consolidated_analysis.get("key_findings", [])
                        if key_findings:
                            file_analysis_data.ai_summary = f"Multi-agent analysis: {'; '.join(key_findings[:3])}"
                    
                    # Update tags with agent insights
                    if hasattr(file_analysis_data, 'ai_tags'):
                        recommendations = consolidated_analysis.get("recommendations", [])
                        for rec in recommendations[:3]:  # Add top 3 recommendations as tags
                            tag = rec.split(":")[0].strip()  # Extract tag from recommendation
                            if tag not in file_analysis_data.ai_tags:
                                file_analysis_data.ai_tags.append(tag)
                    
                    updated_files[doc_path] = file_analysis_data
            
            # Generate case-level insights
            case_analysis = batch_result.get("case_analysis", {})
            
            # Prepare final result
            result = {
                "plugin": self.name,
                "success": True,
                "processed_files_output": updated_files,
                "case_level_analysis": case_analysis,
                "workflow_used": workflow,
                "total_documents_analyzed": len(documents_for_analysis),
                "successful_analyses": batch_result.get("successful_documents", 0),
                "failed_analyses": batch_result.get("failed_documents_count", 0),
                "processing_time": batch_result.get("processing_time", 0),
                "agent_capabilities": self.agent_coordinator.get_agent_capabilities(),
                "status": "completed"
            }
            
            logger.info(f"{self.name}: Completed analysis of {len(documents_for_analysis)} documents")
            
            return result
            
        except Exception as e:
            logger.error(f"{self.name}: Analysis failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "success": False,
                "plugin": self.name
            }
    
    def get_available_workflows(self) -> Dict[str, Any]:
        """Get available analysis workflows"""
        if self.agent_coordinator:
            return self.agent_coordinator.get_available_workflows()
        return {}
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        if not self.agent_coordinator:
            return {"error": "Agent coordinator not initialized"}
        
        status = {}
        for agent_name, agent in self.agent_coordinator.agents.items():
            status[agent_name] = {
                "capabilities": agent.get_capabilities(),
                "available": True
            }
        
        return status