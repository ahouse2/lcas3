
"""
Multi-Agent Analysis Plugin
Integrates the multi-agent system into the LCAS plugin architecture
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from lcas2.core.core import AnalysisPlugin, LCASCore, UIPlugin
from lcas2.agents.agent_coordinator import AgentCoordinator

logger = logging.getLogger(__name__)

class MultiAgentAnalysisPlugin(AnalysisPlugin, UIPlugin):
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
                return {
                    "error": "No processed files provided for analysis",
                    "success": False
                }
            
            logger.info(f"{self.name}: Starting multi-agent analysis of {len(processed_files)} files")
            
            # Prepare documents for agent analysis
            documents = []
            for file_path, file_data in processed_files.items():
                doc_data = {
                    "file_path": file_path,
                    "content": getattr(file_data, 'content', ''),
                    "metadata": getattr(file_data, 'metadata', {}),
                    "file_size": getattr(file_data, 'file_size', 0),
                    "file_type": getattr(file_data, 'file_type', 'unknown'),
                    "hash_sha256": getattr(file_data, 'hash_sha256', '')
                }
                documents.append(doc_data)
            
            # Prepare context for analysis
            context = {
                "case_name": data.get("case_name", "Unknown Case"),
                "case_theory": config.case_theory if config else {},
                "source_directory": data.get("source_directory", ""),
                "target_directory": data.get("target_directory", ""),
                "analysis_timestamp": data.get("timestamp", "")
            }
            
            # Run multi-agent analysis
            workflow_name = getattr(config, 'ai_analysis_depth', 'comprehensive')
            if workflow_name == 'basic':
                workflow_name = 'quick'
            elif workflow_name == 'standard':
                workflow_name = 'strategic'
            
            batch_result = await self.agent_coordinator.analyze_case_batch(
                documents=documents,
                workflow_name=workflow_name,
                context=context,
                max_concurrent=config.max_concurrent_files if config else 3
            )
            
            logger.info(f"{self.name}: Multi-agent analysis completed successfully")
            
            return {
                "status": "completed",
                "success": True,
                "agent_analysis": batch_result,
                "files_analyzed": len(documents),
                "workflow_used": workflow_name,
                "processing_time": batch_result.get("processing_time", 0)
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Analysis failed: {e}", exc_info=True)
            return {
                "error": f"Multi-agent analysis failed: {str(e)}",
                "success": False
            }
    
    def create_ui_elements(self, parent_widget) -> None:
        """Create UI elements for the multi-agent analysis plugin"""
        try:
            import customtkinter as ctk
            
            # Create multi-agent control panel
            agent_frame = ctk.CTkFrame(parent_widget)
            agent_frame.pack(fill="x", pady=5, padx=5)
            
            ctk.CTkLabel(agent_frame, text="Multi-Agent Analysis Controls", 
                        font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=2)
            
            # Workflow selection
            workflow_frame = ctk.CTkFrame(agent_frame)
            workflow_frame.pack(fill="x", padx=5, pady=5)
            
            ctk.CTkLabel(workflow_frame, text="Analysis Workflow:").pack(side="left", padx=5)
            
            self.workflow_var = ctk.StringVar(value="comprehensive")
            workflow_menu = ctk.CTkOptionMenu(
                workflow_frame,
                variable=self.workflow_var,
                values=["quick", "strategic", "comprehensive"]
            )
            workflow_menu.pack(side="left", padx=5)
            
            # Agent status display
            if self.agent_coordinator:
                status_frame = ctk.CTkFrame(agent_frame)
                status_frame.pack(fill="x", padx=5, pady=5)
                
                ctk.CTkLabel(status_frame, text="Available Agents:", 
                            font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5)
                
                for agent_name in self.agent_coordinator.agents.keys():
                    ctk.CTkLabel(status_frame, text=f"✓ {agent_name}").pack(anchor="w", padx=20)
            
            # Run multi-agent analysis button
            def run_multi_agent_analysis():
                if self.lcas_core:
                    self.lcas_core.update_status("Multi-agent analysis requested from UI")
                    # This would trigger through the main analysis pipeline
            
            ctk.CTkButton(
                agent_frame,
                text="Configure Multi-Agent Analysis",
                command=run_multi_agent_analysis
            ).pack(pady=5)
            
        except ImportError:
            logger.warning("CustomTkinter not available for UI creation")
        except Exception as e:
            logger.error(f"Error creating UI elements: {e}")
    
    def get_capabilities(self) -> List[str]:
        """Get plugin capabilities"""
        capabilities = [
            "Multi-agent document analysis",
            "Workflow orchestration",
            "Case-level strategic analysis",
            "Agent coordination and dependency management"
        ]
        
        if self.agent_coordinator:
            agent_caps = self.agent_coordinator.get_agent_capabilities()
            for agent_name, caps in agent_caps.items():
                capabilities.extend([f"{agent_name}: {cap}" for cap in caps])
        
        return capabilities
