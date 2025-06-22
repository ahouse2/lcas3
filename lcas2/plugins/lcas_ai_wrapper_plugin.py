#!/usr/bin/env python3
"""
LCAS AI Wrapper Plugin
Integrates the AiIntegrationOrchestrator into the LCAS plugin system.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

# Corrected import for core and the orchestrator
from lcas2.core.core import AnalysisPlugin, LCASCore
from lcas2.plugins.ai_integration_plugin import AiIntegrationOrchestrator

logger = logging.getLogger(__name__)


# Ensure class name matches what PluginManager expects: LcasAiWrapperPlugin
class LcasAiWrapperPlugin(AnalysisPlugin):
    """
    Wrapper plugin to integrate the advanced AI capabilities from AiIntegrationOrchestrator.
    """

    def __init__(self):
        self.ai_orchestrator: Optional[AiIntegrationOrchestrator] = None
        self.lcas_core: Optional[LCASCore] = None

    @property
    def name(self) -> str:
        return "lcas_ai_wrapper_plugin"

    @property
    def version(self) -> str:
        return "1.1.1" # Incremented version for logging change

    @property
    def description(self) -> str:
        return "Integrates AI analysis capabilities (via AiIntegrationOrchestrator) into LCAS."

    @property
    def dependencies(self) -> List[str]:
        return ["AI Integration Services"]

    async def initialize(self, core_app: LCASCore) -> bool:
        self.lcas_core = core_app
        logger.info(f"[{self.name}] Initializing...")
        try:
            ai_integration_services_plugin = self.lcas_core.plugin_manager.loaded_plugins.get("AI Integration Services")
            if ai_integration_services_plugin and hasattr(ai_integration_services_plugin, 'get_integration_orchestrator'):
                self.ai_orchestrator = ai_integration_services_plugin.get_integration_orchestrator()
                logger.info(f"[{self.name}] Obtained AiIntegrationOrchestrator from 'AI Integration Services' plugin.")

            if not self.ai_orchestrator:
                 logger.warning(f"[{self.name}] AiIntegrationOrchestrator not found via 'AI Integration Services' plugin. Attempting direct creation.")
                 from lcas2.plugins.ai_integration_plugin import create_enhanced_ai_plugin as create_ai_integration_orchestrator
                 self.ai_orchestrator = create_ai_integration_orchestrator(lcas_core_instance=core_app)
                 logger.info(f"[{self.name}] Directly created AiIntegrationOrchestrator.")

            if not self.ai_orchestrator or not self.ai_orchestrator.config:
                logger.error(f"[{self.name}] Failed to obtain or initialize AI Integration Orchestrator or its configuration.")
                return False

            status = self.ai_orchestrator.get_comprehensive_status()
            available_providers = [p for p, s in status.get("providers", {}).items() if s.get("available", False)]
            if not available_providers:
                logger.warning(f"[{self.name}] No AI providers seem to be available/configured via orchestrator. Check 'config/ai_config.json'.")
            else:
                logger.info(f"[{self.name}] Available AI providers via orchestrator: {available_providers}")

            logger.info(f"[{self.name}] Initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"[{self.name}] Error during initialization: {e}", exc_info=True)
            return False

    async def cleanup(self) -> None:
        logger.info(f"[{self.name}] Cleaning up.")
        self.ai_orchestrator = None
        self.lcas_core = None

    async def analyze(self, data: Any) -> Dict[str, Any]:
        logger.info(f"[{self.name}] Analyze method called.")
        if not self.ai_orchestrator:
            logger.error(f"[{self.name}] AI Orchestrator not initialized. Cannot perform analysis.")
            return {"error": "AI Orchestrator not initialized", "success": False, "processed_files_output": data.get("processed_files", {})}

        master_processed_files: Dict[str, Any] = data.get("processed_files", {})
        lcas_config: Optional[LCASCore.config] = data.get("config")

        logger.info(f"[{self.name}] Received {len(master_processed_files)} file(s) for AI analysis.")
        if not master_processed_files:
            logger.info(f"[{self.name}] No files to analyze. Returning.")
            return {"success": True, "message": "No files provided for AI analysis.", "processed_files_output": master_processed_files}

        plugin_operation_results = {"success_count": 0, "failure_count": 0, "skipped_no_content": 0, "details": []}

        files_to_process_count = len(master_processed_files)
        files_processed_count = 0

        for file_path, fad_object in master_processed_files.items():
            files_processed_count +=1
            logger.debug(f"[{self.name}] Processing file {files_processed_count}/{files_to_process_count}: {file_path}")

            file_content = getattr(fad_object, 'content', None) or getattr(fad_object, 'text_content', None)

            if not file_content:
                logger.debug(f"[{self.name}] No content in FileAnalysisData for {file_path}, skipping AI analysis.")
                plugin_operation_results["skipped_no_content"] +=1
                plugin_operation_results["details"].append({"file_path": file_path, "status": "skipped_no_content"})
                # Ensure FAD has a status if skipped
                if hasattr(fad_object, 'ai_analysis_status'): fad_object.ai_analysis_status = "skipped_no_content"
                continue

            logger.debug(f"[{self.name}] Analyzing file: {file_path} with AI Orchestrator. Content length: {len(file_content)} chars.")

            current_case_context = {
                "lcas_case_name": lcas_config.case_name if lcas_config else "Unknown Case",
                "file_category": getattr(fad_object, 'category', None),
            }
            if lcas_config and lcas_config.case_theory:
                 current_case_context["case_theory_objective"] = lcas_config.case_theory.primary_objective
                 current_case_context["case_type_from_lcas"] = lcas_config.case_theory.case_type

            try:
                logger.debug(f"[{self.name}] Calling orchestrator.analyze_file_content for {file_path}.")
                ai_agent_results = await self.ai_orchestrator.analyze_file_content(
                    content=file_content,
                    file_path=file_path,
                    context=current_case_context
                )
                logger.debug(f"[{self.name}] Received AI results for {file_path}: {list(ai_agent_results.keys()) if isinstance(ai_agent_results,dict) else 'Non-dict result'}")


                if ai_agent_results.get("rate_limited"):
                    logger.warning(f"[{self.name}] AI analysis for {file_path} was skipped due to rate limits.")
                    if hasattr(fad_object, 'ai_analysis_status'): fad_object.ai_analysis_status = "skipped_rate_limited"
                    plugin_operation_results["details"].append({"file_path": file_path, "status": "skipped_rate_limited"})
                    plugin_operation_results["failure_count"] +=1 # Count as failure for this plugin's purpose
                    continue

                if hasattr(fad_object, 'ai_analysis_raw'): fad_object.ai_analysis_raw = ai_agent_results
                if hasattr(fad_object, 'ai_analysis_status'): fad_object.ai_analysis_status = "completed"

                doc_intel_results = ai_agent_results.get("document_intelligence", {}) if isinstance(ai_agent_results, dict) else {}
                findings = doc_intel_results.get("findings", {}) if isinstance(doc_intel_results, dict) else {}

                if isinstance(findings, dict) and "summary" in findings:
                    if hasattr(fad_object, 'summary'): fad_object.summary = findings["summary"]

                if isinstance(doc_intel_results, dict) and doc_intel_results.get("tags"):
                    existing_tags = getattr(fad_object, 'tags', [])
                    if existing_tags is None: existing_tags = []
                    new_tags = list(set(existing_tags + doc_intel_results["tags"]))
                    if hasattr(fad_object, 'tags'): fad_object.tags = new_tags

                plugin_operation_results["success_count"] +=1
                plugin_operation_results["details"].append({"file_path": file_path, "status": "success", "agents_run": list(ai_agent_results.keys()) if isinstance(ai_agent_results,dict) else []})

            except Exception as e:
                logger.error(f"[{self.name}] Error during AI analysis for {file_path}: {e}", exc_info=True)
                if hasattr(fad_object, 'ai_analysis_status'): fad_object.ai_analysis_status = "error"
                if hasattr(fad_object, 'ai_analysis_error'): fad_object.ai_analysis_error = str(e)
                plugin_operation_results["failure_count"] +=1
                plugin_operation_results["details"].append({"file_path": file_path, "status": "error", "error_message": str(e)})

        overall_success = plugin_operation_results["failure_count"] == 0 and files_processed_count > 0 # Success if no errors and some files attempted
        if files_processed_count == 0 and plugin_operation_results["skipped_no_content"] == len(master_processed_files) and len(master_processed_files) > 0:
            overall_success = True # No actual processing errors, just no content

        logger.info(f"[{self.name}] Analysis completed. Successes: {plugin_operation_results['success_count']}, Failures: {plugin_operation_results['failure_count']}, Skipped (no content): {plugin_operation_results['skipped_no_content']}.")
        return {
            "success": overall_success,
            "message": f"AI Wrapper analysis ran. Success: {plugin_operation_results['success_count']}, Fail: {plugin_operation_results['failure_count']}, Skip: {plugin_operation_results['skipped_no_content']}.",
            "details": plugin_operation_results["details"],
            "processed_files_output": master_processed_files
        }
