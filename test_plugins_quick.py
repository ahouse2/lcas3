
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lcas2.core.core import LCASCore

async def test_plugins():
    print("Testing LCAS plugin loading...")
    
    # Create core instance
    core = LCASCore.create_with_config()
    
    # Initialize
    if await core.initialize():
        print(f"✅ Core initialized successfully")
        print(f"📁 Plugins directory: {core.config.plugins_directory}")
        print(f"🔍 Discovered plugins: {core.plugin_manager.discover_plugins()}")
        print(f"✅ Loaded plugins: {list(core.plugin_manager.loaded_plugins.keys())}")
        
        # Test AI wrapper plugin specifically
        ai_plugin = core.plugin_manager.loaded_plugins.get("lcas_ai_wrapper_plugin")
        if ai_plugin:
            print(f"🤖 AI Plugin found: {ai_plugin.name} v{ai_plugin.version}")
        else:
            print("❌ AI Plugin not found")
            
        # Test multi-agent plugin
        multi_agent_plugin = core.plugin_manager.loaded_plugins.get("Multi-Agent Analysis")
        if multi_agent_plugin:
            print(f"👥 Multi-Agent Plugin found: {multi_agent_plugin.name}")
        else:
            print("❌ Multi-Agent Plugin not found")
            
    else:
        print("❌ Core initialization failed")
    
    await core.shutdown()

if __name__ == "__main__":
    asyncio.run(test_plugins())
