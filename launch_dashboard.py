#!/usr/bin/env python3
"""
OrthoReduce Dashboard Launcher

CLI tool for launching the interactive dashboard system with various options
for experiment monitoring, result exploration, and analysis.

Usage:
    python launch_dashboard.py [options]
    
Examples:
    python launch_dashboard.py --port 8501 --results-dir ./experiment_results
    python launch_dashboard.py --monitor --auto-refresh
    python launch_dashboard.py --config dashboard_config.yaml
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path
from typing import Optional, List
import logging
import yaml
import json

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def check_dependencies() -> List[str]:
    """Check if required dashboard dependencies are installed."""
    required_packages = [
        'streamlit',
        'plotly',
        'dash',  # Optional alternative
        'colorcet',
        'umap-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages


def install_dependencies(packages: List[str], force: bool = False) -> bool:
    """Install missing dependencies."""
    if not packages:
        return True
    
    if not force:
        print(f"Missing required packages: {', '.join(packages)}")
        response = input("Install missing packages? (y/N): ").strip().lower()
        if response != 'y':
            return False
    
    try:
        cmd = [sys.executable, '-m', 'pip', 'install'] + packages
        subprocess.check_call(cmd)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def load_dashboard_config(config_path: Optional[str] = None) -> dict:
    """Load dashboard configuration from file."""
    default_config = {
        'host': 'localhost',
        'port': 8501,
        'results_dir': 'experiment_results',
        'cache_dir': 'dashboard_cache',
        'auto_refresh': False,
        'refresh_interval': 30,
        'debug': False,
        'theme': 'light',
        'max_upload_size': 200,
        'enable_monitoring': True,
        'enable_real_time': True
    }
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    user_config = yaml.safe_load(f)
                else:
                    user_config = json.load(f)
            
            default_config.update(user_config)
            print(f"✅ Loaded configuration from {config_path}")
            
        except Exception as e:
            print(f"⚠️  Failed to load config {config_path}: {e}")
            print("Using default configuration")
    
    return default_config


def create_streamlit_config(config: dict) -> str:
    """Create Streamlit configuration content."""
    streamlit_config = f"""
[global]
developmentMode = {str(config.get('debug', False)).lower()}
showErrorDetails = {str(config.get('debug', False)).lower()}

[server]
host = "{config.get('host', 'localhost')}"
port = {config.get('port', 8501)}
maxUploadSize = {config.get('max_upload_size', 200)}
enableCORS = false
enableXsrfProtection = true

[theme]
base = "{config.get('theme', 'light')}"
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[browser]
gatherUsageStats = false
serverAddress = "{config.get('host', 'localhost')}"
serverPort = {config.get('port', 8501)}
"""
    return streamlit_config


def setup_environment(config: dict) -> bool:
    """Setup the dashboard environment."""
    # Create necessary directories
    results_dir = Path(config['results_dir'])
    cache_dir = Path(config['cache_dir'])
    
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Streamlit config
    config_dir = Path.home() / '.streamlit'
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / 'config.toml'
    with open(config_file, 'w') as f:
        f.write(create_streamlit_config(config))
    
    # Set environment variables
    os.environ['ORTHOREDUCE_RESULTS_DIR'] = str(results_dir.absolute())
    os.environ['ORTHOREDUCE_CACHE_DIR'] = str(cache_dir.absolute())
    os.environ['ORTHOREDUCE_DEBUG'] = str(config.get('debug', False))
    
    return True


def launch_streamlit_dashboard(config: dict):
    """Launch the Streamlit dashboard."""
    dashboard_script = project_root / 'orthogonal_projection' / 'dashboard.py'
    
    if not dashboard_script.exists():
        print(f"❌ Dashboard script not found: {dashboard_script}")
        return False
    
    # Streamlit command
    cmd = [
        'streamlit', 'run', str(dashboard_script),
        '--server.host', config['host'],
        '--server.port', str(config['port']),
        '--browser.serverAddress', config['host'],
        '--browser.serverPort', str(config['port'])
    ]
    
    if config.get('debug'):
        cmd.extend(['--logger.level', 'debug'])
    
    print(f"🚀 Launching dashboard at http://{config['host']}:{config['port']}")
    print(f"📁 Results directory: {config['results_dir']}")
    print(f"🔄 Auto-refresh: {config.get('auto_refresh', False)}")
    print("\n" + "="*60)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch dashboard: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        return True
    
    return True


def launch_dash_alternative(config: dict):
    """Launch alternative Dash-based dashboard."""
    print("🔄 Launching Dash alternative dashboard...")
    # This would be implemented if Dash alternative is preferred
    print("⚠️  Dash alternative not yet implemented. Use Streamlit dashboard.")
    return False


def create_sample_config():
    """Create a sample configuration file."""
    sample_config = {
        "host": "localhost",
        "port": 8501,
        "results_dir": "experiment_results",
        "cache_dir": "dashboard_cache",
        "auto_refresh": True,
        "refresh_interval": 30,
        "debug": False,
        "theme": "light",
        "max_upload_size": 200,
        "enable_monitoring": True,
        "enable_real_time": True,
        "dashboard_settings": {
            "default_view": "overview",
            "show_advanced_metrics": True,
            "enable_exports": True,
            "cache_embeddings": True
        }
    }
    
    config_file = Path("dashboard_config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Sample configuration created: {config_file}")
    print("Edit this file to customize dashboard settings")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Launch OrthoReduce Interactive Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Launch with defaults
  %(prog)s --port 8502 --host 0.0.0.0       # Launch on specific host/port
  %(prog)s --results-dir ./my_results        # Use custom results directory
  %(prog)s --config my_config.yaml           # Use custom configuration
  %(prog)s --monitor --auto-refresh          # Enable monitoring mode
  %(prog)s --create-config                   # Create sample config file
        """
    )
    
    # Basic options
    parser.add_argument('--host', default='localhost',
                       help='Dashboard host address (default: localhost)')
    parser.add_argument('--port', type=int, default=8501,
                       help='Dashboard port (default: 8501)')
    parser.add_argument('--results-dir', default='experiment_results',
                       help='Directory containing experiment results')
    parser.add_argument('--cache-dir', default='dashboard_cache',
                       help='Directory for dashboard cache')
    
    # Configuration
    parser.add_argument('--config', 
                       help='Configuration file (YAML/JSON)')
    parser.add_argument('--create-config', action='store_true',
                       help='Create sample configuration file')
    
    # Dashboard options
    parser.add_argument('--monitor', action='store_true',
                       help='Enable real-time experiment monitoring')
    parser.add_argument('--auto-refresh', action='store_true',
                       help='Enable automatic page refresh')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    # Installation and setup
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dashboard dependencies')
    parser.add_argument('--install-deps', action='store_true',
                       help='Install missing dependencies')
    parser.add_argument('--force-install', action='store_true',
                       help='Force install dependencies without prompting')
    
    # Alternative backends
    parser.add_argument('--use-dash', action='store_true',
                       help='Use Dash instead of Streamlit (if available)')
    
    # Utility options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Handle special commands
    if args.create_config:
        create_sample_config()
        return 0
    
    if args.check_deps:
        missing = check_dependencies()
        if missing:
            print(f"Missing packages: {', '.join(missing)}")
            return 1
        else:
            print("✅ All dependencies are installed")
            return 0
    
    # Check and install dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        if args.install_deps or args.force_install:
            if not install_dependencies(missing_deps, args.force_install):
                print("❌ Failed to install dependencies")
                return 1
        else:
            print(f"❌ Missing required packages: {', '.join(missing_deps)}")
            print("Run with --install-deps to install them")
            return 1
    
    # Load configuration
    config = load_dashboard_config(args.config)
    
    # Override config with command line arguments
    if args.host != 'localhost':
        config['host'] = args.host
    if args.port != 8501:
        config['port'] = args.port
    if args.results_dir != 'experiment_results':
        config['results_dir'] = args.results_dir
    if args.cache_dir != 'dashboard_cache':
        config['cache_dir'] = args.cache_dir
    if args.monitor:
        config['enable_monitoring'] = True
    if args.auto_refresh:
        config['auto_refresh'] = True
    if args.debug:
        config['debug'] = True
    
    # Setup environment
    if not setup_environment(config):
        print("❌ Failed to setup dashboard environment")
        return 1
    
    # Launch appropriate dashboard
    if args.use_dash:
        success = launch_dash_alternative(config)
    else:
        success = launch_streamlit_dashboard(config)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())