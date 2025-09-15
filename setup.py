#!/usr/bin/env python3
"""
North America Fire Analysis v1.4.3 - Setup Script
Quick setup and validation script for the fire analysis system.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error installing packages")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["results", "logs", "temp"]
    print(f"\n📁 Creating directories: {', '.join(directories)}")
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created: {directory}/")
    
    return True

def validate_config():
    """Validate configuration file"""
    config_path = "config/config_north_america_firms.json"
    print(f"\n⚙️ Validating configuration: {config_path}")
    
    if not os.path.exists(config_path):
        print("❌ Configuration file not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check required keys
        required_keys = ["nasa_firms", "embedding", "clustering"]
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing configuration key: {key}")
                return False
        
        # Check NASA FIRMS settings
        nasa_config = config["nasa_firms"]
        if "map_key" not in nasa_config or nasa_config["map_key"] == "your_map_key_here":
            print("⚠️ Warning: NASA FIRMS API key not configured")
            print("   Please update 'map_key' in config/config_north_america_firms.json")
            print("   Get your key at: https://firms.modaps.eosdis.nasa.gov/api/")
        
        print("✅ Configuration file is valid")
        return True
        
    except json.JSONDecodeError:
        print("❌ Invalid JSON in configuration file")
        return False

def test_imports():
    """Test if all required modules can be imported"""
    print("\n🔍 Testing module imports...")
    
    required_modules = [
        "pandas", "numpy", "sklearn", "matplotlib", "seaborn",
        "requests", "sentence_transformers", "faiss", "hdbscan"
    ]
    
    failed_imports = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def display_summary():
    """Display setup summary"""
    print("\n" + "="*60)
    print("🔥 NORTH AMERICA FIRE ANALYSIS v1.4.3 - SETUP COMPLETE")
    print("="*60)
    print("\n📋 Next Steps:")
    print("1. Update NASA FIRMS API key in config/config_north_america_firms.json")
    print("2. Run the main pipeline: python north_america_firms_pipeline_v143.py")
    print("3. Check results/ directory for outputs")
    print("\n📖 Documentation:")
    print("- README.md: Complete usage guide")
    print("- config/: Configuration files")
    print("- scripts/: Individual analysis modules")
    print("\n🆘 Support:")
    print("- GitHub Issues: Report bugs and feature requests")
    print("- Documentation: Check README.md for detailed instructions")
    print("\n" + "="*60)

def main():
    """Main setup function"""
    print("🔥 North America Fire Analysis v1.4.3 - Setup")
    print("=" * 50)
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Install requirements
    install_choice = input("\n📦 Install required packages? (y/n): ").lower().strip()
    if install_choice in ['y', 'yes']:
        if not install_requirements():
            print("❌ Setup failed at package installation")
            sys.exit(1)
    else:
        print("⏭️ Skipped package installation")
    
    # Step 3: Create directories
    if not create_directories():
        print("❌ Setup failed at directory creation")
        sys.exit(1)
    
    # Step 4: Validate configuration
    if not validate_config():
        print("❌ Setup failed at configuration validation")
        sys.exit(1)
    
    # Step 5: Test imports
    if not test_imports():
        print("❌ Setup failed at import testing")
        sys.exit(1)
    
    # Step 6: Display summary
    display_summary()

if __name__ == "__main__":
    main()