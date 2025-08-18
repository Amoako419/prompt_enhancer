#!/usr/bin/env python3
"""
Test runner script for the prompt enhancer backend.
Run this script to execute all tests.
"""

import subprocess
import sys
import os

def install_test_dependencies():
    """Install test dependencies"""
    print("Installing test dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"
        ], check=True)
        print("✅ Test dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install test dependencies: {e}")
        return False
    return True

def run_tests():
    """Run the test suite"""
    print("\n🧪 Running test suite...")
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "test_main.py", 
            "-v", 
            "--tb=short",
            "--color=yes"
        ], check=True)
        print("\n✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code: {e.returncode}")
        return False

def run_tests_with_coverage():
    """Run tests with coverage report"""
    print("\n📊 Running tests with coverage...")
    try:
        # Install coverage if not available
        subprocess.run([sys.executable, "-m", "pip", "install", "coverage"], check=True)
        
        # Run tests with coverage
        subprocess.run([
            sys.executable, "-m", "coverage", "run", "-m", "pytest", "test_main.py"
        ], check=True)
        
        # Generate coverage report
        subprocess.run([
            sys.executable, "-m", "coverage", "report", "-m"
        ], check=True)
        
        print("\n✅ Coverage report generated!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Coverage analysis failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Prompt Enhancer Backend Test Suite")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("main.py"):
        print("❌ Error: main.py not found. Please run this script from the backend directory.")
        sys.exit(1)
    
    # Install dependencies
    if not install_test_dependencies():
        sys.exit(1)
    
    # Ask user what to run
    print("\nSelect test mode:")
    print("1. Run basic tests")
    print("2. Run tests with coverage")
    print("3. Run both")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    success = True
    
    if choice in ["1", "3"]:
        success &= run_tests()
    
    if choice in ["2", "3"]:
        success &= run_tests_with_coverage()
    
    if choice not in ["1", "2", "3"]:
        print("❌ Invalid choice. Running basic tests...")
        success = run_tests()
    
    if success:
        print("\n🎉 All operations completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Some operations failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
