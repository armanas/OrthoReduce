#!/usr/bin/env python3
"""
test_monitoring.py - Test script for OrthoReduce monitoring system

This script verifies that the comprehensive monitoring system works correctly
with the enhanced experiment runner and core functions.
"""

import sys
import time
from pathlib import Path

# Add the project root to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_monitoring():
    """Test basic monitoring functionality."""
    print("🧪 Testing Basic Monitoring Functionality")
    print("=" * 50)
    
    try:
        from orthogonal_projection.monitoring import (
            SystemMonitor, ProgressTracker, experiment_monitoring,
            format_performance_report, print_system_info
        )
        
        print("✅ Successfully imported monitoring modules")
        
        # Test system information
        print("\n📋 System Information:")
        print_system_info()
        
        # Test basic progress tracking
        print("\n📊 Testing Progress Tracking:")
        methods = ['test_method_1', 'test_method_2', 'test_method_3']
        
        with experiment_monitoring(methods, show_system_stats=True, show_progress_bars=True) as monitor:
            for i, method in enumerate(methods):
                monitor.start_method(method, data_points=100, dimensions=50)
                
                # Simulate work with progress updates
                for progress in [10, 30, 50, 70, 90, 100]:
                    time.sleep(0.1)  # Small delay to simulate work
                    monitor.update_method_progress(progress, f"Processing step {progress//20 + 1}")
                
                monitor.complete_method({'test_metric': i * 10 + 5})
        
        # Get and display performance summary
        performance_summary = monitor.get_performance_summary()
        if performance_summary:
            print("\n📈 Performance Report:")
            print(format_performance_report(performance_summary))
        
        print("✅ Basic monitoring test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Basic monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_experiment_runner():
    """Test monitoring integration with enhanced experiment runner."""
    print("\n🧪 Testing Enhanced Experiment Runner with Monitoring")
    print("=" * 50)
    
    try:
        from orthoreduce_enhanced import EnhancedExperimentRunner
        
        # Create a small test experiment with monitoring enabled
        runner = EnhancedExperimentRunner(
            dataset_size=100,  # Small for quick testing
            dimensions=20,
            epsilon=0.3,
            methods=['pca', 'jll'],  # Just two methods for quick test
            output_dir='test_monitoring_output',
            use_advanced_plots=False,  # Disable plots for speed
            use_interactive=False,
            enable_monitoring=True,
            show_system_stats=True
        )
        
        print(f"📊 Running enhanced experiment with monitoring...")
        results = runner.run_experiment()
        
        # Verify results contain monitoring data
        if 'performance_summary' in results:
            print("✅ Performance monitoring data found in results")
            perf_summary = results['performance_summary']
            if 'total_runtime_seconds' in perf_summary:
                print(f"📈 Total runtime: {perf_summary['total_runtime_seconds']:.2f} seconds")
            if 'method_performance' in perf_summary:
                print(f"📊 Methods monitored: {len(perf_summary['method_performance'])}")
        else:
            print("⚠️  No performance summary found in results")
        
        # Save results to verify monitoring data persistence
        saved_files = runner.save_results(results)
        
        if 'performance' in saved_files:
            print(f"✅ Performance report saved to: {saved_files['performance']}")
        if 'performance_json' in saved_files:
            print(f"✅ Performance data saved to: {saved_files['performance_json']}")
        
        print("✅ Enhanced experiment runner monitoring test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced experiment runner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_core_function_monitoring():
    """Test monitoring integration with core dimensionality reduction functions."""
    print("\n🧪 Testing Core Function Monitoring")
    print("=" * 50)
    
    try:
        from orthogonal_projection.dimensionality_reduction import (
            run_experiment, run_experiment_with_visualization
        )
        
        print("📊 Testing run_experiment with monitoring...")
        
        # Test core experiment function with monitoring
        results = run_experiment(
            n=100,  # Small dataset for testing
            d=20,
            epsilon=0.3,
            methods=['pca', 'jll'],
            enable_monitoring=True,
            show_method_progress=True
        )
        
        print(f"✅ Core experiment completed with {len(results)} methods")
        
        # Test visualization experiment with monitoring  
        print("\n📊 Testing run_experiment_with_visualization with monitoring...")
        
        results, plot_files = run_experiment_with_visualization(
            n=50,  # Even smaller for vis test
            d=15,
            epsilon=0.3,
            methods=['pca', 'jll'],
            create_plots=False,  # Disable actual plotting for speed
            enable_monitoring=True,
            show_method_progress=True
        )
        
        print(f"✅ Visualization experiment completed with {len(results)} methods")
        
        print("✅ Core function monitoring test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Core function monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all monitoring tests."""
    print("🚀 OrthoReduce Monitoring System Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Run individual tests
    test_results.append(test_basic_monitoring())
    test_results.append(test_enhanced_experiment_runner())
    test_results.append(test_core_function_monitoring())
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"✅ Passed: {passed}/{total} tests")
    if passed == total:
        print("🎉 All monitoring tests passed successfully!")
        return 0
    else:
        print(f"❌ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())