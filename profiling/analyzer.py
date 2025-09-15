"""
Profiling analysis tools for identifying bottlenecks in RGB training pipeline.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import argparse


@dataclass
class BottleneckAnalysis:
    """Analysis results for identifying bottlenecks."""
    operation: str
    total_time: float
    call_count: int
    avg_time: float
    percentage_of_total: float
    memory_peak_mb: float
    gpu_memory_peak_mb: float
    bottleneck_score: float  # Combined score for ranking bottlenecks


class ProfilingAnalyzer:
    """Analyzer for profiling data to identify bottlenecks and performance issues."""
    
    def __init__(self, profiling_data: Dict[str, Any]):
        self.data = profiling_data
        self.operations = profiling_data.get('summary', {}).get('operations', {})
        self.total_time = sum(op.get('total_time', 0) for op in self.operations.values())
    
    def analyze_bottlenecks(self, min_calls: int = 10) -> List[BottleneckAnalysis]:
        """Analyze profiling data to identify bottlenecks.
        
        Args:
            min_calls: Minimum number of calls to consider an operation
            
        Returns:
            List of bottleneck analyses sorted by bottleneck score
        """
        bottlenecks = []
        
        for name, stats in self.operations.items():
            if stats.get('call_count', 0) < min_calls:
                continue
            
            total_time = stats.get('total_time', 0)
            call_count = stats.get('call_count', 0)
            avg_time = stats.get('avg_time', 0)
            memory_peak = stats.get('memory_peak_mb', 0)
            gpu_memory_peak = stats.get('gpu_memory_peak_mb', 0)
            
            percentage_of_total = (total_time / self.total_time * 100) if self.total_time > 0 else 0
            
            # Calculate bottleneck score (weighted combination of factors)
            # Higher score = more likely to be a bottleneck
            time_score = percentage_of_total  # Time percentage
            frequency_score = min(avg_time * 1000, 100)  # Average time in ms, capped at 100
            memory_score = min(memory_peak / 100, 10)  # Memory usage, normalized
            gpu_memory_score = min(gpu_memory_peak / 100, 10)  # GPU memory usage, normalized
            
            bottleneck_score = (
                time_score * 0.4 +  # 40% weight on total time
                frequency_score * 0.3 +  # 30% weight on per-call time
                memory_score * 0.15 +  # 15% weight on CPU memory
                gpu_memory_score * 0.15  # 15% weight on GPU memory
            )
            
            bottlenecks.append(BottleneckAnalysis(
                operation=name,
                total_time=total_time,
                call_count=call_count,
                avg_time=avg_time,
                percentage_of_total=percentage_of_total,
                memory_peak_mb=memory_peak,
                gpu_memory_peak_mb=gpu_memory_peak,
                bottleneck_score=bottleneck_score
            ))
        
        # Sort by bottleneck score (descending)
        return sorted(bottlenecks, key=lambda x: x.bottleneck_score, reverse=True)
    
    def categorize_operations(self) -> Dict[str, List[str]]:
        """Categorize operations by their function in the pipeline."""
        categories = {
            'Environment': [],
            'Rendering': [],
            'Model': [],
            'Training': [],
            'Memory': [],
            'Other': []
        }
        
        for name in self.operations.keys():
            if 'env_' in name:
                if 'rgb' in name or 'render' in name:
                    categories['Rendering'].append(name)
                else:
                    categories['Environment'].append(name)
            elif 'convlstm' in name or 'recurrent' in name:
                categories['Model'].append(name)
            elif 'training' in name or 'ppo' in name:
                categories['Training'].append(name)
            elif 'memory' in name or 'gpu' in name:
                categories['Memory'].append(name)
            else:
                categories['Other'].append(name)
        
        return categories
    
    def generate_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        bottlenecks = self.analyze_bottlenecks()
        categories = self.categorize_operations()
        
        report = []
        report.append("="*80)
        report.append("RGB TRAINING PIPELINE PROFILING ANALYSIS")
        report.append("="*80)
        report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS:")
        report.append(f"  Total operations tracked: {len(self.operations)}")
        report.append(f"  Total execution time: {self.total_time:.2f} seconds")
        report.append(f"  Average time per operation: {self.total_time / len(self.operations):.4f} seconds")
        report.append("")
        
        # Top bottlenecks
        report.append("TOP 10 BOTTLENECKS:")
        report.append("-" * 60)
        report.append(f"{'Operation':<30} {'Score':<8} {'Time%':<8} {'Avg(ms)':<10} {'Calls':<8}")
        report.append("-" * 60)
        
        for bottleneck in bottlenecks[:10]:
            report.append(f"{bottleneck.operation:<30} {bottleneck.bottleneck_score:<8.2f} "
                         f"{bottleneck.percentage_of_total:<8.1f} {bottleneck.avg_time*1000:<10.2f} "
                         f"{bottleneck.call_count:<8}")
        report.append("")
        
        # Category breakdown
        report.append("CATEGORY BREAKDOWN:")
        report.append("-" * 40)
        for category, operations in categories.items():
            if operations:
                category_time = sum(self.operations[op].get('total_time', 0) for op in operations)
                category_pct = (category_time / self.total_time * 100) if self.total_time > 0 else 0
                report.append(f"{category:<15}: {category_time:.2f}s ({category_pct:.1f}%) - {len(operations)} operations")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 40)
        
        if bottlenecks:
            top_bottleneck = bottlenecks[0]
            if top_bottleneck.percentage_of_total > 30:
                report.append(f"⚠️  CRITICAL: '{top_bottleneck.operation}' takes {top_bottleneck.percentage_of_total:.1f}% of total time")
                report.append("   Consider optimizing this operation or reducing its frequency")
            
            if top_bottleneck.avg_time > 0.1:  # > 100ms per call
                report.append(f"⚠️  SLOW: '{top_bottleneck.operation}' averages {top_bottleneck.avg_time*1000:.1f}ms per call")
                report.append("   Consider optimizing the implementation")
            
            if top_bottleneck.memory_peak_mb > 100:
                report.append(f"⚠️  MEMORY: '{top_bottleneck.operation}' uses {top_bottleneck.memory_peak_mb:.1f}MB peak memory")
                report.append("   Consider reducing memory usage or batching operations")
            
            if top_bottleneck.gpu_memory_peak_mb > 100:
                report.append(f"⚠️  GPU MEMORY: '{top_bottleneck.operation}' uses {top_bottleneck.gpu_memory_peak_mb:.1f}MB peak GPU memory")
                report.append("   Consider reducing GPU memory usage or using smaller batches")
        
        # RGB-specific recommendations
        rgb_ops = [op for op in self.operations.keys() if 'rgb' in op.lower()]
        if rgb_ops:
            rgb_time = sum(self.operations[op].get('total_time', 0) for op in rgb_ops)
            rgb_pct = (rgb_time / self.total_time * 100) if self.total_time > 0 else 0
            report.append("")
            report.append("RGB-SPECIFIC ANALYSIS:")
            report.append(f"  RGB operations take {rgb_pct:.1f}% of total time")
            if rgb_pct > 20:
                report.append("  ⚠️  RGB rendering is a significant bottleneck")
                report.append("  💡 Consider: reducing image resolution, optimizing PIL operations, or caching")
        
        report.append("")
        report.append("="*80)
        
        return "\n".join(report)
    
    def plot_performance_charts(self, save_path: Optional[str] = None) -> None:
        """Generate performance visualization charts."""
        bottlenecks = self.analyze_bottlenecks()
        categories = self.categorize_operations()
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RGB Training Pipeline Performance Analysis', fontsize=16)
        
        # 1. Top bottlenecks by time percentage
        top_10 = bottlenecks[:10]
        if top_10:
            names = [b.operation for b in top_10]
            percentages = [b.percentage_of_total for b in top_10]
            
            ax1.barh(range(len(names)), percentages)
            ax1.set_yticks(range(len(names)))
            ax1.set_yticklabels(names, fontsize=8)
            ax1.set_xlabel('Percentage of Total Time')
            ax1.set_title('Top 10 Operations by Time Percentage')
            ax1.grid(True, alpha=0.3)
        
        # 2. Category breakdown
        category_times = []
        category_names = []
        for category, operations in categories.items():
            if operations:
                category_time = sum(self.operations[op].get('total_time', 0) for op in operations)
                category_times.append(category_time)
                category_names.append(category)
        
        if category_times:
            ax2.pie(category_times, labels=category_names, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Time Distribution by Category')
        
        # 3. Average time per call
        if top_10:
            avg_times = [b.avg_time * 1000 for b in top_10]  # Convert to ms
            ax3.bar(range(len(names)), avg_times)
            ax3.set_xticks(range(len(names)))
            ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax3.set_ylabel('Average Time (ms)')
            ax3.set_title('Average Time per Call (Top 10)')
            ax3.grid(True, alpha=0.3)
        
        # 4. Memory usage
        if top_10:
            memory_peaks = [b.memory_peak_mb for b in top_10]
            gpu_memory_peaks = [b.gpu_memory_peak_mb for b in top_10]
            
            x = np.arange(len(names))
            width = 0.35
            
            ax4.bar(x - width/2, memory_peaks, width, label='CPU Memory (MB)', alpha=0.8)
            ax4.bar(x + width/2, gpu_memory_peaks, width, label='GPU Memory (MB)', alpha=0.8)
            
            ax4.set_xticks(x)
            ax4.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax4.set_ylabel('Peak Memory Usage (MB)')
            ax4.set_title('Peak Memory Usage (Top 10)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance charts saved to {save_path}")
        else:
            plt.show()
    
    def export_detailed_report(self, filename: str) -> None:
        """Export detailed analysis to JSON file."""
        analysis = {
            'summary': {
                'total_operations': len(self.operations),
                'total_time': self.total_time,
                'bottlenecks': [
                    {
                        'operation': b.operation,
                        'bottleneck_score': b.bottleneck_score,
                        'total_time': b.total_time,
                        'call_count': b.call_count,
                        'avg_time': b.avg_time,
                        'percentage_of_total': b.percentage_of_total,
                        'memory_peak_mb': b.memory_peak_mb,
                        'gpu_memory_peak_mb': b.gpu_memory_peak_mb
                    }
                    for b in self.analyze_bottlenecks()
                ],
                'categories': self.categorize_operations()
            },
            'raw_data': self.data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Detailed analysis exported to {filename}")


def main():
    """Command-line interface for profiling analysis."""
    parser = argparse.ArgumentParser(description="Analyze RGB training pipeline profiling data")
    parser.add_argument("--input", "-i", required=True, help="Path to profiling report JSON file")
    parser.add_argument("--output", "-o", help="Path to save analysis report")
    parser.add_argument("--charts", "-c", help="Path to save performance charts (PNG)")
    parser.add_argument("--detailed", "-d", help="Path to save detailed analysis (JSON)")
    parser.add_argument("--min-calls", type=int, default=10, help="Minimum calls to consider an operation")
    
    args = parser.parse_args()
    
    # Load profiling data
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            profiling_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find profiling file {args.input}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in profiling file {args.input}")
        return
    
    # Create analyzer
    analyzer = ProfilingAnalyzer(profiling_data)
    
    # Generate summary report
    summary = analyzer.generate_summary_report()
    print(summary)
    
    # Save summary report
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Summary report saved to {args.output}")
    
    # Generate charts
    if args.charts:
        analyzer.plot_performance_charts(args.charts)
    
    # Export detailed analysis
    if args.detailed:
        analyzer.export_detailed_report(args.detailed)


if __name__ == "__main__":
    main()
