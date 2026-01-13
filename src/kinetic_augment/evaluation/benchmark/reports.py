"""
Report Generation for KineticAugment Evaluation.

Utilities for generating benchmark reports in various formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import json

from kinetic_augment.evaluation.benchmark.runner import ExperimentResult


class BenchmarkReport:
    """
    Generate reports from benchmark results.

    Supports multiple output formats:
    - Text summary
    - CSV for analysis
    - LaTeX table for papers
    - JSON for further processing
    """

    def __init__(self, results: Dict[str, ExperimentResult]):
        """
        Initialize report generator.

        Args:
            results: Dictionary of experiment results
        """
        self.results = results

    def generate_summary(self) -> str:
        """
        Generate text summary of results.

        Returns:
            Formatted text summary
        """
        lines = []
        lines.append("=" * 70)
        lines.append("  KineticAugment Benchmark Results")
        lines.append("=" * 70)
        lines.append("")

        # Sort by test accuracy if available
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: (
                x[1].test_metrics['accuracy'] if x[1].test_metrics else 0
            ),
            reverse=True,
        )

        # Quality metrics comparison
        lines.append("Quality Metrics:")
        lines.append("-" * 50)
        for name, result in sorted_results:
            lines.append(f"  {name}:")
            for metric_name, metric_result in result.quality_metrics.items():
                lines.append(f"    {metric_name}: {metric_result.value:.4f}")
            lines.append("")

        # Test accuracy comparison
        if any(r.test_metrics for _, r in sorted_results):
            lines.append("")
            lines.append("Test Accuracy Ranking:")
            lines.append("-" * 50)
            for name, result in sorted_results:
                if result.test_metrics:
                    acc = result.test_metrics['accuracy'] * 100
                    top5 = result.test_metrics['top5_accuracy'] * 100
                    lines.append(f"  {name:30s}: {acc:.2f}% (top-5: {top5:.2f}%)")
                else:
                    lines.append(f"  {name:30s}: N/A")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    def generate_csv(self) -> str:
        """
        Generate CSV format for analysis.

        Returns:
            CSV string
        """
        lines = []

        # Get all metric names
        metric_names = set()
        for result in self.results.values():
            metric_names.update(result.quality_metrics.keys())
        metric_names = sorted(metric_names)

        # Header
        header = ['experiment', 'description']
        header.extend(metric_names)
        header.extend(['test_accuracy', 'top5_accuracy', 'test_loss'])
        lines.append(','.join(header))

        # Data rows
        for name, result in self.results.items():
            row = [name, result.config.description.replace(',', ';')]

            # Quality metrics
            for metric_name in metric_names:
                if metric_name in result.quality_metrics:
                    row.append(f"{result.quality_metrics[metric_name].value:.6f}")
                else:
                    row.append('')

            # Test metrics
            if result.test_metrics:
                row.append(f"{result.test_metrics['accuracy']:.6f}")
                row.append(f"{result.test_metrics['top5_accuracy']:.6f}")
                row.append(f"{result.test_metrics['loss']:.6f}")
            else:
                row.extend(['', '', ''])

            lines.append(','.join(row))

        return '\n'.join(lines)

    def generate_latex_table(
        self,
        caption: str = "Augmentation benchmark results",
        label: str = "tab:benchmark",
    ) -> str:
        """
        Generate LaTeX table for papers.

        Args:
            caption: Table caption
            label: Table label for references

        Returns:
            LaTeX table string
        """
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{{label}}}")

        # Determine columns
        has_test = any(r.test_metrics for r in self.results.values())
        if has_test:
            cols = "l" + "c" * 5
            lines.append(f"\\begin{{tabular}}{{{cols}}}")
            lines.append("\\toprule")
            lines.append("Experiment & Limb & Smooth & Variance & Acc (\\%) & Top-5 (\\%) \\\\")
        else:
            cols = "l" + "c" * 3
            lines.append(f"\\begin{{tabular}}{{{cols}}}")
            lines.append("\\toprule")
            lines.append("Experiment & Limb & Smooth & Variance \\\\")

        lines.append("\\midrule")

        # Data rows
        for name, result in self.results.items():
            qm = result.quality_metrics
            limb = qm.get('LimbLengthConsistency')
            smooth = qm.get('SmoothnessMetric')
            var = qm.get('VarianceMetric')

            row = [
                name.replace('_', '\\_'),
                f"{limb.value:.3f}" if limb else "-",
                f"{smooth.value:.3f}" if smooth else "-",
                f"{var.value:.3f}" if var else "-",
            ]

            if has_test:
                if result.test_metrics:
                    row.append(f"{result.test_metrics['accuracy']*100:.1f}")
                    row.append(f"{result.test_metrics['top5_accuracy']*100:.1f}")
                else:
                    row.extend(["-", "-"])

            lines.append(" & ".join(row) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    def generate_markdown(self) -> str:
        """
        Generate Markdown format.

        Returns:
            Markdown string
        """
        lines = []
        lines.append("# Benchmark Results")
        lines.append("")

        # Quality metrics table
        lines.append("## Quality Metrics")
        lines.append("")
        lines.append("| Experiment | Limb Consistency | Smoothness | Variance |")
        lines.append("|------------|------------------|------------|----------|")

        for name, result in self.results.items():
            qm = result.quality_metrics
            limb = qm.get('LimbLengthConsistency')
            smooth = qm.get('SmoothnessMetric')
            var = qm.get('VarianceMetric')

            limb_str = f"{limb.value:.4f}" if limb else "N/A"
            smooth_str = f"{smooth.value:.4f}" if smooth else "N/A"
            var_str = f"{var.value:.4f}" if var else "N/A"

            lines.append(
                f"| {name} | {limb_str} | {smooth_str} | {var_str} |"
            )

        # Test metrics table if available
        if any(r.test_metrics for r in self.results.values()):
            lines.append("")
            lines.append("## Test Accuracy")
            lines.append("")
            lines.append("| Experiment | Accuracy | Top-5 Accuracy |")
            lines.append("|------------|----------|----------------|")

            for name, result in self.results.items():
                if result.test_metrics:
                    lines.append(
                        f"| {name} | "
                        f"{result.test_metrics['accuracy']*100:.2f}% | "
                        f"{result.test_metrics['top5_accuracy']*100:.2f}% |"
                    )

        return "\n".join(lines)

    def save(
        self,
        output_dir: Path,
        formats: Optional[List[str]] = None,
    ) -> None:
        """
        Save reports in multiple formats.

        Args:
            output_dir: Directory to save reports
            formats: List of formats ('txt', 'csv', 'tex', 'md', 'json')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if formats is None:
            formats = ['txt', 'csv', 'md']

        if 'txt' in formats:
            with open(output_dir / 'summary.txt', 'w') as f:
                f.write(self.generate_summary())

        if 'csv' in formats:
            with open(output_dir / 'results.csv', 'w') as f:
                f.write(self.generate_csv())

        if 'tex' in formats:
            with open(output_dir / 'table.tex', 'w') as f:
                f.write(self.generate_latex_table())

        if 'md' in formats:
            with open(output_dir / 'results.md', 'w') as f:
                f.write(self.generate_markdown())

        if 'json' in formats:
            # Convert results to JSON-serializable format
            json_data = {}
            for name, result in self.results.items():
                json_data[name] = {
                    'config': {
                        'name': result.config.name,
                        'description': result.config.description,
                    },
                    'quality_metrics': {
                        k: {'value': v.value}
                        for k, v in result.quality_metrics.items()
                    },
                    'test_metrics': result.test_metrics,
                }
            with open(output_dir / 'results.json', 'w') as f:
                json.dump(json_data, f, indent=2)
