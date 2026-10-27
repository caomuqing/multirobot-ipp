#!/usr/bin/env python3
"""
Script to read evaluation metrics CSV and output in formatted style.
"""

import csv
import numpy as np
import sys
import os

def read_and_format_metrics(csv_file):
    """Read CSV file and output metrics in the specified format."""
    
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        return
    
    # Read CSV file
    metrics_data = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key not in metrics_data:
                    metrics_data[key] = []
                try:
                    metrics_data[key].append(float(value))
                except (ValueError, TypeError):
                    pass
    
    # Compute averages (mean) for each metric
    output_metrics = {}
    for key, values in metrics_data.items():
        if values:
            output_metrics[key] = np.mean(values)
        else:
            output_metrics[key] = np.nan
    
    # Define output order and which metrics to include
    output_order = [
        'avgjsd',
        'avgunc', 
        'minnvisit',
        'avgrmse',
        'stdunc',
        'stdjsd',
        'avgnvisit',
        'stdnvisit',
        'avggapvisit',
        'stdgapvisit'
    ]
    
    # Output in specified format (matching the exact spacing from terminal output)
    for metric in output_order:
        if metric in output_metrics:
            value = output_metrics[metric]
            if np.isnan(value):
                print(f"{metric} :    nan")
            else:
                # Format with consistent spacing: metric name + " : " + spaces + value
                # Using 9 spaces after colon to match the format
                print(f"{metric} :         {value}")
        else:
            print(f"{metric} :    nan")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_metrics.py <csv_file>")
        print("Example: python read_metrics.py decen_train_cent_eval/metric/tgt6_speed20_b30_g200_h100_metric.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    read_and_format_metrics(csv_file)

