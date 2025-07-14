#!/usr/bin/env python3
"""
Script to update existing anomaly datasets with standardized anomaly types using LLM.

This script uses the AnomalyTypeUpdater class to automatically classify and update
all anomaly trace files in the training and evaluation datasets.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the dataset_generator directory to the path
sys.path.append('dataset_generator')

from anomaly_types import AnomalyTypeUpdater, update_all_datasets, print_anomaly_types_summary

def main():
    parser = argparse.ArgumentParser(
        description="Update anomaly datasets with standardized anomaly types using LLM"
    )
    parser.add_argument(
        "--action", 
        choices=["summary", "update", "dry-run"], 
        default="summary",
        help="Action to perform: summary (print types), update (update datasets), or dry-run (test without saving)"
    )
    parser.add_argument(
        "--api-key", 
        type=str, 
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--no-backup", 
        action="store_true", 
        help="Don't create backup files before updating"
    )
    parser.add_argument(
        "--directory", 
        type=str, 
        help="Specific directory to update (optional, defaults to all anomaly directories)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4",
        help="LLM model to use for classification (default: gpt-4)"
    )
    
    args = parser.parse_args()
    
    if args.action == "summary":
        print("=" * 70)
        print("STANDARDIZED ANOMALY TYPES")
        print("=" * 70)
        print_anomaly_types_summary()
        
    elif args.action in ["update", "dry-run"]:
        # Check if we have the required dependencies
        try:
            import openai
        except ImportError:
            print("Error: OpenAI package not installed.")
            print("Please install it with: pip install openai")
            sys.exit(1)
        
        # Check for API key
        api_key = args.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Error: OpenAI API key not provided.")
            print("Please provide it with --api-key or set OPENAI_API_KEY environment variable")
            sys.exit(1)
        
        print("=" * 70)
        print("ANOMALY TYPE UPDATE PROCESS")
        print("=" * 70)
        
        if args.action == "dry-run":
            print("DRY RUN MODE - No files will be modified")
            print("This will test the classification process without saving changes")
            print()
        
        # Initialize the updater
        try:
            updater = AnomalyTypeUpdater(api_key=api_key, model=args.model)
            print(f"✓ Initialized AnomalyTypeUpdater with model: {args.model}")
        except Exception as e:
            print(f"Error initializing updater: {e}")
            sys.exit(1)
        
        # Define directories to update
        if args.directory:
            directories = [args.directory]
        else:
            directories = [
                "training_dataset/anomaly/",
                "eval_dataset/anomaly/"
            ]
        
        # Process each directory
        total_stats = {
            'total_files': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'updated_types': {}
        }
        
        for directory in directories:
            if not os.path.exists(directory):
                print(f"⚠️  Directory {directory} not found, skipping...")
                continue
                
            print(f"\n📁 Processing directory: {directory}")
            
            if args.action == "dry-run":
                # For dry run, just test classification on a few files
                import glob
                import json
                
                files = glob.glob(os.path.join(directory, "*.json"))
                test_files = files[:3]  # Test first 3 files
                
                print(f"Testing classification on {len(test_files)} files...")
                
                for file_path in test_files:
                    try:
                        with open(file_path, 'r') as f:
                            trace_data = json.load(f)
                        
                        anomaly_types = updater.classify_trace(trace_data)
                        current_types = trace_data.get('metadata', {}).get('anomaly_types', [])
                        
                        print(f"  {os.path.basename(file_path)}:")
                        print(f"    Current: {current_types}")
                        print(f"    Proposed: {anomaly_types}")
                        
                        # Update stats
                        total_stats['total_files'] += 1
                        if anomaly_types != current_types:
                            total_stats['successful_updates'] += 1
                            for anomaly_type in anomaly_types:
                                total_stats['updated_types'][anomaly_type] = total_stats['updated_types'].get(anomaly_type, 0) + 1
                        else:
                            total_stats['failed_updates'] += 1
                            
                    except Exception as e:
                        print(f"    Error processing {file_path}: {e}")
                        total_stats['failed_updates'] += 1
            else:
                # Actual update - always set backup=False to avoid creating backup files
                stats = updater.update_dataset_directory(directory, backup=False)
                
                # Aggregate statistics
                total_stats['total_files'] += stats['total_files']
                total_stats['successful_updates'] += stats['successful_updates']
                total_stats['failed_updates'] += stats['failed_updates']
                
                for anomaly_type, count in stats['updated_types'].items():
                    total_stats['updated_types'][anomaly_type] = total_stats['updated_types'].get(anomaly_type, 0) + count
                
                print(f"✓ Directory {directory} completed:")
                print(f"  - Files processed: {stats['total_files']}")
                print(f"  - Successful updates: {stats['successful_updates']}")
                print(f"  - Failed updates: {stats['failed_updates']}")
        
        # Print final summary
        print(f"\n{'='*50}")
        print("FINAL SUMMARY")
        print(f"{'='*50}")
        print(f"Total files processed: {total_stats['total_files']}")
        print(f"Successful updates: {total_stats['successful_updates']}")
        print(f"Failed updates: {total_stats['failed_updates']}")
        
        if total_stats['total_files'] > 0:
            success_rate = (total_stats['successful_updates'] / total_stats['total_files']) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        if total_stats['updated_types']:
            print(f"\nAnomaly type distribution:")
            for anomaly_type, count in sorted(total_stats['updated_types'].items(), key=lambda x: x[1], reverse=True):
                print(f"  - {anomaly_type}: {count}")
        
        if args.action == "dry-run":
            print(f"\n💡 This was a dry run. To apply changes, run with --action update")

if __name__ == "__main__":
    main() 