"""
LeafMachine2 Measurement Validation Script

This script validates measurements from LeafMachine2 output CSVs by:
1. Checking mathematical consistency (circularity, aspect ratio, etc.)
2. Detecting outliers and anomalies
3. Optionally converting pixels to cm using predicted conversion factor
4. Generating a validation report

Usage:
    python validate_measurements.py

The script will automatically find the most recent output folder.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class MeasurementValidator:
    def __init__(self, measurements_csv_path, ruler_csv_path=None):
        """
        Initialize the validator with measurement data.
        
        Args:
            measurements_csv_path: Path to the MEASUREMENTS.csv file
            ruler_csv_path: Optional path to the RULER.csv file
        """
        print(f"Loading measurements from: {measurements_csv_path}")
        self.df_measurements = pd.read_csv(measurements_csv_path)
        self.ruler_csv_path = ruler_csv_path
        
        if ruler_csv_path and os.path.exists(ruler_csv_path):
            print(f"Loading ruler data from: {ruler_csv_path}")
            self.df_ruler = pd.read_csv(ruler_csv_path)
        else:
            self.df_ruler = None
        
        self.validation_results = []
        
    def check_circularity(self, tolerance=0.05):
        """
        Validate circularity calculation: circularity = (4π × area) / (perimeter²)
        """
        print("\n" + "="*60)
        print("VALIDATING CIRCULARITY CALCULATIONS")
        print("="*60)
        
        errors = []
        for idx, row in self.df_measurements.iterrows():
            if pd.notna(row['area']) and pd.notna(row['perimeter']) and row['perimeter'] > 0:
                calculated_circularity = (4 * np.pi * row['area']) / (row['perimeter'] ** 2)
                reported_circularity = row['circularity']
                
                diff = abs(calculated_circularity - reported_circularity)
                
                if diff > tolerance:
                    errors.append({
                        'filename': row['filename'],
                        'component': row['component_name'],
                        'calculated': calculated_circularity,
                        'reported': reported_circularity,
                        'difference': diff
                    })
        
        if errors:
            print(f"⚠️  Found {len(errors)} circularity mismatches:")
            for err in errors[:5]:  # Show first 5
                print(f"   {err['filename']} - {err['component']}")
                print(f"      Calculated: {err['calculated']:.4f}, Reported: {err['reported']:.4f}, Diff: {err['difference']:.4f}")
        else:
            print("✓ All circularity calculations are correct!")
        
        return len(errors) == 0
    
    def check_aspect_ratio(self, tolerance=0.01):
        """
        Validate aspect ratio: should be short_side / long_side
        """
        print("\n" + "="*60)
        print("VALIDATING ASPECT RATIO CALCULATIONS")
        print("="*60)
        
        errors = []
        for idx, row in self.df_measurements.iterrows():
            if pd.notna(row['bbox_min_long_side']) and pd.notna(row['bbox_min_short_side']) and row['bbox_min_long_side'] > 0:
                calculated_ratio = row['bbox_min_short_side'] / row['bbox_min_long_side']
                reported_ratio = row['aspect_ratio']
                
                diff = abs(calculated_ratio - reported_ratio)
                
                if diff > tolerance:
                    errors.append({
                        'filename': row['filename'],
                        'component': row['component_name'],
                        'calculated': calculated_ratio,
                        'reported': reported_ratio,
                        'difference': diff
                    })
        
        if errors:
            print(f"⚠️  Found {len(errors)} aspect ratio mismatches:")
            for err in errors[:5]:
                print(f"   {err['filename']} - {err['component']}")
                print(f"      Calculated: {err['calculated']:.4f}, Reported: {err['reported']:.4f}, Diff: {err['difference']:.4f}")
        else:
            print("✓ All aspect ratio calculations are correct!")
        
        return len(errors) == 0
    
    def check_convexity(self, tolerance=0.001):
        """
        Validate convexity: convexity + concavity should equal 1
        """
        print("\n" + "="*60)
        print("VALIDATING CONVEXITY/CONCAVITY SUM")
        print("="*60)
        
        errors = []
        for idx, row in self.df_measurements.iterrows():
            if pd.notna(row['convexity']) and pd.notna(row['concavity']):
                total = row['convexity'] + row['concavity']
                
                if abs(total - 1.0) > tolerance:
                    errors.append({
                        'filename': row['filename'],
                        'component': row['component_name'],
                        'convexity': row['convexity'],
                        'concavity': row['concavity'],
                        'sum': total
                    })
        
        if errors:
            print(f"⚠️  Found {len(errors)} convexity/concavity sum errors:")
            for err in errors[:5]:
                print(f"   {err['filename']} - {err['component']}")
                print(f"      Convexity: {err['convexity']:.6f}, Concavity: {err['concavity']:.6f}, Sum: {err['sum']:.6f}")
        else:
            print("✓ All convexity/concavity sums are correct!")
        
        return len(errors) == 0
    
    def detect_outliers(self, column, threshold_std=3):
        """
        Detect outliers using standard deviation method.
        """
        if column not in self.df_measurements.columns:
            return []
        
        data = self.df_measurements[column].dropna()
        if len(data) < 3:
            return []
        
        mean = data.mean()
        std = data.std()
        
        outliers = []
        for idx, row in self.df_measurements.iterrows():
            if pd.notna(row[column]):
                z_score = abs((row[column] - mean) / std) if std > 0 else 0
                if z_score > threshold_std:
                    outliers.append({
                        'filename': row['filename'],
                        'component': row['component_name'],
                        'value': row[column],
                        'z_score': z_score,
                        'mean': mean,
                        'std': std
                    })
        
        return outliers
    
    def analyze_outliers(self):
        """
        Detect outliers in key measurements.
        """
        print("\n" + "="*60)
        print("DETECTING MEASUREMENT OUTLIERS")
        print("="*60)
        
        columns_to_check = ['area', 'perimeter', 'bbox_min_long_side', 'bbox_min_short_side', 'aspect_ratio', 'circularity']
        
        for col in columns_to_check:
            outliers = self.detect_outliers(col)
            if outliers:
                print(f"\n⚠️  Found {len(outliers)} outliers in '{col}':")
                for out in outliers[:3]:  # Show first 3
                    print(f"   {out['filename']} - {out['component']}")
                    print(f"      Value: {out['value']:.2f}, Z-score: {out['z_score']:.2f}, Mean: {out['mean']:.2f}")
            else:
                print(f"✓ No outliers detected in '{col}'")
    
    def get_conversion_summary(self):
        """
        Summarize ruler conversion success/failure.
        """
        print("\n" + "="*60)
        print("RULER CONVERSION SUMMARY")
        print("="*60)
        
        if self.df_ruler is None:
            print("⚠️  No ruler data available")
            return
        
        total_rulers = len(self.df_ruler)
        successful = (self.df_ruler['ruler_success'] == True).sum()
        failed = total_rulers - successful
        
        print(f"Total rulers detected: {total_rulers}")
        print(f"Successful conversions: {successful} ({successful/total_rulers*100:.1f}%)")
        print(f"Failed conversions: {failed} ({failed/total_rulers*100:.1f}%)")
        
        if failed > 0:
            print("\nRuler classes that failed:")
            failed_rulers = self.df_ruler[self.df_ruler['ruler_success'] == False]
            class_counts = failed_rulers['ruler_class'].value_counts()
            for ruler_class, count in class_counts.items():
                print(f"   {ruler_class}: {count}")
        
        # Show predicted conversion factors
        if 'predicted_conversion_factor_cm' in self.df_ruler.columns:
            pred_factors = self.df_ruler['predicted_conversion_factor_cm'].dropna()
            if len(pred_factors) > 0:
                print(f"\nPredicted conversion factor:")
                print(f"   Mean: {pred_factors.mean():.2f} pixels/cm")
                print(f"   Std: {pred_factors.std():.2f} pixels/cm")
                print(f"   Range: {pred_factors.min():.2f} - {pred_factors.max():.2f} pixels/cm")
    
    def apply_conversion_factor(self, output_path=None, use_predicted=True):
        """
        Apply conversion factor to measurements and create a new CSV.
        """
        print("\n" + "="*60)
        print("APPLYING CONVERSION FACTORS")
        print("="*60)
        
        df_converted = self.df_measurements.copy()
        
        # Add converted columns
        columns_to_convert = ['area', 'perimeter', 'bbox_min_long_side', 'bbox_min_short_side', 
                             'efd_area', 'efd_perimeter']
        
        for col in columns_to_convert:
            if col in df_converted.columns:
                if col == 'area' or col == 'efd_area':
                    # Area: divide by conversion_factor²
                    df_converted[f'{col}_cm2'] = None
                else:
                    # Linear: divide by conversion_factor
                    df_converted[f'{col}_cm'] = None
        
        # Apply conversion for each row
        for idx, row in df_converted.iterrows():
            conversion_factor = None
            
            # Use actual conversion if successful
            if row['ruler_success'] == True and row['conversion_mean'] > 0:
                conversion_factor = row['conversion_mean']
                df_converted.at[idx, 'conversion_source'] = 'measured'
            # Otherwise use predicted if requested
            elif use_predicted and pd.notna(row['predicted_conversion_factor_cm']) and row['predicted_conversion_factor_cm'] > 0:
                conversion_factor = row['predicted_conversion_factor_cm']
                df_converted.at[idx, 'conversion_source'] = 'predicted'
            else:
                df_converted.at[idx, 'conversion_source'] = 'none'
                continue
            
            # Apply conversions
            for col in columns_to_convert:
                if col in df_converted.columns and pd.notna(row[col]):
                    if col == 'area' or col == 'efd_area':
                        df_converted.at[idx, f'{col}_cm2'] = row[col] / (conversion_factor ** 2)
                    else:
                        df_converted.at[idx, f'{col}_cm'] = row[col] / conversion_factor
        
        # Save if output path provided
        if output_path:
            df_converted.to_csv(output_path, index=False)
            print(f"✓ Converted measurements saved to: {output_path}")
        
        # Summary
        converted_count = (df_converted['conversion_source'] != 'none').sum()
        measured_count = (df_converted['conversion_source'] == 'measured').sum()
        predicted_count = (df_converted['conversion_source'] == 'predicted').sum()
        
        print(f"\nConversion summary:")
        print(f"   Total measurements: {len(df_converted)}")
        print(f"   Converted using measured ruler: {measured_count}")
        print(f"   Converted using predicted factor: {predicted_count}")
        print(f"   Not converted (no factor available): {len(df_converted) - converted_count}")
        
        return df_converted
    
    def generate_summary_stats(self):
        """
        Generate summary statistics for the measurements.
        """
        print("\n" + "="*60)
        print("MEASUREMENT SUMMARY STATISTICS")
        print("="*60)
        
        print(f"\nTotal measurements: {len(self.df_measurements)}")
        print(f"Unique images: {self.df_measurements['filename'].nunique()}")
        
        # Key measurement stats
        stats_columns = ['area', 'perimeter', 'bbox_min_long_side', 'bbox_min_short_side', 'circularity', 'aspect_ratio']
        
        for col in stats_columns:
            if col in self.df_measurements.columns:
                data = self.df_measurements[col].dropna()
                if len(data) > 0:
                    print(f"\n{col}:")
                    print(f"   Mean: {data.mean():.2f}")
                    print(f"   Median: {data.median():.2f}")
                    print(f"   Std: {data.std():.2f}")
                    print(f"   Min: {data.min():.2f}")
                    print(f"   Max: {data.max():.2f}")
    
    def run_full_validation(self, save_converted_csv=True):
        """
        Run all validation checks.
        """
        print("\n" + "█"*60)
        print("█" + " "*58 + "█")
        print("█" + "  LEAFMACHINE2 MEASUREMENT VALIDATION".center(58) + "█")
        print("█" + " "*58 + "█")
        print("█"*60)
        
        # Run checks
        self.check_circularity()
        self.check_aspect_ratio()
        self.check_convexity()
        self.analyze_outliers()
        self.get_conversion_summary()
        self.generate_summary_stats()
        
        # Apply conversion factor if requested
        if save_converted_csv:
            output_dir = os.path.dirname(self.df_measurements.attrs.get('source_path', ''))
            if not output_dir:
                output_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(output_dir, 'measurements_converted_to_cm.csv')
            self.apply_conversion_factor(output_path=output_path, use_predicted=True)
        
        print("\n" + "█"*60)
        print("█" + " "*58 + "█")
        print("█" + "  VALIDATION COMPLETE".center(58) + "█")
        print("█" + " "*58 + "█")
        print("█"*60 + "\n")


def find_latest_output_folder():
    """
    Find the most recent LeafMachine2 output folder.
    """
    demo_output_dir = Path(__file__).parent / 'demo' / 'demo_output'
    
    if not demo_output_dir.exists():
        return None
    
    # Get all subdirectories
    output_folders = [f for f in demo_output_dir.iterdir() if f.is_dir() and not f.name.endswith('.zip')]
    
    if not output_folders:
        return None
    
    # Sort by modification time (most recent first)
    latest_folder = max(output_folders, key=lambda x: x.stat().st_mtime)
    
    return latest_folder


if __name__ == "__main__":
    # Find the latest output folder
    output_folder = find_latest_output_folder()
    
    if output_folder is None:
        print("❌ Could not find any LeafMachine2 output folders in demo/demo_output/")
        print("Please run LeafMachine2 first or specify the path manually.")
        exit(1)
    
    print(f"Found latest output folder: {output_folder.name}")
    
    # Locate the CSV files
    measurements_csv = output_folder / 'Data' / 'Measurements' / f'{output_folder.name}_MEASUREMENTS.csv'
    ruler_csv = output_folder / 'Data' / 'Ruler' / f'{output_folder.name}_RULER.csv'
    
    if not measurements_csv.exists():
        print(f"❌ Measurements CSV not found at: {measurements_csv}")
        exit(1)
    
    # Store source path for later use
    df_temp = pd.read_csv(measurements_csv)
    df_temp.attrs['source_path'] = str(measurements_csv)
    
    # Create validator and run
    validator = MeasurementValidator(
        measurements_csv_path=str(measurements_csv),
        ruler_csv_path=str(ruler_csv) if ruler_csv.exists() else None
    )
    
    validator.df_measurements.attrs['source_path'] = str(measurements_csv)
    validator.run_full_validation(save_converted_csv=True)
