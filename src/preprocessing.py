"""
Data Preprocessing Module
Handles data validation, cleaning, and preparation for feature engineering
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import warnings


class ExamDataPreprocessor:
    """
    Preprocesses exam interaction logs for analysis
    Validates data integrity and prepares for feature extraction
    """
    
    def __init__(self):
        self.required_columns = [
            'student_id', 'question_id', 'time_spent', 
            'selected_option', 'is_correct', 'answer_changed', 'timestamp'
        ]
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input data format and integrity
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        if errors:
            return False, errors
        
        # Check data types and ranges
        if df['time_spent'].min() < 0:
            errors.append("Negative time_spent values found")
        
        if not df['is_correct'].isin([0, 1]).all():
            errors.append("is_correct must be binary (0 or 1)")
        
        if not df['answer_changed'].isin([0, 1]).all():
            errors.append("answer_changed must be binary (0 or 1)")
        
        # Check for missing values
        null_counts = df[self.required_columns].isnull().sum()
        if null_counts.any():
            errors.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
        
        return len(errors) == 0, errors
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove duplicates
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['student_id', 'question_id'])
        if len(df_clean) < initial_count:
            warnings.warn(f"Removed {initial_count - len(df_clean)} duplicate records")
        
        # Handle outliers in time_spent (cap at 99th percentile)
        time_99th = df_clean['time_spent'].quantile(0.99)
        df_clean['time_spent'] = df_clean['time_spent'].clip(upper=time_99th)
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_clean['timestamp']):
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
        
        # Sort by student and question for sequential processing
        df_clean = df_clean.sort_values(['student_id', 'question_id']).reset_index(drop=True)
        
        return df_clean
    
    def get_student_question_matrix(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """
        Create student-question matrix for a specific value
        
        Args:
            df: Input DataFrame
            value_col: Column to pivot (e.g., 'is_correct', 'time_spent')
            
        Returns:
            Pivot table with students as rows, questions as columns
        """
        matrix = df.pivot_table(
            index='student_id',
            columns='question_id',
            values=value_col,
            aggfunc='first'  # Take first value if duplicates
        )
        return matrix
    
    def preprocess(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Raw input DataFrame
            verbose: Print processing information
            
        Returns:
            Preprocessed DataFrame ready for feature engineering
        """
        if verbose:
            print("🔧 Starting data preprocessing...")
        
        # Validate
        is_valid, errors = self.validate_data(df)
        if not is_valid:
            raise ValueError(f"Data validation failed:\n" + "\n".join(errors))
        
        if verbose:
            print("✅ Data validation passed")
        
        # Clean
        df_clean = self.clean_data(df)
        
        if verbose:
            print(f"✅ Data cleaned: {len(df_clean)} records")
            print(f"   Students: {df_clean['student_id'].nunique()}")
            print(f"   Questions: {df_clean['question_id'].nunique()}")
            print(f"   Time range: {df_clean['timestamp'].min()} to {df_clean['timestamp'].max()}")
        
        return df_clean
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Generate comprehensive data summary statistics
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_records': len(df),
            'unique_students': df['student_id'].nunique(),
            'unique_questions': df['question_id'].nunique(),
            'avg_time_per_question': df['time_spent'].mean(),
            'median_time_per_question': df['time_spent'].median(),
            'overall_accuracy': df['is_correct'].mean(),
            'answer_change_rate': df['answer_changed'].mean(),
            'time_range': {
                'min': df['timestamp'].min(),
                'max': df['timestamp'].max()
            }
        }
        return summary


if __name__ == "__main__":
    # Example usage
    from data_generator import ExamDataGenerator
    
    # Generate sample data
    generator = ExamDataGenerator(num_normal_students=50, num_questions=30)
    df, _ = generator.generate_complete_dataset()
    
    # Preprocess
    preprocessor = ExamDataPreprocessor()
    df_clean = preprocessor.preprocess(df)
    
    # Get summary
    summary = preprocessor.get_data_summary(df_clean)
    print("\n📊 Data Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
