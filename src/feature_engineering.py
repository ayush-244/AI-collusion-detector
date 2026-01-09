"""
Feature Engineering Module
Extracts student-level behavioral features from exam interaction logs
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """
    Extracts comprehensive behavioral features at the student level
    Features capture timing patterns, accuracy, and answer-changing behavior
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic statistical features per student
        
        Args:
            df: Preprocessed exam interaction logs
            
        Returns:
            DataFrame with student-level features
        """
        features = df.groupby('student_id').agg({
            'time_spent': ['mean', 'std', 'min', 'max', 'median'],
            'is_correct': ['mean', 'sum'],
            'answer_changed': ['mean', 'sum'],
            'question_id': 'count'
        }).reset_index()
        
        # Flatten column names
        features.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                           for col in features.columns.values]
        
        # Rename for clarity
        features = features.rename(columns={
            'time_spent_mean': 'avg_time_per_question',
            'time_spent_std': 'std_time_per_question',
            'time_spent_min': 'min_time_per_question',
            'time_spent_max': 'max_time_per_question',
            'time_spent_median': 'median_time_per_question',
            'is_correct_mean': 'accuracy_rate',
            'is_correct_sum': 'total_correct',
            'answer_changed_mean': 'answer_change_rate',
            'answer_changed_sum': 'total_answer_changes',
            'question_id_count': 'total_questions_attempted'
        })
        
        return features
    
    def extract_timing_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract timing-based pattern features
        
        Args:
            df: Preprocessed exam interaction logs
            
        Returns:
            DataFrame with timing pattern features
        """
        timing_features = []
        
        for student_id, group in df.groupby('student_id'):
            # Sort by question order
            group = group.sort_values('question_id')
            
            # Time variance across questions
            time_variance = group['time_spent'].var()
            
            # Coefficient of variation (normalized variance)
            cv = group['time_spent'].std() / group['time_spent'].mean() if group['time_spent'].mean() > 0 else 0
            
            # Time trend (are they speeding up or slowing down?)
            time_trend = np.polyfit(range(len(group)), group['time_spent'].values, 1)[0] if len(group) > 1 else 0
            
            # Percentage of questions answered very quickly (< 20 seconds)
            quick_answer_rate = (group['time_spent'] < 20).mean()
            
            # Percentage of questions answered very slowly (> 2 * median)
            median_time = group['time_spent'].median()
            slow_answer_rate = (group['time_spent'] > 2 * median_time).mean() if median_time > 0 else 0
            
            timing_features.append({
                'student_id': student_id,
                'time_variance': time_variance,
                'time_cv': cv,
                'time_trend': time_trend,
                'quick_answer_rate': quick_answer_rate,
                'slow_answer_rate': slow_answer_rate
            })
        
        return pd.DataFrame(timing_features)
    
    def extract_wrong_answer_vectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary vectors of wrong answers for each student
        Used for Jaccard similarity computation
        
        Args:
            df: Preprocessed exam interaction logs
            
        Returns:
            DataFrame with wrong answer vectors
        """
        # Create pivot: 1 if wrong, 0 if correct
        wrong_matrix = df.pivot_table(
            index='student_id',
            columns='question_id',
            values='is_correct',
            aggfunc='first'
        )
        
        # Invert: 1 = wrong, 0 = correct
        wrong_matrix = 1 - wrong_matrix
        wrong_matrix = wrong_matrix.fillna(0)  # Treat missing as correct
        
        return wrong_matrix
    
    def extract_answer_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features related to answer selection patterns
        
        Args:
            df: Preprocessed exam interaction logs
            
        Returns:
            DataFrame with answer pattern features
        """
        pattern_features = []
        
        for student_id, group in df.groupby('student_id'):
            # Option selection distribution
            option_counts = group['selected_option'].value_counts(normalize=True)
            option_entropy = -sum(p * np.log2(p) for p in option_counts if p > 0)
            
            # Consecutive wrong answers
            wrong_streak = 0
            max_wrong_streak = 0
            for is_correct in group.sort_values('question_id')['is_correct']:
                if is_correct == 0:
                    wrong_streak += 1
                    max_wrong_streak = max(max_wrong_streak, wrong_streak)
                else:
                    wrong_streak = 0
            
            # Answer changes on wrong vs correct
            changes_on_wrong = group[group['is_correct'] == 0]['answer_changed'].mean()
            changes_on_correct = group[group['is_correct'] == 1]['answer_changed'].mean()
            
            pattern_features.append({
                'student_id': student_id,
                'option_entropy': option_entropy,
                'max_wrong_streak': max_wrong_streak,
                'changes_on_wrong': changes_on_wrong if not np.isnan(changes_on_wrong) else 0,
                'changes_on_correct': changes_on_correct if not np.isnan(changes_on_correct) else 0
            })
        
        return pd.DataFrame(pattern_features)
    
    def create_feature_matrix(self, df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
        """
        Create complete feature matrix combining all feature types
        
        Args:
            df: Preprocessed exam interaction logs
            normalize: Whether to standardize numerical features
            
        Returns:
            Complete feature matrix with all student-level features
        """
        print("🔨 Extracting features...")
        
        # Extract all feature types
        basic_features = self.extract_basic_features(df)
        timing_features = self.extract_timing_patterns(df)
        pattern_features = self.extract_answer_pattern_features(df)
        
        # Merge all features
        feature_matrix = basic_features.merge(timing_features, on='student_id')
        feature_matrix = feature_matrix.merge(pattern_features, on='student_id')
        
        # Handle missing values
        feature_matrix = feature_matrix.fillna(0)
        
        # Store feature names (excluding student_id)
        self.feature_names = [col for col in feature_matrix.columns if col != 'student_id']
        
        # Normalize if requested
        if normalize:
            numerical_cols = feature_matrix.select_dtypes(include=[np.number]).columns
            numerical_cols = [col for col in numerical_cols if col != 'student_id']
            
            feature_matrix[numerical_cols] = self.scaler.fit_transform(
                feature_matrix[numerical_cols]
            )
        
        print(f"✅ Feature extraction complete: {len(self.feature_names)} features")
        
        return feature_matrix
    
    def get_feature_importance_summary(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for each feature
        
        Args:
            feature_matrix: Complete feature matrix
            
        Returns:
            DataFrame with feature statistics
        """
        summary = []
        
        for col in self.feature_names:
            summary.append({
                'feature': col,
                'mean': feature_matrix[col].mean(),
                'std': feature_matrix[col].std(),
                'min': feature_matrix[col].min(),
                'max': feature_matrix[col].max(),
                'missing_rate': feature_matrix[col].isnull().mean()
            })
        
        return pd.DataFrame(summary)


if __name__ == "__main__":
    # Example usage
    from data_generator import ExamDataGenerator
    from preprocessing import ExamDataPreprocessor
    
    # Generate and preprocess data
    generator = ExamDataGenerator(num_normal_students=50, num_questions=30)
    df, _ = generator.generate_complete_dataset()
    
    preprocessor = ExamDataPreprocessor()
    df_clean = preprocessor.preprocess(df, verbose=False)
    
    # Extract features
    engineer = FeatureEngineer()
    features = engineer.create_feature_matrix(df_clean)
    
    print("\n📊 Feature Matrix:")
    print(features.head())
    print(f"\nShape: {features.shape}")
    
    # Feature summary
    summary = engineer.get_feature_importance_summary(features)
    print("\n📈 Feature Summary:")
    print(summary)
