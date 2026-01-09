"""
Similarity Metrics Module
Computes pairwise student similarity using multiple metrics
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple
from itertools import combinations


class SimilarityCalculator:
    """
    Computes pairwise similarity between students using:
    - Cosine similarity (for numerical behavioral features)
    - Jaccard similarity (for wrong answer overlap)
    - Composite weighted similarity
    """
    
    def __init__(self, cosine_weight: float = 0.4, jaccard_weight: float = 0.6):
        """
        Initialize similarity calculator
        
        Args:
            cosine_weight: Weight for cosine similarity in composite score
            jaccard_weight: Weight for Jaccard similarity in composite score
        """
        self.cosine_weight = cosine_weight
        self.jaccard_weight = jaccard_weight
        
        # Validate weights
        if not np.isclose(cosine_weight + jaccard_weight, 1.0):
            raise ValueError("Weights must sum to 1.0")
    
    def compute_cosine_similarity(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Compute pairwise cosine similarity on behavioral features
        
        Args:
            feature_matrix: Student feature matrix (students × features)
            
        Returns:
            Similarity matrix (students × students)
        """
        # Extract student IDs and numerical features
        student_ids = feature_matrix['student_id'].values
        
        # Get numerical features only
        numerical_features = feature_matrix.select_dtypes(include=[np.number])
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(numerical_features)
        
        # Convert to DataFrame with student IDs
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=student_ids,
            columns=student_ids
        )
        
        return similarity_df
    
    def compute_jaccard_similarity(self, wrong_answer_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Compute pairwise Jaccard similarity on wrong answer patterns
        
        Jaccard(A, B) = |A ∩ B| / |A ∪ B|
        where A and B are sets of questions answered incorrectly
        
        Args:
            wrong_answer_matrix: Binary matrix (students × questions)
                                1 = wrong, 0 = correct
            
        Returns:
            Jaccard similarity matrix (students × students)
        """
        student_ids = wrong_answer_matrix.index.values
        n_students = len(student_ids)
        
        # Initialize similarity matrix
        jaccard_matrix = np.zeros((n_students, n_students))
        
        # Convert to numpy for faster computation
        wrong_answers = wrong_answer_matrix.values
        
        for i in range(n_students):
            for j in range(i, n_students):
                # Intersection: both students got the question wrong
                intersection = np.sum(wrong_answers[i] & wrong_answers[j])
                
                # Union: at least one student got the question wrong
                union = np.sum(wrong_answers[i] | wrong_answers[j])
                
                # Jaccard similarity
                if union > 0:
                    jaccard = intersection / union
                else:
                    jaccard = 0.0
                
                jaccard_matrix[i, j] = jaccard
                jaccard_matrix[j, i] = jaccard  # Symmetric
        
        # Convert to DataFrame
        jaccard_df = pd.DataFrame(
            jaccard_matrix,
            index=student_ids,
            columns=student_ids
        )
        
        return jaccard_df
    
    def compute_composite_similarity(self,
                                      cosine_sim: pd.DataFrame,
                                      jaccard_sim: pd.DataFrame) -> pd.DataFrame:
        """
        Compute weighted composite similarity score
        
        Args:
            cosine_sim: Cosine similarity matrix
            jaccard_sim: Jaccard similarity matrix
            
        Returns:
            Composite similarity matrix
        """
        # Ensure same index/columns
        assert cosine_sim.index.equals(jaccard_sim.index), "Student IDs must match"
        
        # Weighted combination
        composite = (self.cosine_weight * cosine_sim + 
                    self.jaccard_weight * jaccard_sim)
        
        return composite
    
    def get_top_similar_pairs(self,
                              similarity_matrix: pd.DataFrame,
                              threshold: float = 0.7,
                              top_n: int = None) -> pd.DataFrame:
        """
        Extract top similar student pairs above threshold
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            threshold: Minimum similarity threshold
            top_n: Optional limit on number of pairs to return
            
        Returns:
            DataFrame with student pairs and similarity scores
        """
        pairs = []
        student_ids = similarity_matrix.index.values
        
        # Extract upper triangle (avoid duplicates)
        for i, student1 in enumerate(student_ids):
            for j, student2 in enumerate(student_ids[i+1:], start=i+1):
                similarity = similarity_matrix.iloc[i, j]
                
                if similarity >= threshold:
                    pairs.append({
                        'student_1': student1,
                        'student_2': student2,
                        'similarity_score': round(similarity, 4)
                    })
        
        # Convert to DataFrame and sort
        pairs_df = pd.DataFrame(pairs)
        
        if len(pairs_df) > 0:
            pairs_df = pairs_df.sort_values('similarity_score', ascending=False)
            
            if top_n is not None:
                pairs_df = pairs_df.head(top_n)
        
        return pairs_df
    
    def compute_all_similarities(self,
                                  feature_matrix: pd.DataFrame,
                                  wrong_answer_matrix: pd.DataFrame,
                                  verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Compute all similarity metrics
        
        Args:
            feature_matrix: Student behavioral features
            wrong_answer_matrix: Binary wrong answer matrix
            verbose: Print progress information
            
        Returns:
            Dictionary with all similarity matrices
        """
        if verbose:
            print("📊 Computing similarity metrics...")
        
        # Cosine similarity on behavioral features
        cosine_sim = self.compute_cosine_similarity(feature_matrix)
        if verbose:
            print(f"✅ Cosine similarity computed ({cosine_sim.shape})")
        
        # Jaccard similarity on wrong answers
        jaccard_sim = self.compute_jaccard_similarity(wrong_answer_matrix)
        if verbose:
            print(f"✅ Jaccard similarity computed ({jaccard_sim.shape})")
        
        # Composite similarity
        composite_sim = self.compute_composite_similarity(cosine_sim, jaccard_sim)
        if verbose:
            print(f"✅ Composite similarity computed (weights: cosine={self.cosine_weight}, jaccard={self.jaccard_weight})")
        
        return {
            'cosine': cosine_sim,
            'jaccard': jaccard_sim,
            'composite': composite_sim
        }
    
    def get_similarity_statistics(self, similarity_matrix: pd.DataFrame) -> Dict:
        """
        Compute statistics for a similarity matrix
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            
        Returns:
            Dictionary with statistics
        """
        # Extract upper triangle (exclude diagonal)
        upper_triangle = similarity_matrix.values[np.triu_indices_from(similarity_matrix.values, k=1)]
        
        stats = {
            'mean': np.mean(upper_triangle),
            'median': np.median(upper_triangle),
            'std': np.std(upper_triangle),
            'min': np.min(upper_triangle),
            'max': np.max(upper_triangle),
            'q25': np.percentile(upper_triangle, 25),
            'q75': np.percentile(upper_triangle, 75)
        }
        
        return stats


if __name__ == "__main__":
    # Example usage
    from data_generator import ExamDataGenerator
    from preprocessing import ExamDataPreprocessor
    from feature_engineering import FeatureEngineer
    
    # Generate data
    generator = ExamDataGenerator(num_normal_students=30, num_questions=25)
    df, ground_truth = generator.generate_complete_dataset()
    
    # Preprocess
    preprocessor = ExamDataPreprocessor()
    df_clean = preprocessor.preprocess(df, verbose=False)
    
    # Extract features
    engineer = FeatureEngineer()
    features = engineer.create_feature_matrix(df_clean, normalize=True)
    wrong_answers = engineer.extract_wrong_answer_vectors(df_clean)
    
    # Compute similarities
    calculator = SimilarityCalculator(cosine_weight=0.4, jaccard_weight=0.6)
    similarities = calculator.compute_all_similarities(features, wrong_answers)
    
    # Get top pairs
    top_pairs = calculator.get_top_similar_pairs(similarities['composite'], threshold=0.7)
    
    print("\n🔍 Top Similar Student Pairs:")
    print(top_pairs.head(10))
    
    # Statistics
    stats = calculator.get_similarity_statistics(similarities['composite'])
    print("\n📈 Similarity Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value:.4f}")
