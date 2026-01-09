"""
Collusion Detection Module
Implements logic to flag suspicious student groups based on similarity and graph analysis
"""

import pandas as pd
import numpy as np
from typing import List, Set, Dict, Tuple


class CollusionDetector:
    """
    Detects potential collusion by analyzing:
    - Pairwise similarity scores
    - Group sizes from graph analysis
    - Wrong answer overlap patterns
    - Timing correlations
    """
    
    def __init__(self,
                 suspicious_similarity_threshold: float = 0.75,
                 min_group_size: int = 3,
                 wrong_answer_overlap_threshold: float = 0.5):
        """
        Initialize collusion detector
        
        Args:
            suspicious_similarity_threshold: Minimum similarity to flag as suspicious
            min_group_size: Minimum group size to investigate
            wrong_answer_overlap_threshold: Minimum wrong answer overlap ratio
        """
        self.suspicious_similarity_threshold = suspicious_similarity_threshold
        self.min_group_size = min_group_size
        self.wrong_answer_overlap_threshold = wrong_answer_overlap_threshold
    
    def analyze_group(self,
                      group: Set[str],
                      similarity_matrix: pd.DataFrame,
                      wrong_answer_matrix: pd.DataFrame) -> Dict:
        """
        Analyze a group for collusion indicators
        
        Args:
            group: Set of student IDs in the group
            similarity_matrix: Pairwise similarity matrix
            wrong_answer_matrix: Binary wrong answer matrix
            
        Returns:
            Dictionary with analysis results
        """
        group_list = list(group)
        
        # Get pairwise similarities within group
        similarities = []
        for i, s1 in enumerate(group_list):
            for s2 in group_list[i+1:]:
                sim = similarity_matrix.loc[s1, s2]
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 0
        max_similarity = np.max(similarities) if similarities else 0
        min_similarity = np.min(similarities) if similarities else 0
        
        # Analyze wrong answer overlap - compute average pairwise overlap
        wrong_answers_group = wrong_answer_matrix.loc[group_list]
        
        # Calculate pairwise Jaccard similarity for wrong answers within group
        pairwise_overlaps = []
        shared_wrong_count = 0
        for i, s1 in enumerate(group_list):
            for s2 in group_list[i+1:]:
                w1 = wrong_answers_group.loc[s1].values
                w2 = wrong_answers_group.loc[s2].values
                intersection = np.sum(w1 & w2)
                union = np.sum(w1 | w2)
                overlap = intersection / union if union > 0 else 0
                pairwise_overlaps.append(overlap)
                shared_wrong_count += intersection
        
        wrong_overlap_ratio = np.mean(pairwise_overlaps) if pairwise_overlaps else 0
        shared_wrong_questions = int(shared_wrong_count / len(pairwise_overlaps)) if pairwise_overlaps else 0
        
        analysis = {
            'group_size': len(group),
            'avg_similarity': round(avg_similarity, 4),
            'max_similarity': round(max_similarity, 4),
            'min_similarity': round(min_similarity, 4),
            'shared_wrong_questions': shared_wrong_questions,
            'wrong_overlap_ratio': round(wrong_overlap_ratio, 4),
            'is_suspicious': False,
            'suspicion_reasons': []
        }
        
        # Determine if suspicious
        if avg_similarity >= self.suspicious_similarity_threshold:
            analysis['is_suspicious'] = True
            analysis['suspicion_reasons'].append(
                f"High average similarity ({avg_similarity:.3f})"
            )
        
        if wrong_overlap_ratio >= self.wrong_answer_overlap_threshold:
            analysis['is_suspicious'] = True
            analysis['suspicion_reasons'].append(
                f"High wrong answer overlap ({wrong_overlap_ratio:.3f})"
            )
        
        if len(group) >= self.min_group_size and avg_similarity >= self.suspicious_similarity_threshold:
            analysis['is_suspicious'] = True
            analysis['suspicion_reasons'].append(
                f"Large coordinated group (size={len(group)})"
            )
        
        return analysis
    
    def detect_suspicious_groups(self,
                                  groups: List[Set[str]],
                                  similarity_matrix: pd.DataFrame,
                                  wrong_answer_matrix: pd.DataFrame) -> List[Dict]:
        """
        Analyze all groups and flag suspicious ones
        
        Args:
            groups: List of student groups from graph analysis
            similarity_matrix: Pairwise similarity matrix
            wrong_answer_matrix: Binary wrong answer matrix
            
        Returns:
            List of dictionaries with group analysis results
        """
        print("\n🔍 Analyzing groups for collusion...")
        
        results = []
        
        for i, group in enumerate(groups, 1):
            if len(group) < 2:
                continue
            
            analysis = self.analyze_group(group, similarity_matrix, wrong_answer_matrix)
            analysis['group_id'] = i
            analysis['students'] = sorted(list(group))
            
            results.append(analysis)
        
        # Sort by suspicion level (suspicious first, then by avg similarity)
        results.sort(key=lambda x: (not x['is_suspicious'], -x['avg_similarity']))
        
        suspicious_count = sum(1 for r in results if r['is_suspicious'])
        
        print(f"✅ Analysis complete:")
        print(f"   Total groups analyzed: {len(results)}")
        print(f"   Suspicious groups: {suspicious_count}")
        
        return results
    
    def generate_report(self, analysis_results: List[Dict]) -> pd.DataFrame:
        """
        Generate a tabular report of analysis results
        
        Args:
            analysis_results: List of group analysis dictionaries
            
        Returns:
            DataFrame with report
        """
        report_data = []
        
        for result in analysis_results:
            report_data.append({
                'group_id': result['group_id'],
                'students': ', '.join(result['students']),
                'group_size': result['group_size'],
                'avg_similarity': result['avg_similarity'],
                'max_similarity': result['max_similarity'],
                'wrong_overlap_ratio': result['wrong_overlap_ratio'],
                'is_suspicious': result['is_suspicious'],
                'reasons': '; '.join(result['suspicion_reasons']) if result['suspicion_reasons'] else 'None'
            })
        
        return pd.DataFrame(report_data)


if __name__ == "__main__":
    print("Collusion Detector Module - Use via main.py or app.py")
