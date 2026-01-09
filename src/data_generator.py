"""
Synthetic Exam Data Generator
Generates realistic exam interaction logs with normal and colluding student behaviors
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple
import random


class ExamDataGenerator:
    """
    Generates synthetic exam interaction data simulating:
    - Normal student behavior with natural variation
    - Colluding student groups with coordinated patterns
    """
    
    def __init__(self, 
                 num_normal_students: int = 100,
                 num_questions: int = 50,
                 exam_duration_minutes: int = 120,
                 seed: int = 42):
        """
        Initialize the data generator
        
        Args:
            num_normal_students: Number of students with normal behavior
            num_questions: Total questions in the exam
            exam_duration_minutes: Total exam duration
            seed: Random seed for reproducibility
        """
        self.num_normal_students = num_normal_students
        self.num_questions = num_questions
        self.exam_duration_minutes = exam_duration_minutes
        self.seed = seed
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Correct answers for the exam (ground truth)
        self.correct_answers = np.random.randint(1, 5, size=num_questions)
        
    def generate_normal_student(self, student_id: int, base_time: datetime) -> pd.DataFrame:
        """
        Generate exam logs for a normal student with natural variation
        
        Args:
            student_id: Unique student identifier
            base_time: Exam start time
            
        Returns:
            DataFrame with exam interaction logs
        """
        records = []
        current_time = base_time
        
        # Student ability level (affects accuracy)
        ability = np.random.uniform(0.4, 0.95)
        
        # Time management style (affects time distribution)
        avg_time_per_q = (self.exam_duration_minutes * 60) / self.num_questions
        time_variance = np.random.uniform(0.5, 2.0)
        
        for q_id in range(1, self.num_questions + 1):
            # Time spent on this question (with natural variation)
            time_spent = max(10, np.random.normal(
                avg_time_per_q * time_variance, 
                avg_time_per_q * 0.3
            ))
            
            # Answer selection based on ability
            if np.random.random() < ability:
                selected_option = self.correct_answers[q_id - 1]
                is_correct = 1
            else:
                # Wrong answer
                wrong_options = [i for i in range(1, 5) if i != self.correct_answers[q_id - 1]]
                selected_option = np.random.choice(wrong_options)
                is_correct = 0
            
            # Answer change behavior (independent for normal students)
            answer_changed = 1 if np.random.random() < 0.15 else 0
            
            records.append({
                'student_id': f'S{student_id:04d}',
                'question_id': q_id,
                'time_spent': round(time_spent, 2),
                'selected_option': selected_option,
                'is_correct': is_correct,
                'answer_changed': answer_changed,
                'timestamp': current_time
            })
            
            current_time += timedelta(seconds=time_spent)
        
        return pd.DataFrame(records)
    
    def generate_colluding_group(self, 
                                  student_ids: List[int], 
                                  base_time: datetime,
                                  collusion_strength: float = 0.8) -> pd.DataFrame:
        """
        Generate exam logs for a colluding student group
        
        Args:
            student_ids: List of student IDs in the group
            base_time: Exam start time
            collusion_strength: Degree of coordination (0-1)
            
        Returns:
            DataFrame with coordinated exam interaction logs
        """
        all_records = []
        
        # Shared wrong answers (key indicator of collusion)
        shared_wrong_questions = np.random.choice(
            range(1, self.num_questions + 1),
            size=int(self.num_questions * 0.6),  # 60% shared wrong answers for stronger signal
            replace=False
        )
        
        # Generate a shared wrong answer pattern
        shared_wrong_answers = {}
        for q_id in shared_wrong_questions:
            wrong_options = [i for i in range(1, 5) if i != self.correct_answers[q_id - 1]]
            shared_wrong_answers[q_id] = np.random.choice(wrong_options)
        
        # Shared timing pattern (similar time distribution)
        base_time_pattern = np.random.normal(
            (self.exam_duration_minutes * 60) / self.num_questions,
            20,
            size=self.num_questions
        )
        
        for student_id in student_ids:
            records = []
            current_time = base_time
            
            # Individual ability (but similar within group)
            ability = np.random.uniform(0.5, 0.7)
            
            for q_id in range(1, self.num_questions + 1):
                # Time spent - highly correlated with group pattern
                if np.random.random() < collusion_strength:
                    time_spent = max(10, base_time_pattern[q_id - 1] + np.random.normal(0, 5))
                else:
                    time_spent = max(10, np.random.normal(60, 20))
                
                # Answer selection - coordinated wrong answers
                if q_id in shared_wrong_answers and np.random.random() < collusion_strength:
                    selected_option = shared_wrong_answers[q_id]
                    is_correct = 0
                elif np.random.random() < ability:
                    selected_option = self.correct_answers[q_id - 1]
                    is_correct = 1
                else:
                    wrong_options = [i for i in range(1, 5) if i != self.correct_answers[q_id - 1]]
                    selected_option = np.random.choice(wrong_options)
                    is_correct = 0
                
                # Similar answer change behavior
                answer_changed = 1 if np.random.random() < 0.25 else 0
                
                records.append({
                    'student_id': f'S{student_id:04d}',
                    'question_id': q_id,
                    'time_spent': round(time_spent, 2),
                    'selected_option': selected_option,
                    'is_correct': is_correct,
                    'answer_changed': answer_changed,
                    'timestamp': current_time
                })
                
                current_time += timedelta(seconds=time_spent)
            
            all_records.extend(records)
        
        return pd.DataFrame(all_records)
    
    def generate_complete_dataset(self, 
                                   colluding_groups: List[List[int]] = None) -> Tuple[pd.DataFrame, List[List[str]]]:
        """
        Generate complete exam dataset with normal and colluding students
        
        Args:
            colluding_groups: List of student ID lists representing colluding groups
            
        Returns:
            Tuple of (complete DataFrame, list of colluding group IDs)
        """
        if colluding_groups is None:
            colluding_groups = [
                list(range(1, 5)),      # Group 1: 4 students
                list(range(5, 10)),     # Group 2: 5 students
                list(range(10, 13))     # Group 3: 3 students
            ]
        
        all_data = []
        base_time = datetime(2024, 1, 15, 9, 0, 0)  # Exam start time
        
        # Track which students are in colluding groups
        colluding_student_ids = set()
        for group in colluding_groups:
            colluding_student_ids.update(group)
        
        # Generate colluding groups
        ground_truth_groups = []
        for group in colluding_groups:
            group_data = self.generate_colluding_group(group, base_time)
            all_data.append(group_data)
            ground_truth_groups.append([f'S{sid:04d}' for sid in group])
        
        # Generate normal students
        start_id = max(colluding_student_ids) + 1 if colluding_student_ids else 1
        for i in range(start_id, start_id + self.num_normal_students):
            student_data = self.generate_normal_student(i, base_time)
            all_data.append(student_data)
        
        # Combine all data
        complete_df = pd.concat(all_data, ignore_index=True)
        
        # Shuffle to mix colluding and normal students
        complete_df = complete_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        return complete_df, ground_truth_groups
    
    def save_dataset(self, df: pd.DataFrame, filepath: str):
        """Save dataset to CSV"""
        df.to_csv(filepath, index=False)
        print(f"✅ Dataset saved to {filepath}")
        print(f"   Total records: {len(df)}")
        print(f"   Unique students: {df['student_id'].nunique()}")
        print(f"   Questions: {df['question_id'].nunique()}")


if __name__ == "__main__":
    # Example usage
    generator = ExamDataGenerator(
        num_normal_students=100,
        num_questions=50,
        exam_duration_minutes=120
    )
    
    df, ground_truth = generator.generate_complete_dataset()
    
    print("\n📊 Dataset Summary:")
    print(df.head(10))
    print(f"\n🚨 Ground Truth Colluding Groups:")
    for i, group in enumerate(ground_truth, 1):
        print(f"   Group {i}: {group}")
