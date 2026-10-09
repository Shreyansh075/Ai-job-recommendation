import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from collections import OrderedDict

class JobRecommender:
    def __init__(self):
        self.jobs_df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self._query_cache = OrderedDict()
        self.embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        self.embedding_model_type = 'tfidf'
        self.job_embeddings = None
        self._embedding_model = None

    def _ensure_query_cache(self):
        if not isinstance(getattr(self, '_query_cache', None), OrderedDict):
            self._query_cache = OrderedDict(
                getattr(self, '_query_cache', {}) or {}
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_embedding_model'] = None
        state['_query_cache'] = OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'embedding_model_name'):
            self.embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        if not hasattr(self, 'embedding_model_type'):
            self.embedding_model_type = 'tfidf'
        if not hasattr(self, 'job_embeddings'):
            self.job_embeddings = None
        self._embedding_model = None
        self._query_cache = OrderedDict()

    def _get_embedding_model(self):
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    'Semantic model artifacts require sentence-transformers. '
                    'Install project dependencies and run Train_model.py again.'
                ) from error
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def _job_documents(self):
        jobs = self.normalize_jobs_schema(self.jobs_df)
        return (
            jobs['title'].astype(str) + '. ' +
            jobs['skills'].astype(str) + '. ' +
            jobs['description'].astype(str)
        ).tolist()

    @staticmethod
    def normalize_jobs_schema(jobs_df):
        """Keep older CSVs and newly refreshed jobs on the same schema."""
        jobs_df = jobs_df.copy()
        defaults = {
            'title': 'Unknown role',
            'company': 'Company not listed',
            'location': 'Location not listed',
            'experience': 'Mid',
            'salary': 'Not listed',
            'skills': '',
            'job_type': 'Not specified',
            'remote': False,
            'work_mode': '',
            'description': '',
            'posted_date': '',
            'apply_url': ''
        }
        for column, default in defaults.items():
            if column not in jobs_df.columns:
                jobs_df[column] = default
            jobs_df[column] = jobs_df[column].fillna(default)
        jobs_df['remote'] = jobs_df['remote'].map(
            lambda value: value if isinstance(value, bool) else
            str(value).strip().lower() in {'true', 'yes', '1', 'remote'}
        )
        if 'work_mode' not in jobs_df.columns:
            jobs_df['work_mode'] = jobs_df['remote'].map(
                lambda remote: 'Remote' if remote else 'On-site'
            )
        else:
            missing_mode = jobs_df['work_mode'].astype(str).str.strip().eq('')
            jobs_df.loc[missing_mode, 'work_mode'] = jobs_df.loc[
                missing_mode, 'remote'
            ].map(lambda remote: 'Remote' if remote else 'On-site')
            mode_names = {
                'remote': 'Remote',
                'hybrid': 'Hybrid',
                'on-site': 'On-site',
                'onsite': 'On-site',
                'on site': 'On-site'
            }
            jobs_df['work_mode'] = jobs_df['work_mode'].map(
                lambda mode: mode_names.get(str(mode).strip().lower(), str(mode).strip())
            )
        return jobs_df
        
    def load_data(self, filepath='data/jobs.csv'):
        """Load jobs data from CSV"""
        self.jobs_df = self.normalize_jobs_schema(pd.read_csv(filepath))
        return self.jobs_df
    
    def preprocess_skills(self, skills_str):
        """Preprocess skills string"""
        if not isinstance(skills_str, str):
            return set()
        skills = [skill.strip().lower() for skill in skills_str.split(',')]
        return set(skills)
    
    def build_model(self):
        """Build sentence embeddings, retaining TF-IDF as an explicit fallback."""
        if self.jobs_df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        self._ensure_query_cache()

        job_documents = self._job_documents()
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(job_documents)
        self.job_embeddings = None
        self.embedding_model_type = 'tfidf'

        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError:
            print(
                "sentence-transformers is not installed; using TF-IDF fallback. "
                "Install requirements.txt and retrain for semantic matching."
            )
        else:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            self.job_embeddings = self._embedding_model.encode(
                job_documents,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True
            )
            self.embedding_model_type = 'sentence-transformer'

        self._query_cache.clear()
        vectors = (
            self.job_embeddings
            if self.job_embeddings is not None
            else self.tfidf_matrix
        )
        return self.embedding_model_type, vectors
    
    def skill_match_score(self, user_skills, job_skills):
        """Calculate percentage of user skills matching job skills"""
        user_skill_set = self.preprocess_skills(user_skills)
        job_skill_set = self.preprocess_skills(job_skills)
        
        if not job_skill_set:
            return 0
        
        matching_skills = user_skill_set.intersection(job_skill_set)
        score = (len(matching_skills) / len(job_skill_set)) * 100
        
        return round(score)
    
    def get_recommendations(self, user_skills, experience_level, top_n=5):
        """Get job recommendations based on user skills and experience"""
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            raise ValueError("Model not built. Call build_model() first.")
        self._ensure_query_cache()
        
        cache_key = str(user_skills).strip().lower()
        user_tfidf = self._query_cache.get(cache_key)
        if user_tfidf is None:
            if (
                getattr(self, 'embedding_model_type', 'tfidf') == 'sentence-transformer'
                and getattr(self, 'job_embeddings', None) is not None
            ):
                user_vector = self._get_embedding_model().encode(
                    [str(user_skills)],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
            else:
                user_vector = self.tfidf_vectorizer.transform([str(user_skills)])
            self._query_cache[cache_key] = user_vector
            user_tfidf = user_vector
            if len(self._query_cache) > 256:
                self._query_cache.popitem(last=False)
        else:
            self._query_cache.move_to_end(cache_key)

        if (
            getattr(self, 'embedding_model_type', 'tfidf') == 'sentence-transformer'
            and getattr(self, 'job_embeddings', None) is not None
        ):
            similarities = cosine_similarity(
                user_tfidf, self.job_embeddings
            ).flatten()
        else:
            similarities = cosine_similarity(
                user_tfidf, self.tfidf_matrix
            ).flatten()

        recommendations_df = self.normalize_jobs_schema(self.jobs_df)
        recommendations_df['similarity'] = similarities
        
        # Filter by experience level
        if experience_level != 'All':
            recommendations_df = recommendations_df[
                recommendations_df['experience'] == experience_level
            ]
        
        # Sort by similarity and get top N
        recommendations_df = recommendations_df.sort_values(
            'similarity', 
            ascending=False
        ).head(top_n)
        
        return recommendations_df

    def experience_match_score(self, preferred_experience, job_experience):
        """Score experience fit while allowing adjacent career levels."""
        levels = ['Junior', 'Mid', 'Senior']
        preferred = str(preferred_experience).strip().lower()
        actual = str(job_experience).strip().lower()
        if preferred == 'all' or not preferred:
            return 100
        if preferred == actual:
            return 100
        normalized = {level.lower(): index for index, level in enumerate(levels)}
        if preferred in normalized and actual in normalized:
            distance = abs(normalized[preferred] - normalized[actual])
            return 70 if distance == 1 else 40
        return 50

    def overall_match_score(self, similarity, skills_score, experience_score):
        """Combine content similarity, required-skill coverage, and experience fit."""
        return round(
            (float(similarity) * 100 * 0.4) +
            (float(skills_score) * 0.4) +
            (float(experience_score) * 0.2)
        )
    
    def save_model(self, filepath='models/job_recommender.pkl'):
        """Save the trained model"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(self, filepath)
        print(f"✅ Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath='models/job_recommender.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filepath)

        if isinstance(model_data, cls):
            model_data.jobs_df = cls.normalize_jobs_schema(model_data.jobs_df)
            model_data._ensure_query_cache()
            return model_data

        recommender = cls()
        recommender.jobs_df = cls.normalize_jobs_schema(model_data['jobs_df'])
        recommender.tfidf_vectorizer = model_data['tfidf_vectorizer']
        recommender.tfidf_matrix = model_data['tfidf_matrix']
        
        return recommender


if __name__ == '__main__':
    # Test the recommender
    recommender = JobRecommender()
    
    # Load and process data
    print("Loading data...")
    recommender.load_data()
    
    print("Building model...")
    recommender.build_model()
    
    # Test recommendations
    print("\n" + "="*60)
    print("Testing Recommendations")
    print("="*60)
    
    test_cases = [
        ("python, sql, excel", "Junior"),
        ("python, machine learning, tensorflow", "Mid"),
        ("aws, docker, kubernetes, devops", "Senior")
    ]
    
    for skills, exp in test_cases:
        print(f"\nUser Skills: {skills}")
        print(f"Experience: {exp}")
        print("-" * 60)
        
        recs = recommender.get_recommendations(skills, exp, top_n=3)
        
        for idx, row in recs.iterrows():
            match_score = recommender.skill_match_score(skills, row['skills'])
            print(f"📌 {row['title']} at {row['company']}")
            print(f"   Match: {match_score}% | Similarity: {row['similarity']:.2f}")
            print(f"   Salary: {row['salary']} | Location: {row['location']}")
            print()
