import os
from utils import JobRecommender

def evaluate_at_k(recommender, skills, experience, relevant_terms, k=3):
    """Use title-keyword relevance labels for a reproducible demo ranking check."""
    candidates = recommender.jobs_df.copy()
    if experience != 'All':
        candidates = candidates[candidates['experience'] == experience]
    relevant_mask = candidates['title'].str.lower().apply(
        lambda title: any(term in title for term in relevant_terms)
    )
    relevant_count = int(relevant_mask.sum())
    results = recommender.get_recommendations(skills, experience, top_n=k)
    hits = int(results['title'].str.lower().apply(
        lambda title: any(term in title for term in relevant_terms)
    ).sum())
    precision = hits / k if k else 0.0
    recall = hits / relevant_count if relevant_count else 0.0
    return precision, recall


def train_and_save_model():
    """Train the job recommendation model and save it"""
    print("🚀 Training AI Job Recommendation Model...")
    print("=" * 60)
    
    # Initialize recommender
    recommender = JobRecommender()
    
    # Load data
    print("\n📂 Loading jobs data...")
    if not os.path.exists('data/jobs.csv'):
        print("❌ Error: data/jobs.csv not found!")
        return
    
    recommender.load_data()
    print(f"✅ Loaded {len(recommender.jobs_df)} job records")
    
    # Build model
    print("\n🔧 Building semantic recommendation model...")
    recommender.build_model()
    print(f"✅ Model built: {recommender.embedding_model_type}")
    
    # Save model
    print("\n💾 Saving model...")
    recommender.save_model()
    
    # Test the model
    print("\n" + "=" * 60)
    print("🧪 Testing Model with Sample Queries")
    print("=" * 60)
    
    test_cases = [
        ("python, sql, excel", "Junior", "Data Analysis Entry Level", ("data",)),
        ("python, machine learning, tensorflow, aws", "Senior", "Senior ML Engineer", ("ml", "machine learning")),
        ("javascript, react, nodejs", "Mid", "Full Stack Developer", ("full stack", "frontend", "web")),
        ("aws, docker, kubernetes, terraform", "Mid", "DevOps Engineer", ("devops", "site reliability")),
        ("python, django, postgresql", "Mid", "Backend Developer", ("backend",))
    ]
    
    for skills, exp, test_name, relevant_terms in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print(f"Skills: {skills}")
        print(f"Experience: {exp}")
        print(f"{'='*60}")
        
        recs = recommender.get_recommendations(skills, exp, top_n=3)
        precision, recall = evaluate_at_k(
            recommender, skills, exp, relevant_terms, k=3
        )
        print(f"Precision@3: {precision:.2f} | Recall@3: {recall:.2f}")
        
        if recs.empty:
            print("⚠️ No recommendations found for this criteria")
        else:
            for idx, row in recs.iterrows():
                match_score = recommender.skill_match_score(skills, row['skills'])
                print(f"\n📌 {row['title']}")
                print(f"   Company: {row['company']}")
                print(f"   Location: {row['location']} | Salary: {row['salary']}")
                print(f"   Match Score: {match_score}% | Similarity: {row['similarity']:.1%}")
    
    print("\n" + "=" * 60)
    print("✅ Training complete! Model saved to models/job_recommender.pkl")
    print("🚀 You can now run the app with: streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    train_and_save_model()
