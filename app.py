import hashlib
import io
import json
import os
import re
from html import escape
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import JobRecommender
from wordcloud import WordCloud

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)

COMMON_SKILLS = {
    'python', 'java', 'javascript', 'react', 'angular', 'sql', 'pandas', 'numpy',
    'tensorflow', 'aws', 'docker', 'kubernetes', 'html', 'css', 'django', 'flask',
    'excel', 'powerbi', 'tableau', 'r', 'c++', 'c#', 'nodejs', 'git', 'linux',
    'azure', 'spark', 'hadoop', 'machine learning', 'deep learning', 'nlp',
    'pytorch', 'keras', 'data analysis', 'project management'
}

SAMPLE_PROFILES = {
    'Junior Data Analyst': {
        'skills': 'python, sql, excel, tableau, pandas, data analysis',
        'resume': (
            'Data Analyst with experience using Python, SQL, Excel, Tableau and pandas '
            'to clean datasets, build dashboards, and communicate business insights.'
        )
    },
    'Backend Developer': {
        'skills': 'python, django, postgresql, sql, docker, git, api',
        'resume': (
            'Backend Developer experienced with Python, Django, PostgreSQL, SQL, '
            'Docker and Git. Develops APIs and maintains reliable data services.'
        )
    },
    'Machine Learning Engineer': {
        'skills': 'python, machine learning, tensorflow, pytorch, aws, docker',
        'resume': (
            'Machine Learning Engineer experienced in Python, machine learning, '
            'TensorFlow, PyTorch, AWS and Docker. Built and deployed predictive models.'
        )
    },
}


def extract_text_from_resume(uploaded_file):
    if uploaded_file is None:
        return ''

    file_name = uploaded_file.name.lower()
    if file_name.endswith('.txt'):
        return uploaded_file.getvalue().decode('utf-8', errors='ignore')

    if file_name.endswith('.pdf'):
        try:
            import PyPDF2
        except ModuleNotFoundError as error:
            raise RuntimeError('Install PyPDF2 to parse PDF resumes.') from error
        reader = PyPDF2.PdfReader(uploaded_file)
        return '\n'.join(page.extract_text() or '' for page in reader.pages)

    if file_name.endswith('.docx'):
        try:
            from docx import Document
        except ModuleNotFoundError as error:
            raise RuntimeError('Install python-docx to parse DOCX resumes.') from error
        document = Document(uploaded_file)
        return '\n'.join(paragraph.text for paragraph in document.paragraphs)

    raise RuntimeError('Unsupported resume format. Upload TXT, PDF, or DOCX.')


def extract_skills_from_resume(text):
    if not text:
        return ''
    resume_lower = text.lower()
    found_skills = [
        skill for skill in COMMON_SKILLS
        if re.search(
            rf'(?<![a-z0-9]){re.escape(skill)}(?![a-z0-9])',
            resume_lower
        )
    ]
    return ', '.join(sorted(set(found_skills)))


def job_id(job):
    identity = '|'.join(
        str(job.get(key, '')) for key in ('title', 'company', 'location', 'apply_url')
    ).lower()
    return hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]


def read_activity(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def append_activity(filename, row):
    path = DATA_DIR / filename
    pd.DataFrame([row]).to_csv(
        path, mode='a', header=not path.exists(), index=False
    )


def persist_jobs(jobs_df):
    path = DATA_DIR / 'jobs.csv'
    jobs_df = JobRecommender.normalize_jobs_schema(jobs_df)
    jobs_df.to_csv(path, index=False)


def salary_amount(salary):
    salary_text = str(salary).strip()
    if any(symbol in salary_text for symbol in ('€', '£', '₹')) or re.search(
        r'\b(eur|gbp|inr|cad|aud)\b', salary_text, re.IGNORECASE
    ):
        return None
    numbers = re.findall(r'\d[\d,]*(?:\.\d+)?', salary_text)
    if not numbers:
        return None
    values = [float(number.replace(',', '')) for number in numbers]
    if 'k' in salary_text.lower() and max(values) < 1000:
        values = [value * 1000 for value in values]
    return int(sum(values) / len(values))


def feedback_adjustment(identifier, feedback_df):
    """Apply a smoothed, bounded adjustment from prior useful/not-useful votes."""
    if feedback_df.empty or 'job_id' not in feedback_df.columns:
        return 0.0
    votes = feedback_df[feedback_df['job_id'].astype(str) == str(identifier)]
    if votes.empty or 'useful' not in votes.columns:
        return 0.0
    useful = votes['useful'].astype(str).str.lower().isin({'true', '1', 'yes'})
    balance = int(useful.sum()) - int((~useful).sum())
    return round(5 * balance / (len(votes) + 2), 1)


def build_report_pdf(recommendations, skills, experience):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    )

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=0.65 * inch, leftMargin=0.65 * inch,
        topMargin=0.65 * inch, bottomMargin=0.65 * inch
    )
    styles = getSampleStyleSheet()
    elements = [
        Paragraph('AI Job Recommendation Report', styles['Title']),
        Spacer(1, 8),
        Paragraph(f'<b>Candidate skills:</b> {escape(skills)}', styles['BodyText']),
        Paragraph(f'<b>Experience level:</b> {escape(experience)}', styles['BodyText']),
        Spacer(1, 14)
    ]
    for job in recommendations:
        elements.append(Paragraph(
            escape(f"{job.get('title', 'Role')} — {job.get('company', 'Company')}"),
            styles['Heading2']
        ))
        elements.append(Paragraph(
            escape(
                f"{job.get('location', 'Location not listed')} | "
                f"{job.get('salary', 'Salary not listed')} | "
                f"{job.get('work_mode', 'Work mode not listed')}"
            ),
            styles['BodyText']
        ))
        data = [
            ['Overall', 'Similarity', 'Skill coverage', 'Experience fit'],
            [
                f"{job.get('overall_match', 0)}%",
                f"{float(job.get('similarity', 0)):.1%}",
                f"{job.get('skills_match', 0)}%",
                f"{job.get('experience_match', 0)}%"
            ]
        ]
        table = Table(data, colWidths=[1.45 * inch] * 4)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#243b53')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#bcccdc')),
            ('PADDING', (0, 0), (-1, -1), 6)
        ]))
        elements.extend([
            table,
            Paragraph(
                escape(f"Required skills: {job.get('skills', 'Not listed')}"),
                styles['BodyText']
            ),
            Spacer(1, 12)
        ])
    document.build(elements)
    return buffer.getvalue()


def refresh_remote_jobs(recommender):
    request = Request(
        'https://remotive.com/api/remote-jobs?limit=50',
        headers={'User-Agent': 'AIJobRecommendationEngine/1.0'}
    )
    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode('utf-8'))

    refreshed = []
    for job in payload.get('jobs', []):
        refreshed.append({
            'title': job.get('title') or 'Unknown role',
            'company': job.get('company_name') or 'Company not listed',
            'location': job.get('candidate_required_location') or 'Remote',
            'experience': infer_experience(job.get('title', '')),
            'salary': job.get('salary') or 'Not listed',
            'skills': ', '.join(extract_skills_from_resume(
                f"{job.get('title', '')} {job.get('description', '')}"
            ).split(', ')),
            'description': job.get('description', ''),
            'job_type': job.get('job_type') or 'Not specified',
            'remote': True,
            'work_mode': 'Remote',
            'posted_date': job.get('publication_date', ''),
            'apply_url': job.get('url', '')
        })

    if not refreshed:
        return 0

    incoming = JobRecommender.normalize_jobs_schema(pd.DataFrame(refreshed))
    existing = JobRecommender.normalize_jobs_schema(recommender.jobs_df)
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined['_job_id'] = combined.apply(job_id, axis=1)
    combined = combined.drop_duplicates('_job_id', keep='first').drop(columns='_job_id')
    recommender.jobs_df = combined.reset_index(drop=True)
    recommender.build_model()
    persist_jobs(recommender.jobs_df)
    recommender.save_model(str(PROJECT_DIR / 'models' / 'job_recommender.pkl'))
    load_model.clear()
    return len(refreshed)


def infer_experience(title):
    title = str(title).lower()
    if any(term in title for term in ('senior', 'sr.', 'lead', 'principal', 'director')):
        return 'Senior'
    if any(term in title for term in ('junior', 'jr.', 'entry-level', 'intern')):
        return 'Junior'
    return 'Mid'


def generate_cover_letter(job, skills, resume_text):
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError(
            'Set the OPENAI_API_KEY environment variable to generate cover letters.'
        )
    try:
        from openai import OpenAI, OpenAIError
    except ModuleNotFoundError as error:
        raise RuntimeError('Install the openai package from requirements.txt.') from error

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'Write a polished, specific cover letter of 3-4 short paragraphs. '
                        'Map evidence from the candidate resume and skills to the actual '
                        'job responsibilities. Do not invent experience, credentials, '
                        'metrics, or accomplishments. Avoid generic claims and mention '
                        'a skill gap honestly rather than pretending it is present.'
                    )
                },
                {
                    'role': 'user',
                    'content': (
                        f"Candidate skills: {skills}\nResume:\n{resume_text[:6000] or 'Not provided'}\n"
                        f"Job title: {job.get('title')}\nCompany: {job.get('company')}\n"
                        f"Location: {job.get('location')}\n"
                        f"Job description: {str(job.get('description', ''))[:5000]}\n"
                        f"Required skills: {job.get('skills')}\n"
                        f"Work mode: {job.get('work_mode', 'Not specified')}"
                    )
                }
            ],
            temperature=0.5,
            max_tokens=500
        )
    except OpenAIError as error:
        raise RuntimeError(f'OpenAI request failed: {error}') from error
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError('The LLM returned an empty cover letter.')
    return content.strip()


st.set_page_config(
    page_title='AI Job Recommender',
    page_icon='💼',
    layout='wide',
    initial_sidebar_state='expanded'
)


@st.cache_resource
def load_model():
    model_path = PROJECT_DIR / 'models' / 'job_recommender.pkl'
    model_obj = joblib.load(model_path)
    if isinstance(model_obj, dict):
        recommender = JobRecommender()
        recommender.jobs_df = JobRecommender.normalize_jobs_schema(
            model_obj.get('jobs_df')
        )
        recommender.tfidf_vectorizer = model_obj.get('tfidf_vectorizer')
        recommender.tfidf_matrix = model_obj.get('tfidf_matrix')
        return recommender
    model_obj.jobs_df = JobRecommender.normalize_jobs_schema(model_obj.jobs_df)
    if not hasattr(model_obj, '_query_cache'):
        model_obj._query_cache = {}
    return model_obj


def render_favorites():
    favorites = read_activity('favorites.csv')
    if favorites.empty:
        st.caption('No saved jobs yet. Use ☆ Save on a recommendation to keep one here.')
        return
    st.dataframe(
        favorites.drop_duplicates('job_id', keep='last').drop(
            columns=['job_id'], errors='ignore'
        ),
        width='stretch',
        hide_index=True
    )


def main():
    st.title('🤖 AI Job Recommendation Engine')
    st.caption('Personalized job discovery with transparent match scores and actionable insights.')

    try:
        recommender = load_model()
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        st.error(
            f'Unable to load the recommendation model ({error}). '
            'Run `python Train_model.py` from the project folder first.'
        )
        st.stop()

    st.sidebar.header('👤 Your Profile')
    profile_name = st.sidebar.selectbox(
        'Load a sample resume',
        ['Choose a demo profile...'] + list(SAMPLE_PROFILES),
        key='sample_profile'
    )
    if st.sidebar.button('✨ Load sample profile'):
        if profile_name in SAMPLE_PROFILES:
            st.session_state['skills_input'] = SAMPLE_PROFILES[profile_name]['skills']
            st.session_state['sample_resume'] = SAMPLE_PROFILES[profile_name]['resume']
            st.session_state['resume_text'] = st.session_state['sample_resume']
            st.rerun()
        else:
            st.sidebar.info('Choose a sample profile first.')

    resume_file = st.sidebar.file_uploader(
        'Upload your resume (TXT, PDF, DOCX)', type=['txt', 'pdf', 'docx']
    )
    resume_text = st.session_state.get('sample_resume', '')
    if resume_file is not None:
        try:
            with st.spinner('Parsing resume...'):
                resume_text = extract_text_from_resume(resume_file)
            st.session_state['resume_text'] = resume_text
            st.session_state.pop('sample_resume', None)
            st.sidebar.success('Resume parsed successfully.')
        except (RuntimeError, OSError, ValueError) as error:
            st.sidebar.error(f'Could not parse resume: {error}')

    if resume_text:
        with st.sidebar.expander('Resume preview', expanded=False):
            st.write(resume_text[:2000] + ('...' if len(resume_text) > 2000 else ''))
        if st.sidebar.button('📝 Extract skills from resume'):
            extracted_skills = extract_skills_from_resume(resume_text)
            if extracted_skills:
                st.session_state['skills_input'] = extracted_skills
                st.sidebar.success('Skills extracted from resume.')
            else:
                st.sidebar.warning('No recognizable skills were found.')

    skills_input = st.sidebar.text_area(
        'Your skills (comma-separated)',
        placeholder='python, machine learning, sql, pandas, react',
        height=100,
        key='skills_input'
    )
    experience_level = st.sidebar.selectbox(
        'Experience level', ['Junior', 'Mid', 'Senior']
    )
    first, second = st.sidebar.columns(2)
    with first:
        num_recommendations = st.slider('Recommendations', 3, 10, 5)
    with second:
        min_similarity = st.slider('Minimum similarity %', 20, 100, 30)
    high_match_only = st.sidebar.toggle(
        'Only show high match (≥70%)', value=False
    )
    locations = sorted(
        recommender.jobs_df['location'].astype(str).replace('', 'Not listed').unique()
    )
    selected_locations = st.sidebar.multiselect(
        'Preferred locations (leave empty for all)',
        locations,
        default=[]
    )
    work_mode_filter = st.sidebar.selectbox(
        'Work arrangement', ['Any', 'Remote', 'Hybrid', 'On-site']
    )
    salary_values = [
        amount for amount in recommender.jobs_df['salary'].map(salary_amount)
        if amount is not None
    ]
    if salary_values:
        salary_floor = min(salary_values)
        salary_ceiling = max(salary_values)
        if salary_floor == salary_ceiling:
            salary_ceiling += 1
        salary_range = st.sidebar.slider(
            'Annual salary range (USD)',
            min_value=salary_floor,
            max_value=salary_ceiling,
            value=(salary_floor, salary_ceiling),
            step=max(1, min(5000, (salary_ceiling - salary_floor) // 100))
        )
        st.sidebar.caption(
            'Unlisted or non-USD salary listings remain included.'
        )
    else:
        salary_range = None

    st.sidebar.divider()
    st.sidebar.subheader('How it works')
    st.sidebar.caption(
        'Sentence embeddings compare meaning across job titles, skills and descriptions. '
        'The overall score combines semantic similarity (40%), required-skill coverage '
        '(40%), and experience fit (20%). Previous feedback can adjust ranking by up to '
        'five points. TF-IDF is used only if embedding dependencies are unavailable.'
    )
    if st.sidebar.button('🔄 Refresh remote jobs'):
        try:
            with st.spinner('Fetching jobs from Remotive...'):
                added = refresh_remote_jobs(recommender)
            st.sidebar.success(f'Refresh complete: {added} listings received.')
            st.rerun()
        except (
            HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError, RuntimeError
        ) as error:
            st.sidebar.error(f'Could not refresh jobs: {error}')

    st.sidebar.subheader('⭐ Saved jobs')
    with st.sidebar.expander('View favorites'):
        render_favorites()
    with st.sidebar.expander('Recent searches'):
        history = read_activity('user_history.csv')
        if history.empty:
            st.caption('Your searches will appear here.')
        else:
            st.dataframe(
                history.tail(10).iloc[::-1],
                width='stretch',
                hide_index=True
            )

    if st.sidebar.button('🔍 Find jobs', type='primary'):
        if not skills_input.strip():
            st.warning('Add your skills or load a sample profile to find matching jobs.')
        else:
            progress = st.progress(0, text='Preparing your profile...')
            try:
                progress.progress(35, text='Matching skills against job requirements...')
                results = recommender.get_recommendations(
                    skills_input, 'All', top_n=len(recommender.jobs_df)
                )
                progress.progress(75, text='Scoring experience fit and sorting results...')
                results['skills_match'] = results['skills'].apply(
                    lambda job_skills: recommender.skill_match_score(
                        skills_input, job_skills
                    )
                )
                results['experience_match'] = results['experience'].apply(
                    lambda job_experience: recommender.experience_match_score(
                        experience_level, job_experience
                    )
                )
                results['overall_match'] = results.apply(
                    lambda row: recommender.overall_match_score(
                        row['similarity'],
                        row['skills_match'],
                        row['experience_match']
                    ),
                    axis=1
                )
                feedback = read_activity('feedback.csv')
                results['feedback_adjustment'] = results.apply(
                    lambda row: feedback_adjustment(job_id(row), feedback),
                    axis=1
                )
                results['overall_match'] = (
                    results['overall_match'] + results['feedback_adjustment']
                ).clip(0, 100).round().astype(int)
                if selected_locations:
                    results = results[
                        results['location'].astype(str).isin(selected_locations)
                    ]
                if work_mode_filter != 'Any':
                    results = results[
                        results['work_mode'].astype(str).str.lower()
                        == work_mode_filter.lower()
                    ]
                if salary_range:
                    low_salary, high_salary = salary_range
                    amounts = results['salary'].map(salary_amount)
                    results = results[
                        amounts.isna() |
                        amounts.between(low_salary, high_salary, inclusive='both')
                    ]
                results = results[results['similarity'] >= min_similarity / 100]
                if high_match_only:
                    results = results[results['overall_match'] >= 70]
                results = results.sort_values(
                    ['overall_match', 'similarity'], ascending=False
                ).head(num_recommendations)
                st.session_state['recommendations'] = results.to_dict('records')
                st.session_state['search_executed'] = True
                st.session_state['search_skills'] = skills_input
                st.session_state['search_experience'] = experience_level
                append_activity('user_history.csv', {
                    'searched_at': pd.Timestamp.now().isoformat(timespec='seconds'),
                    'skills': skills_input,
                    'experience': experience_level,
                    'results': len(results)
                })
                progress.progress(100, text='Recommendations ready.')
            except (ValueError, KeyError, RuntimeError, OSError) as error:
                st.error(f'Could not calculate recommendations: {error}')
            finally:
                progress.empty()

    recommendations = st.session_state.get('recommendations', [])
    if not recommendations:
        if st.session_state.get('search_executed'):
            st.warning(
                'No jobs met your current filters. Lower the minimum similarity or turn '
                'off “Only show high match (≥70%)”, then search again.'
            )
        else:
            st.info(
                'Start with a sample resume or enter your skills, then select **Find jobs**. '
                'Try lowering the similarity threshold if your results are too narrow.'
            )
        return

    st.header('💼 Your job recommendations')
    st.caption(
        'Overall = semantic similarity 40% + skill coverage 40% + experience fit 20%, '
        'then adjusted by prior job feedback (maximum ±5 points).'
    )
    report_pdf = build_report_pdf(
        recommendations,
        st.session_state.get('search_skills', skills_input),
        st.session_state.get('search_experience', experience_level)
    )
    st.download_button(
        '📄 Download recommendation report (PDF)',
        data=report_pdf,
        file_name='job-recommendations.pdf',
        mime='application/pdf'
    )
    favorites_df = read_activity('favorites.csv')
    favorite_ids = set(favorites_df.get('job_id', pd.Series(dtype=str)).astype(str))
    query_skills = st.session_state.get('search_skills', skills_input)

    for index, row in enumerate(recommendations):
        identifier = job_id(row)
        with st.container(border=True):
            title_col, score_col = st.columns([3, 1])
            with title_col:
                st.subheader(row.get('title', 'Role'))
                st.markdown(
                    f"**{row.get('company', 'Company not listed')}** · "
                    f"{row.get('location', 'Location not listed')} · "
                    f"{row.get('experience', 'Experience not listed')}"
                )
                details = [
                    row.get('salary', 'Not listed'),
                    row.get('job_type', 'Not specified'),
                    row.get('work_mode', 'Not specified'),
                    f"Posted {row.get('posted_date')}" if row.get('posted_date') else ''
                ]
                st.caption(' · '.join(str(item) for item in details if item))
            with score_col:
                st.metric('Overall match', f"{row.get('overall_match', 0)}%")

            score1, score2, score3 = st.columns(3)
            score1.metric('Skills match', f"{row.get('skills_match', 0)}%")
            score2.metric('Experience match', f"{row.get('experience_match', 0)}%")
            score3.metric('Similarity', f"{float(row.get('similarity', 0)):.1%}")
            st.caption(
                f"Score breakdown: Similarity {float(row.get('similarity', 0)):.0%} · "
                f"Skill coverage {row.get('skills_match', 0)}% · "
                f"Experience fit {row.get('experience_match', 0)}% · "
                f"Feedback {float(row.get('feedback_adjustment', 0)):+.1f} pts"
            )

            required_skills = [
                skill.strip() for skill in str(row.get('skills', '')).split(',')
                if skill.strip()
            ]
            candidate_skills = recommender.preprocess_skills(query_skills)
            st.markdown('**Required skills**')
            skill_columns = st.columns(4)
            for skill_index, skill in enumerate(required_skills):
                matched = skill.lower() in candidate_skills
                color = 'green' if matched else 'orange'
                symbol = '✓' if matched else '×'
                with skill_columns[skill_index % len(skill_columns)]:
                    st.markdown(f":{color}[{symbol} {escape(skill)}]")

            actions = st.columns([1, 1, 1, 3])
            with actions[0]:
                saved = identifier in favorite_ids
                if st.button(
                    '★ Saved' if saved else '☆ Save',
                    key=f'favorite-{identifier}'
                ):
                    favorites = read_activity('favorites.csv')
                    if saved and not favorites.empty:
                        favorites = favorites[favorites['job_id'] != identifier]
                        favorites.to_csv(DATA_DIR / 'favorites.csv', index=False)
                    else:
                        saved_job = {
                            'job_id': identifier,
                            'title': row.get('title'),
                            'company': row.get('company'),
                            'location': row.get('location'),
                            'salary': row.get('salary'),
                            'apply_url': row.get('apply_url')
                        }
                        append_activity('favorites.csv', saved_job)
                    st.rerun()
            with actions[1]:
                if st.button('✉️ Cover letter', key=f'cover-{identifier}'):
                    try:
                        with st.spinner('Generating a tailored cover letter...'):
                            letter = generate_cover_letter(
                                row, query_skills,
                                st.session_state.get('resume_text', resume_text)
                            )
                        st.session_state[f'letter-{identifier}'] = letter
                    except RuntimeError as error:
                        st.error(f'Cover letter generation failed: {error}')
            with actions[2]:
                apply_url = row.get('apply_url')
                if not apply_url:
                    apply_url = (
                        'https://www.google.com/search?q=' +
                        quote(f"{row.get('title', '')} {row.get('company', '')} jobs")
                    )
                st.link_button('Apply', apply_url)
            with actions[3]:
                feedback_col1, feedback_col2 = st.columns(2)
                if feedback_col1.button('👍 Useful', key=f'yes-{identifier}'):
                    append_activity('feedback.csv', {
                        'submitted_at': pd.Timestamp.now().isoformat(timespec='seconds'),
                        'job_id': identifier,
                        'title': row.get('title'),
                        'useful': True
                    })
                    st.toast('Thanks for the feedback.')
                if feedback_col2.button('👎 Not useful', key=f'no-{identifier}'):
                    append_activity('feedback.csv', {
                        'submitted_at': pd.Timestamp.now().isoformat(timespec='seconds'),
                        'job_id': identifier,
                        'title': row.get('title'),
                        'useful': False
                    })
                    st.toast('Thanks for the feedback.')

            if st.session_state.get(f'letter-{identifier}'):
                st.text_area(
                    'Generated cover letter',
                    value=st.session_state[f'letter-{identifier}'],
                    height=220,
                    key=f'letter-text-{identifier}'
                )

    if recommendations:
        results_df = pd.DataFrame(recommendations)
        st.header('📊 Match analytics')
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            chart = px.bar(
                results_df,
                x='title',
                y='overall_match',
                color='overall_match',
                color_continuous_scale='viridis',
                title='Overall match score by role'
            )
            chart.update_layout(xaxis_tickangle=35, height=380)
            st.plotly_chart(chart, width='stretch')
        with chart_col2:
            all_skills = ' '.join(results_df['skills'].fillna('').astype(str))
            if all_skills.strip():
                wordcloud = WordCloud(
                    width=500, height=320, background_color='white'
                ).generate(all_skills)
                st.image(wordcloud.to_array(), caption='Skills across recommendations')

        user_skill_set = recommender.preprocess_skills(query_skills)
        all_job_skills = set()
        for skills in results_df['skills'].fillna('').astype(str):
            all_job_skills.update(recommender.preprocess_skills(skills))
        missing = sorted(all_job_skills - user_skill_set)
        if missing:
            st.subheader('🔍 Skills gap')
            st.write('Consider building experience with: ' + ', '.join(missing[:8]))


if __name__ == '__main__':
    main()
