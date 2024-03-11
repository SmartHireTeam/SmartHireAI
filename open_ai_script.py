import pandas as pd
import re
import string # for text cleaning
import contractions # for expanding short form words
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI

client = OpenAI(api_key='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')
from sklearn.metrics.pairwise import cosine_similarity


tqdm.pandas(desc="Progress Bar")

# Ensure to load your OpenAI API key from a secure location or environment variable

extracted_skills = pd.read_csv('./extracted_resume_data.csv')
# print(extracted_skills.head())

jd_data = pd.read_csv('./Data/JD.csv')
# print(jd_data)

jd_df_full = pd.DataFrame(jd_data)
sample_size = 0.01
jd_df = jd_df_full.sample(frac=sample_size, random_state=1).reset_index(drop=True)
# print(jd_df['job_description'][0])

cv_df = extracted_skills[~(extracted_skills['Skills'].isna() 
                           & extracted_skills['Education'].isna()
                            & extracted_skills['Experience'].isna()
                            & extracted_skills['Degree'].isna()
                            & extracted_skills['Designation'].isna()
                        )].reset_index(drop=True)

# print(cv_df.shape)
print(cv_df.head())

attribute_weights = {
    'Skills': 1.0,
    'Education': 1.0,
    'Experience': 1.0,
    'Occupation': 1.0,
    'Total Experience': 1.0,
    'College Name': 0.3,
    'Degree': 0.8,
    'Designation': 0.3
}

# Method for basic text cleaning
def text_cleaning(text:str) -> str:
    if pd.isnull(text):
        return
    
    # lower-case everything, expand short-form words, and remove URLs, emails, phone numbers, punctuations, and non-alphanumeric characters
    text = text.lower().strip()
    translator = str.maketrans('', '', string.punctuation)
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d{1,3}[-./]?\d{1,3}[-./]?\d{1,4}\b', '', text)
    text = text.translate(translator)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    return text.strip()

cv_df = cv_df.fillna(value='').progress_apply(text_cleaning)
# cv_df['CV'] = cv_df['Skills'] + ' ' + cv_df['Education'] + cv_df['Experience'] + ' ' + cv_df['Degree'] + ' ' + cv_df['Designation']
# cv_df['CV'] = cv_df['CV'].progress_apply(text_cleaning)
cv_df.to_csv('./sanitized_resume_data.csv', index=False)

def create_resume_attribute_dict_for_embedding(row):
    return {
        'Skills': row['Skills'],
        'Education': row['Education'],
        'Experience': row['Experience'],
        'Degree': row['Degree'],
        'Designation': row['Designation']
    }

def create_jd_attribute_dict_for_embedding(row):
    return {
        'Description': row['job_description'],
        'Company':row['company_name'],
        'Position':row['position_title'],
    }

# Apply the function to each row in your DataFrame to create a list of dictionaries
resume_dicts = cv_df.apply(create_resume_attribute_dict_for_embedding, axis=1).tolist()
job_description_dicts = jd_df.apply(create_jd_attribute_dict_for_embedding, axis=1).tolist()


job_descriptions = jd_df['job_description'].apply(text_cleaning).to_list()

def embed_attributes_using_openai(attributes_dict):
    attribute_embeddings = {}
    for attribute, text in attributes_dict.items():
        # Ensure text is not empty
        if text.strip():
            response = client.embeddings.create(
                input=[text],
                model="text-embedding-3-small" 
            )
            attribute_embeddings[attribute] = response.data[0].embedding
        else:
            attribute_embeddings[attribute] = None  # Handle empty or invalid texts
    
    return attribute_embeddings

# Embed both job descriptions and resumes
job_description_embeddings = [{**embed_attributes_using_openai(jd_dict), 'Company': jd_dict['Company'], 'Position': jd_dict['Position']}
    for jd_dict in job_description_dicts]
resume_embeddings = [embed_attributes_using_openai(resume_dict) for resume_dict in resume_dicts]

# print(job_description_embeddings)
# print(resume_embeddings)
def calculate_similarity_with_jd(jd_embedding, resume_embeddings, attribute_weights):
    assert isinstance(jd_embedding, dict), "JD embedding must be a dictionary"
    similarity_scores = []
    
    for resume_emb_dict in resume_embeddings:
        total_similarity = 0
        total_weight = 0
        
        for attribute, weight in attribute_weights.items():
            if attribute in resume_emb_dict and resume_emb_dict[attribute] is not None and jd_embedding['Description'] is not None:
                jd_emb_np = np.array(jd_embedding['Description']).reshape(1, -1)
                attribute_emb_np = np.array(resume_emb_dict[attribute]).reshape(1, -1)
                
                if jd_emb_np.size == 0 or attribute_emb_np.size == 0:
                    print(f"Skipping due to empty embedding for {attribute}.")
                    continue
                
                # Ensure the embeddings are 2D and have a shape compatible with cosine_similarity
                assert jd_emb_np.ndim == 2 and attribute_emb_np.ndim == 2, "Embeddings must be 2-dimensional"
                
                similarity = cosine_similarity(jd_emb_np, attribute_emb_np)[0][0]
                total_similarity += (similarity * weight)
                total_weight += weight
        
        if total_weight > 0:
            normalized_similarity = total_similarity / total_weight
        else:
            normalized_similarity = 0
        
        similarity_scores.append(normalized_similarity)
    
    return similarity_scores

# Calculate cosine similarity between job descriptions and resumes
# similarity_scores = calculate_similarity_with_jd(job_description_embeddings, resume_embeddings, attribute_weights)

# Rank candidates for each job description based on similarity scores and print top candidates
num_top_candidates = 5
# for i, job_description in enumerate(job_descriptions):
#     candidates_with_scores = list(enumerate(similarity_scores[i]))
#     candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
#     top_candidates_for_job = candidates_with_scores[:num_top_candidates]
#     print(f"Top candidates for JD {i+1} - Position: {jd_df['position_title'][i]}")
#     for candidate_index, score in top_candidates_for_job:
#         print(f"  Candidate {candidate_index + 1} - Similarity Score: {score:.4f} - {cv_df['Category'][candidate_index]}/{cv_df['ID'][candidate_index]}.pdf")
#         print()

def display_top_candidates(jd_embedding, resume_embeddings, attribute_weights, top_n=5):
    # Calculate similarity scores
    similarity_scores = calculate_similarity_with_jd(jd_embedding, resume_embeddings, attribute_weights)
    
    # Get the top_n candidates
    top_candidates = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Print or return the top candidates
    for candidate in top_candidates:
        print(f"Candidate ID: {candidate[0]}, Score: {candidate[1]}")
    return top_candidates

# Iterate over each job description and display the top candidates
for jd_embedding in job_description_embeddings:
    print(f"Company: {jd_embedding['Company']}, Position: {jd_embedding['Position']}")
    top_candidates = display_top_candidates(jd_embedding, resume_embeddings, attribute_weights)
    print()
# display_top_candidates(job_descriptions, cv_df, similarity_scores, top_n=5)