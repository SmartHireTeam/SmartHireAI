from pydparser import ResumeParser
import os
import pandas as pd

def extract_resume_data(resume_path):
    data = ResumeParser(resume_path).get_extracted_data()
    return data

if __name__ == "__main__":
    data_folder = './Data/clean_resumes/'
    resume_data = []

    for resume_file in os.listdir(data_folder):
        if resume_file.endswith('.pdf') or resume_file.endswith('.docx'):
            resume_path = os.path.join(data_folder, resume_file)
            extracted_data = extract_resume_data(resume_path)
            # print(extracted_data)
            
            # Optionally, customize the data structure as needed
            resume_details = {
                'ID': resume_file.replace('.pdf', '').replace('.docx', ''),
                # Extract other fields as needed
                'Skills': extracted_data.get('skills', []),
                'Education': extracted_data.get('education', []),
                'Experience': extracted_data.get('experience', []),
                'Occupation': extracted_data.get('occupation', 'N/A'),
                'Total Experience': extracted_data.get('total_experience', 'N/A'),
                'College Name': extracted_data.get('college_name', 'N/A'),
                'Degree': extracted_data.get('degree', 'N/A'),
                'Designation': extracted_data.get('designation', 'N/A'),
                # Add more fields as per requirement
            }
            
            resume_data.append(resume_details)

    # Convert the list of dictionaries to a DataFrame
    resume_df = pd.DataFrame(resume_data)
    # Save the DataFrame to a CSV file
    resume_df.to_csv('./extracted_resume_data.csv', index=False)
    print('Resume data extraction complete.')