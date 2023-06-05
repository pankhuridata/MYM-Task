#Author: Pankhuri
#Description: Data Cleansing

import pandas as pd                                                    #import libraries
import datetime
import re

df = pd.read_csv('Uncleaned_DS_jobs.csv')                              #read csv file

df.drop('index', axis=1, inplace=True)
df.info()

columns_to_drop = ['Headquarters', 'Competitors']
df = df.drop(columns=columns_to_drop)                                 #dropping unneccessary columns

missing_values = df.isnull().sum()                                   #dropping the missing values rows

df.drop_duplicates(inplace=True)                                            #dropping the duplicates

df = df[(df != -1).all(axis=1)]                                        #dropping rows with -1 has entry

current_year = datetime.datetime.now().year
df['Company_Age'] = current_year - df['Founded']                         #calculation of company's age

df['Company Name'] = df['Company Name'].apply(lambda x:x.split("\n")[0])

df['Salary Estimate'] = df['Salary Estimate'].str.rstrip('(Glassdoor est.)')   #convert salary estimate to a consistent format
df['Salary Estimate'] = df['Salary Estimate'].str.rstrip('(Employer est.)')
df['Salary Min'] = df['Salary Estimate'].apply(lambda x: x.split('-')[0].replace('$', '').replace('K', '') if 'K' in x else None)
df['Salary Max'] = df['Salary Estimate'].apply(lambda x: x.split('-')[1].replace('$', '').replace('K', '') if 'K' in x else None)
df['Avg Salary'] = (df['Salary Min'].astype(float) + df['Salary Max'].astype(float)) / 2       #calculate average salary estimate

df['Job_States'] = df['Location'].str.split(', ').str[-1]                #extracting the states from the 'Location' column
df['Job_States'] = df['Job_States'].str.strip()
df['Job_States'].replace({'New Jersey': 'NJ', 'Utah': 'UT', 'United States' : 'USA',
                          'Texas': 'TX', 'California': 'CA'}, inplace=True)
avg_rating = df.groupby('Job_States')['Rating'].mean()            #average rating by job states

skills_list = ['Python', 'Excel', 'Hadoop', 'Spark', 'AWS', 'Tableau', 'Big Data', 'Power Bi'] #list of key skills

#function to extract skills from job description and count their occurrences
def count_skills(description):
    skills_count = {skill: 0 for skill in skills_list}  # Initialize count for each skill to 0
    for skill in skills_list:
        if re.search(skill, description, re.IGNORECASE):
            skills_count[skill] += 1
    return skills_count

df['Skills Count'] = df['Job Description'].apply(count_skills)
for skill in skills_list:
    df[skill.capitalize() + ' Count'] = df['Skills Count'].apply(lambda x: x[skill])
df = df.drop('Skills Count', axis=1)

company_stats = df.groupby('Company Name').agg({'Company_Age': 'mean', 'Job Title': 'count', 'Size': 'first'})
print(company_stats)

df.info()

df.to_csv('Cleaned_Jobs.csv', index=False)
