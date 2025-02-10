# Importing Libraries
import pandas as pd
import streamlit as st
#import openai #if using openai api key
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from boltiotai import openai # here i am using boltiotai api key

client = openai.api_key="" #Replace your boltiot api key here

# Load dataset
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)  # Clean data by removing null values
    return df


# Preprocess symptoms (TF-IDF for similarity matching)
def preprocess_data(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    symptom_columns = df.columns[1:-1]  # Exclude 'Disease' and 'Outcome Variable'
    df['Combined_Symptoms'] = df[symptom_columns].astype(str).agg(' '.join, axis=1)
    tfidf_matrix = vectorizer.fit_transform(df['Combined_Symptoms'])
    return vectorizer, tfidf_matrix

# Finding symptoms from dataset
def find_conditions(symptoms, df):
    """Match user input symptoms to possible conditions."""
    matched_conditions = set()
    for symptom in symptoms:
        matches = df[df['Disease'].str.contains(symptom, case=False, na=False)]
        matched_conditions.update(matches['Disease'].tolist())
    return list(matched_conditions)

# Find closest matching symptoms
def match_symptoms(user_input, vectorizer, tfidf_matrix, df):
    user_tfidf = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    top_matches = similarities.argsort()[-3:][::-1]  # Top 3 matches
    return df.iloc[top_matches]



# Generate response using RAG approach
def generate_rag_response(user_input, matching_conditions):
    diseases = matching_conditions['Disease'].tolist()
    context = f"User symptoms: {user_input}. Possible diseases: {', '.join(diseases)}."
    prompt = f"Based on the given symptoms, provide a medical explanation and suggest potential diseases.\n\n{context}\n\nDisclaimer: This tool is for informational purposes only. Consult a medical professional for accurate diagnosis."

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a medical assistant providing symptom analysis."},
                  {"role": "user", "content": prompt}]
    )

    return response['choices'][0]['message']['content']


# Main function
def main():
    """Main function to run the symptom checker."""
    st.title("Medical Symptom Checker ")
    st.write("Disclaimer: This is not a substitute for professional medical advice.")

    csv_path = "Disease_symptom_and_patient_profile_dataset.csv"
    df = load_data(csv_path)
    vectorizer, tfidf_matrix = preprocess_data(df)

    user_input = st.text_input("Enter your symptoms (separated by commas): ")
    symptoms = [s.strip() for s in user_input.split(',')]

    if st.button("Get Response"):
        if user_input:
            with st.spinner("Processing your query, please wait..."):
                matching_conditions = match_symptoms(user_input, vectorizer, tfidf_matrix, df)
                if not matching_conditions.empty:
                    st.write("Possible conditions based on your symptoms:")
                    response = generate_rag_response(user_input, matching_conditions)
                    #st.write(response)

                else:
                    st.write("No matching conditions found. Please consult a medical professional.")
                conditions = find_conditions(symptoms, df)
            st.write("Healthcare Assistant RAG:", response)
            st.write("Healthcare Assistant Dataset:", conditions)
            st.write("Please consult Doctor for accurate advice and diagnosis")
        else:
            st.warning("Please enter a message to get a response.")
    st.markdown("---")





if __name__ == "__main__":
    main()
 
 # run code as streamlit run .\app.py on terminal