# Importing Libraries
import pandas as pd
import faiss
import numpy as np
import streamlit as st
from boltiotai import openai # here i am using boltiotai api key
#import openai #if using openai api key

# Replace with your BoltIoT API key
API_KEY = "" ##replace key
openai.api_key = API_KEY

# Load Dataset
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    return df

# Build FAISS index
def build_faiss_index(data):
    dimension = data.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(data, dtype='float32'))
    return index

# Search in FAISS
def search_faiss_index(query_vector, index, df, top_k=3):
    query_vector = np.array([query_vector], dtype='float32')
    distances, indices = index.search(query_vector, top_k)
    return df.iloc[indices[0]]

# Fetch medical documents
def fetch_medical_documents(matching_conditions):
    documents = []
    for disease in matching_conditions['Disease'].tolist():
        documents.append(f"Relevant medical information about {disease}")
    return documents

# Generate response using RAG
def generate_rag_response(user_input, matching_conditions, medical_docs):
    diseases = matching_conditions['Disease'].tolist()
    context = f"User symptoms: {user_input}. Possible diseases: {', '.join(diseases)}. Relevant medical information: {medical_docs}"
    prompt = f"Based on the given symptoms, provide a medical explanation and suggest potential diseases.\n\n{context}\n\nDisclaimer: This tool is for informational purposes only. Consult a medical professional for accurate diagnosis."
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a medical assistant providing symptom analysis."},
                  {"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']
    

# Main function
def main():
    st.title("Medical Symptom Checker with FAISS and RAG")
    st.write("Disclaimer: This is not a substitute for professional medical advice.")
    
    csv_path = "Disease_symptom_and_patient_profile_dataset.csv"
    df = load_data(csv_path)
    # Ensure only numerical columns are used
    numeric_df = df.select_dtypes(include=[np.number])  
    if numeric_df.empty:
        raise ValueError("No numerical columns found for FAISS indexing.")

    embeddings = numeric_df.to_numpy().astype('float32')  # Convert to float32 for FAISS

    
    # Assuming dataset has precomputed embeddings
    #embeddings = np.array(df.iloc[:, 1:].values, dtype='float32')
    faiss_index = build_faiss_index(embeddings)
    
    user_input = st.text_input("Enter your symptoms (comma-separated):")
    
    if st.button("Get Response") and user_input:
        with st.spinner("Processing your query..."):
            user_vector = np.random.rand(embeddings.shape[1])  # Placeholder for user query embedding
            matched_conditions = search_faiss_index(user_vector, faiss_index, df)
            
            if not matched_conditions.empty:
                medical_docs = fetch_medical_documents(matched_conditions)
                response = generate_rag_response(user_input, matched_conditions, medical_docs)
                st.write("Possible conditions:", matched_conditions[['Disease']])
                st.write("Healthcare Assistant Response:", response)
                st.write("Please consult a medical professional for accurate diagnosis.")
            else:
                st.write("No matching conditions found. Consult a medical professional.")
            
if __name__ == "__main__":
    main()

# run code as streamlit run .\symp.py on terminal