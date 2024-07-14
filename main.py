import asyncio
import streamlit as st
import pickle
from PyPDF2 import PdfReader
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util

load_dotenv()

# Load the Google API key
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    st.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")

# Load FAISS index
faiss_index_path = 'C:/Users/V-Code/Desktop/Intel Unnati/faiss_index/index.pkl'
try:
    with open(faiss_index_path, 'rb') as file:
        data = pickle.load(file)
except FileNotFoundError:
    st.error(f"FAISS index file not found at {faiss_index_path}")

# Load pre-trained models for summarization and classification
summarization_model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Named Entity Recognition
ner_model = pipeline("ner", grouped_entities=True)

# Semantic Similarity
semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Dummy classification and deviation detection (Replace with actual model if available)
classification_model_name = "typeform/distilbert-base-uncased-mnli"
classification_tokenizer = AutoTokenizer.from_pretrained(classification_model_name)
classification_model = AutoModelForSeq2SeqLM.from_pretrained(classification_model_name)
classifier = pipeline("zero-shot-classification", model=classification_model, tokenizer=classification_tokenizer)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def summarize_text(text):
    summaries = summarizer(text, max_length=200, min_length=50, do_sample=False)
    summary = " ".join([s['summary_text'] for s in summaries])
    return summary

def classify_and_detect_deviations(text, template_clauses):
    results = classifier(text, candidate_labels=list(template_clauses.keys()))
    classifications = {label: score for label, score in zip(results['labels'], results['scores'])}
    deviations = {label: "No Deviation" if classifications[label] > 0.5 else "Deviation Detected" for label in template_clauses.keys()}
    return classifications, deviations

def extract_entities(text):
    entities = ner_model(text)
    return entities

def calculate_similarity(text1, text2):
    embeddings1 = semantic_model.encode(text1, convert_to_tensor=True)
    embeddings2 = semantic_model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
    return similarity.item()

def detect_deviations_from_template(text, template_clauses):
    deviations = {}
    for clause, expected_text in template_clauses.items():
        similarity = calculate_similarity(text, expected_text)
        deviations[clause] = "No Deviation" if similarity > 0.8 else f"Deviation Detected (Similarity: {similarity:.2f})"
    return deviations

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    except FileNotFoundError:
        st.error("FAISS index file not found. Please upload and process PDF files first.")
        return

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        template_file = st.file_uploader("Upload your template clauses file (txt)", type=["txt"])

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete!")
            else:
                st.warning("Please upload PDF files first.")

        if st.button("Summarize Contract"):
            if pdf_docs:
                with st.spinner("Summarizing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    summary = summarize_text(raw_text)
                    st.write("Summary: ", summary)
            else:
                st.warning("Please upload PDF files first.")

        if st.button("Classify and Detect Deviations"):
            if pdf_docs and template_file:
                with st.spinner("Classifying and Detecting Deviations..."):
                    raw_text = get_pdf_text(pdf_docs)
                    template_clauses = {}
                    for line in template_file:
                        clause, text = line.split(":")
                        template_clauses[clause.strip()] = text.strip()
                    classified_clauses, deviations = classify_and_detect_deviations(raw_text, template_clauses)
                    st.write("Classified Clauses: ", classified_clauses)
                    st.write("Deviations: ", deviations)

                    st.write("Extracted Entities: ", extract_entities(raw_text))
                    st.write("Semantic Similarities: ", detect_deviations_from_template(raw_text, template_clauses))
            else:
                st.warning("Please upload PDF and template files first.")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    main()
