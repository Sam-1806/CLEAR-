# CLEAR
Contract Language Evaluation and Risk Analysis

Introduction
CLEAR is a powerful tool designed to analyze business contracts by providing comprehensive summaries, classifying content within contract clauses, and detecting deviations from predefined templates. This application leverages state-of-the-art Natural Language Processing (NLP) techniques to offer detailed insights into the contents of uploaded PDF contracts.

Features
PDF Text Extraction: Extracts text from uploaded PDF files.
Text Summarization: Generates concise summaries of the contract.
Content Classification: Classifies contract clauses based on predefined categories.
Deviation Detection: Identifies deviations from provided template clauses using semantic similarity.
Named Entity Recognition (NER): Extracts key entities from the contract text.
Interactive User Interface: Provides an intuitive interface for users to upload files, ask questions, and view results.
Installation
Follow these steps to set up the project on your local machine:

Clone the Repository

sh
Copy code
git clone https://github.com/your-username/chat-with-pdf-gemini.git
cd chat-with-pdf-gemini
Create a Virtual Environment

sh
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install Dependencies

sh
Copy code
pip install -r requirements.txt
Set Up Environment Variables
Create a .env file in the root directory and add your Google API key:

env
Copy code
GOOGLE_API_KEY=your_google_api_key
Run the Application

sh
Copy code
streamlit run app.py
Usage
Upload PDF Files: Use the sidebar to upload one or more PDF files containing business contracts.
Ask Questions: Enter questions about the uploaded contracts in the main input field.
View Results: The application will display the summary, classified clauses, deviations from templates, and extracted entities.
Technical Details
PDF Text Extraction
The get_pdf_text function extracts text from PDF files using the PyPDF2 library.

Text Summarization
The application uses the facebook/bart-large-cnn model from Hugging Face's transformers library to summarize the contract text.

Content Classification
The typeform/distilbert-base-uncased-mnli model is used for zero-shot classification of contract clauses.

Deviation Detection
Semantic similarity is calculated using the sentence-transformers library with the paraphrase-MiniLM-L6-v2 model.

Named Entity Recognition (NER)
The transformers library's NER pipeline is used to extract entities from the contract text.

Vector Store
The FAISS library is used to create and manage a vector store for efficient similarity searches.


OCR Integration: Integrate OCR to handle scanned PDFs.
Custom Model Training: Train custom models for more accurate clause classification and deviation detection.
Enhanced UI: Improve the user interface for better user experience.
Conclusion
CLEAR is a robust tool for business contract analysis, providing valuable insights through advanced NLP techniques. This documentation serves as a comprehensive guide to setting up, using, and understanding the technical implementation of the project.


