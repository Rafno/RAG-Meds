import fitz  # PyMuPDF
import re
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def preprocess_text(text, stopwords_set):
    text = re.sub(r'\s+', ' ', text)
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords_set]
    return ' '.join(tokens)

icelandic_stopwords = {
    "að", "af", "al", "allur", "allt", "annars", "á", "bæði", "betri", "betur", "bíða", 
    "eða", "ef", "ein", "einhver", "einnig", "einn", "einstakur", "ekki", "el", "en", 
    "enda", "er", "ert", "erum", "eru", "eð", "eða", "eigum", "ekki", "enn", "einnig", 
    "einn", "eins", "en", "einhver", "fram", "frá", "fyrir", "gæti", "gott", "hafði", 
    "hafi", "hafa", "hefur", "hefði", "henni", "hennar", "hér", "hún", "hvort", "hver", 
    "hverju", "hverjum", "hvers", "hvað", "hve", "hvenær", "hvernig", "hvor", "hvorki", 
    "hvorri", "hvort", "í", "inn", "inni", "innan", "með", "mig", "mín", "mínar", "mína", 
    "mér", "mér", "mitt", "mína", "mínar", "niður", "og", "sem", "sig", "sína", "sínu", 
    "svo", "það", "þaðan", "þaðar", "þeir", "þeirra", "þeim", "þér", "þess", "þessi", 
    "þessum", "þetta", "þó", "því", "þín", "þínar", "þína", "þínu", "þinn", "þitt", 
    "um", "undir", "upp", "úr", "verið", "við", "voru", "víst", "yfir"
}

def create_qa_pipeline():
    qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
    return qa_pipeline

def answer_question(qa_pipeline, context, question):
    response = qa_pipeline({
        'context': context,
        'question': question
    })
    return response['answer']

def retrieve_relevant_chunks(all_texts, question, top_n=5):
    vectorizer = TfidfVectorizer().fit_transform(all_texts + [question])
    vectors = vectorizer.toarray()
    question_vector = vectors[-1]
    similarities = vectors[:-1].dot(question_vector)
    relevant_indices = similarities.argsort()[-top_n:][::-1]
    return ' '.join([all_texts[i] for i in relevant_indices])

def ask_question(question):
    all_contexts = []
    pdf_directory = 'pdfs/'
    # Step 1: Extract and preprocess text from all PDFs
    for pdf_filename in os.listdir(pdf_directory):
        if pdf_filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, pdf_filename)
            text = extract_text_from_pdf(pdf_path)
            processed_text = preprocess_text(text, icelandic_stopwords)
            all_contexts.append(processed_text)
    
    # Step 2: Retrieve relevant chunks from all contexts
    relevant_context = retrieve_relevant_chunks(all_contexts, question)
    
    # Step 3: Create QA pipeline
    qa_pipeline = create_qa_pipeline()
    
    # Step 4: Answer question
    answer = answer_question(qa_pipeline, relevant_context, question)
    return answer