import streamlit as st
from transformers import pipeline

@st.cache(allow_output_mutation=True)
def load_qa_model():
    model = pipeline("question-answering")
    return model

@st.cache(allow_output_mutation=True)
def load_summarizer_model(): 
    summarizer = pipeline('summarization')
    return summarizer

qa = load_qa_model()
summarizer = load_summarizer_model()


st.title("Get a summary and ask questions about your text.")
maxSummar = st.sidebar.slider('Choose max summary length', 90, 600, step=5, value = 150)
minSummar = st.sidebar.slider('Choose minimum summary length', 10, 250, step=5, value = 60)
do_sample = st.sidebar.checkbox("Do sample", value=False)
sentence = st.text_area('Enter your text', height=30)
summaryButton = st.button('Get summary for your text.')

with st.spinner('Getting summary for your text...'):
    if summaryButton and sentence: 
        summary = summary = summarizer(sentence, max_length=maxSummar, min_length=minSummar, do_sample=do_sample)
        st.write(summary[0]['summary_text'])
question = st.text_input("Enter questions about your text. ")
button = st.button("Get answers")

with st.spinner("Getting your answers..."):
    if button and sentence:
        answers = qa(question=question, context=sentence, do_sample=do_sample)
        st.write(answers['answer'])

