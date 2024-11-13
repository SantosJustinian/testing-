import streamlit as st
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Streamlit UI
st.title("NLP Model with spaCy on Streamlit")
st.write("This app uses spaCy to perform NLP tasks on user input.")

# Text input for the user
text = st.text_area("Enter text to analyze", "Type something here...")

# Button to analyze text
if st.button("Analyze"):
    # Process text with spaCy
    doc = nlp(text)
    
    # Display entities
    st.subheader("Named Entities")
    for ent in doc.ents:
        st.write(f"{ent.text} ({ent.label_})")
    
    # Display part-of-speech tagging
    st.subheader("Token Analysis")
    for token in doc:
        st.write(f"{token.text}: {token.pos_}, {token.dep_}")

st.write("Use the text area to enter your text and click 'Analyze' to see NLP results.")
