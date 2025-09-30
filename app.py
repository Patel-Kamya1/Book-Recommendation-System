import streamlit as st
import pandas as pd
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os # Included for best practice, though st.secrets is preferred

# --- Global Configuration and Initialization ---

# 1. Streamlit Page Setup
st.set_page_config(page_title="Recommendation Dashboard", layout="wide")
st.title("Book Recommendation Dashboard")

# 2. FIXED Gemini Model Configuration
# Use a current, stable, and widely supported model alias
LLM_MODEL = 'gemini-2.5-flash' 
# This model is fast, cost-effective, and works with the v1beta API endpoint.

# 3. Gemini API Authentication
try:
    # Key is securely loaded from Streamlit Secrets (secrets.toml or Cloud settings)
    if "GOOGLE_API_KEY" not in st.secrets:
        st.error(" Gemini API Key not found. Please add GOOGLE_API_KEY to your Streamlit secrets.")
        st.stop()
    else:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}")
    st.stop()


# ---------------------------
# 1. Load Model & Data
# ---------------------------
@st.cache_data
def load_model_and_data():
    """Loads all data and model files from the repository."""
    try:
        # Load CSV
        books = pd.read_csv('books_metadata.csv')
        
        # Load pickle files
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        with open('tfidf_matrix.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
            
        return books, tfidf_vectorizer, tfidf_matrix
    except FileNotFoundError as e:
        st.error(f"Data Error: Required file not found: {e.filename}. Please check your GitHub repository.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models or data: {e}")
        st.stop()


books, tfidf_vectorizer, tfidf_matrix = load_model_and_data()


# ---------------------------
# 2. Recommendation Functions 
# ---------------------------
def get_top_popular_books(top_n=10):
    top_books = books.sort_values(['ratings_count', 'average_rating'], ascending=False)
    return top_books[['title', 'authors', 'average_rating', 'ratings_count', 'image_url']].head(top_n)

def recommend_books_by_title(book_title_query, top_n=5):
    pattern = re.compile(book_title_query, re.IGNORECASE)
    matched_books = books[books['title'].apply(lambda x: bool(pattern.search(x)))]
    
    if matched_books.empty: return pd.DataFrame()
        
    idx = matched_books.index[0]
    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_sim_indices = [i[0] for i in sim_scores[1:top_n+1]]
    
    return books[['title', 'authors', 'average_rating', 'ratings_count', 'image_url']].iloc[top_sim_indices]

def recommend_books_by_author(author_query, top_n=5):
    pattern = re.compile(author_query, re.IGNORECASE)
    matched_books = books[books['authors'].apply(lambda x: bool(pattern.search(x)))]
    
    if matched_books.empty: return pd.DataFrame()
            
    try:
        author_tfidf = tfidf_vectorizer.transform(matched_books['combined_text'])
    except KeyError:
        st.error("Error: 'combined_text' column is missing.")
        return pd.DataFrame()
    
    query_vec = tfidf_vectorizer.transform([author_query.lower()])
    sim_scores = cosine_similarity(query_vec, author_tfidf).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    
    return matched_books[['title', 'authors', 'average_rating', 'ratings_count', 'image_url']].iloc[top_indices]


# ---------------------------
# 3. LLM Functionalities (FIXED)
# ---------------------------

@st.cache_data(show_spinner=False)
def generate_book_summary(title, authors):
    """Generates a summary using the Gemini API."""
    try:
        # CORRECT: Model name is passed positionally
        model = genai.GenerativeModel(LLM_MODEL) 
        prompt = f"Provide a concise plot summary (around 4-5 sentences) for the book '{title}' by '{authors}'. Focus on the main plot points and themes, avoid major spoilers."
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Catch and report the specific error, but the 404 should now be fixed.
        return f"Could not generate summary at this time. (Error: {e})"


def run_chatbot():
    """Runs the interactive book-aware chatbot using the Gemini API."""
    st.header("💬 Ask the Book Bot!")
    st.write("Ask anything about famous books, characters, themes, or authors.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = (
            "You are a friendly, knowledgeable literary assistant specializing in books, authors, and characters. "
            "Answer user questions factually and concisely. "
            "If asked for a recommendation, politely suggest looking at the 'Book Recommendations' tab. "
            "Maintain a helpful, encouraging tone."
        )

    # FIXED: Initialize chat object 
    if "chat_model" not in st.session_state:
        try:
            # CORRECT: Model name (LLM_MODEL) is the first positional argument.
            model_instance = genai.GenerativeModel(
                LLM_MODEL, 
                system_instruction=st.session_state.system_prompt
            )
            st.session_state.chat_model = model_instance.start_chat()
        except Exception as e:
            st.error(f"Could not initialize chat model: {e}. Please check the API key, model name, and quotas.")
            return

    # Display previous messages
    for message in st.session_state.messages:
        role = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask about a book, character, or theme..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Book Bot is thinking..."):
                try:
                    response = st.session_state.chat_model.send_message(prompt)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    error_message = f"An error occurred during chat: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})


# ---------------------------
# 4. Display Books Grid 
# ---------------------------
def display_books_grid(books_to_display, cols_per_row=3):
    """Displays books in a customizable grid layout."""
    
    st.markdown(
        """
        <style>
        .book-card {
            background-color:#1e1e1e; padding:10px; border-radius:10px; text-align:center;
            color:#fff; margin-bottom:20px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            transition: 0.3s; height: 100%; display: flex; flex-direction: column;
            justify-content: space-between;
        }
        .book-card:hover { box-shadow: 0 8px 16px 0 rgba(0,0,0,0.5); }
        .book-title {font-weight:bold; font-size:16px; margin:5px 0 0 0; min-height: 40px;}
        .book-author {font-style:italic; font-size:14px; margin:0;}
        .book-rating {color:#ffdd00; margin-top:5px; margin-bottom: 10px;}
        </style>
        """, unsafe_allow_html=True
    )
    
    for i in range(0, len(books_to_display), cols_per_row):
        cols = st.columns(cols_per_row, gap="large")
        for j, col in enumerate(cols):
            if i + j < len(books_to_display):
                book = books_to_display.iloc[i + j]
                with col:
                    st.markdown(f'<div class="book-card">', unsafe_allow_html=True)
                    
                    st.image(book['image_url'], use_container_width=True)
                    
                    st.markdown(f'<div class="book-title">{book["title"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="book-author">{book["authors"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="book-rating">⭐ {book["average_rating"]:.2f} ({book["ratings_count"]:,.0f})</div>', unsafe_allow_html=True)
                    
                    if st.button(f"Generate Summary", key=f"summary_{i+j}"):
                        summary_text = generate_book_summary(book['title'], book['authors'])
                        
                        if "Could not generate summary" in summary_text:
                            st.warning(summary_text)
                        else:
                            st.expander(f"Summary for {book['title']} (AI Generated)").markdown(summary_text)
                        
                    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------
# 5. Streamlit Layout Tabs and Logic
# ---------------------------
tab1, tab2 = st.tabs(["Book Recommendations", "Book Chatbot"])

with tab1:
    # --- Sidebar for Recommendations ---
    st.sidebar.header("Find Books")
    recommendation_type = st.sidebar.selectbox(
        "Recommendation Type",
        ["Popular Books", "By Book Name", "By Author"],
        index=1 
    )
    
    user_input = ""
    if recommendation_type in ["By Author", "By Book Name"]:
        input_label = f"Enter {recommendation_type.split(' ')[-1]}"
        default_value = "the kite runner" if recommendation_type == "By Book Name" else ""
        user_input = st.sidebar.text_input(input_label, value=default_value)

    top_n = st.sidebar.slider("Number of Recommendations", 3, 12, 6)

    # --- Fetch & Display Logic ---
    books_to_display = pd.DataFrame()
    try:
        if recommendation_type == "Popular Books":
            st.subheader(f"Top {top_n} Most Popular Books")
            books_to_display = get_top_popular_books(top_n)
        elif recommendation_type == "By Author" and user_input:
            st.subheader(f"Recommendations for Author: {user_input}")
            books_to_display = recommend_books_by_author(user_input, top_n)
        elif recommendation_type == "By Book Name" and user_input:
            st.subheader(f"Books Similar to: {user_input}")
            books_to_display = recommend_books_by_title(user_input, top_n)

        if not books_to_display.empty:
            display_books_grid(books_to_display)
        elif user_input and recommendation_type != "Popular Books":
            st.warning(f"No books found for the query '{user_input}'. Try a different search term.")

    except Exception as e:
        st.error(f"An error occurred during recommendation processing: {e}")

with tab2:
    # --- Chatbot Interface ---
    run_chatbot()