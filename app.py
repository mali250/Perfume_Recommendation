import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer

# Load perfume data
perfumes_data = pd.read_csv('data/perfumes.csv')

# Stemming function
ps = PorterStemmer()
def stems(text):
    return " ".join([ps.stem(word) for word in text.split()])

# Apply stemming to the 'Brand' column
perfumes_data['Brand'] = perfumes_data['Brand'].apply(stems)

# Vectorize the 'Brand' column
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(perfumes_data['Brand']).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(vector)

# Function to recommend perfumes
def recommend_perfumes(brand):
    # Get the index of the selected brand
    index = perfumes_data[perfumes_data['Brand'] == brand].index[0]
    
    # Sort distances based on similarity scores
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    
    # Get top recommendations and sort them based on rating
    top_recommendations = [perfumes_data.iloc[i[0]] for i in distances[1:11]]
    
    # Filter out duplicates based on perfume names
    unique_recommendations = []
    perfume_names = set()
    for perfume in top_recommendations:
        if perfume['Brand'] not in perfume_names:
            unique_recommendations.append(perfume)
            perfume_names.add(perfume['Brand'])
    
    # Sort unique recommendations based on rating
    sorted_recommendations = sorted(unique_recommendations, key=lambda x: x['Rating'], reverse=True)
    
    # Return the sorted recommendations
    return sorted_recommendations



# Streamlit app
st.title('Perfume Recommender System')
st.markdown(
    """
    <style>
        body {
            background-color: #F0F0F0;
        }
        .stButton>button {
            color: black;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

selected_brand = st.selectbox("Select a brand:", sorted(perfumes_data['Brand'].unique()))

# Show Recommendations button
if st.button('Show Recommendations'):
    st.subheader('Recommended Perfumes')
    recommended_perfumes = recommend_perfumes(selected_brand)
    
    # Display recommended perfumes in two rows with five columns each
    for i in range(0, len(recommended_perfumes), 5):
        col1, col2, col3, col4, col5 = st.columns(5)
        for j, perfume in enumerate(recommended_perfumes[i:i+5]):
            if j == 0:
                with col1:
                    st.image(perfume['Image Link'], caption=perfume['Brand'], use_column_width=True)
                    st.write(f"<p style='text-align: center; color: #808080;'>Rating: {perfume['Rating']}</p>", unsafe_allow_html=True)
                    st.write(f"<p style='text-align: center; color: #808080;'>Price: {perfume['Price']}</p>", unsafe_allow_html=True)
            elif j == 1:
                with col2:
                    st.image(perfume['Image Link'], caption=perfume['Brand'], use_column_width=True)
                    st.write(f"<p style='text-align: center; color: #808080;'>Rating: {perfume['Rating']}</p>", unsafe_allow_html=True)
                    st.write(f"<p style='text-align: center; color: #808080;'>Price: {perfume['Price']}</p>", unsafe_allow_html=True)
            elif j == 2:
                with col3:
                    st.image(perfume['Image Link'], caption=perfume['Brand'], use_column_width=True)
                    st.write(f"<p style='text-align: center; color: #808080;'>Rating: {perfume['Rating']}</p>", unsafe_allow_html=True)
                    st.write(f"<p style='text-align: center; color: #808080;'>Price: {perfume['Price']}</p>", unsafe_allow_html=True)
            elif j == 3:
                with col4:
                    st.image(perfume['Image Link'], caption=perfume['Brand'], use_column_width=True)
                    st.write(f"<p style='text-align: center; color: #808080;'>Rating: {perfume['Rating']}</p>", unsafe_allow_html=True)
                    st.write(f"<p style='text-align: center; color: #808080;'>Price: {perfume['Price']}</p>", unsafe_allow_html=True)
            elif j == 4:
                with col5:
                    st.image(perfume['Image Link'], caption=perfume['Brand'], use_column_width=True)
                    st.write(f"<p style='text-align: center; color: #808080;'>Rating: {perfume['Rating']}</p>", unsafe_allow_html=True)
                    st.write(f"<p style='text-align: center; color: #808080;'>Price: {perfume['Price']}</p>", unsafe_allow_html=True)
