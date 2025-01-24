# Import the required packages

import streamlit as st
import pandas as pd
import altair as alt

# Page configuration
st.set_page_config(
    page_title="Iris Classification", 
    page_icon="assets/icon/icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

# -------------------------
# Sidebar navigation
with st.sidebar:
    st.title('Iris Classification')
    st.subheader("Pages")

    # Page navigation buttons
    pages = {
        "About": "about",
        "Dataset": "dataset",
        "EDA": "eda",
        "Data Cleaning / Pre-processing": "data_cleaning",
        "Machine Learning": "machine_learning",
        "Prediction": "prediction",
        "Conclusion": "conclusion",
    }

    # Save the selected page in the session state
    selected_page = st.radio("Navigate to:", list(pages.keys()))
    st.session_state.page_selection = pages[selected_page]

    # Abstract and project details
    st.subheader("Abstract")
    st.markdown("A Streamlit dashboard highlighting the results of training two classification models using the Iris flower dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)")
    st.markdown("📗 [Google Colab Notebook](https://colab.research.google.com/drive/1KJDBrx3akSPUW42Kbeepj64ZisHFD-NV?usp=sharing)")
    st.markdown("🐙 [GitHub Repository](https://github.com/Zeraphim/Streamlit-Iris-Classification-Dashboard)")
    st.markdown("by: [`Zeraphim`](https://jcdiamante.com)")

# -------------------------
# Page content logic
def load_data():
    """Function to load the Iris dataset."""
    return pd.read_csv('iris.csv', delimiter=',')

def render_about(): 
    st.title('ISJM BI - Exploration des données des Iris') 
    st.header('Pré-analyse visuelles données données des Iris TP1')
    st.subheader('Description des données') 
    st.write("Cette application explore les données des Iris, met en œuvre des modèles d'apprentissage automatique et visualise les résultats.")
    st.write("Elle inclut une analyse exploratoire, un pré-traitement des données, et des prédictions basées sur des modèles de classification.")
    st.markdown("**Construit avec :** Streamlit, Pandas, Altair")
    st.markdown("**Auteur :** Stéphane C. K. Tékouabou")

def render_dataset(df):
    st.title("Dataset Overview")
    st.write(df.head())
    st.write("Shape of the dataset:", df.shape)

def render_eda(df):
    st.title("Exploratory Data Analysis (EDA)")
    chart = alt.Chart(df).mark_point().encode(
        x='petal_length',
        y='petal_width',
        color='species'
    )
    st.altair_chart(chart, use_container_width=True)

# Mapping page names to their respective functions
page_functions = {
    "about": render_about,
    "dataset": lambda: render_dataset(load_data()),
    "eda": lambda: render_eda(load_data()),
    # Add other pages here...
}

# Display the content for the selected page
page = st.session_state.page_selection
if page in page_functions:
    page_functions[page]()
else:
    st.write("Page not found.")
