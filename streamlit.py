# Import the required packages
import streamlit as st
import pandas as pd
import altair as alt

# Page configuration
st.set_page_config(
    page_title="Iris Classification",
    page_icon="assets/icon/icon.png",  # Assurez-vous que ce chemin est correct
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enable Altair dark theme with error handling
try:
    alt.themes.enable("dark")
except Exception as e:
    st.warning(f"Le thème 'dark' n'a pas pu être activé : {e}")

# Ensure session state is initialized
if "page_selection" not in st.session_state:
    st.session_state.page_selection = "about"

# -------------------------
# Sidebar navigation
with st.sidebar:
    st.title("Iris Classification")
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
    try:
        return pd.read_csv("iris.csv", delimiter=",")
    except FileNotFoundError:
        st.error("Le fichier 'iris.csv' est introuvable. Assurez-vous qu'il est dans le bon répertoire.")
        return pd.DataFrame()  # Return an empty DataFrame if the file is missing

def render_about():
    """Renders the About page."""
    st.title("ISJM BI - Exploration des données des Iris")
    st.subheader("Description des données")
    st.write("Cette application explore les données des Iris, met en œuvre des modèles d'apprentissage automatique et visualise les résultats.")
    st.write("Elle inclut une analyse exploratoire, un pré-traitement des données, et des prédictions basées sur des modèles de classification.")
    st.markdown("**Construit avec :** Streamlit, Pandas, Altair")
    st.markdown("**Auteur :** Stéphane C. K. Tékouabou")

def render_dataset(df):
    """Renders the Dataset page."""
    st.title("Dataset Overview")
    if df.empty:
        st.error("Aucune donnée à afficher. Veuillez vérifier le fichier iris.csv.")
    else:
        st.dataframe(df) 
        st.write("Shape of the dataset:", df.shape)

def render_eda(df):
    """Renders the EDA page."""
    st.title("Exploratory Data Analysis (EDA)")

    # Vérifier si les colonnes nécessaires sont présentes
    required_columns = {"petal_length", "petal_width", "sepal_length", "sepal_width", "species"}
    if not required_columns.issubset(df.columns):
        st.error(f"Le fichier de données doit contenir les colonnes suivantes : {required_columns}")
        return

    # Premier graphique : petal_length vs petal_width
    st.subheader("Relation entre la longueur et la largeur des pétales")
    chart1 = alt.Chart(df).mark_point().encode(
        x="petal_length",
        y="petal_width",
        color="species",
        tooltip=["petal_length", "petal_width", "species"]
    ).interactive()
    st.altair_chart(chart1, use_container_width=True)

    # Deuxième graphique : sepal_length vs sepal_width
    st.subheader("Relation entre la longueur et la largeur des sépales")
    chart2 = alt.Chart(df).mark_circle(size=60).encode(
        x="sepal_length",
        y="sepal_width",
        color="species",
        tooltip=["sepal_length", "sepal_width", "petal_length", "petal_width"]
    ).interactive()
    st.altair_chart(chart2, use_container_width=True)

    # Ajouter un tableau récapitulatif des statistiques descriptives
    st.subheader("Statistiques descriptives")
    st.write(df.describe())

    # Histogrammes des distributions avec Altair
    st.subheader("Histogrammes des caractéristiques")
    numerical_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for col in numerical_columns:
        st.write(f"Distribution de la caractéristique : **{col}**")
        hist_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(col, bin=alt.Bin(maxbins=30)),
            y="count()",
            color="species"
        )
        st.altair_chart(hist_chart, use_container_width=True)

    # Nouveau graphique : distribution de PetalWidth
    st.subheader("Distribution de la largeur des pétales (PetalWidth)")
    st.write("Statistiques descriptives pour **PetalWidth** :")
    st.write(df["petal_width"].describe())

    # Graphique seaborn pour la distribution de PetalWidth
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(15, 5))
    sns.countplot(x="petal_width", data=df)
    plt.title("Distribution de PetalWidth")
    
    # Afficher le graphique dans Streamlit
    st.pyplot(plt)
    plt.clf()  # Nettoyer pour éviter des conflits avec d'autres graphiques
    
def not_implemented():
    """Displays a placeholder for pages under development."""
    st.title("Page en cours de développement")
    st.write("Cette fonctionnalité sera bientôt disponible.")

# Mapping page names to their respective functions
page_functions = {
    "about": render_about,
    "dataset": lambda: render_dataset(load_data()),
    "eda": lambda: render_eda(load_data()),
    "data_cleaning": not_implemented,
    "machine_learning": not_implemented,
    "prediction": not_implemented,
    "conclusion": not_implemented,
}

# Display the content for the selected page
page = st.session_state.page_selection
if page in page_functions:
    page_functions[page]()
else:
    st.error("Page introuvable.") 
