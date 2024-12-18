import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from sklearn.feature_extraction.text import TfidfVectorizer  # Gunakan TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from google_play_scraper import reviews, Sort
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk
import io

# Download data nltk
nltk.download('punkt')
nltk.download('stopwords')

# Memuat model dan vektorisator
model_nb = joblib.load('naive_bayes_model.pkl')  # Model Naive Bayes
model_rf = joblib.load('random_forest_model.pkl')  # Model Random Forest
model_lr = joblib.load('logistic_regression_model.pkl')  # Model Logistic Regression
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Memuat vektorisator yang sesuai


def predict_sentiment(text, model):
    # Mengubah teks input menjadi fitur numerik menggunakan TF-IDF
    text_tfidf = vectorizer.transform([text])
    # Prediksi sentimen menggunakan model yang dipilih
    sentiment = model.predict(text_tfidf)[0]
    return sentiment

# Fungsi pembersihan teks
def clean_text(text):
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Hapus karakter non-huruf
    text = text.lower()  # Ubah ke huruf kecil
    return text

# Fungsi EDA
def perform_eda(data):
    st.subheader('1. Informasi Data')
    st.write("### **Head Data**")
    st.write(data.head())
    st.write(f"**Jumlah Data:** {data.shape[0]}")
    st.write(f"**Jumlah Kolom:** {data.shape[1]}")

    # Menampilkan info data
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.subheader('2. Distribusi Skor Ulasan')
    fig, ax = plt.subplots()
    sns.countplot(x='score', data=data, palette='viridis', ax=ax)
    ax.set_title("Distribusi Skor Ulasan")
    st.pyplot(fig)

    st.subheader('3. Distribusi Panjang Ulasan')
    data['content_length'] = data['content'].apply(len)
    fig, ax = plt.subplots()
    sns.histplot(data['content_length'], bins=30, kde=True, color='blue', ax=ax)
    ax.set_title("Distribusi Panjang Ulasan")
    st.pyplot(fig)

    st.subheader('4. WordCloud Kata Dominan')
    all_words = ' '.join([clean_text(text) for text in data['content']])
    wordcloud = WordCloud(
        stopwords=set(stopwords.words('indonesian')),
        background_color='white',
        width=800,
        height=400
    ).generate(all_words)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("WordCloud Kata Dominan")
    st.pyplot(fig)

# Inisialisasi Stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi Pembersihan Data
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|\S+@\S+', '', text)  # Hapus URL dan email
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Hapus karakter selain huruf
    text = text.lower()  # Ubah ke huruf kecil
    return text

# Fungsi Tokenisasi
def tokenize_text(text):
    return word_tokenize(text)

# Fungsi untuk Hapus Stopwords
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('indonesian'))
    return [word for word in tokens if word.lower() not in stop_words]

# Fungsi untuk Stemming
def stem_tokens(tokens):
    return [stemmer.stem(word) for word in tokens]

# Fungsi untuk Memberikan Label Sentimen
def label_based_on_score(score):
    if score >= 4:
        return 1  # Puas
    elif score <= 3:
        return 0  # Tidak Puas

# Fungsi Preprocessing
def preprocess_data(data):
    if 'content' in data.columns and 'score' in data.columns:
        comments_cleaned = data[['content', 'score']].copy()
    else:
        st.error("File CSV harus memiliki kolom 'content' dan 'score'.")
        return None

    comments_cleaned = comments_cleaned.dropna(subset=['content', 'score'])

    comments_cleaned['content'] = comments_cleaned['content'].apply(clean_text)
    comments_cleaned['tokens'] = comments_cleaned['content'].apply(tokenize_text)
    comments_cleaned['tokens_no_stopwords'] = comments_cleaned['tokens'].apply(remove_stopwords)
    comments_cleaned['stemmed_tokens'] = comments_cleaned['tokens_no_stopwords'].apply(stem_tokens)
    comments_cleaned['sentiment_label'] = comments_cleaned['score'].apply(label_based_on_score)

    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(comments_cleaned['content'])  # Fitur (X)
    y = comments_cleaned['sentiment_label']  # Label (y)

    tfidf_matrix = X.toarray()
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_vectorizer.get_feature_names_out())

    return comments_cleaned, tfidf_df

# Fungsi untuk Split Data, Latih Model, dan Evaluasi
def train_and_evaluate_model(data):
    X = data['content']
    y = data['sentiment_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    model_nb = MultinomialNB()
    model_nb.fit(X_train_tfidf, y_train)
    y_pred_nb = model_nb.predict(X_test_tfidf)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    report_nb = classification_report(y_test, y_pred_nb)
    joblib.dump(model_nb, 'naive_bayes_model.pkl')

    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(X_train_tfidf, y_train)
    y_pred_lr = model_lr.predict(X_test_tfidf)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    report_lr = classification_report(y_test, y_pred_lr)
    joblib.dump(model_lr, 'logistic_regression_model.pkl')

    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train_tfidf, y_train)
    y_pred_rf = model_rf.predict(X_test_tfidf)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf)
    joblib.dump(model_rf, 'random_forest_model.pkl')

    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

    return {
        'Naive Bayes': {'accuracy': acc_nb, 'report': report_nb},
        'Logistic Regression': {'accuracy': acc_lr, 'report': report_lr},
        'Random Forest': {'accuracy': acc_rf, 'report': report_rf}
    }

# Streamlit App Layout
st.title('Sentiment Analysis & EDA')
st.write("Masukkan ulasan untuk menganalisis sentimen atau unggah file CSV untuk melakukan EDA.")

# Tab Navigasi
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 EDA", "📈 Prediksi Sentimen", "Scapping", "Preprocessing Data", "Training"])

# Tab 1: EDA
with tab1:
    st.header('📊 Eksplorasi Data Ulasan (EDA)')
    uploaded_file = st.file_uploader("Unggah file CSV untuk EDA", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if 'content' in data.columns and 'score' in data.columns:
            perform_eda(data[['content', 'score']])
        else:
            st.error("File CSV harus memiliki kolom 'content' dan 'score'.")
    else:
        st.write("Unggah file CSV untuk memulai EDA.")

# Tab 2: Prediksi Sentimen
with tab2:
    st.header('📈 Prediksi Sentimen')
    # Pilihan model di Streamlit
    model_choice = st.selectbox(
        'Pilih Model untuk Prediksi Sentimen:',
        ['Naive Bayes', 'Random Forest', 'Logistic Regression']
    )

    # Menentukan model yang dipilih
    if model_choice == 'Naive Bayes':
        selected_model = model_nb
    elif model_choice == 'Random Forest':
        selected_model = model_rf
    else:
        selected_model = model_lr
        
    user_input = st.text_area("Masukkan teks ulasan")

    # Jika pengguna mengklik tombol, tampilkan prediksi
    if st.button('Prediksi Sentimen'):
        if user_input:
            sentiment = predict_sentiment(user_input, selected_model)
            if sentiment == 1:
                st.success("Sentimen: **Puas** 😄")
            else:
                st.error("Sentimen: **Tidak Puas** 😞")
        else:
            st.warning("Tolong masukkan teks untuk analisis.")

# Tab 3: Scraping
with tab3:
    st.header("Scraping Ulasan Google Play Store")
    app_id = st.text_input("Masukkan ID Aplikasi Google Play", 'ctrip.english')
    jumlah_ulasan = st.number_input("Jumlah ulasan yang ingin diambil", min_value=10, max_value=1000, value=300, step=10)

    if st.button('Scrape Data Ulasan'):
        if app_id:
            with st.spinner('Mengambil ulasan dari Google Play Store...'):
                try:
                    review, _ = reviews(
                        app_id=app_id,
                        lang='id',
                        country='id',
                        count=int(jumlah_ulasan),
                        sort=Sort.MOST_RELEVANT
                    )
                    if isinstance(review, str):
                        st.error(f"Terjadi kesalahan saat scraping: {review}")
                    else:
                        data = pd.DataFrame(review)
                        st.success(f"Berhasil mengambil ulasan dari Google Play Store!")
                        st.write("📋 **Tampilan Data**")
                        st.dataframe(data.head(10))

                        st.download_button(
                            label="📁 Unduh Data Ulasan Sebagai CSV",
                            data=data.to_csv(index=False),
                            file_name='ulasan_google_play.csv',
                            mime='text/csv',
                        )

                        st.session_state['scraped_data'] = data

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat scraping: {e}")
        else:
            st.error("Masukkan ID aplikasi Google Play.")

# Tab 4: Preprocessing Data
with tab4:
    st.header("Preprocessing Data")
    uploaded_file = st.file_uploader("Unggah file CSV untuk Preprocessing", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("📋 **Preview Data Sebelum Preprocessing**")
        st.dataframe(data.head(10))

        if st.button('Mulai Preprocessing'):
            with st.spinner('Proses preprocessing sedang berjalan...'):
                preprocessed_data, tfidf_df = preprocess_data(data)
                if preprocessed_data is not None:
                    st.success('Preprocessing selesai!')
                    st.write("📋 **Preview Data Setelah Preprocessing**")
                    st.dataframe(preprocessed_data.head(10))

                    csv_file = preprocessed_data.to_csv(index=False)
                    st.download_button(
                        label="📁 Unduh Data Preprocessed Sebagai CSV",
                        data=csv_file,
                        file_name='preprocessed_comments.csv',
                        mime='text/csv',
                    )

                    st.write("📋 **Matriks TF-IDF (10 Baris Pertama)**")
                    st.dataframe(tfidf_df.head(10))

                    csv_tfidf_file = tfidf_df.to_csv(index=False)
                    st.download_button(
                        label="📁 Unduh Matriks TF-IDF Sebagai CSV",
                        data=csv_tfidf_file,
                        file_name='tfidf_matrix.csv',
                        mime='text/csv',
                    )
    else:
        st.info("Silakan unggah file CSV dengan kolom 'content' dan 'score'.")

    st.header('📈 Hasil Preprocessing')

    if 'preprocessed_data' in st.session_state:
        preprocessed_data = st.session_state['preprocessed_data']
        st.write("📋 **Data Setelah Preprocessing**")
        st.dataframe(preprocessed_data.head(10))

        st.download_button(
            label="📁 Unduh Data Preprocessed Sebagai CSV",
            data=preprocessed_data.to_csv(index=False),
            file_name='preprocessed_comments.csv',
            mime='text/csv',
        )
    else:
        st.info("Silakan jalankan preprocessing di tab sebelumnya.")

# Tab 5: Training Model
with tab5:
    st.header("Training Model")
    uploaded_file = st.file_uploader("Unggah file CSV hasil preprocessing", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head())
        if st.button('Latih dan Uji Model'):
            results = train_and_evaluate_model(data)
            for model_name, result in results.items():
                st.subheader(f'{model_name}')
                st.write(f'**Akurasi:** {result["accuracy"]:.2f}')
                st.text(result['report'])
