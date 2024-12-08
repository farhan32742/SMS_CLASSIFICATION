
# **Spam Classification Using NLP**

This project involves building and deploying a spam classification model that uses **Natural Language Processing (NLP)** techniques and machine learning models to detect whether a message is spam or not.

### **Project Overview**
The model leverages both **Random Forest** and **LSTM** (Long Short-Term Memory) for spam classification:
- **Random Forest** is used as the primary model due to its high accuracy and interpretability.
- **LSTM** is included for practice, utilizing deep learning techniques for sequence processing.

The project is deployed as a **Streamlit web application**, allowing users to easily classify messages as spam or not in a user-friendly interface.

---

### **Tech Stack**
- **Python 3.x**
- **Streamlit** – for deploying the model as a web application
- **Scikit-learn** – for machine learning algorithms (Random Forest)
- **TensorFlow (Keras)** – for deep learning with LSTM
- **NLTK** – for natural language processing tasks
- **Pandas** – for data manipulation and analysis
- **NumPy** – for numerical computations

---

### **Installation & Setup**

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/farhan32742/spam_classification_using_NLP.git
   cd spam_classification_using_NLP
   ```

2. **Create a Virtual Environment:**
   It's recommended to use a virtual environment to manage dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use 'venv\Scripts\activate'
   ```

3. **Install the Required Dependencies:**
   You can install all the dependencies by running the following command:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App:**
   Once the dependencies are installed, you can start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

   This will launch the app in your browser where you can interact with the spam classification model.

---

### **Project Workflow**

1. **Data Preprocessing:**
   - Text cleaning (removal of stopwords, punctuation, etc.)
   - Tokenization and lemmatization using **NLTK** and **Gensim**
   - Feature extraction using **TF-IDF** vectorization

2. **Model Training:**
   - **Random Forest Classifier**: Used for training the model on processed features.
   - **LSTM**: Deep learning model used to experiment with sequence-based classification.

3. **Model Evaluation:**
   - The models are evaluated based on accuracy and other classification metrics such as precision, recall, and F1-score.

4. **Model Deployment:**
   - The final models are deployed using **Streamlit**, allowing users to input text and classify it as spam or not.

---

### **Contributing**
If you want to contribute to this project, feel free to fork the repository, submit issues, or open pull requests. Contributions are always welcome!

---


### **Contact**
- GitHub: [farhan32742](https://github.com/farhan32742)
- LinkedIn: [Farhan Fayaz][https://www.linkedin.com/in/farhan32742](https://www.linkedin.com/in/farhan-fayaz-1f?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)

