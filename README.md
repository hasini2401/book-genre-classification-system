# Book Genre Classification System Using Machine Learning and Natural Language Processing

A complete, production-ready, diploma-level academic project in Computer Science & Information Technology built with **Python**, **Django**, and **Scikit-Learn**.

---

## 1. Project Overview & Problem Statement

In contemporary digital libraries, online book retailers, and academic archives, thousands of new titles are catalogued each year. Traditional manual classification is slow, subjective, and labor-intensive. 

The **Book Genre Classification System** solves this problem by using Natural Language Processing (NLP) and supervised Machine Learning. Given a book's **title** and **synopsis/description**, the system:
1. Cleans and normalizes the input text.
2. Extracts numerical n-gram features via **TF-IDF Vectorization**.
3. Classifies the text across **10 distinct literary genres** using an empirically evaluated Scikit-learn classification pipeline.
4. Returns the predicted genre along with a calibrated probability distribution.
5. Securely stores predictions and catalogue entries in an SQLite database via Django ORM.

---

## 2. Supported Genres

The machine learning classifier recognizes 10 primary genres:
- **Fiction**
- **Mystery**
- **Romance**
- **Science Fiction**
- **Fantasy**
- **Horror**
- **Thriller**
- **Adventure**
- **Historical Fiction**
- **Biography**

---

## 3. Technology Stack

### Backend
- **Python 3.14+**
- **Django 6.1+** (ORM, Authentication, Sessions, CSRF Protection, Admin)
- **SQLite 3** (Default relational embedded database)

### Machine Learning & NLP
- **Scikit-Learn** (`TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`, `SGDClassifier`, `Pipeline`)
- **Pandas** & **NumPy** (Dataset processing, data validation, array operations)
- **Joblib** (Model serialization and pipeline persistence)

### Frontend
- **HTML5 & CSS3**
- **Bootstrap 5.3** (Responsive design, forms, cards, and modal dialogs)
- **Bootstrap Icons 1.11**
- **Chart.js 4.4** (Interactive dashboard distribution charts)
- **Vanilla JavaScript** (Real-time character counters, preset demonstrations, form UX)

---

## 4. Project Directory Structure

```text
BookGenreClassification/
├── manage.py                          # Django management CLI
├── requirements.txt                   # Pinned project dependencies
├── README.md                          # Main project guide
├── .gitignore                         # Git exclusion rules
├── db.sqlite3                         # Local SQLite database
│
├── bookgenreproject/                  # Django project root configuration
│   ├── __init__.py
│   ├── settings.py                    # Project settings & ML paths
│   ├── urls.py                        # Root URL router
│   ├── asgi.py
│   └── wsgi.py
│
├── classifier/                        # Main web & ML application
│   ├── __init__.py
│   ├── admin.py                       # Django Admin models configuration
│   ├── apps.py                        # Application registry
│   ├── models.py                      # Book & ClassificationHistory models
│   ├── views.py                       # HTTP views & controller logic
│   ├── urls.py                        # Application URL patterns
│   ├── forms.py                       # ModelForms & classification form
│   ├── tests.py
│   ├── ml/                            # Machine Learning & NLP package
│   │   ├── __init__.py
│   │   ├── compat.py                  # Windows Smart App Control compatibility
│   │   ├── dataloader.py              # CSV loading, validation & cleanup
│   │   ├── preprocessing.py           # Text cleaning & feature combinations
│   │   ├── trainmodel.py              # Model training, split & selection
│   │   ├── evaluatemodel.py           # Accuracy, F1, & confusion matrix
│   │   └── predict.py                 # Pipeline loading & inference engine
│   ├── services/                      # Business logic layer
│   │   ├── __init__.py
│   │   ├── classificationservice.py   # Prediction & history orchestration
│   │   ├── bookservice.py             # Book catalogue queries & filters
│   │   └── historyservice.py          # User-scoped history management
│   ├── management/commands/           # Custom manage.py commands
│   │   ├── traingenremodel.py         # python manage.py traingenremodel
│   │   ├── train_genre_model.py       # Alias command
│   │   └── loadbooks.py               # Seed database from books.csv
│   └── templatetags/                  # Custom Django template tags
│       └── classifiertags.py          # Genre badge & confidence formatting
│
├── data/
│   └── books.csv                      # Labeled dataset (200 records, 10 genres)
│
├── artifacts/
│   ├── genreclassifier.joblib         # Serialized Scikit-learn Pipeline
│   └── modelmetrics.json              # Empirical evaluation metrics & matrix
│
├── templates/
│   ├── base.html                      # Layout, navbar, and footer
│   ├── registration/
│   │   ├── login.html                 # User sign-in
│   │   └── register.html              # User registration
│   └── classifier/
│       ├── home.html                  # Landing page & overview
│       ├── classify.html              # Genre prediction form + presets
│       ├── result.html                # Prediction outcome & probabilities
│       ├── booklist.html              # Searchable catalogue + filters
│       ├── bookdetail.html            # Individual book detail view
│       ├── addbook.html               # Add book form
│       ├── editbook.html              # Edit book form
│       ├── confirmdelete.html         # Deletion confirmation
│       ├── history.html               # User-scoped prediction logs
│       ├── dashboard.html             # System statistics & Chart.js graph
│       ├── modelmetrics.html          # Measured evaluation metrics
│       └── about.html                 # Academic project overview
│
├── static/
│   ├── css/style.css                  # Custom responsive styles & badges
│   ├── js/main.js                     # UX interactions & presets
│   └── images/logo.svg                # Vector logo
│
├── tests/                             # Automated test suite (28 tests)
│   ├── testmodels.py                  # Model unit tests
│   ├── testforms.py                   # Form validation unit tests
│   ├── testviews.py                   # HTTP views & auth tests
│   └── testml.py                      # NLP & ML training tests
│
└── docs/                              # Project documentation
    ├── projectoverview.md             # Theoretical project background
    ├── modules.md                     # Major module documentation
    └── installation_guide.md          # Step-by-step setup guide
```

---

## 5. Machine Learning Workflow

```
[Raw Title + Synopsis]
         │
         ▼
[Text Preprocessing] (Remove HTML, normalize whitespace, lowercase)
         │
         ▼
[TF-IDF Feature Extraction] (Sublinear TF, ngram_range=(1,2), English stop-words)
         │
         ▼
[Candidate Model Comparison]
  - Multinomial Naive Bayes (alpha=0.1)
  - Logistic Regression (max_iter=1000)
  - Linear Support Vector Machine (SGDClassifier modified_huber)
         │
         ▼
[Stratified 80/20 Held-Out Evaluation]
  - Accuracy, Macro Precision, Macro Recall, Macro F1-Score
  - Confusion Matrix
         │
         ▼
[Best Pipeline Selection (Highest Macro F1)]
         │
         ▼
[Joblib Persistence] (artifacts/genreclassifier.joblib & artifacts/modelmetrics.json)
```

---

## 6. Installation & Execution Guide

### Step 1: Clone or Navigate to Project
```powershell
cd "c:\Users\lakka\OneDrive\Desktop\New folder"
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run Database Migrations
```powershell
python manage.py makemigrations
python manage.py migrate
```

### Step 4: Seed Initial Book Catalogue (Optional but Recommended)
```powershell
python manage.py loadbooks
```

### Step 5: Train the Machine Learning Model
```powershell
python manage.py traingenremodel
```
*(Or use the alias `python manage.py train_genre_model`)*

### Step 6: Create an Administrator Account (Optional)
```powershell
python manage.py createsuperuser
```

### Step 7: Start Development Server
```powershell
python manage.py runserver
```

Open your web browser and visit: **`http://127.0.0.1:8000/`**

---

## 7. Running the Test Suite

Execute all 28 automated test cases:
```powershell
python manage.py test
```

Expected output:
```text
Creating test database for alias 'default'...
............................
----------------------------------------------------------------------
Ran 28 tests in 14.382s

OK
Destroying test database for alias 'default'...
```

---

## 8. Key URL Routes

| Route | View Name | Description |
|---|---|---|
| `/` | `classifier:home` | Landing page with hero, statistics & genre cards |
| `/classify/` | `classifier:classify` | Submission form to predict book genre |
| `/result/` | `classifier:result` | Predicted genre, probability distribution, & matches |
| `/books/` | `classifier:book_list` | Book catalogue with search, genre filter & pagination |
| `/books/add/` | `classifier:book_add` | Add a new book to the catalogue |
| `/books/<id>/` | `classifier:book_detail`| View book details and related recommendations |
| `/books/<id>/edit/`| `classifier:book_edit` | Edit an existing book |
| `/books/<id>/delete/`| `classifier:book_delete`| Confirm deletion of a book |
| `/history/` | `classifier:history` | View classification history (privacy scoped) |
| `/dashboard/` | `classifier:dashboard` | Aggregated statistics & Chart.js graph |
| `/model-metrics/` | `classifier:model_metrics`| Empirical metrics, comparison & confusion matrix |
| `/about/` | `classifier:about` | Academic documentation & workflow breakdown |
| `/accounts/login/` | `classifier:login` | User authentication sign-in |
| `/accounts/register/` | `classifier:register` | New user registration |
| `/admin/` | `admin:index` | Django administrative interface |

---

## 9. Common Errors & Troubleshooting

1. **"Model Not Trained" Notice on Classification Page**:
   - Cause: The joblib artifact hasn't been compiled yet.
   - Fix: Run `python manage.py traingenremodel` in your terminal.

2. **Windows Smart App Control / Code Integrity Policy**:
   - Cause: Windows 11 blocks unsigned `.pyd` dynamic libraries built for bleeding-edge Python versions (e.g. Python 3.14).
   - Solution: The included `classifier/ml/compat.py` automatically injects pure-NumPy/SciPy compatible implementations, allowing Scikit-learn pipelines to execute cleanly without errors.

3. **Short Description Validation Warning**:
   - Cause: Descriptions under 15 characters don't contain enough tokens for reliable TF-IDF feature extraction.
   - Fix: Provide at least 15-20 words describing the plot or use one of the one-click demo presets on the classify page.

---

## 10. Limitations & Future Enhancements

- **Current Limitations**:
  - Training dataset currently contains 200 labeled books across 10 categories.
  - Multi-label classification (books belonging simultaneously to two genres, e.g., Sci-Fi Romance) is not yet supported in this single-label pipeline.
- **Future Enhancements**:
  - Incorporate multi-label classification using Binary Relevance or Classifier Chains.
  - Integrate book cover image recognition using Convolutional Neural Networks (CNNs).
  - Add REST API endpoints via Django REST Framework (DRF) for mobile integration.
