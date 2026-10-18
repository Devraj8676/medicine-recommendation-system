# Personalized Healthcare & Medicine Recommendation System

A machine-learning app that takes a set of symptoms, predicts the three most likely conditions, and shows the medicines that patients rated best for each condition.

> Educational project, not medical advice. Medicine rankings come from patient reviews, not clinical evidence. Consult a qualified doctor for health concerns.

## How it works

1. **Disease prediction.** The user selects symptoms (132 possible). A Random Forest classifier turns them into probabilities over 41 diseases and the app shows the top 3.
2. **Medicine ranking.** Each disease is linked to condition names in a drug-review dataset. Medicines are ranked by a score that blends each drug's average rating with the disease's overall average, so a drug with only a few reviews is not ranked above a well-reviewed one:

   `score = (reviews x drug_avg + m x disease_avg) / (reviews + m)`, with m = 10 and at least 3 reviews per drug.
3. **Streamlit app.** Shows the three likely diseases with probabilities and a medicine table for each, or "No review data available" when there is none.

## Data

| Dataset | Use | Size |
|---|---|---|
| Disease-symptom dataset (`training_data.csv`) | Train the disease model | 4,920 rows, 132 symptoms, 41 diseases |
| Drugs.com reviews (`drugsComTrain_raw.csv`, `drugsComTest_raw.csv`) | Rank medicines | 215,063 reviews; 212,698 after removing rows with a missing or corrupted condition |

## Results and limitations

**Disease model.**
- Of the 4,920 rows, only 304 are unique. A random split therefore scores 100% only because every test row also appears in training.
- Evaluated on unique rows only (so test patterns are unseen), accuracy is still 100%. The dataset appears to give each disease a fixed set of signature symptoms, so it is very easy. This number should not be read as real-world performance.
- A more realistic test keeps only a few of a test row's symptoms (the situation a real user is in). Top-3 accuracy on 61 test rows:

| Symptoms given | Top-1 | Top-3 |
|---|---|---|
| 1 | 0.39 | 0.51 |
| 2 | 0.56 | 0.74 |
| 3 | 0.64 | 0.85 |
| 4 | 0.73 | 0.91 |
| 5 | 0.86 | 0.96 |

This is why the app shows three possible diseases, not one, and warns when fewer than 3 symptoms are selected.

**Medicine ranking.**
- Only 21 of the 41 diseases have a medicine list. The other diseases have no matching condition in the review data (for example Dengue, Jaundice, Chicken pox, Fungal infection), or no drug with at least 3 reviews (Hypoglycemia, Varicose veins).
- Disease-to-condition matching was done by hand. "Arthritis" is mapped to "Rheumatoid Arthritis" as an assumption, because the symptom dataset does not say which type is meant. "Prevention" conditions (such as Migraine Prevention) are excluded.
- Brand and generic names of the same drug are not merged, so some medicines appear twice.
- Rankings reflect patient ratings, which can differ from clinical effectiveness.

## Project structure

```
app/app.py            Streamlit app
data/                 the four CSV files (download separately)
models/               disease_model.joblib, symptom_list.joblib, medicine_scores.csv
notebooks/            01_load_data.ipynb (cleaning, training, evaluation)
requirements.txt
```

## Run it

```
pip install -r requirements.txt
python -m streamlit run app/app.py
```

The saved files in `models/` are created by running the notebook.
