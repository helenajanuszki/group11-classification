import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from google_play_scraper import reviews


#part 1
df = pd.read_csv("tse_dataset.csv")

df = df[df["category"].isin(["FEATURE", "BUG"])]

X = df["body"]
y = df["category"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC()
svm.fit(X_train, y_train)
svm_predict = svm.predict(X_test)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_predict = knn.predict(X_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_predict = rf.predict(X_test)

model_dict = {
    "Support Vector Machine": svm_predict,
    "K Nearest Neighbors": knn_predict,
    "Random Forest": rf_predict
}

for model, prediction in model_dict.items():
    print({model})

    print("\nConfusion Matrix")
    cm = confusion_matrix(y_test, prediction)
    print(pd.DataFrame(cm, index=["Actual Bug","Actual Feature"], columns=["Predicted as a Bug","Predicted as a Feature"]))

    print("\nResults")
    print(classification_report(y_test, prediction))

#part 2

zoom, cont_token = reviews(
    'us.zoom.videomeetings',
    lang='en',
    country='us',
)
zoom_df = pd.DataFrame(zoom)
zoom_df = zoom_df[["userName", "content", "score", "at"]]

# Filter by latest 2 months
zoom_df['at'] = pd.to_datetime(zoom_df['at'])
df_reviews = zoom_df[zoom_df['at'].dt.month >= 2]

X_zoom = zoom_df["content"]
X_zoom = vectorizer.transform(X_zoom)

zoom_predictions = knn.predict(X_zoom)
zoom_df["category"] = zoom_predictions
zoom_df.to_csv("classified_reviews.csv")