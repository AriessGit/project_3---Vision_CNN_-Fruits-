סיכום תהליך הפיתוח והתובנות
רקע ומטרה: 
הפרויקט החל במטרה לבנות מערכת בינה מלאכותית מבוססת רשת קונבולוציה (CNN) לסיווג אוטומטי של פירות וירקות מתוך מאגר נתונים מורכב, והנגשתו למשתמש קצה באמצעות אפליקציית אינטרנט ומובייל מודרנית ב-Streamlit.
מה לא עבד והבעיות אשר נתקלתי בהן:
1.	פיצול יתר של מחלקות (Over-segmentation): בתחילת הדרך, המודל אומן על 60 מחלקות שונות. התברר שמאגר הנתונים פוצל לתתי-קטגוריות קטנות מדי (לדוגמה: 20 סוגים שונים של תפוחים ותפוזים כגון Orange 1, Orange 2, Apple Red וכו'). הדבר גרם לבלבול קשה במודל ולדיוק נמוך מאוד במציאות.
2.	אובדן הגדרות קומפילציה בטעינה: בעת טעינת המודל באפליקציה, התקבלה אזהרת ארכיטקטורה שהמודל נטען ללא פונקציית הפסד ואופטימייזר, מה שגרם לשכבת ה-Softmax להחזיר ניחושים אקראיים לחלוטין (סביב 1.9% דיוק).
3.	עיוותי צבע ופורמט במובייל: בעת הרצת האפליקציה מהטלפון הנייד, התמונות מהמצלמה החזירו פורמטים זרים והבדלי ערוצי צבע (RGB לעומת BGR), מה שגרם לחיזויים שגויים בעולם האמיתי למרות דיוק גבוה כביכול.
4.	אובייקטים זרים:  המודל נטה לסווג חפצים אקראיים בחדר (כמו מקלדת או יד) כאילו היו פירות בביטחון מלא.
מה עבד ואיך שיפרתי:
1.	איחוד תיקיות אוטומטי (Data Restructuring): נכתב סקריפט פייתון שסרק את המאגר ואיחד את כל תתי-הקטגוריות ל-8 מחלקות אב ברורות ונקיות (Apple, Banana, Orange וכו'). צעד זה הפחית את הסיבוכיות.
2.	קומפילציה ידנית מחדש: תיקון שלב הטעינה באפליקציה וביצוע קומפילציה לשכבות הנוירונים.
3.	סטנדרטיזציה של התמונות: שימוש בפונקציה img_to_array של Keras אפשר לעבד בצורה אחידה כל תמונת מובייל, ולהביא אותה לפורמט ה-RGB המדויק ($100 \times 100 \times 3$) שהמודל מצפה לקבל.
4.	מנגנון סינון אובייקטים זרים (Thresholding): הגדרתי סף ביטחון דינמי (Confidence Threshold). אם המודל מזהה אובייקט ברמת ביטחון נמוכה מ-60%, האפליקציה עוצרת, מציגה הודעת שגיאה מובנית ומבקשת מהמשתמש לצלם שוב פרי אמיתי מהרשימה.


# 🍎 Fruit & Vegetable Classifier — CNN + Streamlit

A deep learning project that classifies fruit images using a custom-built Convolutional Neural Network (CNN), with a mobile-friendly web interface powered by Streamlit.

---

## 📌 Overview

This project trains a CNN model on a dataset of 7+ fruit categories and deploys a real-time classifier accessible from any device — including smartphones (via live camera).

The model can identify the following fruits:
`Apple` · `Banana` · `Grape` · `Grapefruit` · `Mandarine` · `Mango` · `Orange` · `Peach`

---

## 🏗️ Model Architecture

The CNN is built with TensorFlow/Keras using the following layers:

```
Input (100x100x3)
  → Conv2D(32, 3x3, ReLU) → MaxPooling2D(2x2)
  → Conv2D(64, 3x3, ReLU) → MaxPooling2D(2x2)
  → Conv2D(128, 3x3, ReLU) → MaxPooling2D(2x2)
  → Flatten
  → Dense(128, ReLU)
  → Dense(8, Softmax)
```

- **Optimizer:** Adam  
- **Loss:** Categorical Crossentropy  
- **Epochs:** 15  
- **Input size:** 100×100 RGB

---

## 🗂️ Dataset

- **Dataset:** `fruits_7_dataset_100x100.zip`
- **Image size:** 100×100 pixels
- **Structure after preprocessing:**
```
dataset/
└── 7_fruits/
    ├── Training/
    │   ├── Apple/
    │   ├── Banana/
    │   └── ...
    └── Test/
        ├── Apple/
        ├── Banana/
        └── ...
```

> The notebook includes an automatic folder restructuring script that merges subcategories (e.g., `Apple 10`, `Apple 20`) into a single `Apple/` folder.

---

## 🔧 Data Augmentation

Applied to the **training set** only:

| Technique | Value |
|---|---|
| Rescale | 1/255 |
| Shear range | 0.2 |
| Zoom range | 0.2 |
| Horizontal flip | True |

---

## 📱 Streamlit Web App

The app (`app.py`) provides a mobile-friendly interface with two input modes:

- 📸 **Live Camera** — take a photo directly from your phone
- 🖼️ **Gallery Upload** — upload an existing image (JPG, PNG, WEBP)

### Features:
- Displays a live image preview
- Runs AI prediction with confidence score
- Rejects low-confidence predictions (threshold: 40%)
- Saves uploaded images to the server

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow streamlit pillow numpy pyngrok
```

### Run on Google Colab

1. Upload the dataset zip to your Google Drive
2. Mount Drive and run all cells in the notebook
3. The final cell launches the Streamlit app via **ngrok** and prints a public URL

```python
public_url = ngrok.connect(8501)
print(f"🚀 YOUR APP LINK: {public_url}")
!streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > /dev/null &
```

### Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Install dependencies
pip install -r requirements.txt

# Run the app (make sure fruits_cnn_model.h5 exists)
streamlit run app.py
```

---

## 📁 Project Structure

```
├── Project_3_Vision__CNN_.ipynb   # Main training notebook
├── app.py                          # Streamlit web application
├── fruits_cnn_model.h5             # Saved trained model (generated after training)
├── saved_images/                   # Directory for uploaded images
└── README.md
```

---

## 📦 Requirements

```
tensorflow
streamlit
pillow
numpy
pandas
matplotlib
seaborn
pyngrok
```

> To generate a `requirements.txt` automatically from your environment:
> ```bash
> pip freeze > requirements.txt
> ```

---

## 🧪 Example Prediction Flow

1. User uploads or captures a fruit image
2. Image is resized to 100×100 and normalized to [0, 1]
3. Model outputs a probability vector over 8 classes
4. The class with the highest probability is returned
5. If confidence < 40%, the image is rejected as unrecognized

---

## 📊 Training Results

Training was performed for **15 epochs** on Google Colab with GPU acceleration. Loss and accuracy curves can be found in the notebook.

---

## 🙏 Acknowledgements

- Dataset: Fruits 7 Dataset (100×100)
- Framework: [TensorFlow / Keras](https://www.tensorflow.org/)
- App: [Streamlit](https://streamlit.io/)
- Tunneling: [ngrok](https://ngrok.com/)
