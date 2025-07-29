import os
import json
import pickle
import random
import logging
import sqlite3
from datetime import datetime, timezone
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

import xgboost as xgb

=== CONSTANTES ===

TOKEN = "8124789563:AAGB2kMmzGQzy6pHoglJDKYqE7NqNEOBtOw"
PAYMENT_NUMBER = "+2250779984519"
ADMIN_ID = 7798029774
ADMIN_CODE = "Diom19035696fsin"
ADMIN_ACCESS_MONTHS = 4

DB_FILE = "data.db"
USERS_FILE = "users.json"
SETTINGS_FILE = "settings.json"
MODEL_FILE = "stacked_model.pkl"
LSTM_MODEL_FILE = "lstm_model.h5"

FREE_TRIAL_DAYS = 2
REFERRAL_REQUIRED = 3
PAID_REFERRAL_REQUIRED = 2
BONUS_PREDICTIONS_REWARD = 2

users = {}
settings = {}
predictions_history = []

=== CHARGEMENT / SAUVEGARDE ===

def save_data():
try:
with open(USERS_FILE, "w") as f:
json.dump(users, f, default=str)
with open(SETTINGS_FILE, "w") as f:
json.dump(settings, f)
except Exception as e:
logging.error(f"Erreur lors de la sauvegarde des données : {e}")

def load_data():
global users, settings
try:
if os.path.exists(USERS_FILE):
with open(USERS_FILE, "r") as f:
raw_users = json.load(f)
for k, v in raw_users.items():
v["referrals"] = set(v.get("referrals", []))
v["free_trial_start"] = datetime.fromisoformat(v["free_trial_start"]) if v.get("free_trial_start") else None
v["paid_until"] = datetime.fromisoformat(v["paid_until"]) if v.get("paid_until") else None
v["admin_until"] = datetime.fromisoformat(v["admin_until"]) if v.get("admin_until") else None
users[int(k)] = v
if os.path.exists(SETTINGS_FILE):
with open(SETTINGS_FILE, "r") as f:
settings = json.load(f)
except Exception as e:
logging.error(f"Erreur lors du chargement des données : {e}")

def init_db():
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS results (
id INTEGER PRIMARY KEY AUTOINCREMENT,
user_id INTEGER,
prediction_id TEXT,
game_type TEXT,
date TEXT,
confidence REAL,
result TEXT
)""")
conn.commit()
conn.close()

def store_prediction(user_id, prediction_id, game_type, date, confidence, result):
try:
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute("INSERT INTO results (user_id, prediction_id, game_type, date, confidence, result) VALUES (?, ?, ?, ?, ?, ?)",
(user_id, prediction_id, game_type, date.isoformat(), confidence, result))
conn.commit()
conn.close()
except Exception as e:
logging.warning(f"Erreur en enregistrant une prédiction : {e}")

=== SCRAPING ===

def scrape_1xbet(game_type):
try:
logging.info(f"📡 Scraping simulé pour {game_type} à {datetime.now()}")
# Simulation : ajouter aux résultats une entrée fictive
sample_result = f"{game_type}result{random.randint(1, 99)}"
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()

now = datetime.utcnow().isoformat()  
    c.execute("INSERT INTO results (user_id, prediction_id, game_type, date, confidence, result) VALUES (?, ?, ?, ?, ?, ?)",  
              (-1, "SCRAPER", game_type, now, 1.0, sample_result))  
    conn.commit()  
    conn.close()  
except Exception as e:  
    logging.warning(f"Erreur scraping : {e}")

=== MACHINE LEARNING : STACKING + LSTM ===

def get_dataset(game_type):
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute("SELECT result FROM results WHERE game_type = ? AND user_id = -1 ORDER BY date DESC LIMIT 100", (game_type,))
rows = c.fetchall()
conn.close()
return [row[0] for row in rows]

def encode_features(data):
return np.array([[ord(char) % 10 for char in entry[-4:]] for entry in data])

def train_model():
dataset_poker = get_dataset("POKER")
dataset_wheel = get_dataset("WHEEL")

if len(dataset_poker) < 20 or len(dataset_wheel) < 20:  
    return None  

def prepare_dataset(dataset):  
    X = encode_features(dataset[:-1])  
    y = [ord(v[-1]) % 2 for v in dataset[1:]]  
    return X, y  

X1, y1 = prepare_dataset(dataset_poker)  
X2, y2 = prepare_dataset(dataset_wheel)  

X = np.vstack((X1, X2))  
y = y1 + y2  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  

# === Base Models  
rf = RandomForestClassifier()  
gb = GradientBoostingClassifier()  
mlp = MLPClassifier(max_iter=300)  
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")  

estimators = [  
    ("rf", rf),  
    ("gb", gb),  
    ("mlp", mlp),  
    ("xgb", xgb_clf)  
]  

stack = StackingClassifier(  
    estimators=estimators,  
    final_estimator=LogisticRegression(),  
    cv=5  
)  

pipe = make_pipeline(StandardScaler(), stack)  
pipe.fit(X_train, y_train)  

y_pred = pipe.predict(X_test)  
acc = accuracy_score(y_test, y_pred)  
logging.info(f"🎯 Stacked model accuracy: {round(acc*100, 2)}%")  

with open(MODEL_FILE, "wb") as f:  
    pickle.dump(pipe, f)  

# === LSTM Model  
X_lstm = np.array([x for x in X])  
y_lstm = np.array(y)  

X_lstm = X_lstm.reshape((X_lstm.shape[0], 1, X_lstm.shape[1]))  

lstm_model = Sequential()  
lstm_model.add(LSTM(64, input_shape=(1, X_lstm.shape[2]), activation='relu'))  
lstm_model.add(Dense(1, activation='sigmoid'))  
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  

lstm_model.fit(X_lstm, y_lstm, epochs=10, verbose=0, callbacks=[EarlyStopping(patience=2)])  

lstm_model.save(LSTM_MODEL_FILE)  
return pipe

def predict_with_model(game_type):
try:
data = get_dataset(game_type)
if len(data) < 5:
return "Pas assez de données"

X = encode_features([data[-1]])  
    with open(MODEL_FILE, "rb") as f:  
        model = pickle.load(f)  
    pred = model.predict(X)[0]  

    lstm = tf.keras.models.load_model(LSTM_MODEL_FILE)  
    X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))  
    pred_lstm = lstm.predict(X_lstm, verbose=0)[0][0]  

    final = round((pred + pred_lstm) / 2)  
    return "PAIR" if final == 1 else "IMPAIR"  
except Exception as e:  
    logging.error(f"Erreur prediction : {e}")  
    return "Erreur"

=== GESTION UTILISATEUR ===

banned_users = set()

def update_referral(user_id, referred_by):
if user_id != referred_by and referred_by in users:
users[referred_by]["referrals"].add(user_id)

def give_referral_reward(referred_by):
referrals = users[referred_by]["referrals"]
new_users = [uid for uid in referrals if uid in users]
if len(new_users) >= REFERRAL_REQUIRED:
users[referred_by]["bonus_predictions"] += BONUS_PREDICTIONS_REWARD
return True
return False

def is_admin(user_id):
u = users.get(user_id)
return u and u.get("admin_until") and u["admin_until"] > datetime.now()

async def is_user_allowed(user_id):
if user_id in banned_users:
return False
u = users.get(user_id)
if not u:
return False
if u.get("paid_until") and u["paid_until"] > datetime.now():
return True
if u.get("free_trial_start"):
days_used = (datetime.now() - u["free_trial_start"]).days
return days_used <= FREE_TRIAL_DAYS
return False

=== UTILITAIRES ===

def tr(fr, en, lang="fr"):
return fr if lang == "fr" else en

def get_lang(update):
user_lang = update.effective_user.language_code
return "en" if user_lang.startswith("en") else "fr"

=== COMMANDES SUPPLÉMENTAIRES ===

async def ban(update: Update, context: ContextTypes.DEFAULT_TYPE):
if not is_admin(update.effective_user.id):
await update.message.reply_text("⛔️ Tu n’as pas les droits.")
return
if len(context.args) != 1:
await update.message.reply_text("Utilise : /ban <user_id>")
return
try:
uid = int(context.args[0])
banned_users.add(uid)
await update.message.reply_text(f"🚫 Utilisateur {uid} banni.")
except:
await update.message.reply_text("❌ ID invalide.")

async def unban(update: Update, context: ContextTypes.DEFAULT_TYPE):
if not is_admin(update.effective_user.id):
await update.message.reply_text("⛔️ Tu n’as pas les droits.")
return
if len(context.args) != 1:
await update.message.reply_text("Utilise : /unban <user_id>")
return
try:
uid = int(context.args[0])
banned_users.discard(uid)
await update.message.reply_text(f"✅ Utilisateur {uid} débanni.")
except:
await update.message.reply_text("❌ ID invalide.")

=== SCRAPING 1XBET ===

def scrape_1xbet(game_type):
try:
# ✏️ Exemple fictif - à adapter selon le vrai scraping
now = datetime.now()
result = f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(10,99)}"
store_prediction(-1, f"{game_type[:2]}SCR{random.randint(1000,9999)}", game_type, now, 1.0, result)
except Exception as e:
logging.error(f"Scraping échoué : {e}")

=== SAUVEGARDE / BASE DE DONNÉES ===

DB_FILE = "data.db"
MODEL_FILE = "ml_model.pkl"
LSTM_MODEL_FILE = "lstm_model.h5"

def init_db():
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS results (
user_id INTEGER,
prediction_id TEXT,
game_type TEXT,
date TEXT,
precision REAL,
predicted_value TEXT
)""")
conn.commit()
conn.close()

def store_prediction(user_id, prediction_id, game_type, date, precision, predicted_value):
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?)", (user_id, prediction_id, game_type, date.isoformat(), precision, predicted_value))
conn.commit()
conn.close()

def load_data():
global users, settings, predictions_history
try:
with open("users.pkl", "rb") as f:
users = pickle.load(f)
except:
users = {}
try:
with open("settings.pkl", "rb") as f:
settings = pickle.load(f)
except:
settings = {"auto_prediction_enabled": True}
try:
with open("history.pkl", "rb") as f:
predictions_history = pickle.load(f)
except:
predictions_history = []

def save_data():
with open("users.pkl", "wb") as f:
pickle.dump(users, f)
with open("settings.pkl", "wb") as f:
pickle.dump(settings, f)
with open("history.pkl", "wb") as f:
pickle.dump(predictions_history, f)

=== LANCEMENT DU BOT ===

def main():
logging.basicConfig(level=logging.INFO)
load_data()
init_db()
train_model()

app = ApplicationBuilder().token(TOKEN).build()  

app.add_handler(CommandHandler("start", start))  
app.add_handler(CommandHandler("acheter", acheter))  
app.add_handler(CommandHandler("bonus", bonus))  
app.add_handler(CommandHandler("monlien", my_link))  
app.add_handler(CommandHandler("admin", admin))  
app.add_handler(CommandHandler("help", help_command))  
app.add_handler(CommandHandler("confirmpayment", admin_confirm_payment))  
app.add_handler(CommandHandler("ban", ban))  
app.add_handler(CommandHandler("unban", unban))  

asyncio.create_task(periodic_tasks(app))  

print("✅ Bot en cours d'exécution...")  
app.run_polling()

=== BOUCLE DE FOND ===

async def periodic_tasks(app):
while True:
try:
now = datetime.utcnow().timestamp()
if int(now) % (5 * 60) < 30:
scrape_1xbet("POKER")
scrape_1xbet("WHEEL")
train_model()
if settings.get("auto_prediction_enabled", True) and int(now) % (15 * 60) < 30:
await send_predictions_to_all(app)
except Exception as e:
logging.error(f"Boucle échouée : {e}")
await asyncio.sleep(60)

if name == "main":
main()
