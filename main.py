import cv2
import joblib
import numpy as np
from PIL import Image
from google.colab import files
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("Loading CIFAR-10 dataset...")
(X_train_full, _), (_, _) = cifar10.load_data()

# Use a highly representative subset for fast, clean training
SUBSET_SIZE = 10000
X_train = X_train_full[:SUBSET_SIZE]

def lsb_embed(image, payload_rate=0.2):
    img = image.copy()
    total_bits = int(img.size * payload_rate)
    positions = np.random.choice(img.size, total_bits, replace=False)
    flat = img.flatten()
    secret_bits = np.random.randint(0, 2, total_bits)

    flat[positions] = (flat[positions] & 254) | secret_bits
    return flat.reshape(img.shape)

print("Generating Stego Images via LSB embedding...")
stego_images = np.array([lsb_embed(img_train, payload_rate=0.3) for img_train in X_train])

# Combine Clean and Stego sets
X_all = np.concatenate([X_train, stego_images])
y_all = np.concatenate([np.zeros(len(X_train)), np.ones(len(stego_images))])

# Shuffle and Split Data using Scikit-Learn
X, X_val, y, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42, shuffle=True)

# Feature Extraction Functions
def lsb_ratio(img):
    return (img & 1).mean()

def horizontal_transition(img):
    bits = img & 1
    return np.mean(bits[:, 1:] != bits[:, :-1])

def vertical_transition(img):
    bits = img & 1
    return np.mean(bits[1:, :] != bits[:-1, :])

def rf_features(img):
    return [lsb_ratio(img), horizontal_transition(img), vertical_transition(img)]

print("Extracting features for Random Forest...")
RF_X = np.array([rf_features(img) for img in X])
RF_X_val = np.array([rf_features(img) for img in X_val])

print("Training Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
rf.fit(RF_X, y)

# Residual Processing for CNN
def residual_map(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    return img.astype(np.float32) - blur.astype(np.float32)

print("Generating residual maps for CNN...")
residuals_train = np.array([residual_map(img) for img in X])
residuals_val = np.array([residual_map(img) for img in X_val])

# Build CNN Model
cnn = Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(32, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(64, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Training CNN with mixed validation data...")
cnn.fit(
    residuals_train,
    y,
    epochs=6,
    batch_size=128,
    validation_data=(residuals_val, y_val)
)

# Save models in modern format
joblib.dump(rf, "rf.pkl")
cnn.save("cnn.keras")
print("\n[✓] Model training completed and models saved.")

# Single Image Analysis Pipeline
def analyze_image(img_input):
    if len(img_input.shape) == 2:
        img_input = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
    elif img_input.shape[2] == 4:
        img_input = img_input[:, :, :3]

    image_rf_features = np.array([rf_features(img_input)])
    image_residual_map = np.array([residual_map(img_input)])

    rf_prob_single = rf.predict_proba(image_rf_features)[0][1]
    cnn_prob_single = cnn.predict(image_residual_map, verbose=0)[0][0]

    final_score_single = (0.3 * rf_prob_single) + (0.7 * cnn_prob_single)

    print("\n--- Analysis Result ---")
    print(f"Random Forest Prediction: {rf_prob_single:.2%}")
    print(f"CNN Prediction: {cnn_prob_single:.2%}")
    print(f"Combined Threat Score: {final_score_single:.2%}")

    if final_score_single > 0.5:
        print("Result: [!] STEGO DETECTED")
    else:
        print("Result: [✓] CLEAN IMAGE")

# --- MODIFIED INTERACTIVE LOOP ---
while True:
    print("\n" + "="*50)
    print("OPTIONS: 1 = Generate Live Stego Sample | 2 = Upload Custom File | exit = Stop Program")
    choice = input("Enter choice: ").strip().lower()

    if choice == 'exit':
        print("Exiting loop...")
        break
    elif choice == '1':
        print("\nTaking a clean sample from CIFAR, embedding a payload, and running analysis...")
        random_idx = np.random.randint(0, len(X_train_full))
        test_img = X_train_full[random_idx]
        stego_test = lsb_embed(test_img, payload_rate=0.3)
        analyze_image(stego_test)

    elif choice == '2':
        print("\nPlease upload your image file:")
        uploaded = files.upload()
        if uploaded:
            file_name = list(uploaded.keys())[0]
            img = Image.open(file_name)

            # --- FIX: CROP INSTEAD OF RESIZE ---
            # Get dimensions
            width, height = img.size

            if width >= 32 and height >= 32:
                # Take a 32x32 patch from the center to preserve raw pixel data
                left = (width - 32) // 2
                top = (height - 32) // 2
                right = left + 32
                bottom = top + 32
                img = img.crop((left, top, right, bottom))
                print("[✓] Extracted an unblended 32x32 patch from the center.")
            else:
                # Fallback if image is somehow smaller than 32x32
                img = img.resize((32, 32))

            analyze_image(np.array(img))
    else:
        print("Invalid entry. Please type 1, 2, or exit.")
