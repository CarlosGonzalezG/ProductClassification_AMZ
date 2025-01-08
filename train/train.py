from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from labelEncoder import LabelEncoderWrapper
from pathlib import Path
import pandas as pd
import numpy as np
import gzip, json, scipy, pickle, yaml, os
import lightgbm as lgb

########################################
# Utility functions
########################################
def load_config(config_file="config_train.yaml"):
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_file}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)
            
def build_dataframe(path):
    products = []
    for product_json in parse(path):
        brand = product_json.get("brand", "")
        title = product_json.get("title", "")
        description = " ".join(product_json.get("description", []))
        feature = " ".join(product_json.get("feature", []))
        main_cat = product_json.get("main_cat", "")

        combined_text = " ".join([str(brand), str(title), str(description), str(feature)])
        combined_text = combined_text.lower()
        
        products.append({
            "text": combined_text,
            "main_cat": main_cat
        })
    return pd.DataFrame(products)

def load_or_generate_embeddings(embedder, text_data, file_path):
    """
    Load embeddings from file if it exists, otherwise generate and save them.
    """
    if os.path.exists(file_path):
        print(f"Loading embeddings from {file_path}")
        with open(file_path, "rb") as f:
            embeddings = pickle.load(f)
    else:
        print(f"Generating embeddings for {file_path}")
        embeddings = embedder.encode(text_data, show_progress_bar=True)
        with open(file_path, "wb") as f:
            pickle.dump(embeddings, f)
    return embeddings

def load_or_save_labels(labels, file_path):
    """
    Load labels from file if it exists, otherwise save them.
    """
    if os.path.exists(file_path):
        print(f"Loading labels from {file_path}")
        with open(file_path, "rb") as f:
            loaded_labels = pickle.load(f)
    else:
        print(f"Saving labels to {file_path}")
        with open(file_path, "wb") as f:
            pickle.dump(labels, f)
        loaded_labels = labels
    return loaded_labels

def save_classification_report(report, output_path, model_name):
    """
    Save classification report to a .txt file.
    """
    report_path = os.path.join(output_path, f"{model_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

########################################
# Main function
########################################
def main():
    # 1. Load the YAML configuration file
    config_path = Path(__file__).parent.parent / "configuration/config_train.yaml"
    # 2. Extract settings from the config    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise RuntimeError(f"Error parsing configuration file: {e}")

    try:
        data_path = config["dataset"]["path"]  
        output_path = config["output_path"]["path"]
        model = config["model"]["name"]
        embedder_name = config["embedding"]["model_name"]         
    except KeyError as e:
        raise KeyError(f"Missing required configuration key: {e}")
    
    # 3. Build the dataframe
    df = build_dataframe(data_path)
    
    # 4. Split dataset 
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["main_cat"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["main_cat"])
    
    # 5. Compute class weights
    classes = np.unique(train_df["main_cat"])
    class_weights = compute_class_weight(
        "balanced",
        classes=classes,
        y=train_df["main_cat"]
    )
    # 6. Label encoding
    label_encoder = LabelEncoderWrapper()
    label_encoder.fit(train_df["main_cat"])
    
    encoded_classes = label_encoder.transform(classes)
    class_weights_dict = dict(zip(encoded_classes, class_weights))
    
    class_weights_dict_xgb = dict(zip(classes, class_weights))

    #train_encoded_main_cat = label_encoder.transform(train_df["main_cat"])
    #train_sample_weights = np.array([class_weights_dict[label] for label in train_encoded_main_cat])
    
    # Encode tags for boosting algorithms
    y_train_encoded = label_encoder.transform(train_df["main_cat"])
        
    # 7. Prepare output folders
    os.makedirs(output_path, exist_ok=True)
    embeddings_folder = os.path.join(output_path, "embeddings")
    os.makedirs(embeddings_folder, exist_ok=True)
    
    # 8. Load the SentenceTransformer
    embedder = SentenceTransformer(embedder_name)

    # 9. Generate or load embeddings
    X_train_file = os.path.join(embeddings_folder, "X_train.pkl")
    y_train_file = os.path.join(embeddings_folder, "y_train.pkl")
    X_val_file = os.path.join(embeddings_folder, "X_val.pkl")
    y_val_file = os.path.join(embeddings_folder, "y_val.pkl")
    X_test_file = os.path.join(embeddings_folder, "X_test.pkl")
    y_test_file = os.path.join(embeddings_folder, "y_test.pkl")

    # Generate or load embeddings train
    X_train_embeddings = load_or_generate_embeddings(embedder, train_df["text"].tolist(), X_train_file)
    y_train = load_or_save_labels(train_df["main_cat"].values, y_train_file)
    
    # Validation and test embeddings
    X_val_embeddings = load_or_generate_embeddings(embedder, val_df["text"].tolist(), X_val_file)
    y_val = load_or_save_labels(val_df["main_cat"].values, y_val_file)
    X_test_embeddings = load_or_generate_embeddings(embedder, test_df["text"].tolist(), X_test_file)
    y_test = load_or_save_labels(test_df["main_cat"].values, y_test_file)

    # 10. Create a model-specific folder
    model_folder = os.path.join(output_path, model)
    os.makedirs(model_folder, exist_ok=True)
    
    # 11. Train the chosen model
    if model == "logistic_regression":
        clf = LogisticRegression(
            max_iter=1000,
            class_weight=class_weights_dict,
            random_state=0
        ).fit(
            X_train_embeddings,
            y_train
        )

        y_val_pred = clf.predict(X_val_embeddings)
        y_test_pred = clf.predict(X_test_embeddings)

    elif model == "xgboost":
        train_sample_weights = train_df["main_cat"].map(class_weights_dict_xgb).values
        clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(classes),
            random_state=0
        ).fit(
            X_train_embeddings,
            y_train_encoded,
            sample_weight=train_sample_weights
            
        )
        
        y_val_pred_encoded = clf.predict(X_val_embeddings)
        y_test_pred_encoded = clf.predict(X_test_embeddings)

        y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)
        y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)

    elif model == "catboost":
        clf = CatBoostClassifier(
            iterations=10,
            depth=3,
            class_weights=list(class_weights),
            verbose=0
        ).fit(
            X_train_embeddings,
            y_train_encoded
        )

        y_val_pred_encoded = clf.predict(X_val_embeddings)
        y_test_pred_encoded = clf.predict(X_test_embeddings)

        y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)
        y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)
    
    elif model == "lightgbm":
        clf = lgb.LGBMClassifier(
            class_weight=class_weights_dict, 
            random_state=0  
        )
        
        # Train the model
        clf.fit(
            X_train_embeddings,
            y_train_encoded
        )

        # Make predictions
        y_val_pred_encoded = clf.predict(X_val_embeddings)
        y_test_pred_encoded = clf.predict(X_test_embeddings)

        # Decode predictions back to original labels
        y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)
        y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)
    
    else:
        raise ValueError("Model not found, please choose between 'logistic_regression', 'xgboost' or 'catboost'")
        return

    # 12. Generate and save classification reports
    val_report = classification_report(y_val, y_val_pred, output_dict=False)
    test_report = classification_report(y_test, y_test_pred, output_dict=False)
    save_classification_report(val_report, model_folder, f"{model}_val")
    save_classification_report(test_report, model_folder, f"{model}_test")

    # 13. Save the trained model
    pickle.dump(clf, open(os.path.join(model_folder, "model.pkl"), "wb"))

    # 14. Save the SentenceTransformer model to ensure reproducibility
    embedder_folder = os.path.join(model_folder, "sbert_model")
    embedder.save(embedder_folder)

    print(f"\nTraining complete for {model}.")
    print(f"Model artifacts saved to: {model_folder}")

if __name__ == "__main__":
    main()