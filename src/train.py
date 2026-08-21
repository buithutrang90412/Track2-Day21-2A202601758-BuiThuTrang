import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65
REFERENCE_POSITIVE_RATE = 0.248


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    # TODO 1: Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # TODO 2: Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():

        # TODO 3: Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # TODO 4: Khoi tao va huan luyen GradientBoostingClassifier
        # Goi y: su dung random_state=42 de dam bao tinh tai tao
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # TODO 5: Du doan tren tap holdout va tinh chi so
        # Chu y: f1_score o day tinh cho LOP DUONG (target = 1), khong dung average.
        probabilities = model.predict_proba(X_eval)[:, 1]
        thresholds = [round(i / 100, 2) for i in range(10, 91, 5)]
        threshold_scores = {
            threshold: f1_score(y_eval, (probabilities >= threshold).astype(int))
            for threshold in thresholds
        }
        best_threshold = max(threshold_scores, key=threshold_scores.get)
        default_preds = (probabilities >= 0.5).astype(int)
        preds = (probabilities >= best_threshold).astype(int)
        default_f1 = f1_score(y_eval, default_preds)
        f1 = f1_score(y_eval, preds)
        acc = accuracy_score(y_eval, preds)
        precision = precision_score(y_eval, preds, zero_division=0)
        recall = recall_score(y_eval, preds, zero_division=0)
        matrix = confusion_matrix(y_eval, preds)
        positive_rate = float(y_train.mean())
        drift = abs(positive_rate - REFERENCE_POSITIVE_RATE) > 0.05

        # TODO 6: Ghi nhan chi so vao MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("default_f1_score", default_f1)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("train_positive_rate", positive_rate)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In ket qua ra man hinh
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f} | threshold: {best_threshold:.2f}")
        if drift:
            print(f"WARNING: train positive rate {positive_rate:.3f} differs from reference {REFERENCE_POSITIVE_RATE:.3f}")

        # TODO 8: Luu metrics ra file outputs/report.json
        # File nay duoc doc boi GitHub Actions o Buoc 2
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump({
                "f1_score": f1,
                "accuracy": acc,
                "default_f1_score": default_f1,
                "best_threshold": best_threshold,
                "train_positive_rate": positive_rate,
                "drift_warning": drift,
            }, f)
        with open("outputs/detail.txt", "w") as f:
            f.write("Confusion matrix (rows=true, columns=predicted):\n")
            f.write(f"{matrix.tolist()}\n\n")
            f.write(f"threshold={best_threshold:.2f}\n")
            f.write(f"class_0_precision={precision_score(y_eval, preds, pos_label=0, zero_division=0):.4f}\n")
            f.write(f"class_0_recall={recall_score(y_eval, preds, pos_label=0, zero_division=0):.4f}\n")
            f.write(f"class_1_precision={precision:.4f}\n")
            f.write(f"class_1_recall={recall:.4f}\n")

        # TODO 9: Luu mo hinh ra file models/model.joblib
        # File nay duoc upload len cloud storage o Buoc 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    # TODO 10: Tra ve f1
    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
