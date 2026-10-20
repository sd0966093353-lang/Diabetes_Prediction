############ Import the required Libraries ##################
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score

def main():
    st.title("Predicting Diabetes Web App")
    st.markdown("Affected by Diabetes or not ?")
    
    st.sidebar.title("Model & Data Selection Panel")
    
    @st.cache_data
    def load_data():  # Function to load dataset
        data = pd.read_csv("diabetes.csv")
        return data

    df = load_data() 
    class_names = ['Non-Diabetes', 'Diabetes']

    # 1. 可設定訓練資料的比例
    st.sidebar.subheader("Data Split Parameter")
    train_size = st.sidebar.slider("Training Data Ratio (%)", min_value=10, max_value=90, value=70, step=5) / 100.0

    def split(df, train_ratio):  # Split data based on user input ratio
        y = df.Outcome
        x = df.drop(columns=['Outcome'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_ratio, random_state=0)
        return x, x_train, x_test, y_train, y_test

    x_all, x_train, x_test, y_train, y_test = split(df, train_size)

    # 繪圖函式（已刪除 Precision-Recall Curve）
    def plot_metrics(metrics_list, model, x_test, y_test, class_names):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, model.predict(x_test))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            fig, ax = plt.subplots()
            disp.plot(cmap="Blues", ax=ax)
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            y_prob = model.predict_proba(x_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
            ax.plot([0, 1], [0, 1], "k--", label="Random guess")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Receiver Operating Characteristic")
            ax.legend(loc="lower right")
            st.pyplot(fig)

    # 2. 可選擇模型與設定模型參數
    st.sidebar.subheader("Select your Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Decision Tree", "Logistic Regression", "Random Forest"))

    # 根據選取的模型建立模型物件
    if classifier == 'Decision Tree':
        st.sidebar.subheader("Model Parameters")
        criterion = st.sidebar.radio("Criterion", ("gini", "entropy"), key='criterion')
        splitter = st.sidebar.radio("Splitter", ("best", "random"), key='splitter')
        model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, random_state=0)

    elif classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Parameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, value=1.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, value=200, key='max_iter')
        model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter, random_state=0)

    elif classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 10, 1000, value=100, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, value=10, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=(bootstrap == 'True'), n_jobs=-1, random_state=0)

    # 評估指標圖表勾選（無 Precision-Recall Curve）
    metrics = st.sidebar.multiselect("Select evaluation metrics to plot:", ('Confusion Matrix', 'ROC Curve'))

    # 直接擬合模型（不需再按按鈕手動重新訓練）
    model.fit(x_train, y_train)

    # 展示模型評估結果
    st.subheader(f"{classifier} Results")
    accuracy = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{round(accuracy * 100, 2)}%")
    col2.metric("Precision", f"{round(precision_score(y_test, y_pred), 2)}")
    col3.metric("Recall", f"{round(recall_score(y_test, y_pred), 2)}")
    # 3. 增加 F1 的指標輸出
    col4.metric("F1 Score", f"{round(f1_score(y_test, y_pred), 2)}")

    plot_metrics(metrics, model, x_test, y_test, class_names)

    # 4. 單筆預測功能 (含機率分布圓餅圖)
    st.markdown("---")
    st.subheader("Single Patient Diabetes Prediction")
    st.write("Adjust the parameters on the sidebar to make a real-time prediction for a single patient.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Single Prediction Input Features")
    
    # 動態產生數值輸入介面，預設值取該特徵值的平均值
    input_data = {}
    for col in x_all.columns:
        default_val = float(x_all[col].mean())
        min_val = float(x_all[col].min())
        max_val = float(x_all[col].max())
        input_data[col] = st.sidebar.number_input(f"{col}", min_value=min_val, max_value=max_val, value=default_val)

    # 轉換成 DataFrame 進行模型預測
    single_sample = pd.DataFrame([input_data])
    single_pred = model.predict(single_sample)[0]
    single_prob = model.predict_proba(single_sample)[0]

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.write("### Prediction Result")
        pred_label = class_names[single_pred]
        if single_pred == 1:
            st.error(f"**Predicted Status:** {pred_label}")
        else:
            st.success(f"**Predicted Status:** {pred_label}")

        st.write(f"- Probability of Non-Diabetes: **{single_prob[0]*100:.2f}%**")
        st.write(f"- Probability of Diabetes: **{single_prob[1]*100:.2f}%**")

    with res_col2:
        # 繪製單筆預測機率分布的圓餅圖
        fig_pie, ax_pie = plt.subplots()
        colors = ['#66b3ff', '#ff9999']
        ax_pie.pie(single_prob, labels=class_names, autopct='%1.1f%%', startangle=90, colors=colors)
        ax_pie.axis('equal')  # 確保圓餅圖為圓形
        plt.title("Prediction Probability Distribution")
        st.pyplot(fig_pie)

    # 顯示原始資料表選項
    st.markdown("---")
    if st.checkbox("Show raw data", False):
        st.subheader("Diabetes Raw Dataset")
        st.write(df)

if __name__ == '__main__':
    main()