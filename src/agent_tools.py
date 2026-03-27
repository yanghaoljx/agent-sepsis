import os
from dotenv import load_dotenv
import sys
import joblib
import shap
import numpy as np
load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

from langchain_core.tools import tool
from src.validators.agent_validators import *
from typing import  Literal
import pandas as pd
import json
from src.vector_database.main import PineconeManagment
from src.utils import format_retrieved_docs

class PatientIDInput(BaseModel):
    patient_id: str = Field(description="The unique patient identifier, typically starting with 'IP' (e.g., IP2333).")


pinecone_conn = PineconeManagment()
pinecone_conn.loading_vdb(index_name = 'sepsis')
retriever = pinecone_conn.vdb.as_retriever(search_type="similarity", 
                                    search_kwargs={"k": 2})
rag_chain = retriever | format_retrieved_docs

@tool
def retrieve_faq_info(question:str):
    """
    Retrieve documents or additional info from general questions about the medical clinic.
    Call this tool if question is regarding center:
    For example: is it open? Do you have parking? Can  I go with bike? etc...
    """
    return rag_chain.invoke(question)


@tool("get_patient_basic_info", args_schema=PatientIDInput)
def get_patient_basic_info(patient_id: str):
    """
    获取患者基础信息及异常检验结果。
    适用于：用户询问“患者情况”、“某患者是谁”或“有哪些异常指标”。
    """
    file_path = 'data/patient_sample_with_diag.json'
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 注意：如果 JSON 中包含原始的 NaN，json.load 可能会报错
            # 建议在生成 JSON 时将 NaN 替换为 null
            patient_data_list = json.load(f)
    except FileNotFoundError:
        return {"error": f"找不到数据文件: {file_path}"}
    except json.JSONDecodeError:
        return {"error": "JSON 格式错误，请检查文件中是否存在非法字符（如未加引号的 NaN）"}

    # 1. 查找患者
    patient_record = next((p for p in patient_data_list if p.get("就诊号") == patient_id), None)

    # 2. 如果没有找到患者数据，直接返回错误信息
    if not patient_record:
        return {"error": f"未找到就诊号为 {patient_id} 的患者信息"}

    # 3. 提取基本信息
    demo_info = patient_record.get("人口学信息", {})
    
    # 4. 提取并格式化异常检验情况
    abnormal_tests = []
    for item in patient_record.get("异常检验情况", []):
        test_detail = {
            "项目": item.get("项目"),
            "结果": item.get("结果"),
            "单位": item.get("单位"),
            "标志": "偏高 (H)" if item.get("标志") == "h" else "偏低 (L)"
        }
        abnormal_tests.append(test_detail)

    # 5. 返回整合后的数据
    return {
        "status": "success",
        "patient_id": patient_record.get("就诊号"),
        "basic_info": {
            "age": demo_info.get("年龄"),
            "bmi": demo_info.get("BMI"),
            "diagnosis": patient_record.get("诊断信息")
        },
        "abnormal_indicators": abnormal_tests
    }


@tool("sepsis_early_screening", args_schema=PatientIDInput)
def sepsis_early_screening(patient_id: str):
    """
    Determines if the patient CURRENTLY has sepsis or is developing it right now.
    Focus: DIAGNOSIS and DETECTION.
    
    USE THIS TOOL WHEN USER ASKS:
    - "Does the patient have sepsis?"
    - "Is the patient suspected of sepsis?"
    - "What is the screening result?"
    - "Check for early signs of infection."
    
    OUTPUT: Returns positive/negative screening result.
    """
    df = pd.read_csv('data/combined_sepsis_icu_data.csv')
    print(patient_id)
    patient_data = df[df['就诊号'] == patient_id]
    if patient_data.empty:
        return {"error": f"未找到就诊号为 {patient_id} 的患者信息"}
    # 去除就诊号列和标签列
    X_patient = patient_data.drop(['就诊号', 'mortality_label'], axis=1)
    # 载入训练好的模型
    model = joblib.load('models/best_sepsis_model.pkl')
    # 进行预测
    mortality_prob = model.predict_proba(X_patient)[:, 1][0]
    # 根据概率给出风险等级
    if mortality_prob >= 0.7:
        risk_level = "High Risk of Deterioration"
    elif mortality_prob >= 0.4:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk" 
    # 这是一个lightgbm模型,基于SHAP对其个体的特征贡献结果进行展示

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_patient)
    # 获取特征名称
    feature_names = X_patient.columns.tolist()
    # 得到前10影响最大的特征
    shap_abs_mean = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(shap_abs_mean)[-10:][::-1]
    top_features = [(feature_names[i], shap_values[0][i]) for i in top_indices]
    return {
        "patient_id": patient_id,
        "sepsis_risk_probability": round(float(mortality_prob), 4),
        "risk_level": risk_level,
        "top_contributing_factors": top_features
    }


@tool("predict_sepsis_prognosis_mortality", args_schema=PatientIDInput)
def predict_sepsis_prognosis_mortality(patient_id: str):
    """
    Predicts the FUTURE outcome, mortality rate, or survival chance.
    Focus: PROGNOSIS and DETERIORATION (What will happen next?).
    
    USE THIS TOOL WHEN USER ASKS:
    - "What is the survival rate?"
    - "Will the patient die?"
    - "What is the risk of death/mortality?"
    - "Predict the outcome."
    - "Is the condition getting worse?"
    
    DO NOT USE for checking if they *have* sepsis (use screening for that).
    """

    df = pd.read_csv('data/combined_sepsis_icu_data.csv')
    patient_data = df[df['就诊号'] == patient_id]
    if patient_data.empty:
        return {"error": f"未找到就诊号为 {patient_id} 的患者信息"}
    # 去除就诊号列和标签列
    X_patient = patient_data.drop(['就诊号', 'mortality_label'], axis=1)
    # 载入训练好的模型
    model = joblib.load('models/best_death_model.pkl')
    # 进行预测
    mortality_prob = model.predict_proba(X_patient)[:, 1][0]
    # 根据概率给出风险等级
    if mortality_prob >= 0.7:
        risk_level = "High Risk of Deterioration"
    elif mortality_prob >= 0.4:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk" 
    # 这是一个lightgbm模型,基于SHAP对其个体的特征贡献结果进行展示

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_patient)
    # 获取特征名称
    feature_names = X_patient.columns.tolist()
    # 得到前10影响最大的特征
    shap_abs_mean = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(shap_abs_mean)[-10:][::-1]
    top_features = [(feature_names[i], shap_values[0][i]) for i in top_indices]
    return {
        "patient_id": patient_id,
        "mortality_risk_probability": round(float(mortality_prob), 4),
        "risk_level": risk_level,
        "top_contributing_factors": top_features
    }