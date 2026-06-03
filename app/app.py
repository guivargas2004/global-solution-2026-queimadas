"""
Classificação de Áreas Propensas a Queimadas
Global Solution 2026 - Applied Computer Vision
"""

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Configuração página
st.set_page_config(
    page_title="Detecção de Risco de Queimadas",
    layout="wide",
)

# Contantes
IMG_SIZE = (128, 128)
CLASSES = ["nowildfire", "wildfire"]
CLASSES_PT = {
    "nowildfire": "Sem Risco de Queimada",
    "wildfire": "Risco de Queimada",
}

MODEL_PATH = Path(__file__).parent.parent / "models" / "modelo_avancado_melhor.keras"


# Carregamento do modelo
@st.cache_resource
def carregar_modelo():
    return tf.keras.models.load_model(MODEL_PATH)


# Predição
def prever(modelo, imagem_pil):
    # Redimensiona pra 128x128
    img = imagem_pil.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    # Predição
    prob = float(modelo.predict(arr, verbose=0)[0][0])
    classe_idx = 1 if prob > 0.5 else 0
    confianca = prob if classe_idx == 1 else 1 - prob
    return CLASSES[classe_idx], confianca, prob


# Interface do programa

st.title("Detecção de Áreas Propensas a Queimadas")
st.markdown(
    "**Global Solution 2026 — Applied Computer Vision** | "
    "Classificação de imagens de satelites com Visão Computacional"
)
st.divider()

# Sidebar com informações do projeto
with st.sidebar:
    st.header("Sobre o Projeto")
    st.markdown("""
    Esta solução utiliza uma Rede Neural Convolucional treinada do zero 
    para classificar imagens de satelites ao risco de queimadas.
    
    Tema: Space Connect — Tecnologia espacial aplicada a desafios reais.""")
    
    st.divider()
    st.header("Modelo")
    st.markdown("""
    - Arquitetura: CNN com 4 blocos Conv2D
    - Técnicas: BatchNorm + Dropout + GAP + Data Augmentation
    - Acurácia no teste: 97.30%
    - AUC: 0.997
    - Parâmetros: 423K
    """)
    
    st.divider()

# Layout principal
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Envie uma imagem de satelite")
    arquivo = st.file_uploader(
        "Formatos suportados: JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"],
    )
    
    if arquivo is not None:
        imagem = Image.open(arquivo)
        st.image(imagem, caption="Imagem enviada", use_container_width=True)

with col2:
    st.subheader("Resultado da Análise")
    
    if arquivo is None:
        st.info("Envie uma imagem para iniciar a análise.")
    else:
        # Carrega modelo
        try:
            with st.spinner("Carregando modelo..."):
                modelo = carregar_modelo()
            
            # Prediz
            with st.spinner("Analisando imagem..."):
                classe, confianca, prob_raw = prever(modelo, imagem)
            
            # Exibe resultado destacado
            if classe == "wildfire":
                st.error(f"### {CLASSES_PT[classe]}")
                st.markdown(
                    f"O modelo identificou características associadas a "
                    f"áreas com **risco de queimadas**."
                )
            else:
                st.success(f"### {CLASSES_PT[classe]}")
                st.markdown(
                    f"O modelo não identificou características associadas a "
                    f"áreas com risco de queimadas."
                )
            
            # Métricas
            st.metric(
                label="Confiança da predição",
                value=f"{confianca*100:.2f}%",
            )
            
            # Barra de probabilidade
            st.markdown("**Distribuição da probabilidade:**")
            prob_no = (1 - prob_raw) * 100
            prob_yes = prob_raw * 100
            
            st.markdown(f"🟢 **Sem risco:** {prob_no:.2f}%")
            st.progress(prob_no / 100)
            
            st.markdown(f"🔴 **Com risco:** {prob_yes:.2f}%")
            st.progress(prob_yes / 100)
            
            # Aviso técnico
            st.divider()
            st.caption(
                "⚙️ **Nota técnica:** o modelo foi treinado com imagens de satelites "
                "de alta resolução com classes pré-anotadas. Resultados em imagens "
                "muito diferentes do conjunto de treino (ex: fotos terrestres) "
                "podem não ser confiáveis e gerar resultados distintos."
            )
            
        except FileNotFoundError:
            st.error(
                f" Modelo não encontrado em `{MODEL_PATH}`. "
                "Certifique-se de treinar o modelo antes (notebook 03)."
            )
        except Exception as e:
            st.error(f" Erro ao processar a imagem: {str(e)}")

# Rodapé
st.divider()
st.caption(
    "🛰️ Imagens recomendadas: vistas aéreas/satelitais de áreas naturais ou rurais. "
)