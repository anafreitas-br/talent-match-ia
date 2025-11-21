import os
import json
import re
from io import BytesIO

from flask import Flask, request, Response
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
from openai import OpenAI
import joblib
import pdfplumber



load_dotenv()

app = Flask(__name__)
CORS(app, origins="*")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    pipeline_clf = joblib.load("pipeline_TalentMatch_Classifier.joblib")
    print(" Pipeline DE CLASSIFICAÇÃO carregado!")
except Exception as e:
    print(" Erro ao carregar pipeline:", e)
    pipeline_clf = None


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    def fix(b):
        b = re.sub(r"\n+", " ", b)
        b = re.sub(r"[ ]{2,}", " ", b)
        return b.strip()

    return "\n\n".join(fix(b) for b in text.split("\n\n")).strip()


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        txt = ""
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                txt += (page.extract_text() or "") + "\n"
        return clean_text(txt)
    except:
        return ""



def classificar_nivel(score_percent: float) -> str:
    if score_percent >= 70:
        return "Alta compatibilidade"
    elif score_percent >= 50:
        return "Média compatibilidade"
    else:
        return "Baixa compatibilidade"


def resposta_json(data, status=200):
    return Response(
        json.dumps(data, ensure_ascii=False, indent=2),
        content_type="application/json; charset=utf-8",
        status=status
    )


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return resposta_json({
        "status": "API TalentMatch CLASSIFIER Online",
        "rotas": ["/predict-score", "/analyze-fit", "/analyze"]
    })



@app.route("/predict-score", methods=["POST"])
@cross_origin()
def predict_score():
    if not pipeline_clf:
        return resposta_json({"error": "Pipeline não carregado."}, 500)

    data = request.json or {}
    texto = data.get("text", "")

    if not texto:
        return resposta_json({"error": "Envie o campo 'text'."}, 400)


    probs = pipeline_clf.predict_proba([texto])[0]
    classe = int(probs.argmax())
    score_percent = round(max(probs) * 100, 2)

    return resposta_json({
        "classe_predita": classe,
        "nivel_compatibilidade": classificar_nivel(classe),
        "score_percent": score_percent,
        "probs": probs.tolist()
    })


@app.route("/analyze-fit", methods=["POST"])
@cross_origin()
def analyze_fit():
    data = request.json or {}
    curriculo = data.get("curriculo", "")
    vaga = data.get("vaga", "")

    if not curriculo or not vaga:
        return resposta_json({"error": "Envie 'curriculo' e 'vaga'."}, 400)

    texto_completo = curriculo + "\n\n" + vaga


    probs = pipeline_clf.predict_proba([texto_completo])[0]
    classe = int(probs.argmax())
    score_percent = round(max(probs) * 100, 2)
    nivel = classificar_nivel(classe)


    prompt = f"""
Você é um especialista em RH Senior.

Retorne somente um JSON, no resumo_final retorne um resumo do curriculo com suas pralavras resumido e o seu parecer, no formato:

{{
  "nome": "Nome do candidato",
  "score_percent": {score_percent},
  "nivel_compatibilidade": "{nivel}",
  "pontos_fortes": "...",
  "pontos_a_melhorar_riscos": "...",
  "recomendacao": "...",
  "resumo_final": "..."
}}

CURRÍCULO:
{curriculo}

VAGA:
{vaga}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )

    analise = json.loads(resp.choices[0].message.content)

    analise["score_percent"] = score_percent
    analise["nivel_compatibilidade"] = nivel

    return resposta_json(analise)


@app.route("/analyze", methods=["POST"])
@cross_origin()
def analyze():
    if "file" not in request.files:
        return resposta_json({"error": "Envie um PDF."}, 400)

    pdf_file = request.files["file"]
    vaga = request.form.get("vaga", "")

    texto_pdf = extract_text_from_pdf(pdf_file.read())

    texto_completo = texto_pdf + "\n\n" + vaga

    probs = pipeline_clf.predict_proba([texto_completo])[0]
    classe = int(probs.argmax())
    score_percent = round(max(probs) * 100, 2)
    nivel = classificar_nivel(classe)

    prompt = f"""
Você é um especialista em RH Senior.

Retorne somente um JSON, no resumo_final retorne um resumo do curriculo com suas pralavras resumido e o seu parecer, no formato:


{{
  "nome": "Nome do candidato",
  "score_percent": {score_percent},
  "nivel_compatibilidade": "{nivel}",
  "pontos_fortes": "...",
  "pontos_a_melhora_riscos": "...",
  "recomendacao": "...",
  "resumo_final": "..."
}}

CURRÍCULO:
{texto_pdf}

VAGA:
{vaga}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )

    analise = json.loads(resp.choices[0].message.content)

    analise["score_percent"] = score_percent
    analise["nivel_compatibilidade"] = nivel

    return resposta_json(analise)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
