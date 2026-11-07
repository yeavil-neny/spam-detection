"""
Aplicativo web para la detección de spam en mensajes.

Integra el modelo preentrenado Goodmotion/spam-mail-classifier
(Hugging Face), un codificador Multilingual-MiniLM ajustado (fine-tuned)
para clasificar textos en dos clases: NOSPAM y SPAM.

El usuario ingresa un mensaje en el formulario y la aplicación devuelve
la etiqueta estimada junto con el nivel de confianza del modelo.

Proyecto: Detección de Spam
Integrantes del grupo: Fredy Alvarez, Sebastian Orozco, Cristian Yepes,
Yenny Villarreal.
"""

from flask import Flask, render_template, request
from transformers import pipeline

# Identificador del modelo preentrenado en Hugging Face.
MODEL_NAME = "Goodmotion/spam-mail-classifier"

# Longitud máxima aceptada de un mensaje, para evitar peticiones con
# entradas excesivamente largas que degradan el tiempo de respuesta.
MAX_TEXT_LENGTH = 5000

app = Flask(__name__)

# El pipeline se instancia una sola vez y se reutiliza en cada petición,
# dado que la carga del modelo (pesos + tokenizador) es la parte más lenta.
_CLASSIFIER = None


def get_classifier():
    """Carga el pipeline de clasificación la primera vez que se necesita."""
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _CLASSIFIER = pipeline("text-classification", model=MODEL_NAME)
    return _CLASSIFIER


def predict_message(text):
    """
    Clasifica un mensaje de texto y entrega un diccionario con:
    spam, score, model y texto. La etiqueta del modelo se normaliza a
    los valores legibles SPAM / NO SPAM.
    """
    text = (text or "").strip()
    if not text:
        return {
            "spam": False,
            "score": 0.0,
            "model": MODEL_NAME,
            "text": "",
            "error": "El mensaje está vacío. Escriba un texto antes de analizarlo.",
        }

    # Truncado defensivo; el tokenizador del modelo también recorta a
    # 128 tokens, pero se limita la entrada para no enviar texto de más.
    text = text[:MAX_TEXT_LENGTH]

    result = get_classifier()(text, truncation=True)[0]

    # El repositorio publica las etiquetas SPAM (índice 1) y NOSPAM
    # (índice 0). Para hacer el código más robusto también se aceptan
    # las variantes LABEL_1 / LABEL_0 que genera transformers cuando el
    # mapeo id2label no está disponible.
    label = result["label"].upper().strip()
    is_spam = label in {"SPAM", "1", "LABEL_1"}
    return {
        "spam": is_spam,
        "score": float(result["score"]),
        "model": MODEL_NAME,
        "text": text,
        "error": None,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    """Página principal: formulario de análisis y presentación del resultado."""
    result = None
    if request.method == "POST":
        raw_text = request.form.get("message", "")
        result = predict_message(raw_text)
    return render_template("index.html", result=result)


if __name__ == "__main__":
    # debug=False para uso general; en desarrollo se puede poner en True.
    app.run(host="0.0.0.0", port=5000, debug=False)
