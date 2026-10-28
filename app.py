
import os
import json
import time
import base64
import io

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from PIL import Image, ImageOps

st.set_page_config(
    page_title="NutriLens - Meal Nutrition Analyzer",
    page_icon="🥗",
    layout="wide",
)

DEFAULT_MODEL = "gemini-3.8-flash"
API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)

st.title("🥗 NutriLens")
st.subheader("AI Meal Nutrition Analyzer")
st.write("Enter a meal description or upload a photo to estimate nutrition.")

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input(
        "Gemini model",
        value=DEFAULT_MODEL,
    ).strip()
    st.caption("Choose a model supported by your API key.")
    st.caption("Nutrition values are estimates only.")


def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    return str(key or os.getenv("GEMINI_API_KEY", "")).strip()


def call_gemini(prompt, api_key, model, image=None):
    parts = [{"text": prompt}]

    if image is not None:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((768, 768))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": encoded,
            }
        })

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "maxOutputTokens": 1200,
        },
    }

    url = API_URL.format(model=model)
    response = None

    for attempt in range(4):
        try:
            response = requests.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                },
                json=payload,
                timeout=(10, 90),
            )

            if response.ok:
                break

            if response.status_code in (408, 429, 500, 502, 503, 504):
                if attempt < 3:
                    delay = 2 ** (attempt + 1)
                    st.warning(
                        f"Gemini is busy. Retrying in {delay} seconds "
                        f"(attempt {attempt + 2}/4)."
                    )
                    time.sleep(delay)
                    continue

            try:
                message = response.json().get(
                    "error", {}
                ).get("message", response.text[:500])
            except ValueError:
                message = response.text[:500]

            if response.status_code == 503:
                raise RuntimeError(
                    "Gemini is overloaded. Wait a few minutes or select "
                    "another supported model. Details: " + message
                )

            if response.status_code == 429:
                raise RuntimeError(
                    "Gemini rate limit or quota reached. Check your API "
                    "quota and billing. Details: " + message
                )

            if response.status_code == 404:
                raise RuntimeError(
                    f"Model '{model}' is unavailable. Choose a supported "
                    "model in the sidebar. Details: " + message
                )

            raise RuntimeError(
                f"Gemini API error {response.status_code}: {message}"
            )

        except requests.exceptions.Timeout as exc:
            if attempt < 3:
                delay = 2 ** (attempt + 1)
                st.warning(f"Request timed out. Retrying in {delay} seconds.")
                time.sleep(delay)
                continue
            raise RuntimeError(
                "Gemini did not respond in time. Try again later."
            ) from exc

        except requests.exceptions.RequestException as exc:
            if attempt < 3:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(f"Connection error: {exc}") from exc

    if response is None or not response.ok:
        raise RuntimeError("Gemini could not complete the request.")

    try:
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini returned no answer.")

        output = "".join(
            part.get("text", "")
            for part in candidates[0].get("content", {}).get("parts", [])
        ).strip()

        if output.startswith("```"):
            output = output.strip("`").strip()
            if output.lower().startswith("json"):
                output = output[4:].strip()

        result = json.loads(output)
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object.")
        return result

    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Could not read Gemini's response. Please try again."
        ) from exc


def analyze_meal(description, api_key, model, image=None):
    prompt = f"""
Estimate the nutrition in the meal described or shown in the image.
Return ONLY valid JSON in this exact structure:
{{
  "meal_name": "Meal name",
  "summary": "Brief explanation and uncertainty",
  "items": [
    {{
      "name": "Food item",
      "portion": "Estimated portion",
      "calories": 100,
      "protein_g": 5,
      "carbs_g": 10,
      "fat_g": 3,
      "fiber_g": 2
    }}
  ],
  "advice": ["Practical nutrition suggestion"],
  "confidence": "Low, Medium, or High"
}}

Rules:
- Nutrition values must be numeric and non-negative.
- Estimate realistic portions.
- Include identifiable foods, drinks, toppings, and sauces.
- Explain uncertainty; do not claim exact accuracy.
- If no food is identifiable, return an empty items list.
- Do not diagnose medical conditions.
- Return JSON only, without Markdown fences.

Meal description:
{description.strip() if description.strip() else "Analyze the attached meal photo."}
"""
    return call_gemini(prompt, api_key, model, image)


api_key = get_api_key()

if not api_key:
    st.warning("Gemini API key is not configured.")
    st.markdown(
        "Get a key at [Google AI Studio]"
        "(https://aistudio.google.com/apikey), then add it to Streamlit Secrets."
    )
    st.code('GEMINI_API_KEY = "YOUR_REAL_API_KEY"', language="toml")
    st.caption("Keep your API key private. Do not commit it to GitHub.")

with st.form("meal_form"):
    description = st.text_area(
        "Describe your meal",
        placeholder=(
            "Example: 2 boiled eggs, 2 slices of bread, "
            "and a cup of tea with milk and sugar."
        ),
        height=120,
    )

    uploaded_file = st.file_uploader(
        "Upload a meal photo (optional)",
        type=["png", "jpg", "jpeg", "webp"],
    )

    submitted = st.form_submit_button(
        "Analyze meal",
        type="primary",
        use_container_width=True,
    )

image = None
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded meal", width=350)
    except Exception:
        st.error("Could not open this image.")
        image = None

if submitted:
    if not api_key:
        st.error("Add your Gemini API key to Streamlit Secrets first.")
    elif not description.strip() and image is None:
        st.error("Enter a meal description or upload a photo.")
    elif not model_name:
        st.error("Enter a Gemini model name in the sidebar.")
    else:
        try:
            with st.spinner("Analyzing your meal..."):
                result = analyze_meal(
                    description,
                    api_key,
                    model_name,
                    image,
                )
                st.session_state["nutrilens_result"] = result
        except Exception as exc:
            st.error(str(exc))

result = st.session_state.get("nutrilens_result")

if result:
    st.divider()
    st.header(result.get("meal_name", "Nutrition results"))
    st.write(result.get("summary", ""))

    items = result.get("items", [])

    if not isinstance(items, list) or not items:
        st.info("No food items identified. Please add more details.")
    else:
        df = pd.DataFrame(items)

        for col in ["name", "portion"]:
            if col not in df.columns:
                df[col] = ""

        numeric_cols = [
            "calories", "protein_g", "carbs_g", "fat_g", "fiber_g"
        ]

        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0)
                .clip(lower=0)
            )

        calories = float(df["calories"].sum())
        protein = float(df["protein_g"].sum())
        carbs = float(df["carbs_g"].sum())
        fat = float(df["fat_g"].sum())
        fiber = float(df["fiber_g"].sum())

        st.subheader("Nutrition totals")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Calories", f"{calories:.0f} kcal")
        c2.metric("Protein", f"{protein:.1f} g")
        c3.metric("Carbs", f"{carbs:.1f} g")
        c4.metric("Fat", f"{fat:.1f} g")
        c5.metric("Fiber", f"{fiber:.1f} g")

        labels = {
            "name": "Food",
            "portion": "Estimated portion",
            "calories": "Calories (kcal)",
            "protein_g": "Protein (g)",
            "carbs_g": "Carbs (g)",
            "fat_g": "Fat (g)",
            "fiber_g": "Fiber (g)",
        }

        report_df = df[list(labels)].rename(columns=labels)

        st.subheader("Nutrition by food item")
        st.dataframe(
            report_df,
            use_container_width=True,
            hide_index=True,
        )

        chart_df = pd.DataFrame({
            "Macronutrient": ["Protein", "Carbohydrates", "Fat"],
            "Grams": [protein, carbs, fat],
        })

        st.subheader("Macronutrient chart")
        st.plotly_chart(
            px.bar(
                chart_df,
                x="Macronutrient",
                y="Grams",
                title="Macronutrients in grams",
            ),
            use_container_width=True,
        )

        advice = result.get("advice", [])
        if isinstance(advice, list) and advice:
            st.subheader("Suggestions")
            for tip in advice:
                st.write(f"- {tip}")

        st.caption(
            f"Confidence: {result.get('confidence', 'Not specified')}. "
            "Values are estimates only."
        )

        csv_data = report_df.to_csv(index=False).encode("utf-8")
        json_data = json.dumps(
            result, indent=2, ensure_ascii=False
        ).encode("utf-8")

        col1, col2 = st.columns(2)

        col1.download_button(
            "Download CSV report",
            data=csv_data,
            file_name="nutrilens_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

        col2.download_button(
            "Download JSON report",
            data=json_data,
            file_name="nutrilens_report.json",
            mime="application/json",
            use_container_width=True,
        )

st.divider()
st.caption(
    "Nutrition estimates are for general information and are not medical advice."
)
