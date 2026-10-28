
import json
import os
import time
import base64
import io

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(
    page_title="NutriLens",
    page_icon="🥗",
    layout="wide",
)

DEFAULT_MODEL = "gemini-3.8-flash"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

st.title("🥗 NutriLens")
st.subheader("AI Meal Nutrition Analyzer")
st.write("Enter a meal description or upload a meal photo.")

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("Gemini model", value=DEFAULT_MODEL).strip()
    st.caption("Use a model supported by your Gemini API key.")
    st.caption("Nutrition values are estimates, not medical advice.")


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
        image.save(buffer, format="JPEG", quality=82, optimize=True)
        encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")

        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": encoded_image,
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
                    wait = 2 ** (attempt + 1)
                    st.warning(
                        f"Gemini is temporarily busy. "
                        f"Retrying in {wait} seconds "
                        f"(attempt {attempt + 2}/4)."
                    )
                    time.sleep(wait)
                    continue

            try:
                details = response.json().get("error", {}).get(
                    "message", response.text[:500]
                )
            except ValueError:
                details = response.text[:500]

            if response.status_code == 503:
                raise RuntimeError(
                    "Gemini is overloaded after several attempts. "
                    "Wait a few minutes or try another supported model. "
                    + details
                )

            if response.status_code == 429:
                raise RuntimeError(
                    "Gemini rate limit or quota reached. Check your API "
                    "quota and billing. " + details
                )

            if response.status_code == 404:
                raise RuntimeError(
                    f"The model '{model}' is unavailable or not found. "
                    "Choose a supported model in the sidebar. " + details
                )

            raise RuntimeError(
                f"Gemini API returned HTTP {response.status_code}: {details}"
            )

        except requests.exceptions.Timeout as exc:
            if attempt < 3:
                wait = 2 ** (attempt + 1)
                st.warning(
                    f"Request timed out. Retrying in {wait} seconds."
                )
                time.sleep(wait)
                continue

            raise RuntimeError(
                "Gemini did not respond in time. Please try again later."
            ) from exc

        except requests.exceptions.RequestException as exc:
            if attempt < 3:
                time.sleep(2 ** (attempt + 1))
                continue

            raise RuntimeError(
                f"Could not connect to Gemini: {exc}"
            ) from exc

    if response is None or not response.ok:
        raise RuntimeError("Gemini could not complete the request.")

    try:
        data = response.json()
        output = "".join(
            part.get("text", "")
            for part in data["candidates"][0]["content"]["parts"]
        ).strip()

        if output.startswith("```"):
            output = output.strip("`").strip()
            if output.lower().startswith("json"):
                output = output[4:].strip()

        return json.loads(output)

    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Gemini returned an unreadable response. Please try again."
        ) from exc


def analyze_meal(description, api_key, model, image=None):
    prompt = f"""
Estimate the nutrition of this meal.

Return ONLY valid JSON in this format:
{{
  "meal_name": "Meal name",
  "summary": "Brief explanation of the meal",
  "items": [
    {{
      "name": "Food name",
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
- Use realistic nutrition estimates.
- Use numeric, non-negative nutrition values.
- Include each identifiable food and drink.
- State portion-size assumptions.
- Do not claim estimates are exact.
- If the meal is unclear, explain the uncertainty.
- Return an empty items list if no food can be identified.

User's meal description:
{description or "Analyze the attached meal photo."}
"""
    return call_gemini(prompt, api_key, model, image)


api_key = get_api_key()

if not api_key:
    st.warning("Your Gemini API key is not configured.")
    st.markdown(
        "Create a key at "
        "[Google AI Studio](https://aistudio.google.com/apikey)."
    )
    st.write("Then add it to Streamlit Secrets:")
    st.code('GEMINI_API_KEY = "your_actual_api_key"', language="toml")

description = st.text_area(
    "Describe your meal",
    placeholder=(
        "Example: 2 boiled eggs, 2 slices of whole-wheat bread, "
        "and a cup of tea with milk and sugar."
    ),
    height=110,
)

uploaded_file = st.file_uploader(
    "Or upload a meal photo (optional)",
    type=["png", "jpg", "jpeg", "webp"],
)

image = None

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded meal", width=320)
    except Exception:
        st.error("Could not open this image. Please upload another one.")
        image = None

if st.button("Analyze meal", type="primary", use_container_width=True):
    if not api_key:
        st.error("Configure your Gemini API key first.")

    elif not description.strip() and image is None:
        st.error("Enter a meal description or upload a photo.")

    elif not model_name:
        st.error("Enter a Gemini model name in the sidebar.")

    else:
        try:
            with st.spinner("Analyzing your meal..."):
                result = analyze_meal(
                    description.strip(),
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
        st.info("No food items were identified. Add more meal details.")

    else:
        df = pd.DataFrame(items)

        numeric_columns = [
            "calories", "protein_g", "carbs_g", "fat_g", "fiber_g"
        ]

        for column in numeric_columns:
            if column not in df.columns:
                df[column] = 0

            df[column] = (
                pd.to_numeric(df[column], errors="coerce")
                .fillna(0)
                .clip(lower=0)
            )

        for column in ["name", "portion"]:
            if column not in df.columns:
                df[column] = ""

        calories = df["calories"].sum()
        protein = df["protein_g"].sum()
        carbs = df["carbs_g"].sum()
        fat = df["fat_g"].sum()
        fiber = df["fiber_g"].sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Calories", f"{calories:.0f} kcal")
        c2.metric("Protein", f"{protein:.1f} g")
        c3.metric("Carbs", f"{carbs:.1f} g")
        c4.metric("Fat", f"{fat:.1f} g")

        st.subheader("Nutrition by food item")

        columns = {
            "name": "Food",
            "portion": "Estimated portion",
            "calories": "Calories (kcal)",
            "protein_g": "Protein (g)",
            "carbs_g": "Carbs (g)",
            "fat_g": "Fat (g)",
            "fiber_g": "Fiber (g)",
        }

        report_df = df[list(columns)].rename(columns=columns)

        st.dataframe(
            report_df,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Macronutrients")

        chart_df = pd.DataFrame({
            "Macronutrient": ["Protein", "Carbohydrates", "Fat"],
            "Grams": [protein, carbs, fat],
        })

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

        if advice:
            st.subheader("Suggestions")
            for tip in advice:
                st.write(f"- {tip}")

        st.caption(
            f"Confidence: {result.get('confidence', 'Unspecified')}. "
            "Nutrition figures are estimates only."
        )

        csv_data = report_df.to_csv(index=False).encode("utf-8")
        json_data = json.dumps(
            result, indent=2, ensure_ascii=False
        ).encode("utf-8")

        d1, d2 = st.columns(2)

        d1.download_button(
            "Download CSV report",
            data=csv_data,
            file_name="nutrilens_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

        d2.download_button(
            "Download JSON report",
            data=json_data,
            file_name="nutrilens_report.json",
            mime="application/json",
            use_container_width=True,
        )
