
import io
import json
import os
from datetime import date
from math import isfinite

import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image, ImageOps, UnidentifiedImageError


st.set_page_config(
    page_title="NutriLens - Meal Nutrition Analyzer",
    page_icon="🥗",
    layout="wide",
)

DEFAULT_MODEL = "gemini-2.5-flash"
MAX_IMAGE_BYTES = 10 * 1024 * 1024

NUTRIENTS = [
    "calories", "protein_g", "carbs_g",
    "fat_g", "fiber_g", "sugar_g",
]

LABELS = {
    "calories": "Calories (kcal)",
    "protein_g": "Protein (g)",
    "carbs_g": "Carbohydrates (g)",
    "fat_g": "Fat (g)",
    "fiber_g": "Fiber (g)",
    "sugar_g": "Sugar (g)",
}


def get_setting(name, default=""):
    """Read configuration from Streamlit secrets or environment."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


def get_client():
    api_key = get_setting("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Configure it in "
            ".streamlit/secrets.toml or Streamlit Cloud Secrets."
        )
    return genai.Client(api_key=api_key)


def safe_number(value):
    try:
        number = float(value)
        if not isfinite(number) or number < 0:
            return 0.0
        return round(number, 2)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def normalize_report(data):
    if not isinstance(data, dict):
        raise ValueError("The AI response is not a JSON object.")

    raw_foods = data.get("foods", [])
    if not isinstance(raw_foods, list):
        raw_foods = []

    foods = []
    for item in raw_foods:
        if not isinstance(item, dict):
            continue

        food = {
            "name": str(item.get("name", "Unknown food"))[:120],
            "portion": str(
                item.get("portion", "Portion unknown")
            )[:120],
        }

        for nutrient in NUTRIENTS:
            food[nutrient] = safe_number(item.get(nutrient, 0))

        foods.append(food)

    raw_totals = data.get("totals", {})
    if not isinstance(raw_totals, dict):
        raw_totals = {}

    # Prefer totals calculated from individual food items.
    if foods:
        totals = {
            n: round(sum(f[n] for f in foods), 2)
            for n in NUTRIENTS
        }
    else:
        totals = {
            n: safe_number(raw_totals.get(n, 0))
            for n in NUTRIENTS
        }

    assumptions = data.get("assumptions", [])
    if not isinstance(assumptions, list):
        assumptions = [str(assumptions)]

    return {
        "foods": foods,
        "totals": totals,
        "summary": str(
            data.get("summary", "No summary available.")
        )[:2000],
        "assumptions": [str(x)[:500] for x in assumptions[:10]],
    }


def analyze_meal(description, image, meal_type, model):
    client = get_client()

    prompt = f"""
You are a food nutrition estimation assistant.

Meal category: {meal_type}
User description: {description or "Not provided"}

Analyze the supplied text and/or image. Return valid JSON only,
using this exact structure:

{{
  "foods": [
    {{
      "name": "Food name",
      "portion": "Estimated serving size",
      "calories": 150,
      "protein_g": 5,
      "carbs_g": 20,
      "fat_g": 4,
      "fiber_g": 3,
      "sugar_g": 2
    }}
  ],
  "totals": {{
    "calories": 150,
    "protein_g": 5,
    "carbs_g": 20,
    "fat_g": 4,
    "fiber_g": 3,
    "sugar_g": 2
  }},
  "summary": "Brief nutritional summary",
  "assumptions": ["Explain uncertainties"]
}}

Rules:
- Estimate each identifiable food separately.
- Use realistic approximate portions.
- Do not claim a photo reveals exact weights or hidden ingredients.
- Mention uncertainty about oils, sauces, and preparation.
- Use non-negative numeric nutrient values.
- Totals should approximately equal the sum of food items.
- If the image is unclear, state that limitation.
- Do not give medical diagnoses or invent exact measurements.
- Return JSON without Markdown fences.
"""

    contents = [prompt]

    if image is not None:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        contents.append(
            types.Part.from_bytes(
                data=buffer.getvalue(),
                mime_type="image/jpeg",
            )
        )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    try:
        result = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise ValueError("Gemini returned invalid JSON.") from exc

    return normalize_report(result)


def sum_meals(meals):
    totals = {n: 0.0 for n in NUTRIENTS}

    for meal in meals:
        for nutrient in NUTRIENTS:
            totals[nutrient] += meal["report"]["totals"][nutrient]

    return {n: round(v, 2) for n, v in totals.items()}


def make_csv(meals):
    rows = []

    for meal in meals:
        for food in meal["report"]["foods"]:
            row = {
                "date": meal["date"],
                "meal": meal["meal_type"],
                "food": food["name"],
                "portion": food["portion"],
            }
            for n in NUTRIENTS:
                row[LABELS[n]] = food[n]
            rows.append(row)

    columns = ["date", "meal", "food", "portion"] + [
        LABELS[n] for n in NUTRIENTS
    ]
    return pd.DataFrame(rows, columns=columns).to_csv(index=False)


if "meals" not in st.session_state:
    st.session_state.meals = []

if "latest_report" not in st.session_state:
    st.session_state.latest_report = None


with st.sidebar:
    st.title("🥗 NutriLens")
    st.caption("AI meal nutrition estimates")

    model = st.text_input(
        "Gemini model",
        value=get_setting("GEMINI_MODEL", DEFAULT_MODEL),
        help="Use a model supported by your Gemini API key.",
    ).strip()

    st.metric("Meals in this session", len(st.session_state.meals))

    if st.button("Clear session history"):
        st.session_state.meals = []
        st.session_state.latest_report = None
        st.rerun()

    st.caption("History is temporary and may disappear after a session reset.")


st.title("AI Meal Nutrition Analyzer")
st.write(
    "Describe your meal or upload a photo to estimate calories, "
    "protein, carbohydrates, fats, fiber, and sugar."
)

st.warning(
    "All results are estimates. Photos cannot reliably determine "
    "exact portion weights, ingredients, or cooking oil."
)

analyze_tab, daily_tab, about_tab = st.tabs(
    ["🍽️ Analyze meal", "📊 Daily report", "ℹ️ About"]
)

with analyze_tab:
    with st.form("meal_form"):
        meal_type = st.selectbox(
            "Meal type",
            ["Breakfast", "Lunch", "Dinner", "Snack", "Other"],
        )
        meal_date = st.date_input("Meal date", value=date.today())

        description = st.text_area(
            "What did you eat?",
            placeholder=(
                "Example: Two chapatis, one bowl of daal, "
                "one boiled egg and a cup of tea."
            ),
        )

        uploaded_file = st.file_uploader(
            "Upload a meal photo (optional)",
            type=["jpg", "jpeg", "png"],
        )

        submitted = st.form_submit_button(
            "Analyze nutrition",
            type="primary",
            use_container_width=True,
        )

    image = None

    if uploaded_file is not None:
        if uploaded_file.size > MAX_IMAGE_BYTES:
            st.error("Image must be smaller than 10 MB.")
        else:
            try:
                image = Image.open(uploaded_file)
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((1600, 1600))
                st.image(image, caption="Uploaded meal", width=400)
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("Could not read this image. Try another file.")
                image = None

    if submitted:
        if not description.strip() and image is None:
            st.warning("Describe your meal or upload a valid photo.")
        elif not model:
            st.error("Enter a Gemini model name in the sidebar.")
        else:
            try:
                with st.spinner("Analyzing your meal..."):
                    report = analyze_meal(
                        description=description.strip(),
                        image=image,
                        meal_type=meal_type,
                        model=model,
                    )

                record = {
                    "date": meal_date.isoformat(),
                    "meal_type": meal_type,
                    "description": description.strip(),
                    "report": report,
                }

                st.session_state.meals.append(record)
                st.session_state.latest_report = record
                st.success("Nutrition report generated.")

            except Exception as exc:
                st.error(
                    "Analysis failed. Check your API key, model access, "
                    "internet connection, and API limits."
                )
                st.caption(f"Technical details: {exc}")

    latest = st.session_state.latest_report

    if latest:
        report = latest["report"]
        totals = report["totals"]

        st.divider()
        st.subheader(f"{latest['meal_type']} nutrition report")
        st.caption(f"Date: {latest['date']}")

        cols = st.columns(3)
        cols[0].metric("Calories", f"{totals['calories']:.0f} kcal")
        cols[1].metric("Protein", f"{totals['protein_g']:.1f} g")
        cols[2].metric("Carbs", f"{totals['carbs_g']:.1f} g")

        cols = st.columns(3)
        cols[0].metric("Fat", f"{totals['fat_g']:.1f} g")
        cols[1].metric("Fiber", f"{totals['fiber_g']:.1f} g")
        cols[2].metric("Sugar", f"{totals['sugar_g']:.1f} g")

        if report["foods"]:
            food_df = pd.DataFrame(report["foods"]).rename(
                columns={
                    "name": "Food",
                    "portion": "Estimated portion",
                    "calories": "Calories (kcal)",
                    "protein_g": "Protein (g)",
                    "carbs_g": "Carbs (g)",
                    "fat_g": "Fat (g)",
                    "fiber_g": "Fiber (g)",
                    "sugar_g": "Sugar (g)",
                }
            )

            st.subheader("Food breakdown")
            st.dataframe(food_df, use_container_width=True, hide_index=True)

            macro_df = pd.DataFrame({
                "Nutrient": ["Protein", "Carbohydrates", "Fat"],
                "Grams": [
                    totals["protein_g"],
                    totals["carbs_g"],
                    totals["fat_g"],
                ],
            })

            st.subheader("Macronutrients")
            fig = px.bar(
                macro_df,
                x="Nutrient",
                y="Grams",
                color="Nutrient",
                text="Grams",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Nutrition summary")
        st.write(report["summary"])

        if report["assumptions"]:
            with st.expander("Assumptions and uncertainties"):
                for item in report["assumptions"]:
                    st.write(f"- {item}")

        st.download_button(
            "Download meal report (JSON)",
            data=json.dumps(latest, indent=2, ensure_ascii=False),
            file_name=f"{latest['meal_type'].lower()}_report.json",
            mime="application/json",
        )


with daily_tab:
    selected_date = st.date_input(
        "Choose report date",
        value=date.today(),
        key="daily_report_date",
    )

    daily_meals = [
        m for m in st.session_state.meals
        if m["date"] == selected_date.isoformat()
    ]

    if not daily_meals:
        st.info("No meals recorded for this date in this session.")
    else:
        totals = sum_meals(daily_meals)

        cols = st.columns(3)
        cols[0].metric("Calories", f"{totals['calories']:.0f} kcal")
        cols[1].metric("Protein", f"{totals['protein_g']:.1f} g")
        cols[2].metric("Carbohydrates", f"{totals['carbs_g']:.1f} g")

        cols = st.columns(3)
        cols[0].metric("Fat", f"{totals['fat_g']:.1f} g")
        cols[1].metric("Fiber", f"{totals['fiber_g']:.1f} g")
        cols[2].metric("Sugar", f"{totals['sugar_g']:.1f} g")

        meal_rows = []
        for meal in daily_meals:
            row = {"Meal": meal["meal_type"]}
            row.update({
                LABELS[n]: meal["report"]["totals"][n]
                for n in NUTRIENTS
            })
            meal_rows.append(row)

        st.subheader("Meal summary")
        st.dataframe(
            pd.DataFrame(meal_rows),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download daily report (CSV)",
            data=make_csv(daily_meals),
            file_name=f"nutrition_{selected_date.isoformat()}.csv",
            mime="text/csv",
        )

        daily_report = {
            "date": selected_date.isoformat(),
            "meals": daily_meals,
            "totals": totals,
        }

        st.download_button(
            "Download daily report (JSON)",
            data=json.dumps(daily_report, indent=2, ensure_ascii=False),
            file_name=f"nutrition_{selected_date.isoformat()}.json",
            mime="application/json",
        )


with about_tab:
    st.subheader("About NutriLens")
    st.markdown(
        """
        - Enter a meal description, upload a photo, or use both.
        - Gemini Flash identifies foods and estimates portions.
        - The app displays nutrition totals and charts.
        - Add multiple meals to create a daily report.

        **Limitations:** Estimates can be inaccurate when portions,
        ingredients, sauces, or cooking methods are unknown.
        This is not a medical diagnostic tool.
        """
    )

st.caption(
    "NutriLens | AI-generated estimates | Verify important "
    "dietary decisions with reliable nutrition information."
)
