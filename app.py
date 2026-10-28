
import os
import json
import base64
import io
from datetime import datetime, date

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    import plotly.express as px
except ImportError:
    px = None


st.set_page_config(
    page_title="NutriLens | AI Meal Nutrition Analyzer",
    page_icon="🥗",
    layout="wide",
)

API_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/{model}:generateContent"
)
DEFAULT_MODEL = "gemini-2.5-flash"

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.45rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        color: #64748b;
        margin-bottom: 1.4rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(120,120,120,0.08);
        padding: 12px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_api_key():
    """Get the Gemini API key from Streamlit secrets or environment."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY", "").strip()


def call_gemini_rest(
    api_key,
    model,
    prompt,
    image_bytes=None,
    mime_type=None,
):
    """Call Gemini directly over HTTPS without the Google SDK."""
    parts = [{"text": prompt}]

    if image_bytes:
        parts.append(
            {
                "inline_data": {
                    "mime_type": mime_type or "image/jpeg",
                    "data": base64.b64encode(
                        image_bytes
                    ).decode("utf-8"),
                }
            }
        )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }

    url = API_URL_TEMPLATE.format(model=model.strip())

    try:
        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            json=payload,
            timeout=90,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not connect to Gemini API: {exc}"
        ) from exc

    if not response.ok:
        try:
            details = response.json().get(
                "error", {}
            ).get("message", response.text)
        except Exception:
            details = response.text

        raise RuntimeError(
            f"Gemini API returned HTTP "
            f"{response.status_code}: {details}"
        )

    try:
        data = response.json()

        text_parts = [
            part.get("text", "")
            for candidate in data.get("candidates", [])
            for part in candidate.get(
                "content", {}
            ).get("parts", [])
            if part.get("text")
        ]

        if not text_parts:
            reason = (
                data.get("promptFeedback", {}).get(
                    "blockReason"
                )
                or "The model returned no text."
            )
            raise RuntimeError(
                f"No usable response from Gemini: {reason}"
            )

        raw_text = "\n".join(text_parts).strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").strip()
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:].strip()

        return json.loads(raw_text)

    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "Gemini returned a response that could not "
            "be read as JSON. Please try again."
        ) from exc


def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def normalize_food_items(items):
    normalized = []

    if not isinstance(items, list):
        return normalized

    for item in items:
        if not isinstance(item, dict):
            continue

        normalized.append(
            {
                "food": str(
                    item.get(
                        "food",
                        item.get("name", "Unknown food"),
                    )
                ),
                "portion": str(
                    item.get("portion", "Not specified")
                ),
                "calories": max(
                    0.0, safe_float(item.get("calories", 0))
                ),
                "protein_g": max(
                    0.0,
                    safe_float(
                        item.get(
                            "protein_g",
                            item.get("protein", 0),
                        )
                    ),
                ),
                "carbs_g": max(
                    0.0,
                    safe_float(
                        item.get(
                            "carbs_g",
                            item.get("carbs", 0),
                        )
                    ),
                ),
                "fat_g": max(
                    0.0,
                    safe_float(
                        item.get(
                            "fat_g",
                            item.get("fat", 0),
                        )
                    ),
                ),
                "fiber_g": max(
                    0.0,
                    safe_float(
                        item.get(
                            "fiber_g",
                            item.get("fiber", 0),
                        )
                    ),
                ),
            }
        )

    return normalized


def analyze_meal(
    api_key,
    model,
    description,
    image_bytes=None,
    mime_type=None,
):
    prompt = f"""
You are a careful food and nutrition estimation assistant.

Meal description:
{description or "No text description supplied."}

If an image is supplied, identify visible foods and estimate
reasonable portions.

Return ONLY valid JSON with this structure:
{{
  "meal_name": "short descriptive meal name",
  "summary": "one or two sentence overview",
  "items": [
    {{
      "food": "food name",
      "portion": "estimated serving size",
      "calories": 0,
      "protein_g": 0,
      "carbs_g": 0,
      "fat_g": 0,
      "fiber_g": 0
    }}
  ],
  "advice": ["short practical nutrition note"],
  "confidence": "low, medium, or high"
}}

Use numeric values for nutrients. Include each visible or
mentioned food. Give plausible estimates, not falsely precise
values. If the image is unclear, explain that in the summary
and use low confidence.

This is an estimate, not medical advice.
"""

    result = call_gemini_rest(
        api_key,
        model,
        prompt,
        image_bytes,
        mime_type,
    )

    if not isinstance(result, dict):
        raise RuntimeError(
            "The API response had an unexpected format."
        )

    items = normalize_food_items(result.get("items", []))

    if not items:
        raise RuntimeError(
            "No food items were returned. Try a clearer photo "
            "or add a meal description."
        )

    totals = {
        "calories": sum(x["calories"] for x in items),
        "protein_g": sum(x["protein_g"] for x in items),
        "carbs_g": sum(x["carbs_g"] for x in items),
        "fat_g": sum(x["fat_g"] for x in items),
        "fiber_g": sum(x["fiber_g"] for x in items),
    }

    return {
        "meal_name": str(
            result.get("meal_name", "Meal analysis")
        ),
        "summary": str(result.get("summary", "")),
        "items": items,
        "totals": totals,
        "advice": (
            result.get("advice", [])
            if isinstance(result.get("advice", []), list)
            else []
        ),
        "confidence": str(
            result.get("confidence", "medium")
        ),
        "date": date.today().isoformat(),
        "time": datetime.now().strftime("%H:%M"),
        "description": description,
    }


def item_dataframe(items):
    return pd.DataFrame(
        items,
        columns=[
            "food",
            "portion",
            "calories",
            "protein_g",
            "carbs_g",
            "fat_g",
            "fiber_g",
        ],
    )


def meal_record_for_export(meal):
    return {
        "meal_name": meal.get("meal_name", ""),
        "date": meal.get("date", ""),
        "time": meal.get("time", ""),
        "summary": meal.get("summary", ""),
        "confidence": meal.get("confidence", ""),
        "totals": meal.get("totals", {}),
        "items": meal.get("items", []),
        "advice": meal.get("advice", []),
    }


if "meal_history" not in st.session_state:
    st.session_state.meal_history = []

if "latest_meal" not in st.session_state:
    st.session_state.latest_meal = None


# Sidebar
with st.sidebar:
    st.title("🥗 NutriLens")
    st.caption("AI Meal Nutrition Analyzer")
    st.divider()

    model_name = st.text_input(
        "Gemini model",
        value=DEFAULT_MODEL,
        help="Default: gemini-2.5-flash",
    )

    st.caption(
        "The API key is read from Streamlit secrets "
        "or the GEMINI_API_KEY environment variable."
    )

    with st.expander("API key setup"):
        st.markdown(
            "In Streamlit Cloud, open **Manage app → "
            "Settings → Secrets** and add:"
        )
        st.code(
            'GEMINI_API_KEY = "your_api_key_here"',
            language="toml",
        )
        st.markdown(
            "Keep your API key private. Do not put it "
            "in app.py or publish it on GitHub."
        )

    st.divider()
    st.caption(
        "Nutrition values are AI estimates and may be inaccurate."
    )


st.markdown(
    '<div class="main-title">🥗 NutriLens</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Estimate meal calories, macros, '
    'and nutrition from a photo or description.</div>',
    unsafe_allow_html=True,
)

analysis_tab, report_tab, about_tab = st.tabs(
    ["🍽️ Analyze a meal", "📊 Daily report", "ℹ️ About"]
)


# Analyze meal
with analysis_tab:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("1. Add your meal")

        uploaded_file = st.file_uploader(
            "Upload a meal photo",
            type=["jpg", "jpeg", "png", "webp"],
            help="Use a clear photo with the food visible.",
        )

        preview_bytes = None
        preview_mime = None

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                image = ImageOps.exif_transpose(image).convert("RGB")

                st.image(
                    image,
                    caption="Uploaded meal",
                    use_container_width=True,
                )

                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=90)
                preview_bytes = buffer.getvalue()
                preview_mime = "image/jpeg"

            except (UnidentifiedImageError, OSError):
                st.error(
                    "This image could not be opened. "
                    "Please upload a JPG, PNG, or WEBP image."
                )

        description = st.text_area(
            "Describe the meal (optional)",
            placeholder=(
                "Example: grilled chicken, rice, salad, "
                "and a glass of water"
            ),
            height=110,
        )

        st.caption(
            "For better estimates, mention portion sizes, "
            "cooking method, sauces, and drinks."
        )

        analyze_clicked = st.button(
            "✨ Analyze nutrition",
            type="primary",
            use_container_width=True,
        )

    with right:
        st.subheader("2. Nutrition results")

        if analyze_clicked:
            api_key = get_api_key()

            if not api_key:
                st.error(
                    "Gemini API key not found. Add GEMINI_API_KEY "
                    "in Streamlit Cloud → Manage app → "
                    "Settings → Secrets."
                )

            elif not description.strip() and preview_bytes is None:
                st.warning(
                    "Upload a meal photo or enter a meal description first."
                )

            else:
                with st.spinner("Analyzing your meal..."):
                    try:
                        meal = analyze_meal(
                            api_key=api_key,
                            model=model_name or DEFAULT_MODEL,
                            description=description.strip(),
                            image_bytes=preview_bytes,
                            mime_type=preview_mime,
                        )

                        st.session_state.latest_meal = meal
                        st.session_state.meal_history.append(meal)

                    except Exception as exc:
                        st.error(str(exc))

        meal = st.session_state.latest_meal

        if meal:
            st.markdown(f"### {meal['meal_name']}")

            if meal.get("summary"):
                st.write(meal["summary"])

            totals = meal["totals"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Calories", f"{totals['calories']:.0f} kcal")
            m2.metric("Protein", f"{totals['protein_g']:.1f} g")
            m3.metric("Carbs", f"{totals['carbs_g']:.1f} g")
            m4.metric("Fat", f"{totals['fat_g']:.1f} g")

            st.caption(
                "Estimate confidence: "
                f"{meal.get('confidence', 'medium').capitalize()}"
            )

            st.markdown("**Food breakdown**")

            df = item_dataframe(meal["items"])

            display_df = df.rename(
                columns={
                    "food": "Food",
                    "portion": "Estimated portion",
                    "calories": "Calories (kcal)",
                    "protein_g": "Protein (g)",
                    "carbs_g": "Carbs (g)",
                    "fat_g": "Fat (g)",
                    "fiber_g": "Fiber (g)",
                }
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

            macro_df = pd.DataFrame(
                {
                    "Macro": ["Protein", "Carbs", "Fat"],
                    "Grams": [
                        totals["protein_g"],
                        totals["carbs_g"],
                        totals["fat_g"],
                    ],
                }
            )

            if px is not None:
                fig = px.pie(
                    macro_df,
                    names="Macro",
                    values="Grams",
                    hole=0.45,
                    title="Macronutrient split (by grams)",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(macro_df.set_index("Macro"))

            if meal.get("advice"):
                st.markdown("**Nutrition notes**")
                for tip in meal["advice"]:
                    st.write(f"• {tip}")

            st.download_button(
                "Download this meal as JSON",
                data=json.dumps(
                    meal_record_for_export(meal),
                    indent=2,
                    ensure_ascii=False,
                ),
                file_name="nutrilens_meal_report.json",
                mime="application/json",
                use_container_width=True,
            )

        else:
            st.info(
                "Your results will appear here after you analyze a meal."
            )


# Daily report
with report_tab:
    st.subheader("Daily nutrition report")

    history = st.session_state.meal_history

    if not history:
        st.info("Analyze at least one meal to build your report.")

    else:
        report_date = st.date_input(
            "Report date",
            value=date.today(),
        )

        daily_meals = [
            meal
            for meal in history
            if meal.get("date") == report_date.isoformat()
        ]

        if not daily_meals:
            st.warning(
                "No analyzed meals for this date in the current session."
            )

        else:
            totals = {
                key: sum(
                    meal.get("totals", {}).get(key, 0)
                    for meal in daily_meals
                )
                for key in [
                    "calories",
                    "protein_g",
                    "carbs_g",
                    "fat_g",
                    "fiber_g",
                ]
            }

            st.write(f"Meals analyzed: **{len(daily_meals)}**")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Calories", f"{totals['calories']:.0f} kcal")
            c2.metric("Protein", f"{totals['protein_g']:.1f} g")
            c3.metric("Carbs", f"{totals['carbs_g']:.1f} g")
            c4.metric("Fat", f"{totals['fat_g']:.1f} g")

            summary_rows = []

            for meal in daily_meals:
                summary_rows.append(
                    {
                        "Time": meal.get("time", ""),
                        "Meal": meal.get("meal_name", ""),
                        "Calories (kcal)": round(
                            meal["totals"]["calories"]
                        ),
                        "Protein (g)": round(
                            meal["totals"]["protein_g"], 1
                        ),
                        "Carbs (g)": round(
                            meal["totals"]["carbs_g"], 1
                        ),
                        "Fat (g)": round(
                            meal["totals"]["fat_g"], 1
                        ),
                        "Fiber (g)": round(
                            meal["totals"]["fiber_g"], 1
                        ),
                    }
                )

            summary_df = pd.DataFrame(summary_rows)

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download daily report (CSV)",
                data=summary_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=(
                    f"nutrilens_daily_report_"
                    f"{report_date.isoformat()}.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

            daily_json = {
                "date": report_date.isoformat(),
                "daily_totals": totals,
                "meals": [
                    meal_record_for_export(meal)
                    for meal in daily_meals
                ],
            }

            st.download_button(
                "Download daily report (JSON)",
                data=json.dumps(
                    daily_json,
                    indent=2,
                    ensure_ascii=False,
                ),
                file_name=(
                    f"nutrilens_daily_report_"
                    f"{report_date.isoformat()}.json"
                ),
                mime="application/json",
                use_container_width=True,
            )

            if st.button(
                "Clear meal history for this session",
                type="secondary",
            ):
                st.session_state.meal_history = []
                st.session_state.latest_meal = None
                st.rerun()


# About
with about_tab:
    st.subheader("About NutriLens")

    st.write(
        "NutriLens uses Google's Gemini API to estimate foods and "
        "nutrition from a meal photo or text description. The app "
        "calls Gemini's HTTPS REST endpoint directly using Python "
        "requests, so it does not import google.genai or require "
        "the google-genai package."
    )

    st.warning(
        "Nutrition estimates are approximate. Photo-based estimates "
        "can be wrong, especially for hidden ingredients, oil, sauces, "
        "and portion sizes. Do not use this app as a substitute for "
        "professional medical or dietary advice."
    )

    st.markdown(
        "For details, see the "
        "[official Gemini API documentation](https://ai.google.dev/api)."
    )
