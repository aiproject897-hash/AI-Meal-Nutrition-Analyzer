
# 🥗 NutriLens — AI Meal Nutrition Analyzer

NutriLens is a Streamlit web app that uses Google's Gemini Flash
model to estimate the nutritional content of meals.

Users can describe what they ate, upload a meal photo, or provide
both. The app generates a nutrition report and can combine multiple
meal reports into a daily summary.

## Features

- Analyze breakfast, lunch, dinner, snacks, and other meals.
- Accept text descriptions and JPG, JPEG, or PNG images.
- Estimate calories, protein, carbohydrates, fats, fiber, and sugar.
- Display a food-by-food nutrition breakdown.
- Visualize macronutrients with a chart.
- Build daily nutrition summaries from multiple meals.
- Download individual reports as JSON.
- Download daily reports as CSV or JSON.
- Keep the Gemini API key outside the source code.

## Technology stack

- Python
- Streamlit
- Google Gen AI Python SDK
- Gemini Flash
- Pandas
- Plotly
- Pillow

## Project structure

    meal-nutrition-analyzer/
    ├── app.py
    ├── README.md
    └── requirements.txt

## Requirements

- Python 3.10 or a newer Python version supported by the
  selected dependencies.
- A Google Gemini API key.
- An internet connection for Gemini API requests.

## 1. Get a Gemini API key

Visit Google AI Studio:

https://aistudio.google.com/

Create or obtain an API key and confirm that your account has access
to the selected Gemini Flash model.

Never publish your API key in your source code or GitHub repository.

## 2. Download or clone the project

If you have Git installed:

    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
    cd YOUR_REPOSITORY

Replace the placeholders with your actual GitHub username and
repository name.

Alternatively, download the repository ZIP from GitHub and extract it.

## 3. Create a virtual environment

Windows:

    python -m venv .venv
    .venv\Scripts\activate

macOS or Linux:

    python3 -m venv .venv
    source .venv/bin/activate

## 4. Install dependencies

    python -m pip install --upgrade pip
    pip install -r requirements.txt

## 5. Configure the API key locally

Create this file:

    .streamlit/secrets.toml

Put the following inside it:

    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

Optionally, select a different Gemini model:

    GEMINI_MODEL = "gemini-2.5-flash"

Create the .streamlit directory if it does not already exist.

Do not commit secrets.toml to GitHub.

## 6. Run the application

From the project root:

    streamlit run app.py

Streamlit will show a local address, usually:

    http://localhost:8501

Open that address in your browser.

## 7. How to use the app

1. Select the meal category.
2. Choose the date of the meal.
3. Enter a food description, upload a photo, or do both.
4. Click Analyze nutrition.
5. Review the estimated nutrient totals and food breakdown.
6. Download the individual report if required.
7. Analyze more meals and open the Daily report tab.

For more useful estimates, include portion sizes and preparation
details. For example:

"Two medium chapatis, one bowl of daal, one boiled egg,
and one cup of tea with milk and one teaspoon of sugar."

## 8. Push the project to GitHub

1. Sign in to GitHub.
2. Create a new repository.
3. Give it a name such as meal-nutrition-analyzer.
4. Choose Public or Private.
5. Create the repository.
6. Add app.py, README.md, and requirements.txt.
7. Commit the files to the main branch.

For the first upload, you can use GitHub's website:

- Open your repository.
- Click Add file, then Upload files.
- Drag the three project files into the upload area.
- Enter a commit message.
- Click Commit changes.

Make sure you have not uploaded your API key or secrets.toml.

## 9. Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io/
2. Sign in using your GitHub account.
3. Click Create app.
4. Select your repository and the main branch.
5. Set the main file path to app.py.
6. Open Advanced settings.
7. Select a compatible Python version.
8. Add the secret below in the Secrets field:

       GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
       GEMINI_MODEL = "gemini-2.5-flash"

9. Click Save or Deploy.
10. Wait for the app to build and start.
11. Open the generated app URL and test a meal.

If deployment fails, open the app's logs and check for missing
packages, invalid secrets, unsupported model names, or API errors.

Official deployment guide:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## 10. Testing checklist

Before sharing the app, verify:

- [ ] The application starts without import errors.
- [ ] Missing API keys produce a readable error.
- [ ] Text-only meal analysis works.
- [ ] Image-only meal analysis works.
- [ ] Text and image together work.
- [ ] Invalid and oversized images are handled.
- [ ] Nutrition values appear in the report.
- [ ] Multiple meals contribute to daily totals.
- [ ] CSV and JSON downloads open correctly.
- [ ] The deployed app can access the Gemini API.
- [ ] No API keys are present in GitHub source files.

## Important limitations

Nutrition values are estimates, not laboratory measurements.
A photo cannot reliably reveal exact food weights, ingredients,
cooking oil, or recipe details.

This first version stores meal history in Streamlit session state.
It does not provide permanent user accounts or a database. The log
may disappear when the session resets.

The app is not a medical diagnostic tool. Users with medical dietary
requirements should verify important nutrition information using
reliable nutrition databases or a qualified professional.

## Future improvements

- Add a nutrition database for more consistent nutrient calculations.
- Let users correct detected foods and serving sizes.
- Add permanent meal history with a database.
- Support user accounts and personalized daily targets.
- Generate PDF reports.
- Add micronutrients such as sodium, potassium, calcium, and iron
  when supported by reliable data.
