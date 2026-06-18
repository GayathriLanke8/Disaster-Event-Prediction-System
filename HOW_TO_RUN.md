# HOW TO RUN — DisasterSense AI

## STEP 1 — Open the project folder in VS Code

File → Open Folder → select the DisasterSense-AI folder


## STEP 2 — Open terminal in VS Code

Press:
```
Ctrl + `
```


## STEP 3 — Create virtual environment

```bash
python -m venv venv
```


## STEP 4 — Activate virtual environment

Windows:
```bash
venv\Scripts\activate
```

Mac / Linux:
```bash
source venv/bin/activate
```

You will see (venv) appear in your terminal. That means it worked.


## STEP 5 — Install all libraries

```bash
pip install -r requirements.txt
```

Wait for everything to finish installing.


## STEP 6 — Place the dataset

Make sure this file is in the root project folder:
```
synthetic_disaster_events_2025.csv
```


## STEP 7 — Train the model

```bash
python train_model.py
```

This will train the model and save:
- models/disaster_model.pkl
- models/ml_results.json


## STEP 8 — Run the Streamlit app

```bash
streamlit run app.py
```

Then open your browser and go to:
```
http://localhost:8501
```


---


## EVERY TIME YOU COME BACK (after first setup)

Just run these two:

Windows:
```bash
venv\Scripts\activate
streamlit run app/app.py
```

Mac / Linux:
```bash
source venv/bin/activate
streamlit run app/app.py
```


---


## IF SOMETHING GOES WRONG

Library missing error:
```bash
pip install -r requirements.txt
```

Model file not found error:
```bash
python disaster_classification.py
```

Port already in use error:
```bash
streamlit run app/app.py --server.port 8502
```

Then open:
```
http://localhost:8502
```
