# App - Waste Image Classification Demo

## Run the app

From the project root directory:

```bash
streamlit run app/app.py
```

If your terminal uses `python3`, run:

```bash
python3 -m streamlit run app/app.py
```

The app opens in your browser at `http://localhost:8501`.

## Requirements

Install all project dependencies first:

```bash
pip install -r requirements.txt
```

## Model checkpoint

The trained model checkpoint is not committed to GitHub.

To run predictions, download or copy `mobilenet_v2_v1.pt` into the `models/` folder:

```text
models/mobilenet_v2_v1.pt
```

If the checkpoint is missing, the app still opens and shows a clear message instead of crashing.

## What the app does

1. Upload a JPG, JPEG, or PNG image of a waste item.
2. The app shows a preview of the uploaded image.
3. If the model checkpoint is present, it predicts one of 12 waste classes.
4. It shows the predicted class name, confidence score, and top 3 predictions.
5. It shows a short disposal note for the predicted class.

## Waste classes

The model classifies images into 12 classes:

| Class | Notes |
|---|---|
| battery | Handle separately |
| biological | Organic waste, can be composted |
| brown-glass | Recyclable glass |
| cardboard | Recyclable if dry |
| clothes | Donate or textile recycling |
| green-glass | Recyclable glass |
| metal | Recyclable |
| paper | Recyclable if dry |
| plastic | Recyclable if clean |
| shoes | Donate or special disposal |
| trash | General waste |
| white-glass | Recyclable glass |
