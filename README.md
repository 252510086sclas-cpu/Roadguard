# RoadGuard – AI Road Damage Detection

A Flask website for uploading road images and running a YOLO object-detection model.

## Run on Windows

1. Install Python 3.11.
2. Open Command Prompt in this folder.
3. Create and activate a virtual environment:

   ```bat
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   ```

4. Install dependencies:

   ```bat
   pip install -r requirements.txt
   ```

5. Put your trained YOLO model weights in this folder and name it `best.pt`.
6. Start the app:

   ```bat
   python app.py
   ```

7. Open http://127.0.0.1:5000

## Model requirement

The website interface is complete, but real road-damage detection requires trained YOLO weights (`best.pt`) whose classes match your road-damage dataset. The app does not fabricate detections: if the weights are missing, it displays a clear setup message.

If your model has another filename or path, set the `MODEL_PATH` environment variable before starting the app.

## Deploy on Render

Push this folder to a GitHub repository and create a Render Web Service from that repository. Render uses `requirements.txt` and `Procfile`/`render.yaml`.

Important: large model weights may exceed free hosting memory, disk, or deployment limits. For production, use a persistent model download/storage strategy or a paid instance with adequate RAM. Do not commit private credentials or very large model files to a public GitHub repository.

## Notes

- Accepted formats: JPG, JPEG, PNG, WEBP
- Maximum upload size: 10 MB
- Detection confidence threshold: 0.25
- This is a prototype, not a certified road-safety inspection tool.
