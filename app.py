import os
import time
from flask import Flask, render_template, request, redirect, url_for
from src.extractor import extract_and_save_slices
from src.validator import ClinicalValidator
from src.reconstruction import generate_3d_volume

app = Flask(__name__)
app.secret_key = "qwertyuiop1234567890"

BASE_DIR = r"C:/Users/vibee/gvspcos"
DATA_ROOT = os.path.join(BASE_DIR, "data", "processed_results")
os.makedirs(DATA_ROOT, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_path = os.path.join(DATA_ROOT, f"Result_{timestamp}")
    os.makedirs(session_path, exist_ok=True)

    file_path = os.path.join(session_path, file.filename)
    file.save(file_path)

    try:
        file_hash = ClinicalValidator.get_sha256(file_path)
        
        # Extract 50 slices for the 'Selection' engine to choose from
        slices = extract_and_save_slices(file_path, session_path, num_slices=50)
        
        # SECURITY GATE: Authenticity check
        is_real, reason = ClinicalValidator.is_clinical_ultrasound(slices[0])
        if not is_real:
            return render_template('failure.html', reason=reason, file_hash=file_hash)

        # RECONSTRUCTION: Picks follicle-rich slices and stacks as ONE
        output_name = "Unified_Follicle_Volume.ply"
        output_path = os.path.join(session_path, output_name)
        
        generate_3d_volume(slices, output_path)

        return render_template('success.html', filename=file.filename, 
                               file_hash=file_hash, folder_loc=session_path)

    except Exception as e:
        return render_template('failure.html', reason=str(e), file_hash="N/A")

if __name__ == '__main__':
    app.run(debug=True, port=5000)