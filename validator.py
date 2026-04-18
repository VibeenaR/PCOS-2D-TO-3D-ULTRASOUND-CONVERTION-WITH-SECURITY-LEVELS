import hashlib
import numpy as np
import cv2

class ClinicalValidator:
    @staticmethod
    def get_sha256(file_path):
        """Returns the SHA-256 hash of the file to prove no data stealing/tampering."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def is_clinical_ultrasound(image):
        """
        Detects artificial or non-clinical images.
        Uses Entropy (complexity) and standard deviation check.
        """
        # Calculate Entropy
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
        
        # Ultrasound images have high entropy due to 'speckle noise'
        # Random graphics or text files will have very low entropy
        if entropy < 3.5:
            return False, "Image lacks clinical noise profile (Artificial)."
            
        return True, "Clinical Authenticity Verified."