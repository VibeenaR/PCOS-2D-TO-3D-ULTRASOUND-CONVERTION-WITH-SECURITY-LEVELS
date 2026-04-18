import numpy as np

class MedicalSecurity:
    def __init__(self, shape):
        self.shape = shape
        # Step 2: Generate secret key matrix B using chirp function
        self.B = self._generate_chirp_matrix(shape)

    def _generate_chirp_matrix(self, shape):
        """Generates a frequency-swept sine wave (Chirp) as a secret key."""
        t = np.linspace(0, 1, shape[0] * shape[1])
        # Chirp signal logic
        chirp = np.sin(2 * np.pi * (0.1 * t + 0.5 * 0.5 * t**2))
        return chirp.reshape(shape)

    def generate_hash(self, I, alpha=1.0):
        """Step 3: Apply hash transformation -> H = aI + B"""
        I_norm = I.astype(float) / 255.0 if I.max() > 1 else I.astype(float)
        H = (alpha * I_norm) + self.B
        return H

    def verify_integrity(self, I, H, alpha=1.0):
        """Step 4 & 5: Compute error matrix E = H - I and compare to B"""
        I_norm = I.astype(float) / 255.0 if I.max() > 1 else I.astype(float)
        E = H - (alpha * I_norm)
        
        # If the image was tampered with, E will not match the original B
        diff = np.abs(E - self.B)
        is_authentic = np.mean(diff) < 1e-6
        return is_authentic