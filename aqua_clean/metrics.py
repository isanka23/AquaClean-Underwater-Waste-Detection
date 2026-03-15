import cv2
import numpy as np

def calculate_uiqm(img_np):
    """Calculates UIQM given an RGB image array in [0, 1] range."""
    img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # 1. UICM (Colorfulness)
    rg = img_bgr[:,:,2].astype(np.float32) - img_bgr[:,:,1].astype(np.float32)
    yb = 0.5 * (img_bgr[:,:,2].astype(np.float32) + img_bgr[:,:,1].astype(np.float32)) - img_bgr[:,:,0].astype(np.float32)
    uicm = 0.02 * np.sqrt(np.mean(rg**2) + np.mean(yb**2))
    
    # 2. UISM (Sharpness)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    uism = cv2.Laplacian(gray, cv2.CV_64F).var() / 100.0 
    
    # 3. UIConM (Contrast)
    uiconm = np.std(gray) / 255.0
    
    return (0.3 * uicm) + (0.5 * uism) + (0.2 * uiconm)