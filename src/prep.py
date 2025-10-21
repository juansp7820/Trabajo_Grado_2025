#### Convertir DICOM a PNG (8bits) Normalización Min-Max
import pydicom
import numpy as np
import cv2
from PIL import Image

def dcm_to_png(dcm_path):
    """Lee un DICOM, normaliza a 8 bits y retorna un objeto PIL.Image en memoria (no guarda archivo)."""
    ds = pydicom.dcmread(dcm_path)
    pixel_array = ds.pixel_array

    # Escalado lineal a 8 bits (basado en rango real)
    if pixel_array.dtype != np.uint8:
        img_8bit = ((pixel_array - pixel_array.min()) *
                   (255.0 / (pixel_array.max() - pixel_array.min()))).astype(np.uint8)
    else:
        img_8bit = pixel_array

    # Crear imagen PIL en memoria
    img = Image.fromarray(img_8bit)

    # Si se requiere DPI para mantener mm/pixel, se puede devolver como atributo
    dpi = None
    if hasattr(ds, 'PixelSpacing'):
        dpi = int(25.4 / float(ds.PixelSpacing[0]))  # mm/pixel → DPI

    return img, dpi

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img)
    return clahe_img


# === FUNCIONES DE SEGMENTACIÓN Y RECORTE DE MAMOGRAFÍA ===
def preprocess_mammogram(img):
    h, w = img.shape
    top_crop = int(h * 0.05)
    bottom_crop = int(h * 0.05)
    cropped_img = img[top_crop:h-bottom_crop, :]
    crop_info = {'top': top_crop, 'left': 0, 'bottom': bottom_crop, 'right': 0}
    return cropped_img, crop_info

def adjust_coordinates_to_original(coordinates, crop_info):
    if isinstance(coordinates, tuple) and len(coordinates) == 4:
        x1, y1, x2, y2 = coordinates
        x1_orig = x1 + crop_info['left']
        y1_orig = y1 + crop_info['top']
        x2_orig = x2 + crop_info['left']
        y2_orig = y2 + crop_info['top']
        return (x1_orig, y1_orig, x2_orig, y2_orig)
    elif hasattr(coordinates, 'shape'):
        adjusted_contour = coordinates.copy()
        adjusted_contour[:, 0, 0] += crop_info['left']
        adjusted_contour[:, 0, 1] += crop_info['top']
        return adjusted_contour
    else:
        return coordinates

def adaptive_roi_crop(img, bbox, threshold=0.95, band_percent=0.05):
    x1, y1, x2, y2 = bbox
    roi_img = img[y1:y2, x1:x2]
    h, w = roi_img.shape
    band_h = int(h * band_percent)
    # Recorte inferior
    bottom = h
    while bottom - band_h > 0:
        band = roi_img[bottom-band_h:bottom, :]
        black_ratio = np.mean(band < 10)
        if black_ratio >= threshold:
            bottom -= band_h
        else:
            break
    # Recorte superior
    top = 0
    while top + band_h < bottom:
        band = roi_img[top:top+band_h, :]
        black_ratio = np.mean(band < 10)
        if black_ratio >= threshold:
            top += band_h
        else:
            break
    # Nueva ROI
    new_y1 = y1 + top
    new_y2 = y1 + bottom
    return (x1, new_y1, x2, new_y2)

def get_mama_contour(img):
    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    pts = largest.squeeze()
    if pts.ndim != 2 or len(pts) < 20:
        return largest
    N = max(10, int(0.08 * len(pts)))
    def find_cut_index(seg, threshold_dev=8, threshold_len=0.12):
        if len(seg) < 2:
            return False
        x = seg[:,0]
        y = seg[:,1]
        if np.unique(x).size < 2 or np.unique(y).size < 2:
            return False
        try:
            fit = np.polyfit(x, y, 1)
            y_fit = np.polyval(fit, x)
            deviation = np.std(y - y_fit)
            y_span = np.max(y) - np.min(y)
            if deviation < threshold_dev and y_span < threshold_len * img.shape[0]:
                return True
        except Exception:
            return False
        return False
    start_idx = 0
    end_idx = len(pts)
    for i in range(N, 0, -1):
        if find_cut_index(pts[:i]):
            start_idx = i
            break
    for i in range(N, 0, -1):
        if find_cut_index(pts[-i:]):
            end_idx = len(pts) - i
            break
    if end_idx > start_idx and (end_idx - start_idx) > 10:
        pts = pts[start_idx:end_idx]
        largest = pts.reshape(-1, 1, 2)
    return largest

def detect_breast_roi(img):
    cropped_img, crop_info = preprocess_mammogram(img)
    contour_cropped = get_mama_contour(cropped_img)
    if contour_cropped is None:
        return None, None
    contour_original = adjust_coordinates_to_original(contour_cropped, crop_info)
    pts = contour_original.squeeze()
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])
    breast_bbox = (min_x, min_y, max_x, max_y)
    breast_bbox = adaptive_roi_crop(img, breast_bbox, threshold=0.90, band_percent=0.05)
    return breast_bbox, contour_original

def get_adaptive_quadrant_crops(img, mask_paths=None):
    """
    Recorta las seis áreas delimitadas por la caja amarilla y sus segmentos internos.
    Devuelve una lista de recortes (arrays) para la mamografía y para cada máscara.
    """
    breast_bbox, _ = detect_breast_roi(img)
    if breast_bbox is None:
        print('No se pudo detectar la región de interés de la mama.')
        return None, None
    x1, y1, x2, y2 = breast_bbox
    x_v1 = int(x1 + (x2 - x1) / 3)
    x_v2 = int(x1 + 2 * (x2 - x1) / 3)
    y_nipple = int((y1 + y2) / 2)
    crop1 = img[y1:y_nipple, x1:x_v1]
    crop2 = img[y1:y_nipple, x_v1:x_v2]
    crop3 = img[y1:y_nipple, x_v2:x2]
    crop4 = img[y_nipple:y2, x1:x_v1]
    crop5 = img[y_nipple:y2, x_v1:x_v2]
    crop6 = img[y_nipple:y2, x_v2:x2]
    # Redimensionar cada recorte a 512x512 (igual que en entrenamiento)
    target_size = (512, 512)
    crops_mammo = [cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR) for crop in [crop1, crop2, crop3, crop4, crop5, crop6]]
    crops_masks = []
    if mask_paths:
        for mask_path in mask_paths:
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            m1 = mask_img[y1:y_nipple, x1:x_v1]
            m2 = mask_img[y1:y_nipple, x_v1:x_v2]
            m3 = mask_img[y1:y_nipple, x_v2:x2]
            m4 = mask_img[y_nipple:y2, x1:x_v1]
            m5 = mask_img[y_nipple:y2, x_v1,x_v2]
            m6 = mask_img[y_nipple:y2, x_v2,x2]
            # Redimensionar cada recorte de máscara a 512x512
            crops_masks.append([cv2.resize(m, target_size, interpolation=cv2.INTER_NEAREST) for m in [m1, m2, m3, m4, m5, m6]])
    return crops_mammo, crops_masks