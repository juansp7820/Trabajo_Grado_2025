
# --- NUEVA INTERFAZ PARA DOS MAMOGRAFÍAS (MLO y CC) ---
import gradio as gr
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Importar funciones comprobadas del pipeline
from src.prep import dcm_to_png, apply_clahe, detect_breast_roi

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "models", "unet_custom2_46235.pth"))
from src.arc.unet import UNet
model = UNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

def process_dicom(dicom_file):
    # 1. Convertir DICOM a PNG (8 bits)
    img_png, _ = dcm_to_png(dicom_file.name)
    img_8bit = np.array(img_png)
    # 2. Aplicar CLAHE solo para segmentación
    img_clahe = apply_clahe(img_8bit)
    # 3. Convertir a RGB para visualización (sobre original)
    img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)
    return img_rgb, img_8bit, img_clahe

def segment_six_regions(img_clahe):
    # Usar pipeline anatómico comprobado
    #cropped_img, crop_info = preprocess_mammogram(img_clahe)
    breast_bbox, _ = detect_breast_roi(img_clahe)
    if breast_bbox is None:
        raise ValueError('No se pudo detectar la región de interés de la mama.')
    x1, y1, x2, y2 = breast_bbox
    x_v1 = int(x1 + (x2 - x1) / 3)
    x_v2 = int(x1 + 2 * (x2 - x1) / 3)
    y_nipple = int((y1 + y2) / 2)
    crops = [
        img_clahe[y1:y_nipple, x1:x_v1],
        img_clahe[y1:y_nipple, x_v1:x_v2],
        img_clahe[y1:y_nipple, x_v2:x2],
        img_clahe[y_nipple:y2, x1:x_v1],
        img_clahe[y_nipple:y2, x_v1:x_v2],
        img_clahe[y_nipple:y2, x_v2:x2]
    ]
    preds = []
    for crop in crops:
        crop_resized = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR)
        input_tensor = torch.from_numpy(crop_resized).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.sigmoid(output).cpu().numpy().squeeze()
        pred_bin = (pred > 0.5).astype(np.uint8)
        preds.append(pred_bin)
    # Return crops, preds, and anatomical coordinates
    coords = [
        (y1, y_nipple, x1, x_v1),
        (y1, y_nipple, x_v1, x_v2),
        (y1, y_nipple, x_v2, x2),
        (y_nipple, y2, x1, x_v1),
        (y_nipple, y2, x_v1, x_v2),
        (y_nipple, y2, x_v2, x2)
    ]
    grid_coords = {'x_v1': x_v1, 'x_v2': x_v2, 'y_nipple': y_nipple}
    return preds, crops, coords, grid_coords

def overlay_preds(img_rgb, preds, crops, coords, grid_coords, colors, show_grid, show_overlays, opacity):
    # img_rgb: original RGB image (DICOM 8bit RGB)
    overlay = np.zeros_like(img_rgb)
    # Validar colores
    for i in range(len(colors)):
        if not isinstance(colors[i], (list, tuple)) or len(colors[i]) != 3:
            colors[i] = [255,0,0]
    for i, (pred, crop, (y0, y1, x0, x1)) in enumerate(zip(preds, crops, coords)):
        if show_overlays[i]:
            crop_h, crop_w = crop.shape
            pred_resized = cv2.resize(pred, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
            color_mask = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
            color_mask[pred_resized > 0] = colors[i]
            overlay[y0:y1, x0:x1] = overlay[y0:y1, x0:x1] + color_mask
    # Superponer la grilla como capa adicional usando cv2.line
    if show_grid:
        x_v1 = int(grid_coords['x_v1'])
        x_v2 = int(grid_coords['x_v2'])
        y_nipple = int(grid_coords['y_nipple'])
        h, w = overlay.shape[:2]
        # Verticales
        cv2.line(overlay, (x_v1, 0), (x_v1, h-1), (255,255,255), 2)
        cv2.line(overlay, (x_v2, 0), (x_v2, h-1), (255,255,255), 2)
        # Horizontal
        cv2.line(overlay, (0, y_nipple), (w-1, y_nipple), (255,255,255), 2)
    img_result = cv2.addWeighted(img_rgb, 1.0, overlay, opacity, 0)
    return Image.fromarray(img_result)

def hex_to_rgb(hex_color):
    try:
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            return (255,0,0)
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        return (255,0,0)

def default_colors(selected_hex):
    rgb = hex_to_rgb(selected_hex)
    # Validar rango de color
    rgb = tuple(max(0, min(255, int(c))) for c in rgb)
    return [list(rgb)]*6

with gr.Blocks() as demo:
    gr.Markdown("# Herramienta de segmentación automática de masas en mamografías (MLO y CC)")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Mamografía MLO/CC")
            mlo_input = gr.File(label="Cargar DICOM MLO/CC")
            mlo_overlay_enable = gr.Checkbox(label="Mostrar superposición de segmentación MLO/CC", value=True)
            mlo_grid_chk = gr.Checkbox(label="Mostrar grilla anatómica MLO/CC", value=False)
            mlo_overlay_checks = gr.CheckboxGroup(choices=[f"Recorte {i+1}" for i in range(6)], value=[f"Recorte {i+1}" for i in range(6)], label="Mostrar recortes MLO/CC")
            mlo_img = gr.Image(label="Resultado MLO/CC", elem_id="mlo_img")
        with gr.Column():
            gr.Markdown("## Mamografía CC/MLO")
            cc_input = gr.File(label="Cargar DICOM CC/MLO")
            cc_overlay_enable = gr.Checkbox(label="Mostrar superposición de segmentación CC/MLO", value=True)
            cc_grid_chk = gr.Checkbox(label="Mostrar grilla anatómica CC/MLO", value=False)
            cc_overlay_checks = gr.CheckboxGroup(choices=[f"Recorte {i+1}" for i in range(6)], value=[f"Recorte {i+1}" for i in range(6)], label="Mostrar recortes CC/MLO")
            cc_img = gr.Image(label="Resultado CC/MLO", elem_id="cc_img")
    # --- Caching for MLO and CC ---
    mlo_cache = {}
    cc_cache = {}

    def mlo_upload_callback(file, show_grid, overlay_checks, overlay_enable):
        try:
            img_rgb, img_8bit, img_clahe = process_dicom(file)
            preds, crops, coords, grid_coords = segment_six_regions(img_clahe)
            mlo_cache['img_rgb'] = img_rgb
            mlo_cache['preds'] = preds
            mlo_cache['crops'] = crops
            mlo_cache['coords'] = coords
            mlo_cache['grid_coords'] = grid_coords
            # UI state
            mlo_cache['file'] = file
            mlo_cache['show_grid'] = show_grid
            mlo_cache['overlay_checks'] = overlay_checks
            mlo_cache['overlay_enable'] = overlay_enable
            show_overlays = [f"Recorte {i+1}" in overlay_checks for i in range(6)]
            colors = default_colors("#FF0000")
            opacity = 1.0 if overlay_enable else 0.0
            return overlay_preds(img_rgb, preds, crops, coords, grid_coords, colors, show_grid, show_overlays, opacity)
        except Exception:
            error_img = Image.new('RGB', (512, 512), color=(255,0,0))
            return error_img

    def mlo_update_display(file, show_grid, overlay_checks, overlay_enable):
        # Only update overlays/grid using cached results
        try:
            if not mlo_cache or mlo_cache.get('file') != file:
                # If cache is empty or file changed, fallback to upload callback
                return mlo_upload_callback(file, show_grid, overlay_checks, overlay_enable)
            img_rgb = mlo_cache['img_rgb']
            preds = mlo_cache['preds']
            crops = mlo_cache['crops']
            coords = mlo_cache['coords']
            grid_coords = mlo_cache['grid_coords']
            show_overlays = [f"Recorte {i+1}" in overlay_checks for i in range(6)]
            colors = default_colors("#FF0000")
            opacity = 1.0 if overlay_enable else 0.0
            return overlay_preds(img_rgb, preds, crops, coords, grid_coords, colors, show_grid, show_overlays, opacity)
        except Exception:
            error_img = Image.new('RGB', (512, 512), color=(255,0,0))
            return error_img

    def cc_upload_callback(file, show_grid, overlay_checks, overlay_enable):
        try:
            img_rgb, img_8bit, img_clahe = process_dicom(file)
            preds, crops, coords, grid_coords = segment_six_regions(img_clahe)
            cc_cache['img_rgb'] = img_rgb
            cc_cache['preds'] = preds
            cc_cache['crops'] = crops
            cc_cache['coords'] = coords
            cc_cache['grid_coords'] = grid_coords
            cc_cache['file'] = file
            cc_cache['show_grid'] = show_grid
            cc_cache['overlay_checks'] = overlay_checks
            cc_cache['overlay_enable'] = overlay_enable
            show_overlays = [f"Recorte {i+1}" in overlay_checks for i in range(6)]
            colors = default_colors("#FF0000")
            opacity = 1.0 if overlay_enable else 0.0
            return overlay_preds(img_rgb, preds, crops, coords, grid_coords, colors, show_grid, show_overlays, opacity)
        except Exception:
            error_img = Image.new('RGB', (512, 512), color=(255,0,0))
            return error_img

    def cc_update_display(file, show_grid, overlay_checks, overlay_enable):
        try:
            if not cc_cache or cc_cache.get('file') != file:
                return cc_upload_callback(file, show_grid, overlay_checks, overlay_enable)
            img_rgb = cc_cache['img_rgb']
            preds = cc_cache['preds']
            crops = cc_cache['crops']
            coords = cc_cache['coords']
            grid_coords = cc_cache['grid_coords']
            show_overlays = [f"Recorte {i+1}" in overlay_checks for i in range(6)]
            colors = default_colors("#FF0000")
            opacity = 1.0 if overlay_enable else 0.0
            return overlay_preds(img_rgb, preds, crops, coords, grid_coords, colors, show_grid, show_overlays, opacity)
        except Exception:
            error_img = Image.new('RGB', (512, 512), color=(255,0,0))
            return error_img

    # Connect Gradio events
    mlo_input.upload(mlo_upload_callback, inputs=[mlo_input, mlo_grid_chk, mlo_overlay_checks, mlo_overlay_enable], outputs=[mlo_img])
    mlo_grid_chk.change(mlo_update_display, inputs=[mlo_input, mlo_grid_chk, mlo_overlay_checks, mlo_overlay_enable], outputs=[mlo_img])
    mlo_overlay_checks.change(mlo_update_display, inputs=[mlo_input, mlo_grid_chk, mlo_overlay_checks, mlo_overlay_enable], outputs=[mlo_img])
    mlo_overlay_enable.change(mlo_update_display, inputs=[mlo_input, mlo_grid_chk, mlo_overlay_checks, mlo_overlay_enable], outputs=[mlo_img])
    cc_input.upload(cc_upload_callback, inputs=[cc_input, cc_grid_chk, cc_overlay_checks, cc_overlay_enable], outputs=[cc_img])
    cc_grid_chk.change(cc_update_display, inputs=[cc_input, cc_grid_chk, cc_overlay_checks, cc_overlay_enable], outputs=[cc_img])
    cc_overlay_checks.change(cc_update_display, inputs=[cc_input, cc_grid_chk, cc_overlay_checks, cc_overlay_enable], outputs=[cc_img])
    cc_overlay_enable.change(cc_update_display, inputs=[cc_input, cc_grid_chk, cc_overlay_checks, cc_overlay_enable], outputs=[cc_img])

    demo.css = """
    #mlo_img img, #cc_img img {
        max-width: 48vw;
        max-height: 80vh;
        display: block;
        margin: auto;
    }
    """

demo.launch(share=False, server_port=7860, inbrowser=True)