# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from PIL import Image, ImageDraw, ImageFont
# from transformers import DetrImageProcessor
# import customtkinter as ctk
# from tkinter import filedialog, messagebox # Added messagebox for error alerts
# import time
# import threading # Added for the loading state
# from models import JointModel

# # ==========================================
# # . GUI APPLICATION
# # ==========================================
# class AquaCleanGUI(ctk.CTk):
#     def __init__(self):
#         super().__init__()
#         self.title("AquaClean: Underwater Waste Detection System")
#         self.geometry("1400x700")
#         ctk.set_appearance_mode("dark")

#         # Load Model
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = JointModel().to(self.device)
#         weights_path = "../epoch_files/joint_model_epoch_60.pt"
        
#         try:
#             self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
#             self.model.eval()
#         except Exception as e:
#             print(f"Error loading weights: {e}")

#         self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        
#         try:
#             self.font = ImageFont.truetype("arial.ttf", 22) 
#         except:
#             self.font = ImageFont.load_default()

#         # --- UI LAYOUT ---
#         self.sidebar = ctk.CTkFrame(self, width=200)
#         self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

#         self.btn_select = ctk.CTkButton(self.sidebar, text="Select Image", command=self.start_processing_thread)
#         self.btn_select.pack(pady=20, padx=10)

#         self.btn_db = ctk.CTkButton(self.sidebar, text="Store to Database", 
#                                     fg_color="#2c3e50", hover_color="#34495e",
#                                     command=self.save_to_db)
#         self.btn_db.pack(pady=10, padx=10)

#         # 🟢 Stats Label (Will show "Processing..." during load)
#         self.lbl_stats = ctk.CTkLabel(self.sidebar, text="Stats:\nWaiting...", justify="left", text_color="white")
#         self.lbl_stats.pack(pady=20, padx=10)

#         self.img_container = ctk.CTkFrame(self)
#         self.img_container.pack(side="right", fill="both", expand=True, padx=10, pady=10)

#         # Labels
#         self.title_orig = ctk.CTkLabel(self.img_container, text="1. Original Image", font=("Arial", 16, "bold"))
#         self.title_orig.grid(row=0, column=0, pady=(5, 0))
#         self.title_enh = ctk.CTkLabel(self.img_container, text="2. Enhanced Image", font=("Arial", 16, "bold"))
#         self.title_enh.grid(row=0, column=1, pady=(5, 0))
#         self.title_det = ctk.CTkLabel(self.img_container, text="3. Detection Image", font=("Arial", 16, "bold"))
#         self.title_det.grid(row=0, column=2, pady=(5, 0))

#         # Panels
#         self.panel_orig = ctk.CTkLabel(self.img_container, text="No Image", fg_color="gray20")
#         self.panel_orig.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
#         self.panel_enh = ctk.CTkLabel(self.img_container, text="No Image", fg_color="gray20")
#         self.panel_enh.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
#         self.panel_det = ctk.CTkLabel(self.img_container, text="No Image", fg_color="gray20")
#         self.panel_det.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")

#         self.img_container.grid_columnconfigure((0, 1, 2), weight=1)
#         self.img_container.grid_rowconfigure(1, weight=1)

#     def save_to_db(self):
#         print("💾 Database storage triggered!")

#     def start_processing_thread(self):
#         # Trigger the file dialog
#         path = filedialog.askopenfilename()
#         if not path:
#             return
        
#         # 🟢 Validation: Check if it's an image file
#         valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
#         if not path.lower().endswith(valid_extensions):
#             messagebox.showerror("Invalid File", "Error: This is not an image file!")
#             return

#         # 🟢 UI Loading State
#         self.lbl_stats.configure(text="Stats:\n🔄 Processing...", text_color="#f1c40f")
#         self.btn_select.configure(state="disabled") # Prevent double clicking

#         # Start the AI logic in a background thread
#         thread = threading.Thread(target=self.process_image, args=(path,))
#         thread.start()

#     def process_image(self, path):
#         try:
#             orig_img = Image.open(path).convert("RGB")
#             start_t = time.time()
            
#             # AI Inference
#             inputs = self.processor(images=orig_img, return_tensors="pt").to(self.device)
#             with torch.no_grad():
#                 outputs, enh_tensor, _ = self.model(pixel_values=inputs['pixel_values'])
            
#             # 1. Prepare Enhanced Image
#             enh_np = enh_tensor[0].cpu().permute(1, 2, 0).numpy()
#             enh_np = (enh_np * 255).clip(0, 255).astype('uint8')
#             enh_img = Image.fromarray(enh_np)

#             # 2. Draw Detections
#             det_img = orig_img.copy() 
#             draw = ImageDraw.Draw(det_img)
#             target_sizes = torch.tensor([orig_img.size[::-1]]).to(self.device)
#             results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]
            
#             categories = ['plastic', 'bio', 'rov']
#             for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#                 box = box.tolist()
#                 draw.rectangle(box, outline="#00FF00", width=4)
#                 label_text = f"{categories[label]}: {round(score.item() * 100, 1)}%"
#                 draw.text((box[0], box[1] - 30), label_text, fill="red", font=self.font)

#             latency = (time.time() - start_t) * 1000
            
#             # 🟢 Update UI (Must be done in a safe way or back in main thread)
#             self.after(0, lambda: self.update_ui_results(orig_img, enh_img, det_img, results, latency))

#         except Exception as e:
#             self.after(0, lambda: messagebox.showerror("Error", f"Could not process image: {e}"))
#             self.after(0, lambda: self.lbl_stats.configure(text="Stats:\nError", text_color="red"))
#             self.after(0, lambda: self.btn_select.configure(state="normal"))

#     def update_ui_results(self, orig, enh, det, results, latency):
#         # Update Stats
#         self.lbl_stats.configure(text=f"Stats:\nObjects: {len(results['labels'])}\nTime: {latency:.1f}ms\nDevice: {self.device}", text_color="white")
        
#         # Refresh UI Panels
#         self.display_image(orig, self.panel_orig)
#         self.display_image(enh, self.panel_enh)
#         self.display_image(det, self.panel_det)
        
#         # Re-enable button
#         self.btn_select.configure(state="normal")

#     def display_image(self, img, panel):
#         w, h = img.size
#         aspect = w / h
#         new_w = 450
#         new_h = int(new_w / aspect)
#         ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(new_w, new_h))
#         panel.configure(image=ctk_img, text="")

# if __name__ == "__main__":
#     app = AquaCleanGUI()
#     app.mainloop()
