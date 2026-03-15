import threading
from PIL import Image
import customtkinter as ctk
from tkinter import filedialog, messagebox

# Import our custom modules
from database import MongoHandler
from engine import InferenceEngine

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class AquaCleanGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AquaClean - Underwater Waste Detection")
        self.geometry("1450x850")
        
        self.current_image_path = None
        self.original_pil = None
        self.result_pil = None
        self.metrics = {}
        
        self._cached_orig_img = None
        self._cached_res_img = None
        
        self.db = MongoHandler()
        self.build_ui()
        
        self.withdraw() 
        self.show_loading_screen()
        threading.Thread(target=self.init_ai_engine, daemon=True).start()

    def init_ai_engine(self):
        try:
            print("[SYSTEM] Starting InferenceEngine initialization...")
            self.engine = InferenceEngine()
            self.after(500, self.finish_loading)
            print("[SYSTEM] InferenceEngine initialized successfully.")
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Failed to start AI engine: {e}")
            self.after(0, lambda: messagebox.showerror("Engine Error", str(e)))
            self.after(0, self.destroy)

    def show_loading_screen(self):
        self.splash = ctk.CTkToplevel(self)
        self.splash.title("Initializing...")
        self.splash.geometry("450x200")
        self.splash.attributes("-topmost", True)
        
        ctk.CTkLabel(self.splash, text="Loading AquaClean Engine...\nPlease wait.", font=("Arial", 18, "bold")).pack(expand=True)
        self.progress = ctk.CTkProgressBar(self.splash, mode="indeterminate")
        self.progress.pack(pady=20, padx=40, fill="x")
        self.progress.start()

    def finish_loading(self):
        self.progress.stop()
        self.splash.destroy()
        self.deiconify()

    def build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar.grid_rowconfigure(5, weight=1)

        ctk.CTkLabel(self.sidebar, text="AquaClean Dashboard", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(30, 30))

        self.btn_select = ctk.CTkButton(self.sidebar, text="🖼️ Select Image", command=self.select_image, height=40)
        self.btn_select.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.btn_clear = ctk.CTkButton(self.sidebar, text="🗑️ Clear Screen", command=self.clear_screen, height=40, fg_color="transparent", border_width=2)
        self.btn_clear.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.btn_save = ctk.CTkButton(self.sidebar, text="💾 Store to DB", command=self.save_to_db, height=40, state="disabled")
        self.btn_save.grid(row=3, column=0, padx=20, pady=40, sticky="ew")

        self.status_lbl = ctk.CTkLabel(self.sidebar, text="Status: Ready", text_color="gray")
        self.status_lbl.grid(row=6, column=0, padx=20, pady=20, sticky="s")

        # --- Images Area ---
        self.image_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.image_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.image_frame.grid_columnconfigure((0, 1), weight=1)
        self.image_frame.grid_rowconfigure(0, weight=1)

        ctk.CTkLabel(self.image_frame, text="Original Input", font=ctk.CTkFont(size=16)).grid(row=0, column=0, sticky="n", pady=10)
        self.lbl_orig_img = ctk.CTkLabel(self.image_frame, text="No Image Selected", fg_color="#1e1e1e", corner_radius=10)
        self.lbl_orig_img.grid(row=0, column=0, sticky="nsew", padx=10, pady=(40, 10))

        ctk.CTkLabel(self.image_frame, text="Enhanced + Detected", font=ctk.CTkFont(size=16)).grid(row=0, column=1, sticky="n", pady=10)
        self.lbl_res_img = ctk.CTkLabel(self.image_frame, text="No Image Selected", fg_color="#1e1e1e", corner_radius=10)
        self.lbl_res_img.grid(row=0, column=1, sticky="nsew", padx=10, pady=(40, 10))

        # --- Metrics Area ---
        self.metrics_frame = ctk.CTkFrame(self, height=120)
        self.metrics_frame.grid(row=1, column=1, sticky="ew", padx=20, pady=(0, 20))
        self.metrics_frame.grid_columnconfigure((0, 1, 2, 3, 4, 5), weight=1)

        self.lbl_stat_objects = self.create_metric_box(0, "Objects Found", "0")
        self.lbl_stat_classes = self.create_metric_box(1, "Classes", "-")
        self.lbl_stat_in_uiqm = self.create_metric_box(2, "Input UIQM", "0.000")
        self.lbl_stat_out_uiqm = self.create_metric_box(3, "Enhanced UIQM", "0.000")
        self.lbl_stat_psnr = self.create_metric_box(4, "PSNR", "0.00 dB")
        self.lbl_stat_time = self.create_metric_box(5, "Inference Time", "Awaiting")

    def create_metric_box(self, col, title, default_val):
        frame = ctk.CTkFrame(self.metrics_frame, fg_color="transparent")
        frame.grid(row=0, column=col, pady=20, padx=5)
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=14, weight="bold"), text_color="gray").pack()
        lbl_val = ctk.CTkLabel(frame, text=default_val, font=ctk.CTkFont(size=22, weight="bold"), text_color="#1f6aa5")
        lbl_val.pack()
        return lbl_val

    def select_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.clear_screen()
            self.current_image_path = file_path
            
            temp_pil = Image.open(file_path).convert("RGB")
            self.display_image(temp_pil, self.lbl_orig_img, is_original=True)
            
            self.lbl_res_img.configure(image=None, text="Processing...")
            self.lbl_stat_time.configure(text="Processing...")
            self.status_lbl.configure(text="Status: Processing...", text_color="yellow")
            self.btn_select.configure(state="disabled")
            
            threading.Thread(target=self.run_inference_thread, args=(file_path,), daemon=True).start()

    def run_inference_thread(self, filepath):
        try:
            print(f"[UI THREAD] Attempting to process image: {filepath}")
            # All the hard work is passed to the engine!
            self.original_pil, self.result_pil, self.metrics = self.engine.process_image(filepath)
            self.after(0, self.update_ui_post_inference)
            print("[UI THREAD] Image processed and UI update triggered.")
        except Exception as e:
            print(f"\n[ERROR] run_inference_thread failed: {e}")
            self.after(0, lambda: messagebox.showerror("Processing Error", str(e)))
            self.after(0, self.reset_ui_state)

    def update_ui_post_inference(self):
        self.display_image(self.result_pil, self.lbl_res_img, is_original=False)
        
        self.lbl_stat_objects.configure(text=str(self.metrics["count"]))
        self.lbl_stat_classes.configure(text=self.metrics["classes"])
        self.lbl_stat_in_uiqm.configure(text=f"{self.metrics['in_uiqm']}")
        self.lbl_stat_out_uiqm.configure(text=f"{self.metrics['out_uiqm']}")
        self.lbl_stat_psnr.configure(text=f"{self.metrics['psnr']} dB")
        self.lbl_stat_time.configure(text=f"{self.metrics['time']}s")
        
        self.status_lbl.configure(text="Status: Complete", text_color="#2ecc71")
        self.btn_select.configure(state="normal")
        self.btn_save.configure(state="normal")

    def save_to_db(self):
        try:
            print(f"[DB LOG] Attempting to save results to MongoDB...")
            self.db.save_detection(self.current_image_path, self.metrics, self.original_pil, self.result_pil)
            print("[DB LOG] Save successful.")
            messagebox.showinfo("Success", "Results and Images successfully saved to MongoDB!")
            self.btn_save.configure(state="disabled")
        except Exception as e:
            print(f"\n[DB ERROR] Failed to save to database: {e}")
            messagebox.showerror("Database Error", str(e))

    def display_image(self, pil_img, target_label, is_original=True):
        img_copy = pil_img.copy()
        img_copy.thumbnail((500, 500), Image.Resampling.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=img_copy, dark_image=img_copy, size=(img_copy.width, img_copy.height))
        target_label.configure(image=ctk_img, text="")
        
        if is_original: self._cached_orig_img = ctk_img
        else: self._cached_res_img = ctk_img

    def clear_screen(self):
        self.current_image_path = None
        self.metrics = {}
        placeholder = ctk.CTkImage(Image.new("RGBA", (1, 1), (0, 0, 0, 0)), size=(1, 1))
        
        self.lbl_orig_img.configure(image=placeholder, text="No Image Selected")
        self.lbl_res_img.configure(image=placeholder, text="No Image Selected")
        self._cached_orig_img = self._cached_res_img = placeholder 
        
        self.lbl_stat_objects.configure(text="0")
        self.lbl_stat_classes.configure(text="-")
        self.lbl_stat_in_uiqm.configure(text="0.000")
        self.lbl_stat_out_uiqm.configure(text="0.000")
        self.lbl_stat_psnr.configure(text="0.00 dB")
        self.lbl_stat_time.configure(text="Awaiting")
        
        self.btn_save.configure(state="disabled")
        self.status_lbl.configure(text="Status: Ready", text_color="gray")

    def reset_ui_state(self):
        self.status_lbl.configure(text="Status: Error", text_color="red")
        self.btn_select.configure(state="normal")

if __name__ == "__main__":
    app = AquaCleanGUI()
    app.mainloop()