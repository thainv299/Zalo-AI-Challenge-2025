import os
import torch
# Tối ưu cho RTX 5060 Ti (sm_120)
os.environ['TORCH_CUDA_ARCH_LIST'] = "12.0"
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
# Tránh phân mảnh VRAM và tăng tốc độ cấp phát
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
if torch.cuda.is_available():
    # TF32 giúp tăng tốc độ tính toán trên RTX 30/40/50 mà không mất nhiều độ chính xác
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Cho phép cuDNN tự tìm thuật toán tối ưu nhất cho phần cứng hiện tại
    torch.backends.cudnn.benchmark = True
import time
import json
import threading
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk

# --- IMPORT MODULES ---
from load_models import load_models
from modules.extract_frames import extract_frames_to_queue
from modules.vlm import generate_video_description
from modules.llm import llm_choise_answer
from utils.cached_helper import get_vlm_cache, save_vlm_cache
from ultralytics import YOLO
from modules.tracker import BestFrameTracker
# --- CONSTANTS ---
YOLO_JSON_DIR = 'yolo_json'

# --- GLOBAL VARIABLES ---
MODELS = None
TRACKER = BestFrameTracker()
ALL_QUESTIONS = []
CURRENT_VIDEO_FOLDER = None

# ==========================================
# BACKEND LOGIC
# ==========================================

def save_yolo_json(question_id, yolo_data_json):
    """Lưu chuỗi JSON YOLO"""
    os.makedirs(YOLO_JSON_DIR, exist_ok=True)
    file_path = os.path.join(YOLO_JSON_DIR, f"{question_id}.json")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            data = json.loads(yolo_data_json)
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Đã lưu YOLO JSON: {file_path}")
    except Exception as e:
        print(f"Lỗi khi lưu YOLO JSON: {e}")

def process_yolo_tracker(frames_queue, model: YOLO, tracker: BestFrameTracker, batch_size=12):
    """
    Sửa lỗi No Frames: Thêm cơ chế đợi dữ liệu vào Queue
    """
    # Xóa sạch dữ liệu cũ trong tracker trước khi bắt đầu câu hỏi mới
    tracker.best_frames.clear()
    frames_buffer = []
    
    # Ép kiểu Half để tối ưu RTX 5060 Ti
    model.to('cuda').half()

    # Cơ chế đợi: Nếu Queue rỗng, đợi tối đa 5 giây để frame đầu tiên xuất hiện
    wait_time = 0
    while frames_queue.empty() and wait_time < 5:
        time.sleep(0.1)
        wait_time += 0.1

    while True:
        try:
            # Sử dụng timeout để tránh kẹt nếu thread trích xuất frame gặp sự cố
            frame = frames_queue.get(timeout=2)
            
            if frame is None: # Tín hiệu kết thúc video
                if frames_buffer:
                    _run_batch_inference(frames_buffer, model, tracker)
                break
                
            frames_buffer.append(frame)
            
            if len(frames_buffer) >= batch_size:
                _run_batch_inference(frames_buffer, model, tracker)
                frames_buffer = []
        except: # Hết frame hoặc queue bị timeout
            if frames_buffer:
                _run_batch_inference(frames_buffer, model, tracker)
            break

    # Trích xuất dữ liệu sau khi xử lý xong
    yolo_data_list = []
    frames = []
    
    # Kiểm tra xem có bắt được đối tượng nào không
    if not tracker.best_frames:
        print("⚠️ YOLO không tìm thấy đối tượng nào trong các frame đã trích xuất.")
        return [], "[]"

    for track_id, frame_data in tracker.best_frames.items():
        yolo_data_list.append({
            "track_id": track_id,
            "object_type": frame_data.box_info.class_name,
            "bbox": frame_data.box_info.bbox.tolist(),
            "confidence": round(frame_data.box_info.confidence, 3)
        })
        frames.append(frame_data.frame)

    video_info = json.dumps(yolo_data_list, ensure_ascii=False, indent=2)
    return frames, video_info

def _run_batch_inference(batch_frames, model, tracker):
    with torch.no_grad():
        # Vẫn dùng autocast để tối ưu nhân Tensor trên RTX 5060 Ti
        with torch.amp.autocast('cuda'):
            results = model.track(
                batch_frames, 
                tracker="bytetrack.yaml", 
                verbose=False, 
                persist=True
            )
            
        for i, res in enumerate(results):
            if not res.boxes or res.boxes.id is None:
                continue
            for box in res.boxes:
                if box.id is None: continue
                # Trích xuất track_id
                tracker.update_track(
                    batch_frames[i], 
                    int(box.id), 
                    box.xyxy[0].cpu().numpy().astype(int), 
                    float(box.conf), 
                    res.names[int(box.cls)]
                )

def process_single_question_logic(question_data, models, tracker: BestFrameTracker, enable_thinking: bool = False):
    """
    Xử lý logic chính
    enable_thinking: Bật/tắt thinking mode
    """
    video_rel_path = question_data['video_path']
    question_id = question_data['id']
    
    if CURRENT_VIDEO_FOLDER:
        base_dir = os.path.dirname(CURRENT_VIDEO_FOLDER)
        video_path = os.path.join(base_dir, video_rel_path)
    else:
        video_path = video_rel_path

    if not os.path.exists(video_path):
        return {
            'error': f"❌ Không tìm thấy video tại: {video_path}",
            'vlm_description': "N/A",
            'llm_output': "N/A",
            'thinking': None
        }

    try:
        torch.cuda.empty_cache()
        # 1. Check Cache VLM
        vlm_description, video_info = get_vlm_cache(video_path)
        
        # 2. Nếu chưa có Cache
        if vlm_description is None:
            print(f"▶️ Xử lý video: {os.path.basename(video_path)}")
            
            frames_queue = extract_frames_to_queue(video_path)
            frames, video_info = process_yolo_tracker(frames_queue, models['yolo'], tracker, batch_size=12)
            save_yolo_json(question_id, video_info)

            choices_str = "\n".join(question_data['choices'])
            vlm_prompt_content = f"{question_data['question']}\n{choices_str}"
            vlm_description = generate_video_description(frames, models, video_info, vlm_prompt_content)
            
            save_vlm_cache(video_path, vlm_description, video_info)
        else:
            print(f"Sử dụng Cache VLM: {question_id}")

        # 3. Chạy LLM
        print(f"Chạy LLM (Thinking: {'ON' if enable_thinking else 'OFF'})...")
        llm_answer, thinking, cache_data = llm_choise_answer(
            models, 
            vlm_description, 
            question_data, 
            video_info,
            enable_thinking=enable_thinking
        )
        
        return {
            'error': None,
            'vlm_description': vlm_description,
            'llm_output': llm_answer,
            'thinking': thinking,
            'cache_data': cache_data
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'error': f"Lỗi: {str(e)}",
            'vlm_description': "Error",
            'llm_output': "Error",
            'thinking': None
        }

# ==========================================
# GUI
# ==========================================

# Thêm style cho progress bar
def setup_styles():
    style = ttk.Style()
    style.theme_use('clam')  # Hoặc 'alt', 'default', 'classic'
    style.configure("Custom.Horizontal.TProgressbar",
                   foreground='#4CAF50',
                   background='#4CAF50',
                   troughcolor='#E0E0E0',
                   bordercolor='#BDBDBD',
                   lightcolor='#4CAF50',
                   darkcolor='#388E3C')
class TrafficQAGUI:
    def __init__(self, master):
        self.master = master
        setup_styles()
        master.title("🚦 Hệ thống VQA Luật Giao Thông")
        master.geometry("1200x900")
        
        self.selected_idx = None
        self.enable_thinking = tk.BooleanVar(value=False)  # Thinking mode
        
        # --- MAIN FRAME ---
        main_frame = tk.Frame(master, padx=10, pady=10)
        main_frame.pack(fill='both', expand=True)

        # 1. CONTROL PANEL
        control_frame = tk.LabelFrame(main_frame, text="⚙️ Điều khiển & Dữ liệu", 
                                       padx=5, pady=5, font=('Arial', 10, 'bold'))
        control_frame.pack(fill='x', pady=5)

        btn_load = tk.Button(control_frame, text="📂 Chọn File JSON", 
                            command=self.load_json_file, width=15)
        btn_load.pack(side='left', padx=5)
        
        self.lbl_status = tk.Label(control_frame, text="Chưa tải dữ liệu", fg="red")
        self.lbl_status.pack(side='left', padx=10)

        # Listbox
        list_frame = tk.Frame(control_frame)
        list_frame.pack(side='left', fill='both', expand=True, padx=10)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.lst_questions = tk.Listbox(list_frame, height=4, 
                                        yscrollcommand=scrollbar.set, exportselection=False)
        self.lst_questions.pack(side='left', fill='both', expand=True)
        self.lst_questions.bind('<<ListboxSelect>>', self.on_select_question)
        scrollbar.config(command=self.lst_questions.yview)

        # Thinking Mode Checkbox
        chk_thinking = tk.Checkbutton(control_frame, text="🧠 Thinking Mode", 
                                      variable=self.enable_thinking,
                                      font=('Arial', 10, 'bold'))
        chk_thinking.pack(side='right', padx=10)

        # Process Button
        self.btn_process = tk.Button(control_frame, text="▶️ XỬ LÝ CÂU HỎI", 
                                     command=self.start_processing, 
                                     state='disabled', bg='#4CAF50', fg='white', 
                                     font=('Arial', 11, 'bold'))
        self.btn_process.pack(side='right', padx=10, fill='y')

        # 2. QUESTION DISPLAY
        qa_frame = tk.LabelFrame(main_frame, text="📝 Nội dung Câu hỏi", 
                                 padx=5, pady=5, font=('Arial', 10, 'bold'))
        qa_frame.pack(fill='x', pady=5)
        
        self.txt_question = scrolledtext.ScrolledText(qa_frame, height=6, 
                                                      font=('Consolas', 10))
        self.txt_question.pack(fill='both', expand=True)
        self.txt_question.config(state='disabled')

        # 3. VLM DESCRIPTION
        vlm_frame = tk.LabelFrame(main_frame, text="🎥 Mô tả Video (VLM)", 
                                  padx=5, pady=5, font=('Arial', 10, 'bold'))
        vlm_frame.pack(fill='both', expand=True, pady=5)
        
        self.txt_vlm = scrolledtext.ScrolledText(vlm_frame, height=8, 
                                                 font=('Consolas', 10))
        self.txt_vlm.pack(fill='both', expand=True)
        self.txt_vlm.config(state='disabled')

        # 4. THINKING PROCESS (Collapsible)
        self.thinking_frame = tk.LabelFrame(main_frame, text="💭 Quá trình Suy luận (Thinking)", 
                                           padx=5, pady=5, font=('Arial', 10, 'bold'), fg='purple')
        # Don't pack initially - will show/hide dynamically
        
        self.txt_thinking = scrolledtext.ScrolledText(self.thinking_frame, height=10, 
                                                      font=('Consolas', 9), 
                                                      bg='#FFF9E6', fg='#333')
        self.txt_thinking.pack(fill='both', expand=True)
        self.txt_thinking.config(state='disabled')

        # 5. LLM RESULT
        llm_frame = tk.LabelFrame(main_frame, text="🎯 Kết quả Suy luận", 
                                  padx=5, pady=5, font=('Arial', 10, 'bold'), fg="blue")
        llm_frame.pack(fill='x', pady=5)
        
        self.lbl_result = tk.Label(llm_frame, text="Đáp án: ---", 
                           font=('Arial', 18, 'bold'), # Giảm nhẹ size nếu text dài
                           fg="blue",
                           anchor='w',       # Căn nội dung trong Label sang trái
                           justify='left')   # Căn lề trái nếu text có nhiều dòng
        self.lbl_result.pack(side='left', padx=20, pady=10, fill='x', expand=True) 
        
        self.lbl_time = tk.Label(llm_frame, text="Thời gian: ---", 
                                font=('Arial', 10))
        self.lbl_time.pack(side='right', padx=20)

        # Status Bar
        self.statusbar = tk.Label(master, text="Đang khởi tạo...", 
                                 bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.start_loading_models()
    def start_yolo_video(self, video_path):
        """Chỉ mở thread chạy OpenCV, không tạo Toplevel của Tkinter nữa"""
        if MODELS is None:
            messagebox.showwarning("Chưa sẵn sàng", "Mô hình chưa tải xong.")
            return

        # Start thread chạy video OpenCV trực tiếp
        thread = threading.Thread(target=self.thread_run_yolo_video, args=(video_path,))
        thread.daemon = True
        thread.start()

    def thread_run_yolo_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps > 0 else 33

        # Đặt tên cửa sổ và cho phép thay đổi kích thước
        window_name = f"YOLO Viewer - {os.path.basename(video_path)}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540) # Kích thước mặc định vừa mắt

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize để xử lý nhanh hơn trên RTX 5060 Ti
            frame_resized = cv2.resize(frame, (960, 540))

            try:
                # Chạy inference với TensorRT cực nhanh
                with torch.no_grad():
                    # model.predict sẽ tự nhận diện .engine hoặc .pt
                    results = MODELS['yolo'].predict(
                        frame_resized,
                        conf=0.4,
                        verbose=False
                    )[0]

                if results.boxes is not None:
                    frame_resized = results.plot()
            except Exception:
                pass

            cv2.imshow(window_name, frame_resized)

            # Nhấn 'q' hoặc đóng cửa sổ bằng dấu X để thoát
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyWindow(window_name)

    def start_loading_models(self):
        """Hiển thị cửa sổ loading với progress bar"""
        self.statusbar.config(text="⏳ Đang khởi tạo...")
        
        # Tạo cửa sổ loading popup
        self.loading_window = tk.Toplevel(self.master)
        self.loading_window.title("Đang tải mô hình AI")
        self.loading_window.geometry("500x250")
        self.loading_window.resizable(False, False)
        self.loading_window.transient(self.master)
        self.loading_window.grab_set()
        
        # Center window
        self.loading_window.update_idletasks()
        x = (self.loading_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (self.loading_window.winfo_screenheight() // 2) - (250 // 2)
        self.loading_window.geometry(f"500x250+{x}+{y}")
        
        # Content frame
        content_frame = tk.Frame(self.loading_window, padx=30, pady=30)
        content_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = tk.Label(content_frame, 
                              text="🤖 Đang tải mô hình AI...",
                              font=('Arial', 16, 'bold'),
                              fg='#2196F3')
        title_label.pack(pady=(0, 20))
        
        # Current task label
        self.loading_label = tk.Label(content_frame,
                                     text="Đang chuẩn bị...",
                                     font=('Arial', 11),
                                     fg='#666')
        self.loading_label.pack(pady=(0, 10))
        
        # Progress bar (indeterminate style)
        self.progress_bar = ttk.Progressbar(content_frame,
                                   mode='determinate',
                                   style="Custom.Horizontal.TProgressbar",  # Thêm style
                                   length=400,
                                   maximum=100)
        self.progress_bar.pack(pady=(0, 10))
        
        # Percentage label
        self.percentage_label = tk.Label(content_frame,
                                        text="0%",
                                        font=('Arial', 12, 'bold'),
                                        fg='#4CAF50')
        self.percentage_label.pack()
        
        # Detail info
        self.detail_label = tk.Label(content_frame,
                                     text="Vui lòng đợi, quá trình này có thể mất 30-60 giây...",
                                     font=('Arial', 9),
                                     fg='#999')
        self.detail_label.pack(pady=(10, 0))
        
        # Start loading thread
        thread = threading.Thread(target=self.thread_load_models)
        thread.daemon = True
        thread.start()

    def update_loading_progress(self, percentage, message):
        """Cập nhật progress bar từ thread"""
        self.progress_bar['value'] = percentage
        self.loading_label.config(text=message)
        self.percentage_label.config(text=f"{percentage}%")

    def thread_load_models(self):
        global MODELS
        try:
            def progress_callback(percentage, message):
                self.master.after(0, lambda p=percentage, m=message: 
                                self.update_loading_progress(p, m))
            
            MODELS = load_models(progress_callback=progress_callback)
            
            self.master.after(0, lambda: self.on_models_loaded(True))
            
        except Exception as e:
            import traceback
            traceback.print_exc()  # IN RA LỖI THẬT
            err_msg = str(e) if str(e) else "Lỗi không xác định khi load mô hình"
            self.master.after(0, lambda: self.on_models_loaded(False, err_msg))

    def on_models_loaded(self, success, error_msg=""):
        """Đóng loading window và thông báo kết quả"""
        if hasattr(self, 'loading_window') and self.loading_window.winfo_exists():
            self.loading_window.destroy()
        
        if success:
            self.statusbar.config(text="Mô hình đã sẵn sàng.")
            messagebox.showinfo("Thành công", 
                              "Đã tải xong tất cả mô hình AI!\n\nBạn có thể bắt đầu sử dụng hệ thống.")
        else:
            self.statusbar.config(text="Lỗi tải mô hình.")
            messagebox.showerror("Lỗi", f"Không thể tải mô hình:\n\n{error_msg}")

    def load_json_file(self):
        global ALL_QUESTIONS, CURRENT_VIDEO_FOLDER
        
        filename = filedialog.askopenfilename(
            title="Chọn file JSON câu hỏi",
            filetypes=(("JSON Files", "*.json"), ("All Files", "*.*"))
        )
        
        if not filename:
            return

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'data' in data:
                ALL_QUESTIONS = data['data']
            elif isinstance(data, list):
                ALL_QUESTIONS = data
            else:
                raise ValueError("Format JSON không hợp lệ.")

            CURRENT_VIDEO_FOLDER = filename
            
            self.lst_questions.delete(0, tk.END)
            for q in ALL_QUESTIONS:
                self.lst_questions.insert(tk.END, f"{q['id']}")
            
            self.lbl_status.config(text=f"Đã tải: {os.path.basename(filename)} ({len(ALL_QUESTIONS)} câu)", 
                                  fg="green")
            self.statusbar.config(text=f"Đã tải {len(ALL_QUESTIONS)} câu hỏi.")
            self.btn_process.config(state='normal')
            
        except Exception as e:
            messagebox.showerror("Lỗi File", f"Không đọc được file JSON:\n{e}")

    def on_select_question(self, event):
        selection = self.lst_questions.curselection()
        if not selection:
            return
        
        index = selection[0]
        self.selected_idx = index
        q_data = ALL_QUESTIONS[index]
        
        self.txt_question.config(state='normal')
        self.txt_question.delete('1.0', tk.END)

        self.txt_question.insert(tk.END, f"ID: {q_data['id']}\n")
        self.txt_question.insert(tk.END, f"Video: {q_data['video_path']}\n")
        self.txt_question.insert(tk.END, "-"*50 + "\n")
        self.txt_question.insert(tk.END, f"CÂU HỎI: {q_data['question']}\n\n")
        self.txt_question.insert(tk.END, "LỰA CHỌN:\n")

        true_answer = q_data.get("answer")
        if true_answer:
            true_answer = true_answer.strip().upper()
            if '.' in true_answer:
                true_answer = true_answer.split('.')[0]

        # Tag cho đáp án đúng
        self.txt_question.tag_configure(
            "correct_answer",
            foreground="red",
            font=('Consolas', 10, 'bold')
        )

        for choice in q_data['choices']:
            start = self.txt_question.index(tk.END)

            key, val = choice.split('.', 1)
            key = key.strip().upper()
            text = f"{key}: {val.strip()}\n"

            self.txt_question.insert(tk.END, text)

            if true_answer and key == true_answer:
                end = self.txt_question.index(tk.END)
                self.txt_question.tag_add("correct_answer", start, end)
        self.txt_question.config(state='disabled')

        
    def start_processing(self):
        if self.selected_idx is None:
            messagebox.showwarning("Chưa chọn", "Vui lòng chọn câu hỏi.")
            return
        
        if MODELS is None:
            messagebox.showwarning("Chưa sẵn sàng", "Mô hình chưa tải xong.")
            return

        self.btn_process.config(state='disabled', text="⏳ Đang xử lý...")
        self.statusbar.config(text=f"Đang xử lý câu hỏi ID: {ALL_QUESTIONS[self.selected_idx]['id']}...")
        
        thread = threading.Thread(target=self.thread_process_question)
        thread.daemon = True
        thread.start()

    def thread_process_question(self):
        start_time = time.time()
        q_data = ALL_QUESTIONS[self.selected_idx]
        
        result = process_single_question_logic(
            q_data, 
            MODELS, 
            TRACKER,
            enable_thinking=self.enable_thinking.get()
        )
        
        elapsed_time = time.time() - start_time
        
        self.master.after(0, lambda: self.update_ui_after_process(result, elapsed_time))

    def update_ui_after_process(self, result, elapsed_time):
        self.btn_process.config(state='normal', text="XỬ LÝ CÂU HỎI")
        
        if result.get('error'):
            self.statusbar.config(text="Xảy ra lỗi khi xử lý.")
            messagebox.showerror("Lỗi Xử lý", result['error'])
            return

        # HIỂN THỊ VIDEO TẠI ĐÂY (Sau khi chạy xong LLM)
        q_data = ALL_QUESTIONS[self.selected_idx]
        video_rel_path = q_data['video_path']
        base_dir = os.path.dirname(CURRENT_VIDEO_FOLDER) if CURRENT_VIDEO_FOLDER else ""
        video_path = os.path.join(base_dir, video_rel_path)

        if os.path.exists(video_path):
            self.start_yolo_video(video_path)

        # VLM Description
        self.txt_vlm.config(state='normal')
        self.txt_vlm.delete('1.0', tk.END)
        self.txt_vlm.insert('1.0', result['vlm_description'])
        self.txt_vlm.config(state='disabled')
        
        # Thinking Process
        if result.get('thinking'):
            self.txt_thinking.config(state='normal')
            self.txt_thinking.delete('1.0', tk.END)
            self.txt_thinking.insert('1.0', result['thinking'])
            self.txt_thinking.config(state='disabled')
            self.thinking_frame.pack(fill='both', expand=True, pady=5, after=self.txt_vlm.master)
        else:
            self.thinking_frame.pack_forget()
        
        # LLM Result
        ans_key = result['llm_output']
        q_data = ALL_QUESTIONS[self.selected_idx]

        # Lấy đầy đủ văn bản cho đáp án dự đoán
        pred_answer = map_answer_text(ans_key, q_data['choices'])
        
        # Lấy đáp án thực tế từ JSON
        raw_true_answer = q_data.get("answer", "")
        
        # Nếu đáp án thực tế chỉ là "A", "B"... thì map nó sang văn bản đầy đủ
        # Nếu nó đã là văn bản đầy đủ "A. Nội dung...", hàm map sẽ xử lý chuẩn hóa
        true_answer_full = map_answer_text(raw_true_answer, q_data['choices'])
        
        if raw_true_answer:
            # So sánh dựa trên ký tự đầu (A, B, C...) để chính xác tuyệt đối
            is_correct = ans_key.strip().upper() == raw_true_answer.strip().upper()
            
            display_text = f"Dự đoán: {pred_answer}\nThực tế:  {true_answer_full}"
            self.lbl_result.config(
                text=display_text,
                fg="#2E7D32" if is_correct else "#C62828"
            )
        else:
            self.lbl_result.config(
                text=f"Dự đoán: {pred_answer}",
                fg="blue"
            )
        self.lbl_time.config(text=f"Thời gian: {elapsed_time:.2f}s")
        
        think_mode = "ON" if self.enable_thinking.get() else "OFF"
        self.statusbar.config(text=f"Hoàn tất. Thời gian: {elapsed_time:.2f}s | Thinking: {think_mode}")
def map_answer_text(answer_key, choices):
    """
    answer_key: 'A', 'B', 'C', ...
    choices: ['A. ...', 'B. ...']
    """
    if not answer_key:
        return answer_key

    answer_key = answer_key.strip().upper()

    for c in choices:
        if c.strip().startswith(answer_key + "."):
            return c.strip()

    return answer_key

# ==========================================
# MAIN
# ==========================================
def main():
    root = tk.Tk()
    app = TrafficQAGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
