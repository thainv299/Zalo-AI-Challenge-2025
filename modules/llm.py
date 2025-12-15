from utils.cached_helper import save_json
import re
import torch

def rerank(reranker, query, docs, k=5):
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:k]]

def format_docs(docs):
    out = ""
    for d in docs:
        bien = d.metadata.get("bien_so", "")
        out += f"[Biển số: {bien}]\n{d.page_content.strip()}\n\n"
    return out.strip()

def normalize_vehicle_names(yolo_keywords):
    """Chuẩn hóa tên phương tiện"""
    synonym_map = {
        "xe lam": ["xe ba bánh", "xe máy ba bánh", "xe 3 bánh"],
        "auto_rickshaw": ["xe ba bánh", "xe máy ba bánh", "xe lam"],
        "xe mô tô": ["xe máy", "xe gắn máy", "mô tô"],
        "xe máy": ["xe mô tô", "xe gắn máy", "mô tô"],
        "motorcycle": ["xe máy", "xe mô tô", "mô tô"],
        "xe đạp": ["bicycle", "bike"],
        "bicycle": ["xe đạp"],
        "ô tô": ["xe hơi", "xe ô tô", "car"],
        "car": ["ô tô", "xe hơi"],
        "xe tải": ["truck"],
        "truck": ["xe tải"],
    }
    
    expanded = list(yolo_keywords)
    for kw in yolo_keywords:
        kw_lower = kw.lower()
        for term, syns in synonym_map.items():
            if term in kw_lower:
                expanded.extend(syns)
    return list(set(expanded))

def create_messages(context, video_description, question, choices, enable_thinking=False):
    """
    ✅ FULL VIETNAMESE - Consistent language throughout
    enable_thinking: Nếu True, cho phép LLM suy luận trước khi trả lời
    """
    
    if enable_thinking:
        system_prompt = """Bạn là chuyên gia luật giao thông Việt Nam.

NHIỆM VỤ: Trả lời câu hỏi trắc nghiệm về luật giao thông.

KIẾN THỨC QUAN TRỌNG:
1. Biển phụ S.501 "Phạm vi tác dụng của biển":
   - Biển này CHỈ chỉ khoảng cách (50m, 100m, 200m, 500m, v.v.)
   - Ví dụ: W.225 (Cảnh báo trẻ em) + S.501 (200m) = "Cảnh báo trẻ em trong phạm vi 200 mét phía trước"
   - Khi thấy biển chính + S.501 có số, quy định áp dụng cho khoảng cách đó

2. Biển phụ S.508a "Biểu thị thời gian":
   - Biển này CHỈ thời gian áp dụng của biển chính
   - Ví dụ: P.124b1 (Cấm quay đầu) + S.508a = "Cấm quay đầu trong khung thời gian quy định"
   - Khi thấy biển cấm/hạn chế + S.508a → Thêm "trong khung thời gian" vào đáp án

3. Phân biệt RẼ TRÁI vs QUAY ĐẦU (QUAN TRỌNG!):
   - "Rẽ trái" = Turn left = Chuyển sang đường bên trái (P.123)
   - "Quay đầu/Quay xe" = U-turn = Quay ngược lại 180 độ (P.124)
   - Đây là HAI HÀNH ĐỘNG HOÀN TOÀN KHÁC NHAU!

4. Tên gọi khác nhau của cùng 1 loại xe:
   - "Xe lam" = "Xe ba bánh" = "Xe máy ba bánh" = "AUTO_RICKSHAW" (cùng loại xe 3 bánh)
   - "Xe máy" = "Xe mô tô" = "Xe gắn máy" = "MOTORCYCLE" (cùng loại)
   - "Xe đạp" = "BICYCLE"
   - "Ô tô" = "CAR" = "Xe hơi"
   - "Xe tải" = "TRUCK"

QUY TẮC:
1. Trước tiên, PHÂN TÍCH câu hỏi trong phần <thinking>...</thinking>
2. Sau đó, trả lời MỘT chữ cái (A, B, C, hoặc D) trong phần <answer>...</answer>
3. **ĐỌC KỸ TÊN BIỂN trong YOLO detections - Tên biển ĐÃ CHO BIẾT ý nghĩa!**
4. **Với câu hỏi "tồn tại" (Có... không?): Ưu tiên thông tin YOLO TRƯỚC**
5. Với câu hỏi về luật/quy định: Ưu tiên NGỮ CẢNH PHÁP LÝ trước

FORMAT TRẢ LỜI:
<thinking>
[Phân tích câu hỏi, YOLO detections, ngữ cảnh pháp lý...]
</thinking>

<answer>A</answer>"""
    else:
        system_prompt = """Bạn là chuyên gia luật giao thông Việt Nam.

NHIỆM VỤ: Trả lời câu hỏi trắc nghiệm về luật giao thông.

KIẾN THỨC QUAN TRỌNG:
1. Biển phụ S.501 "Phạm vi tác dụng của biển":
   - Biển này CHỈ chỉ khoảng cách (50m, 100m, 200m, 500m, v.v.)
   - Ví dụ: W.225 (Cảnh báo trẻ em) + S.501 (200m) = "Cảnh báo trẻ em trong phạm vi 200 mét phía trước"

2. Biển phụ S.508a "Biểu thị thời gian":
   - Biển này CHỈ thời gian áp dụng của biển chính

3. Phân biệt RẼ TRÁI vs QUAY ĐẦU (QUAN TRỌNG!):
   - "Rẽ trái" = Turn left = P.123
   - "Quay đầu/Quay xe" = U-turn = P.124

4. Tên gọi khác nhau của cùng 1 loại xe:
   - "Xe lam" = "Xe ba bánh" = "AUTO_RICKSHAW"
   - "Xe máy" = "Xe mô tô" = "MOTORCYCLE"

QUY TẮC:
1. Chỉ trả lời MỘT chữ cái (A, B, C, hoặc D)
2. KHÔNG giải thích, KHÔNG văn bản thêm
3. **ĐỌC KỸ TÊN BIỂN trong YOLO detections**
4. **Ưu tiên: YOLO > VIDEO > NGỮ CẢNH**"""

    # Examples (rút gọn khi không thinking)
    if enable_thinking:
        ex1 = """mô_tả_video: YOLO: - TRAFFIC_LIGHT_GREEN
Cảnh: Đường phố thành phố ban đêm.

câu_hỏi: Phía trước có đèn giao thông không?
lựa_chọn:
A. Có
B. Không

<ngữ_cảnh>
[Biển số: Đèn vàng]
Tín hiệu đèn màu vàng phải dừng lại...
</ngữ_cảnh>

Trả lời:"""
        ex1_answer = """<thinking>
YOLO phát hiện: TRAFFIC_LIGHT_GREEN (đèn xanh)
Câu hỏi: Có đèn giao thông không?
→ YOLO đã phát hiện đèn → Có đèn
</thinking>

<answer>A</answer>"""
    else:
        ex1 = """mô_tả_video: YOLO: - TRAFFIC_LIGHT_GREEN
câu_hỏi: Phía trước có đèn giao thông không?
lựa_chọn: A. Có | B. Không
Trả lời:"""
        ex1_answer = "A"

    # Câu hỏi thực tế
    real_content = f"""mô_tả_video: {video_description}

câu_hỏi: {question}
lựa_chọn:
{choices}

<ngữ_cảnh>
{context}
</ngữ_cảnh>

Trả lời:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ex1},
        {"role": "assistant", "content": ex1_answer},
        {"role": "user", "content": real_content}
    ]
    
    return messages

def parse_llm_output(raw_output, enable_thinking=False):
    """
    Phân tích output của LLM
    Returns: (answer: str, thinking: str or None)
    """
    thinking = None
    answer = None
    
    if enable_thinking:
        # Trích xuất thinking
        think_match = re.search(r'<thinking>(.*?)</thinking>', raw_output, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
        
        # Trích xuất answer
        ans_match = re.search(r'<answer>([ABCD])</answer>', raw_output, re.IGNORECASE)
        if ans_match:
            answer = ans_match.group(1).upper()
    
    # Fallback: Tìm A/B/C/D bất kỳ
    if not answer:
        match = re.search(r'\b([ABCD])\b', raw_output.upper())
        if match:
            answer = match.group(1)
        else:
            answer = raw_output[0].upper() if raw_output and raw_output[0].upper() in "ABCD" else "A"
    
    return answer, thinking

def llm_choise_answer(models, vlm_description: str, question_data, box_info: str = "", enable_thinking: bool = False) -> tuple:
    """
    Returns: (answer: str, thinking: str or None, cache_data: dict)
    """
    llm = models['llm']
    tokenizer = models['llm_tokenizer']
    retriever = models['retriever']
    reranker = models['reranker']

    question = question_data["question"]
    choices = question_data["choices"]

    if isinstance(choices, list):
        choices = "\n".join(choices)
    
    # ========== EXTRACT YOLO KEYWORDS ==========
    yolo_keywords = []
    distance_info = None
    
    if "YOLO:" in vlm_description:
        yolo_part = vlm_description.split("YOLO:")[1]
        if "Scene:" in vlm_description or "Cảnh:" in vlm_description:
            yolo_part = yolo_part.split("Scene:")[0] if "Scene:" in vlm_description else yolo_part.split("Cảnh:")[0]
        
        for line in yolo_part.split("\n"):
            if line.strip().startswith("-"):
                obj = line.strip()[1:].strip()
                if obj and obj not in ["None", "Không có"]:
                    yolo_keywords.append(obj)
                    
                    # Extract S.501 distance
                    if "S.501" in obj or "Phạm vi" in obj:
                        match = re.search(r'(\d+)\s*m', obj)
                        if match:
                            distance_info = f"{match.group(1)}m"
                        else:
                            numbers = re.findall(r'\d+', obj)
                            if numbers:
                                distance_info = f"{numbers[0]}m"
    
    # ========== RETRIEVAL OPTIMIZATION ==========
    sign_codes = []
    for kw in yolo_keywords:
        match = re.match(r'([A-Z]\.\d+[a-z]?\d?)', kw)
        if match:
            sign_codes.append(match.group(1))
    
    # Strategy 1: Sign codes (exact match)
    if sign_codes:
        sign_query = " ".join(sign_codes)
        docs_signs = retriever.invoke(sign_query)[:8]
    else:
        docs_signs = []
    
    # Strategy 2: Question (semantic)
    docs_question = retriever.invoke(question)[:4]
    
    docs = docs_signs + docs_question
    
    # Normalize vehicle names
    yolo_keywords_normalized = normalize_vehicle_names(yolo_keywords)
    
    # Rerank
    if sign_codes:
        yolo_str = ", ".join(sign_codes) + " | " + ", ".join(yolo_keywords_normalized[:3])
    else:
        yolo_str = ", ".join(yolo_keywords_normalized[:5]) if yolo_keywords_normalized else "Không có"
    
    rank_prompt = f"Câu hỏi: {question}\nYOLO: {yolo_str}"
    if distance_info:
        rank_prompt += f"\nKhoảng cách: {distance_info}"
    
    top_docs = rerank(reranker, rank_prompt, docs, k=5)
    context = format_docs(top_docs)
    
    # ========== CREATE MESSAGES ==========
    messages = create_messages(context, vlm_description, question, choices, enable_thinking)
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Internal parameter, không dùng
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(llm.device)
    
    # ========== GENERATION ==========
    max_tokens = 256 if enable_thinking else 3  # Thinking mode cần nhiều tokens hơn
    
    with torch.inference_mode():  # Faster than torch.no_grad()
        generated_ids = llm.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True  # Enable KV cache
        )

    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    raw_answer = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
    
    # ========== PARSE OUTPUT ==========
    answer, thinking = parse_llm_output(raw_answer, enable_thinking)
    
    # Verify "đầy đủ nhất"
    if "đầy đủ" in question.lower():
        if isinstance(choices, str):
            choice_lines = [l.strip() for l in choices.split('\n') if l.strip()]
            choice_dict = {}
            for line in choice_lines:
                if line and line[0] in "ABCD":
                    choice_dict[line[0]] = len(line[2:].strip())
            
            if choice_dict:
                longest = max(choice_dict, key=choice_dict.get)
                curr_len = choice_dict.get(answer, 0)
                if curr_len < choice_dict[longest] * 0.7:
                    print(f"Chọn {answer} ({curr_len}ch) < {longest} ({choice_dict[longest]}ch)")
    
    # ========== SAVE CACHE ==========
    data_cache = {
        "vlm_description": vlm_description,
        "yolo_keywords": yolo_keywords,
        "yolo_normalized": yolo_keywords_normalized,
        "distance_info": distance_info,
        "context": context,
        "question": question,
        "choices": choices,
        "raw_output": raw_answer,
        "thinking": thinking,
        "final_answer": answer,
        "thinking_enabled": enable_thinking
    }
    save_json(data_cache, f"{question_data['id']}.json")
    
    dist_str = f" | S.501: {distance_info}" if distance_info else ""
    think_str = f" | Thinking: {len(thinking)} chars" if thinking else ""
    print(f"LLM: {repr(raw_answer[:50])}... → {answer}{dist_str}{think_str}")
    
    return answer, thinking, data_cache