from qwen_vl_utils import process_vision_info
from PIL import Image
import json

def format_yolo_for_vlm(box_info):
    """Format YOLO data with special markers to preserve Vietnamese."""
    if isinstance(box_info, str):
        try:
            box_info = json.loads(box_info)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return "No valid detection data"
    
    if not box_info or len(box_info) == 0:
        return "NO OBJECTS DETECTED"
    
    formatted_lines = []
    for obj in box_info:
        obj_type = obj.get('object_type', 'Unknown')
        confidence = obj.get('confidence', 0.0)
        track_id = obj.get('track_id', 'N/A')
        
        line = f"• Object {track_id}: «{obj_type}» (confidence: {confidence:.0%})"
        formatted_lines.append(line)
    
    return "\n".join(formatted_lines)


def generate_video_description(frames, models, box_info, question):
    try:
        model = models['vlm']
        processor = models['vlm_processor']
        
        # Validate frames
        if not frames or len(frames) == 0:
            return "No frames available."
            
        valid_frames = [frame for frame in frames if frame is not None]
        if not valid_frames:
            return "No valid frames."
            
        # Convert to PIL Images
        if hasattr(valid_frames[0], 'shape'):
            frames = [Image.fromarray(frame.astype("uint8")).convert("RGB") 
                     for frame in valid_frames]
        else:
            frames = valid_frames
            
        if len(frames) == 0:
            return "No valid frames."
        
        # Parse YOLO JSON
        if isinstance(box_info, str):
            try:
                box_data = json.loads(box_info)
            except:
                box_data = []
        else:
            box_data = box_info
        
        # Format YOLO detections
        if not box_data:
            yolo_text = "None"
        else:
            yolo_lines = []
            for obj in box_data:
                yolo_lines.append(f"- {obj.get('object_type', 'Unknown')}")
            yolo_text = "\n".join(yolo_lines)
        
        print("\n" + "="*50)
        print("YOLO DETECTIONS:")
        print(yolo_text)
        print("="*50 + "\n")
        
        # ✅ IMPROVED PROMPT - Focus on text reading + key info
        instruction_text = (
            f"Detected objects:\n{yolo_text}\n\n"
            
            "Describe the traffic scene. Answer these questions:\n\n"
            
            "1. ROAD & MARKINGS:\n"
            "   - Road type? (highway/city street/intersection/rural)\n"
            "   - Lane markings? (solid/dashed, white/yellow)\n"
            "   - Number of lanes?\n\n"
            
            "2. TRAFFIC SIGNS (IMPORTANT - READ TEXT!):\n"
            "   - Traffic lights? (red/yellow/green, position)\n"
            "   - Direction signs:\n"
            "     * READ TEXT ON SIGNS! (road names, destinations)\n"
            "     * Examples: 'Xa Lộ Hà Nội', 'Dầu Giây', 'Đường Đỗ Xuân Hợp'\n"
            "     * Arrow directions (straight/left/right)\n"
            "   - Warning/regulatory signs: type and position\n\n"
            
            "3. VEHICLES:\n"
            "   - Vehicles around? (ego vehicle, front, rear, left, right)\n"
            "   - Vehicle types and actions\n\n"
            
            "4. OTHER:\n"
            "   - Time of day? (day/dusk/night)\n"
            "   - Any visible violations?\n\n"
            
            "**CRITICAL: If you see text on direction signs, READ IT EXACTLY!**\n"
            "Answer in Vietnamese for better accuracy. Be specific about sign text and locations."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": instruction_text}
                ]
            }
        ]
        
        # Preparation
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # ✅ Generation - Increased tokens for text reading
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=200,      # Tăng từ 120 lên 200 cho OCR
            do_sample=False,         # Greedy decoding
            repetition_penalty=1.3,  # Giảm từ 1.5 để cho phép lặp tên đường
            no_repeat_ngram_size=3,  # Giảm từ 5 xuống 3
            eos_token_id=processor.tokenizer.eos_token_id, 
            pad_token_id=processor.tokenizer.pad_token_id
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        result = output_text[0].strip() if isinstance(output_text, list) else str(output_text).strip()
        
        # Kết hợp YOLO + VLM description
        final_output = f"YOLO: {yolo_text}\n\nScene: {result}"
        
        print("VLM OUTPUT (with text reading):", final_output[:300])
        
        return final_output
            
    except Exception as e:
        print(f"❌ VLM Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Error"