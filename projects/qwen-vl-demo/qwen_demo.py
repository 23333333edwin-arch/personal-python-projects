from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import gradio as gr

# === 加载模型（只加载一次）===
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

print("正在加载模型，请稍候...")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)
print("✅ 模型加载完成！")

# === 预设系统 Prompt 模板列表 ===
SYSTEM_PROMPT_TEMPLATES = {
    "通用助手": "你是一个有用的人工智能助手，能够回答问题、描述图像、进行推理等。",
    "翻译专家": "你是一个专业的翻译助手，请将用户输入准确翻译成目标语言，保持语义和语气一致。",
    "数学老师": "你是一个耐心的数学老师，请用清晰、分步的方式解释数学问题，适合中学生理解。",
    "图像分析师": "你是一个图像分析专家，请详细描述图片中的所有物体、场景、颜色、动作和可能的上下文。",
    "科普作家": "你是一个科普作家，用通俗易懂、生动有趣的语言解释复杂的科学概念，避免术语。",
    "编程助手": "你是一个资深程序员，请帮助用户编写、调试或解释代码，提供最佳实践建议。",
    "创意写手": "你是一个创意写手，请根据用户要求创作故事、诗歌、广告文案或社交媒体内容，风格自由。",
    "冷静AI": "你是一个冷静、理性、不带情感的AI助手，只提供事实和逻辑分析，不安慰、不鼓励。"
}

# === 推理函数 ===
def predict(image, user_prompt, system_prompt):
    # 如果系统 Prompt 为空，使用默认值
    if not system_prompt.strip():
        system_prompt = SYSTEM_PROMPT_TEMPLATES["通用助手"]
    
    if image is None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text_input],
            return_tensors="pt"
        ).to(model.device)
    else:
        image_pil = image.convert("RGB")
        image_pil.thumbnail((384, 384), Image.Resampling.LANCZOS)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            images=image_pil,
            text=[text_input],
            return_tensors="pt"
        ).to(model.device)

    input_len = inputs["input_ids"].shape[1]  # ⭐ 关键：记录输入长度

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # ⭐ 只取新生成的部分（跳过输入）
    generated_tokens = outputs[0][input_len:]
    response = processor.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()

# === 应用模板函数：点击按钮后自动填充系统 Prompt ===
def apply_template(template_key):
    return SYSTEM_PROMPT_TEMPLATES.get(template_key, SYSTEM_PROMPT_TEMPLATES["通用助手"])

# === Gradio 界面 ===
with gr.Blocks(title="Qwen-VL 助手") as demo:
    gr.Markdown("# 🖼️ Qwen2.5-VL 视觉语言助手")
    gr.Markdown("上传一张图片并提问，或直接输入文本问题。选择预设角色，一键优化模型行为。")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="上传图片（可选）")
        
        with gr.Column(scale=2):
            # 👇 新增：角色模板选择器（单选按钮）
            template_radio = gr.Radio(
                choices=list(SYSTEM_PROMPT_TEMPLATES.keys()),
                value="通用助手",
                label="选择角色模板",
                interactive=True
            )
            
            # 👇 新增：应用模板按钮
            apply_btn = gr.Button("✨ 应用模板", variant="primary")
            
            # 👇 系统 Prompt 输入框（由按钮自动填充，也可手动修改）
            system_prompt_input = gr.Textbox(
                label="系统 Prompt（可编辑）",
                placeholder="选择模板后自动填充，也可手动修改...",
                value=SYSTEM_PROMPT_TEMPLATES["通用助手"],
                lines=4,
                max_lines=12
            )
            
            text_input = gr.Textbox(
                label="你的问题", 
                placeholder="例如：描述这张图片", 
                lines=2
            )
            submit_btn = gr.Button("🚀 提交", variant="primary")
    
    output = gr.Textbox(
        label="模型回答",
        interactive=False,
        lines=6,
        max_lines=30,
        placeholder="模型的回答将显示在这里...",
        show_copy_button=True
    )

    # 👇 绑定：点击“应用模板”按钮 → 填充系统 Prompt
    apply_btn.click(
        fn=apply_template,
        inputs=template_radio,
        outputs=system_prompt_input
    )

    # 👇 绑定：提交按钮 → 执行推理
    submit_btn.click(
        fn=predict,
        inputs=[image_input, text_input, system_prompt_input],
        outputs=output
    )

    # 可选：保留一个简单示例（仅作演示）
    gr.Markdown("### 💡 小贴士")
    gr.Markdown("- 上传图片 + 选择「图像分析师」→ 获取详细描述\n- 输入中文 + 选择「翻译专家」→ 自动翻译\n- 提问数学题 + 选择「数学老师」→ 分步讲解")

# 启动（局域网可访问）
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

# 在浏览器通过 http://localhost:7860 访问此项目
    