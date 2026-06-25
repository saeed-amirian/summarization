import os
import warnings

# تنظیمات محیطی برای حذف هشدارهای اضافه
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

from transformers import BartForConditionalGeneration, BartTokenizer

# انتخاب مدل اصلی و قدرتمند فیس‌بوک
model_name = "facebook/bart-large-cnn"

print(f"Loading {model_name}...")
print("Please wait, this might take a moment if it's the first time...")

# بارگذاری Tokenizer و Model
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# متن ورودی
text = """
The field of Natural Language Processing (NLP) has seen incredible growth 
in recent years. Modern architectures like BART, which stands for 
Bidirectional and Auto-Regressive Transformers, have set new standards 
in sequence-to-sequence tasks. These models are pre-trained on massive 
amounts of data, allowing them to understand context and nuance in human 
language better than ever before. This enables applications like 
high-quality translation and precise text summarization.
"""

# آماده‌سازی متن برای مدل
inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)

# تولید خلاصه با تنظیمات بهینه برای کوتاه کردن متن
summary_ids = model.generate(
    inputs["input_ids"], 
    num_beams=4, 
    max_length=50,      # حداکثر طول خلاصه
    min_length=10,     # حداقل طول خلاصه
    length_penalty=2.5, # عدد بزرگتر باعث خلاصه شدن بیشتر می‌شود
    early_stopping=True
)

# تبدیل کد به متن قابل خواندن
summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

print("\n" + "="*30)
print("ORIGINAL TEXT LENGTH:", len(text.split()), "words")
print("-" * 30)
print("FINAL SUMMARY:")
print(summary[0])
print("-" * 30)
print("SUMMARY LENGTH:", len(summary[0].split()), "words")
print("="*30)
