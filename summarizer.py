import os
import warnings

# ۱. تنظیمات محیطی برای حذف هشدارهای اضافه
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

from transformers import BartForConditionalGeneration, BartTokenizer

# ۲. انتخاب مدل اصلی و قدرتمند فیس‌بوک
model_name = "facebook/bart-large-cnn"

print(f"Loading {model_name}...")
print("Please wait, this might take a moment if it's the first time...")

# ۳. بارگذاری Tokenizer و Model
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# ۴. متن ورودی (می‌توانی متن طولانی‌تری هم اینجا بگذاری)
text = """
For decades, leaders of Arab nations in the Persian Gulf viewed their relationship with the United States as a strategic partnership. Donald Trump often saw it differently.

“King, we’re protecting you. You might not be there for two weeks without us. You have to pay for your military,” Trump said in 2018, speaking of the Saudi monarch and encapsulating a more transactional vision of a relationship that Gulf leaders had long regarded as a cornerstone of their security.

A year later, Saudi Arabia suffered the biggest attack on its territory in decades when strikes on key oil facilities temporarily knocked out roughly half of the kingdom’s crude production, sending global oil prices soaring. While Washington blamed Iran and condemned the attack, Gulf states were left with lingering questions about the extent of American willingness to confront Tehran on their behalf.

By Trump’s second term, Gulf leaders had taken note. As Gulf states pledged trillions of dollars in investment in the US economy, Trump chose the region for his first official trip abroad.

“We are going to protect this country,” the US president declared in the Qatari capital Doha during his Gulf tour last May.
That pledge faced its biggest test this year. Despite Gulf states’ efforts to avoid a regional conflict, the US – alongside Israel – launched a war against Iran, triggering ferocious retaliatory attacks across the Gulf and forcing regional governments to confront once again the question of what American protection really means.
US Secretary of State Marco Rubio arrived in the region on Tuesday, with the unenviable task of convincing Gulf states that Washington’s security commitments remain intact. Yet for many in the Gulf, the question is no longer whether Washington remains committed to their security, but whether the emerging agreement with Iran leaves them better or worse off than they were before the war.

“From the Arab Gulf states’ perspective, the Iran war is a disastrous turning point for the regional security order,” said Hasan Alhasan, senior fellow at the International Institute for Strategic Studies (IISS), who sees the agreement as part of a broader US retrenchment from the region. “US disengagement from the Gulf and the flow of financial and economic resources to Iran are likely to embolden Tehran further.”

“Nonetheless, the Arab Gulf states have facilitated and supported the Iran-US ceasefire deal. For them, a bad deal is still preferable to war,” he told CNN.
‘We want to hear their thoughts’
Rubio’s tour includes the United Arab Emirates, Bahrain and Kuwait, three Gulf nations that bore the brunt of Iranian attacks during the war and are likely among the most skeptical of the emerging détente between Washington and Tehran.
“We want to hear their thoughts, especially in the aftermath of this weekend in Switzerland, and make sure that their views are taken into account in every decision that we make, because they’re our partners,” Rubio told reporters upon landing in Abu Dhabi, referring to the agreement.
Gulf states had opposed the 2015 Iran nuclear agreement reached under the Obama administration and cheered Trump when he tore it up in 2018 because it didn’t address their concerns. The emerging US-Iran pact is likely to generate even greater unease in Gulf capitals, not only because it leaves many of those concerns unresolved, but because it comes amid what Alhasan described as a “major loss of confidence in the US.” A senior Gulf diplomat told CNN the conflict showed that “Iran had a well-developed plan to target” Gulf states.

The agreement grants Tehran a formal role in overseeing commercial traffic through the Strait of Hormuz alongside Oman. This means that much of the Gulf states’ maritime trade – and crucially their energy exports – could be conducted with Iranian oversight.
The pact also fails to address Iran’s missile program and its network of proxy militant groups – concerns many Gulf states consider more immediate than Tehran’s nuclear activities. Rubio said in Abu Dhabi on Tuesday that Iran’s missile program would “most certainly come up in these conversations.” Yet Trump appeared to downplay the issue last week, saying it was only fair for Iran to have missiles if Saudi Arabia does.
The pact also requires Gulf buy-in because it includes a $300 billion reconstruction fund for Iran. Trump committed Gulf funding to the initiative, but there is little evidence that Gulf states have done the same. Saudi Arabia has said it has “no details” about the proposal, while Qatar has expressed interest without formally signing on.

Rubio said he would not be asking allies for monetary help with the $300 billion Iran reconstruction fund during his trip, calling that “far down the road.”
Accommodating Iran
Gulf states recognize that, for now, they have few alternatives to the US as their primary security partner. And even as the US security role is perceived to be waning, its economic partnership with individual regional states remains robust, with nations like the UAE pledging to “double down” on their ties with the US.
How Gulf states’ relationship with the Trump administration evolves after the war remains unclear, the senior Gulf diplomat told CNN before the deal was signed, including whether it develops into a more formalized security arrangement that would obligate Washington to intervene if Gulf security were threatened.

Even so, some Gulf states are already looking to diversify their military procurement, particularly by turning to Turkey as an alternative supplier of arms, the diplomat said.
The war has also forced Gulf leaders to think more seriously about a long-term accommodation with Iran. While no regional power is currently capable of replacing the US as the Gulf’s security guarantor, officials are increasingly contemplating a future in which Washington plays a much smaller role in the regional security architecture, the diplomat said. One possible framework could involve a regional non-aggression pact with Iran.

How Iran could be persuaded to enter such an arrangement is another matter. As confidence in US security guarantees wanes, Gulf states have few tools for influencing Tehran beyond trade, investment and economic cooperation.
Analysts caution that diplomacy alone is unlikely to provide the security assurances Gulf states are seeking.

Alhasan, of IISS, doubts Iran would abide by a non-aggression pact “in the absence of a credible Arab Gulf deterrence capability,” arguing that Gulf states must first create “the right strategic conditions to incentivize Iran.”

“A nonaggression pact is unlikely to change Iran’s strategic calculus,” he said. “To do so, the Arab Gulf states must first redress the strategic imbalance with Iran through credible deterrence, enhanced and integrated defense and robust resilience measures.”

Rethinking relationships
Gulf commentators in state-linked media, too, are increasingly grappling with deeper questions about Iran’s role in the region, moving beyond the confrontational rhetoric that once dominated much of the discourse.
An opinion piece in the Saudi Arabia’s Asharq al-Awsat newspaper this week suggested that Iran’s circumstances may have forced it to take a confrontational regional posture and asked whether it can be moderated through diplomacy.
Even before the war, prominent Saudi commentator Abdulrahman Alrashed rejected in an article the idea that a weak, isolated Iran is good for the Gulf. The objective, he said, is not to permanently weaken the Islamic Republic but rather change its behavior and integrate it into a more stable regional order.

If Gulf states are rethinking their relationship with Iran, it is partly because they are rethinking their relationship with Washington.
“(The idea that) America as a strategic ally that can be relied upon is now very much in question in the Gulf states,” said Firas Maksad, managing director for the Middle East and North Africa at Eurasia Group, who argued that the war capped years of disappointments that had steadily undermined Gulf faith in US security guarantees.

“Gulf countries… need to come to an accommodation with Iran because they do not fully trust the United States. In the longer term, it’s not just détente, it’s also deterrence. They have to stand up their own military capabilities.”

CNN’s Jennifer Hansler and Becky Anderson contributed to this report.
"""

# ۵. آماده‌سازی متن برای مدل
inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)

# ۶. تولید خلاصه با تنظیمات بهینه برای کوتاه کردن متن
summary_ids = model.generate(
    inputs["input_ids"], 
    num_beams=4, 
    max_length=100,      # حداکثر طول خلاصه
    min_length=10,     # حداقل طول خلاصه
    length_penalty=2.5, # عدد بزرگتر باعث خلاصه شدن بیشتر می‌شود
    early_stopping=True
)

# ۷. تبدیل کد به متن قابل خواندن
summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

print("\n" + "="*30)
print("ORIGINAL TEXT LENGTH:", len(text.split()), "words")
print("-" * 30)
print("FINAL SUMMARY:")
print(summary[0])
print("-" * 30)
print("SUMMARY LENGTH:", len(summary[0].split()), "words")
print("="*30)
