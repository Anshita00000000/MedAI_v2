"""Few-shot role classification prompts for MedPhi-Instruct."""

ROLE_CLASSIFIER_SYSTEM_PROMPT = """
You are a clinical conversation analyst. Your task is to identify the role
of a speaker in a medical consultation.

Roles:
- DOCTOR: The primary treating physician or specialist
- PATIENT: The person receiving medical care
- NURSE: A nursing staff member assisting in the consultation
- FAMILY: A family member or caregiver accompanying the patient

You will be given:
- The target turn to classify
- Up to 2 turns of context before and after

Respond with ONLY the role label. No explanation.
"""

ROLE_CLASSIFIER_FEW_SHOT_EXAMPLES = [
    # 1. Doctor giving clinical instructions
    {
        "context_before": [
            "Patient: I've been having chest pain for three days.",
            "Patient: It gets worse when I take deep breaths.",
        ],
        "target_turn": "I want you to take aspirin 81mg daily and avoid strenuous activity until we get the ECG results back. Come back immediately if the pain radiates to your left arm.",
        "context_after": [
            "Patient: Should I stop my other medications?",
            "Patient: What about exercise?",
        ],
        "label": "DOCTOR",
    },
    # 2. Doctor asking diagnostic questions
    {
        "context_before": [
            "Patient: I've been feeling really tired lately.",
        ],
        "target_turn": "How long have you been experiencing this fatigue? Is it worse in the morning or evening? Have you noticed any changes in your appetite or weight?",
        "context_after": [
            "Patient: About two weeks now.",
            "Patient: It's worse in the morning and I've lost about 5 pounds.",
        ],
        "label": "DOCTOR",
    },
    # 3. Patient describing symptoms in first person
    {
        "context_before": [
            "Doctor: What brings you in today?",
        ],
        "target_turn": "I've had this sharp stabbing pain in my lower right abdomen since yesterday evening. It started around my belly button and moved down. I also feel nauseous and I threw up twice this morning.",
        "context_after": [
            "Doctor: On a scale of 1 to 10, how would you rate the pain?",
            "Patient: About an 8. It's pretty severe.",
        ],
        "label": "PATIENT",
    },
    # 4. Patient asking about medication
    {
        "context_before": [
            "Doctor: I'm going to prescribe metformin for your diabetes.",
            "Doctor: Start with 500mg twice daily with meals.",
        ],
        "target_turn": "Can I take metformin with my blood pressure medication? I'm on lisinopril 10mg. And what happens if I miss a dose?",
        "context_after": [
            "Doctor: Yes, metformin and lisinopril are safe to take together.",
        ],
        "label": "PATIENT",
    },
    # 5. Nurse taking vitals mid-conversation
    {
        "context_before": [
            "Doctor: So how long have you had the headache?",
            "Patient: Since this morning when I woke up.",
        ],
        "target_turn": "Excuse me, I just need to check your blood pressure quickly. It's reading 158 over 96. Heart rate 88. Temperature 37.2 Celsius. I'll note these in the chart.",
        "context_after": [
            "Doctor: Thank you. So your blood pressure is elevated today.",
            "Patient: Is that bad?",
        ],
        "label": "NURSE",
    },
    # 6. Nurse relaying information from doctor
    {
        "context_before": [
            "Patient: When will my test results be ready?",
        ],
        "target_turn": "Dr. Patel asked me to let you know that your blood work from Monday came back. Your HbA1c is 7.8 and your kidney function is within normal range. The doctor will review everything with you in a few minutes.",
        "context_after": [
            "Patient: Thank you. Should I be worried about the HbA1c?",
            "Doctor: Let's discuss those results now.",
        ],
        "label": "NURSE",
    },
    # 7. Family member speaking on behalf of patient
    {
        "context_before": [
            "Doctor: Can you tell me what happened?",
        ],
        "target_turn": "She couldn't speak when I found her this morning. Her right arm was drooping and she couldn't lift it. I think she may have had a stroke in the night. She has a history of atrial fibrillation but she's been taking her warfarin.",
        "context_after": [
            "Doctor: How long ago did you find her like this?",
            "Doctor: When did you last see her acting normally?",
        ],
        "label": "FAMILY",
    },
    # 8. Family member asking prognosis question
    {
        "context_before": [
            "Doctor: The biopsy results confirm stage 2 breast cancer.",
            "Patient: Oh no.",
        ],
        "target_turn": "Doctor, what are the chances of full recovery? My mother had the same thing and she didn't survive. What treatment options are we looking at and how long will it take?",
        "context_after": [
            "Doctor: Stage 2 breast cancer has a very good prognosis with treatment.",
            "Doctor: The five-year survival rate is over 85 percent.",
        ],
        "label": "FAMILY",
    },
    # 9. Ambiguous short acknowledgement resolved by context
    {
        "context_before": [
            "Patient: So I take it twice a day with food?",
            "Doctor: That's right, once in the morning and once in the evening with your meals.",
        ],
        "target_turn": "Got it.",
        "context_after": [
            "Doctor: Any questions about the dosing schedule?",
            "Patient: No, I think I understand.",
        ],
        "label": "PATIENT",
    },
    # 10. Patient speaking about someone else (not themselves)
    {
        "context_before": [
            "Doctor: Any family history of heart disease?",
        ],
        "target_turn": "Yes, my father had a massive heart attack at 55 and my older brother was diagnosed with coronary artery disease last year. He's on statins now.",
        "context_after": [
            "Doctor: That's important family history. Given that background, we should monitor your cholesterol closely.",
            "Patient: Does that mean I'm at higher risk?",
        ],
        "label": "PATIENT",
    },
    # 11. Doctor addressing nurse directly
    {
        "context_before": [
            "Doctor: The patient's potassium is low.",
        ],
        "target_turn": "Please arrange an IV potassium infusion, 40mEq over four hours, and recheck the electrolytes in six hours. Also make sure the cardiac monitor is connected before you start the infusion.",
        "context_after": [
            "Nurse: Understood, I'll set that up right away.",
            "Doctor: Thank you. Now, regarding your symptoms...",
        ],
        "label": "DOCTOR",
    },
    # 12. Multiple speakers in one turn — classify dominant speaker
    {
        "context_before": [
            "Doctor: How have you been managing the pain at home?",
        ],
        "target_turn": "I've been taking ibuprofen but it only helps a little. My husband — he's here with me — he said the pain woke me up twice last night and I didn't even remember, but mostly I just can't sit for long periods.",
        "context_after": [
            "Doctor: The ibuprofen is only providing partial relief, that's important.",
            "Doctor: How long has this been going on?",
        ],
        "label": "PATIENT",
    },
]


def build_role_classifier_prompt(
    target_turn: str,
    context_before: list,
    context_after: list,
) -> str:
    """
    Assembles the few-shot prompt for a single turn classification.
    Returns the full prompt string ready to send to MedPhi.
    """
    lines = [ROLE_CLASSIFIER_SYSTEM_PROMPT.strip(), ""]

    # Append few-shot examples
    for i, ex in enumerate(ROLE_CLASSIFIER_FEW_SHOT_EXAMPLES, 1):
        lines.append(f"--- Example {i} ---")
        if ex["context_before"]:
            lines.append("Context before:")
            for t in ex["context_before"]:
                lines.append(f"  {t}")
        lines.append(f"Target turn: {ex['target_turn']}")
        if ex["context_after"]:
            lines.append("Context after:")
            for t in ex["context_after"]:
                lines.append(f"  {t}")
        lines.append(f"Label: {ex['label']}")
        lines.append("")

    # Append the actual query
    lines.append("--- Classify this turn ---")
    if context_before:
        lines.append("Context before:")
        for t in context_before:
            lines.append(f"  {t}")
    lines.append(f"Target turn: {target_turn}")
    if context_after:
        lines.append("Context after:")
        for t in context_after:
            lines.append(f"  {t}")
    lines.append("Label:")

    return "\n".join(lines)
