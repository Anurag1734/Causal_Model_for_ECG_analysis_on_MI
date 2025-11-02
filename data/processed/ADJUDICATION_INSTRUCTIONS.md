# Clinical Adjudication Protocol
## ECG-Based Myocardial Infarction Label Validation

---

## 📋 OVERVIEW

**Purpose**: Validate automated MI detection algorithm by comparing algorithmic labels with expert clinical judgment.

**Your Role**: Review 100 ECG cases from MIMIC-IV and determine whether each represents an **Acute MI** or **No Acute MI** based on all available clinical data.

**Time Required**: ~30-45 minutes (approximately 30 seconds per case)

**Deliverable**: Return the CSV file with your assessments in the `clinician_label` column.

---

## 🎯 WHAT YOU NEED TO KNOW

### Automated Algorithm Being Validated:
- **MI Definition**: Troponin T > 0.10 ng/mL within 24 hours of ECG
- **Acute Presentation**: ECG obtained within ±24 hours of peak troponin
- **Control**: No troponin elevation during admission

### Your Task:
Determine if each case represents **Acute MI** or **No Acute MI** based on:
1. Troponin values and timing
2. Admission context (Emergency vs Elective)
3. Time relationship between ECG and troponin
4. Clinical plausibility

---

## 📝 ADJUDICATION CRITERIA

### **ACUTE MI** (Label as: `MI_Acute`)
✅ Troponin elevation AND at least ONE of:
- Emergency admission with chest pain syndrome
- Troponin timing consistent with acute event
- ECG obtained during acute presentation window

### **NO ACUTE MI** (Label as: `No_MI`)
❌ Any of the following:
- Chronic troponin elevation (CKD, HF)
- Troponin elevation inconsistent with ECG timing
- Elective admission without acute symptoms
- Troponin < 0.10 ng/mL
- Missing troponin data (likely control case)

### **UNCERTAIN** (Label as: `Uncertain`)
❓ Use sparingly when:
- Conflicting information
- Insufficient data to make confident determination
- Borderline case requiring full chart review

---

## 📊 DATA DICTIONARY

| Column | Description |
|--------|-------------|
| `record_id` | ECG identifier |
| `subject_id` | Patient identifier |
| `hadm_id` | Hospital admission identifier |
| `ecg_time` | Date/time ECG was obtained |
| `automated_label` | Algorithm's classification |
| `index_mi_time` | Time of peak troponin (if MI detected) |
| `hours_from_mi` | Hours between ECG and peak troponin |
| `admission_type` | EW EMER. (Emergency), URGENT, ELECTIVE |
| `troponin_count` | Number of troponin tests in admission |
| `max_troponin` | Highest troponin value (ng/mL) |
| **`clinician_label`** | **YOUR ASSESSMENT** (MI_Acute, No_MI, Uncertain) |
| `notes` | Optional: Your reasoning (especially for disagreements) |

---

## 🔍 ADJUDICATION DECISION TREE

```
START
  ↓
[1] Is max_troponin > 0.10 ng/mL?
  ↓ NO → Label: No_MI
  ↓ YES
  ↓
[2] Is admission_type = "EW EMER." or "URGENT"?
  ↓ NO (ELECTIVE) → Consider No_MI (unless clear acute event)
  ↓ YES
  ↓
[3] Is |hours_from_mi| < 24 hours?
  ↓ NO → Label: No_MI (wrong timing)
  ↓ YES
  ↓
[4] Is troponin_count >= 2 (serial testing)?
  ↓ NO → Consider Uncertain (need more data)
  ↓ YES
  ↓
[5] Is this clinically plausible for acute MI?
  ↓ YES → Label: MI_Acute
  ↓ NO → Label: No_MI
```

---

## 💡 EXAMPLE CASES

### Example 1: Clear Acute MI ✅
```
record_id: 12345678
ecg_time: 2160-04-16 02:23:00
automated_label: MI_Acute_Presentation
index_mi_time: 2160-04-16 01:05:00
hours_from_mi: 1.3
admission_type: EW EMER.
troponin_count: 3
max_troponin: 2.45

→ clinician_label: MI_Acute
→ notes: Emergency admission, serial troponins, ECG 1.3h after peak. Consistent with STEMI/NSTEMI.
```

### Example 2: Chronic Elevation (Not Acute MI) ❌
```
record_id: 87654321
ecg_time: 2161-02-08 18:44:00
automated_label: MI_Acute_Presentation
index_mi_time: 2161-02-08 17:21:00
hours_from_mi: 1.4
admission_type: ELECTIVE
troponin_count: 1
max_troponin: 0.12

→ clinician_label: No_MI
→ notes: Elective admission, borderline troponin, single test. Likely CKD-related elevation.
```

### Example 3: Control Case (No Troponin) ✅
```
record_id: 11223344
ecg_time: 2165-06-12 10:30:00
automated_label: Control_Symptomatic
index_mi_time: NaN
hours_from_mi: NaN
admission_type: EW EMER.
troponin_count: 0
max_troponin: NaN

→ clinician_label: No_MI
→ notes: No troponin tested. True control case.
```

---

## 🚀 HOW TO COMPLETE THE ADJUDICATION

### Step 1: Open the File
- Open `adjudication_sample.csv` in Excel, Google Sheets, or any CSV editor

### Step 2: Review Each Case
- Read across the row to understand the clinical context
- Apply the decision tree above

### Step 3: Fill in Your Assessment
- In the `clinician_label` column, enter ONE of:
  - `MI_Acute` (if you believe this is an acute MI)
  - `No_MI` (if you believe this is NOT an acute MI)
  - `Uncertain` (only if truly ambiguous)

### Step 4: Add Notes (Optional but Helpful)
- In the `notes` column, briefly explain your reasoning
- **Especially important** when you disagree with the automated label

### Step 5: Save and Return
- Save the file as `adjudication_sample_REVIEWED.csv`
- Return via email or shared drive

---

## 📈 QUALITY METRICS

Your adjudications will be used to calculate:

| Metric | Calculation | Target |
|--------|-------------|--------|
| **Overall Agreement** | (Agree cases / Total cases) × 100 | ≥80% |
| **Positive Agreement** | Algorithm MI ∩ Clinician MI | ≥75% |
| **Negative Agreement** | Algorithm No-MI ∩ Clinician No-MI | ≥85% |
| **Cohen's Kappa** | Agreement beyond chance | ≥0.70 |

**If agreement ≥ 80%**: Algorithm validated, proceed to analysis  
**If agreement < 80%**: Revise algorithm, re-adjudicate

---

## ❓ FREQUENTLY ASKED QUESTIONS

**Q: What if I disagree with the automated label?**  
A: That's expected! Your expert judgment is the gold standard. Just document your reasoning in the notes column.

**Q: Can I look up more information about the patient?**  
A: For this exercise, please base decisions ONLY on the data provided in the CSV. This simulates the algorithm's information constraint.

**Q: What if troponin_count = 0 but automated_label = MI_Acute?**  
A: This is likely an algorithm error. Label as `No_MI` and note "Missing troponin data."

**Q: How should I handle borderline troponin (0.09-0.12)?**  
A: Consider clinical context. Emergency admission with serial rise → likely MI. Elective admission, single test → likely not MI.

**Q: What's the difference between MI_Acute_Presentation and Control_Symptomatic?**  
A: These are the algorithm's labels. Ignore them and make your own independent assessment.

---

## 📧 CONTACT

**Questions during review?**  
- Email: [Your Email]
- Phone: [Your Phone]

**Expected turnaround**: 3-5 days

---

## 🙏 THANK YOU!

Your clinical expertise is invaluable for validating this AI-powered MI detection system. This adjudication will strengthen the scientific rigor of the study and ensure patient safety in future applications.

**Your contribution matters!** 🩺

---

**Document Version**: 1.0  
**Date**: October 24, 2025  
**Project**: ECG-Based Causal Inference for Myocardial Infarction
