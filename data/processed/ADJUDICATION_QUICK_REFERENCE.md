# QUICK REFERENCE CARD
## ECG MI Adjudication - 1-Page Guide

---

## YOUR TASK
Review 100 cases and determine: **Acute MI** or **No Acute MI**

---

## DECISION CHECKLIST

For each case, ask yourself:

- ☐ **Is troponin > 0.10 ng/mL?** (If NO → Label: `No_MI`)
- ☐ **Is admission Emergency/Urgent?** (If ELECTIVE → Likely `No_MI`)
- ☐ **Is ECG within ±24h of peak troponin?** (If NO → Label: `No_MI`)
- ☐ **Are there serial troponins?** (If only 1 test → Suspicious)
- ☐ **Does this make clinical sense?** (Use your judgment)

If ALL boxes checked → `MI_Acute`  
Otherwise → `No_MI`  
Truly unsure → `Uncertain`

---

## WHAT TO FILL IN

**Column: `clinician_label`**
- Enter: `MI_Acute` OR `No_MI` OR `Uncertain`

**Column: `notes`** (Optional)
- Brief reason, especially if you disagree with `automated_label`

---

## RED FLAGS for Non-MI

🚩 Elective admission  
🚩 Troponin 0.10-0.15 (borderline)  
🚩 Only 1 troponin test  
🚩 ECG >24h from troponin peak  
🚩 No troponin data (`troponin_count = 0`)

---

## COMMON SCENARIOS

| Scenario | Label | Why |
|----------|-------|-----|
| Emergency, troponin 2.5, ECG at 2h | `MI_Acute` | Classic acute MI |
| Elective, troponin 0.12, single test | `No_MI` | Chronic elevation |
| troponin_count = 0 | `No_MI` | True control |
| Emergency, no troponin data | `No_MI` | No biochemical proof |

---

## TIME REQUIRED
~30 minutes for 100 cases (18 seconds each)

---

## RETURN FILE AS
`adjudication_sample_REVIEWED.csv`

---

**Questions?** Contact [Your Name] at [Your Email]

**Target Agreement**: ≥80% with algorithm
