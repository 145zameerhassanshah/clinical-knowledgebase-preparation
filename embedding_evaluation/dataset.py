
# embedding_evaluation/dataset.py

PAIRS = [
    # SYNONYMS: Same or closely equivalent clinical meaning
    ("S01", "Synonyms",
     "The patient has hypertension.",
     "The patient has high blood pressure.", 1),

    ("S02", "Synonyms",
     "The patient has dyspnea.",
     "The patient experiences shortness of breath.", 1),

    ("S03", "Synonyms",
     "The patient has myocardial infarction.",
     "The patient has suffered a heart attack.", 1),

    ("S04", "Synonyms",
     "The patient has edema in both legs.",
     "The patient has swelling in both legs.", 1),

    ("S05", "Synonyms",
     "The patient has hyperglycemia.",
     "The patient has elevated blood glucose.", 1),

    ("S06", "Synonyms",
     "The patient has hypotension.",
     "The patient has low blood pressure.", 1),

    ("S07", "Synonyms",
     "The patient has tachycardia.",
     "The patient has an abnormally fast heart rate.", 1),

    ("S08", "Synonyms",
     "The patient has bradycardia.",
     "The patient has an abnormally slow heart rate.", 1),

    # ABBREVIATIONS: Medical abbreviation vs full expression
    ("A01", "Abbreviations",
     "The patient has COPD.",
     "The patient has chronic obstructive pulmonary disease.", 1),

    ("A02", "Abbreviations",
     "The patient has CKD.",
     "The patient has chronic kidney disease.", 1),

    ("A03", "Abbreviations",
     "The patient has CHF.",
     "The patient has congestive heart failure.", 1),

    ("A04", "Abbreviations",
     "The patient has DVT.",
     "The patient has deep vein thrombosis.", 1),

    ("A05", "Abbreviations",
     "The patient has UTI.",
     "The patient has a urinary tract infection.", 1),

    ("A06", "Abbreviations",
     "The patient has T2DM.",
     "The patient has type 2 diabetes mellitus.", 1),

    ("A07", "Abbreviations",
     "The patient has SOB.",
     "The patient has shortness of breath.", 1),

    ("A08", "Abbreviations",
     "The patient has AKI.",
     "The patient has acute kidney injury.", 1),

    # NEGATION: Presence vs explicit absence
    ("N01", "Negation",
     "The patient has pneumonia.",
     "The patient does not have pneumonia.", 0),

    ("N02", "Negation",
     "The patient reports chest pain.",
     "The patient denies chest pain.", 0),

    ("N03", "Negation",
     "The patient has a fever.",
     "The patient has no fever.", 0),

    ("N04", "Negation",
     "The patient has diabetes.",
     "The patient does not have diabetes.", 0),

    ("N05", "Negation",
     "The patient has a known drug allergy.",
     "The patient has no known drug allergy.", 0),

    ("N06", "Negation",
     "The patient is experiencing nausea.",
     "The patient is not experiencing nausea.", 0),

    ("N07", "Negation",
     "The patient has a pulmonary embolism.",
     "The patient has no pulmonary embolism.", 0),

    ("N08", "Negation",
     "The patient is taking anticoagulants.",
     "The patient is not taking anticoagulants.", 0),

    # NUMERICAL DIFFERENCES: Same measurement, changed value
    ("D01", "Numerical",
     "The patient's blood pressure is 120/80 mmHg.",
     "The patient's blood pressure is 180/110 mmHg.", 0),

    ("D02", "Numerical",
     "The patient's potassium level is 4.0 mmol/L.",
     "The patient's potassium level is 6.5 mmol/L.", 0),

    ("D03", "Numerical",
     "The patient's temperature is 37 degrees Celsius.",
     "The patient's temperature is 40 degrees Celsius.", 0),

    ("D04", "Numerical",
     "The patient's oxygen saturation is 98%.",
     "The patient's oxygen saturation is 82%.", 0),

    ("D05", "Numerical",
     "The patient is prescribed 5 mg of the medication.",
     "The patient is prescribed 50 mg of the medication.", 0),

    ("D06", "Numerical",
     "The patient's heart rate is 70 beats per minute.",
     "The patient's heart rate is 140 beats per minute.", 0),

    ("D07", "Numerical",
     "The patient's blood glucose is 90 mg/dL.",
     "The patient's blood glucose is 300 mg/dL.", 0),

    ("D08", "Numerical",
     "The patient's hemoglobin level is 14 g/dL.",
     "The patient's hemoglobin level is 7 g/dL.", 0),
]

assert len(PAIRS) >= 25
assert len({p[0] for p in PAIRS}) == len(PAIRS)
