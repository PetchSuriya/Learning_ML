
import os, re, json
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Config
DATA_PATH = os.getenv("DATA_PATH", "./data/ResumeDataSet.csv")

DEFAULT_JD = """
We are hiring a Software Developer responsible for building web applications.
Requirements: experience with JavaScript/TypeScript, React, Node.js or Python (FastAPI/Django),
REST APIs, databases (MongoDB/PostgreSQL), Git/GitHub, testing, CI/CD, problem solving,
communication skills, and Agile/Scrum. Nice to have: Docker, cloud (AWS/GCP/Azure).
""".strip()


TOKEN_PATTERN = re.compile(r"[A-Za-z]+")

def simple_preprocess(text: str) -> str:
    """
    ทำการประมวลผลข้อความเบื้องต้น (Text Preprocessing)
    - แปลงข้อความเป็นตัวพิมพ์เล็กทั้งหมด
    - แยกเอาเฉพาะตัวอักษรภาษาอังกฤษออกมา (ไม่รวมตัวเลข, สัญลักษณ์)
    - คืนค่าเป็นข้อความที่ทำความสะอาดแล้ว เชื่อมด้วยช่องว่าง
    
    Args:
        text (str): ข้อความที่ต้องการประมวลผล
    
    Returns:
        str: ข้อความที่ทำความสะอาดแล้ว
    """
    if not isinstance(text, str):
        return ""
    tokens = TOKEN_PATTERN.findall(text.lower())
    return " ".join(tokens)


# Train TF-IDF from CSV 

def train_vectorizer_from_csv(csv_path: str) -> TfidfVectorizer:
    """
    สร้างและฝึก TF-IDF Vectorizer จากข้อมูลเรซูเม่ในไฟล์ CSV
    
    การทำงาน:
    1. อ่านข้อมูลจากไฟล์ CSV
    2. ค้นหาคอลัมน์ที่มีข้อความเรซูเม่ (โดยดูจากชื่อคอลัมน์หรือความยาวข้อความ)
    3. ทำความสะอาดข้อความเรซูเม่ทั้งหมด
    4. รวมข้อความเรซูเม่กับ Job Description เริ่มต้น
    5. ฝึก TF-IDF Vectorizer ด้วยข้อมูลทั้งหมด
    
    Args:
        csv_path (str): path ไปยังไฟล์ CSV ที่มีข้อมูลเรซูเม่
    
    Returns:
        TfidfVectorizer: vectorizer ที่ฝึกแล้ว
        
    Raises:
        FileNotFoundError: หากไม่พบไฟล์ CSV
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # หา column ข้อความ
    text_col = None
    for c in df.columns:
        if c.strip().lower() in ["resume", "resume_text", "resume_texts", "summary", "text"]:
            text_col = c
            break
    if text_col is None:
        text_col = max(df.columns, key=lambda c: df[c].astype(str).str.len().sum())

    resumes = df[text_col].astype(str).fillna("").tolist()
    resumes_clean = [simple_preprocess(t) for t in resumes]

    corpus = resumes_clean + [simple_preprocess(DEFAULT_JD)]

    vectorizer = TfidfVectorizer(
        min_df=3,
        max_df=0.9,
        ngram_range=(1,2),
        strip_accents="unicode",
        sublinear_tf=True
    )
    vectorizer.fit(corpus)
    return vectorizer

def score_resume_against_job(resume_text: str, job_description: Optional[str], vectorizer: TfidfVectorizer) -> float:
    """
    คำนวณคะแนนความเหมาะสมของเรซูเม่กับงาน (Resume-Job Matching Score)
    
    การทำงาน:
    1. ใช้ Job Description ที่ให้มา หรือใช้ DEFAULT_JD หากไม่มี
    2. ทำความสะอาดข้อความทั้งเรซูเม่และ Job Description
    3. แปลงข้อความเป็น TF-IDF vectors
    4. คำนวณ cosine similarity ระหว่างสอง vectors
    5. แปลงค่าเป็นเปอร์เซ็นต์ (0-100%)
    
    Args:
        resume_text (str): ข้อความเรซูเม่
        job_description (Optional[str]): ข้อความ Job Description (หากเป็น None จะใช้ DEFAULT_JD)
        vectorizer (TfidfVectorizer): TF-IDF vectorizer ที่ฝึกแล้ว
    
    Returns:
        float: คะแนนความเหมาะสม (0.00-100.00%)
    """
    jd = (job_description or "").strip() or DEFAULT_JD
    inputs = [simple_preprocess(resume_text), simple_preprocess(jd)]
    vecs = vectorizer.transform(inputs)
    sim = cosine_similarity(vecs[0], vecs[1])[0][0]  # 0..1
    percent = float(np.clip(sim * 100.0, 0, 100))
    return round(percent, 2)

# ----------------------------
# CLI Demo

def main():
    """
    ฟังก์ชันหลักสำหรับการรันโปรแกรม Resume Matching แบบ Command Line Interface
    
    การทำงาน:
    1. โหลดและฝึก TF-IDF vectorizer จากข้อมูล CSV
    2. รับ input จากผู้ใช้ (resume text และ job description)
    3. คำนวณคะแนนความเหมาะสมและแสดงผล
    4. วนซ้ำจนกว่าผู้ใช้จะเลือกออก
    
    Features:
    - รองรับการใส่ resume text หลายบรรทัด
    - สามารถใส่ job description ได้ หรือใช้ default JD
    - แสดงผลคะแนนเป็นเปอร์เซ็นต์
    - สามารถทดสอบซ้ำได้หลายครั้ง
    """
    print("Loading & training TF-IDF from:", DATA_PATH)
    vectorizer = train_vectorizer_from_csv(DATA_PATH)
    print("Ready! Type your resume and (optionally) JD to see the % suitability.")
    print("- Leave JD empty to use default Software Developer JD.")
    print("- Press Enter twice to submit each block.\n")

    while True:
        try:
            print("="*60)
            print("Paste RESUME text (end with a blank line):")
            resume_lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                resume_lines.append(line)
            resume_text = "\n".join(resume_lines).strip()

            if not resume_text:
                print("No resume text. Exiting.")
                break

            print("\nPaste JOB DESCRIPTION (optional, end with a blank line):")
            jd_lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                jd_lines.append(line)
            jd_text = "\n".join(jd_lines).strip() or None

            percent = score_resume_against_job(resume_text, jd_text, vectorizer)
            used_default = jd_text is None

            print("\nResult:")
            print(f"  Suitability: {percent:.2f}%")
            print(f"  Using default JD: {used_default}")
            print("  Notes: Cosine similarity over TF-IDF (1–2 grams)")

            print("\nTry again? (Y/n): ", end="")
            ans = input().strip().lower()
            if ans == "n":
                break
        except KeyboardInterrupt:
            break

    print("Bye!")

if __name__ == "__main__":
    main()
