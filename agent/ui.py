"""Streamlit interface of the ISET Charguia assistant.

Run from the repository root:  streamlit run app.py
All academic facts (dates, rooms, articles) come from the regulation PDF
through scolarite_agent; nothing academic is hard-coded here.
"""
from __future__ import annotations

import base64
import html
import re
import uuid
from datetime import date, datetime

import streamlit as st

import scolarite_agent as core

# ---------------------------------------------------------------------------
# TRANSLATIONS
# ---------------------------------------------------------------------------

LANGS = {"fr": "FR", "ar": "عربي", "en": "EN"}

T = {
    "fr": {
        "yesterday": "Hier",
        "week": "7 derniers jours",
        "older": "Plus ancien",
        "search_chats": "Rechercher une conversation",
        "no_match": "Aucune conversation trouvée.",
        "title": "ISET Charguia – Assistant de scolarité",
        "student": "Étudiant",
        "hello": "Bonjour {name}",
        "subtitle": "Règlement, examens, absences et démarches — en un seul endroit.",
        "next_exam": "Prochain examen",
        "no_exam": "Aucun examen à venir",
        "days": "jours",
        "today": "Aujourd'hui",
        "profile": "Mon profil",
        "profile_help": "Utilisé pour vous saluer et pré-remplir vos demandes. "
                        "Conservé seulement pendant cette session.",
        "full_name": "Nom et prénom",
        "student_id": "Numéro d'étudiant",
        "email": "Votre e-mail (copie des demandes, facultatif)",
        "save": "Enregistrer",
        "saved": "Profil enregistré.",
        "bad_email": "Adresse e-mail invalide.",
        "conversations": "Conversations",
        "new_chat": "Nouvelle conversation",
        "delete_chat": "Supprimer cette conversation",
        "no_chats": "Aucune conversation enregistrée pour l'instant.",
        "status": "État du service",
        "api_ok": "Modèle : {model}",
        "api_missing": "Clé GROQ_API_KEY manquante",
        "mail_ok": "E-mail : configuré",
        "mail_missing": "E-mail : non configuré",
        "assistant": "Assistant de scolarité",
        "empty": "Posez une question sur le règlement, un examen, une absence, un "
                 "certificat, un rattrapage ou un stage — ou choisissez un raccourci.",
        "input": "Écrivez votre question…",
        "thinking": "Recherche dans le règlement…",
        "steps": "Étapes suivies par l'agent",
        "error": "Une erreur est survenue. Vérifiez la clé API et la connexion.",
        "error_detail": "Détail technique",
        "pending": "Un brouillon est prêt. Écrivez « oui » pour l'envoyer, ou « non » pour annuler.",
        "planning": "Calendrier des examens",
        "source_pdf": "Source : règlement PDF",
        "exam": "Examen",
        "resit": "Rattrapage",
        "deadline": "Justificatif avant le {d}",
        "past": "Passé",
        "handin": "Remise",
        "cal_missing": "Calendrier indisponible (PDF introuvable ou illisible).",
        "rules": "Règles à connaître",
        "rule_titles": {4: "Certificat médical", 6: "Hospitalisation",
                        8: "Demande de rattrapage", 12: "Convention de stage"},
        "request": "Demande de rattrapage",
        "request_intro": "Préparez votre demande écrite (article 8), téléchargez-la ou "
                         "envoyez-la à la scolarité.",
        "request_btn": "Préparer une demande",
        "module": "Module",
        "hospital": "J'ai été hospitalisé(e) (article 6 : 7 jours ouvrables)",
        "exam_info": "Examen du {exam} · justificatif à déposer avant le {d}",
        "fill_profile": "Complétez votre nom et votre numéro d'étudiant.",
        "bad_name": "Nom invalide (3 à 60 lettres).",
        "bad_id": "Numéro d'étudiant invalide (3 à 15 caractères, lettres, chiffres, tirets).",
        "preview": "Aperçu",
        "subject": "Objet",
        "download": "Télécharger (.txt)",
        "confirm": "Je confirme l'envoi de cette demande à {to}",
        "send": "Envoyer à la scolarité",
        "sent": "Demande envoyée à {to}.",
        "mail_off": "L'envoi par e-mail n'est pas configuré : téléchargez la demande et "
                    "déposez-la au bureau de la scolarité.",
        "no_resit_modules": "Aucun module avec rattrapage dans le calendrier.",
        "footer": "ISET Charguia · Année universitaire 2026–2027 · "
                  "Documents fictifs à usage pédagogique",
        "shortcuts": [
            ("Calendrier", ":material/event:", "Donne-moi le calendrier des examens"),
            ("Rattrapages", ":material/autorenew:", "Quelles sont les dates des rattrapages ?"),
            ("Certificat médical", ":material/medical_services:",
             "Quel est le délai pour déposer un certificat médical ?"),
            ("Absences", ":material/event_busy:",
             "Je suis absent à un examen, que dois-je faire ?"),
            ("Stage", ":material/work:", "Combien de semaines dure le stage ?"),
        ],
        "weekdays": ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"],
        "months": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
                   "août", "septembre", "octobre", "novembre", "décembre"],
        "months_short": ["JANV", "FÉVR", "MARS", "AVR", "MAI", "JUIN", "JUIL",
                         "AOÛT", "SEPT", "OCT", "NOV", "DÉC"],
    },
    "en": {
        "yesterday": "Yesterday",
        "week": "Previous 7 days",
        "older": "Older",
        "search_chats": "Search chats",
        "no_match": "No matching conversation.",
        "title": "ISET Charguia – Academic assistant",
        "student": "Student",
        "hello": "Hello {name}",
        "subtitle": "Regulations, exams, absences and paperwork — all in one place.",
        "next_exam": "Next exam",
        "no_exam": "No upcoming exam",
        "days": "days",
        "today": "Today",
        "profile": "My profile",
        "profile_help": "Used to greet you and pre-fill your requests. "
                        "Kept for this session only.",
        "full_name": "Full name",
        "student_id": "Student number",
        "email": "Your email (copy of requests, optional)",
        "save": "Save",
        "saved": "Profile saved.",
        "bad_email": "Invalid email address.",
        "conversations": "Conversations",
        "new_chat": "New conversation",
        "delete_chat": "Delete this conversation",
        "no_chats": "No saved conversations yet.",
        "status": "Service status",
        "api_ok": "Model: {model}",
        "api_missing": "GROQ_API_KEY missing",
        "mail_ok": "Email: configured",
        "mail_missing": "Email: not configured",
        "assistant": "Academic assistant",
        "empty": "Ask about the regulations, an exam, an absence, a medical certificate, "
                 "a resit or an internship — or pick a shortcut.",
        "input": "Type your question…",
        "thinking": "Searching the regulations…",
        "steps": "Steps taken by the agent",
        "error": "Something went wrong. Check the API key and the connection.",
        "error_detail": "Technical details",
        "pending": "A draft is ready. Reply “yes” to send it, or “no” to cancel.",
        "planning": "Exam calendar",
        "source_pdf": "Source: regulation PDF",
        "exam": "Exam",
        "resit": "Resit",
        "deadline": "Certificate due by {d}",
        "past": "Past",
        "handin": "Hand-in",
        "cal_missing": "Calendar unavailable (PDF missing or unreadable).",
        "rules": "Key rules",
        "rule_titles": {4: "Medical certificate", 6: "Hospitalisation",
                        8: "Resit request", 12: "Internship agreement"},
        "request": "Resit request",
        "request_intro": "Prepare your written request (Article 8), download it or send "
                         "it to the registrar's office.",
        "request_btn": "Prepare a request",
        "module": "Module",
        "hospital": "I was hospitalised (Article 6: 7 working days)",
        "exam_info": "Exam on {exam} · certificate due by {d}",
        "fill_profile": "Fill in your name and student number.",
        "bad_name": "Invalid name (3 to 60 letters).",
        "bad_id": "Invalid student number (3 to 15 letters, digits or dashes).",
        "preview": "Preview",
        "subject": "Subject",
        "download": "Download (.txt)",
        "confirm": "I confirm sending this request to {to}",
        "send": "Send to the registrar",
        "sent": "Request sent to {to}.",
        "mail_off": "Email sending is not configured: download the request and hand it "
                    "in at the registrar's office.",
        "no_resit_modules": "No module with a resit in the calendar.",
        "footer": "ISET Charguia · Academic year 2026–2027 · "
                  "Fictitious documents for teaching purposes",
        "shortcuts": [
            ("Calendar", ":material/event:", "Give me the exam calendar"),
            ("Resits", ":material/autorenew:", "When are the resit exams?"),
            ("Medical certificate", ":material/medical_services:",
             "What is the deadline to hand in a medical certificate?"),
            ("Absences", ":material/event_busy:",
             "I will miss an exam, what should I do?"),
            ("Internship", ":material/work:", "How many weeks does the internship last?"),
        ],
        "weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                     "Saturday", "Sunday"],
        "months": ["January", "February", "March", "April", "May", "June", "July",
                   "August", "September", "October", "November", "December"],
        "months_short": ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG",
                         "SEP", "OCT", "NOV", "DEC"],
    },
    "ar": {
        "yesterday": "أمس",
        "week": "آخر 7 أيام",
        "older": "أقدم",
        "search_chats": "البحث في المحادثات",
        "no_match": "لا توجد محادثة مطابقة.",
        "title": "المعهد العالي للدراسات التكنولوجية بالشرقية – مساعد الشؤون الدراسية",
        "student": "طالب",
        "hello": "مرحبا {name}",
        "subtitle": "النظام الدراسي، الامتحانات، الغيابات والإجراءات — في مكان واحد.",
        "next_exam": "الامتحان القادم",
        "no_exam": "لا يوجد امتحان قادم",
        "days": "يوما",
        "today": "اليوم",
        "profile": "ملفي",
        "profile_help": "يُستعمل للترحيب بك وتعبئة مطالبك مسبقا. يُحفظ خلال هذه الجلسة فقط.",
        "full_name": "الاسم واللقب",
        "student_id": "رقم الطالب",
        "email": "بريدك الإلكتروني (نسخة من المطالب، اختياري)",
        "save": "حفظ",
        "saved": "تم حفظ الملف.",
        "bad_email": "عنوان بريد غير صالح.",
        "conversations": "المحادثات",
        "new_chat": "محادثة جديدة",
        "delete_chat": "حذف هذه المحادثة",
        "no_chats": "لا توجد محادثات محفوظة بعد.",
        "status": "حالة الخدمة",
        "api_ok": "النموذج: {model}",
        "api_missing": "مفتاح GROQ_API_KEY غير موجود",
        "mail_ok": "البريد: مُفعّل",
        "mail_missing": "البريد: غير مُفعّل",
        "assistant": "مساعد الشؤون الدراسية",
        "empty": "اطرح سؤالا حول النظام الدراسي، امتحان، غياب، شهادة طبية، "
                 "دورة تدارك أو تربص — أو اختر اختصارا.",
        "input": "اكتب سؤالك…",
        "thinking": "جارٍ البحث في النظام الدراسي…",
        "steps": "الخطوات التي اتبعها المساعد",
        "error": "حدث خطأ. تحقق من مفتاح API والاتصال.",
        "error_detail": "التفاصيل التقنية",
        "pending": "المسودة جاهزة. اكتب « oui » لإرسالها أو « non » للإلغاء.",
        "planning": "روزنامة الامتحانات",
        "source_pdf": "المصدر: النظام الدراسي PDF",
        "exam": "الامتحان",
        "resit": "التدارك",
        "deadline": "الشهادة قبل {d}",
        "past": "انتهى",
        "handin": "تسليم",
        "cal_missing": "الروزنامة غير متوفرة (ملف PDF غير موجود أو غير مقروء).",
        "rules": "قواعد مهمة",
        "rule_titles": {4: "الشهادة الطبية", 6: "الإقامة بالمستشفى",
                        8: "مطلب التدارك", 12: "اتفاقية التربص"},
        "request": "مطلب دورة التدارك",
        "request_intro": "حضّر مطلبك الكتابي (الفصل 8)، حمّله أو أرسله إلى مصلحة الدراسات.",
        "request_btn": "تحضير مطلب",
        "module": "المادة",
        "hospital": "أقمت بالمستشفى (الفصل 6: 7 أيام عمل)",
        "exam_info": "امتحان {exam} · تقديم الشهادة قبل {d}",
        "fill_profile": "أكمل اسمك ورقم الطالب.",
        "bad_name": "اسم غير صالح (من 3 إلى 60 حرفا).",
        "bad_id": "رقم طالب غير صالح (من 3 إلى 15 حرفا أو رقما).",
        "preview": "معاينة",
        "subject": "الموضوع",
        "download": "تحميل (.txt)",
        "confirm": "أؤكد إرسال هذا المطلب إلى {to}",
        "send": "إرسال إلى مصلحة الدراسات",
        "sent": "تم إرسال المطلب إلى {to}.",
        "mail_off": "الإرسال بالبريد غير مُفعّل: حمّل المطلب وقدّمه لمكتب الدراسات.",
        "no_resit_modules": "لا توجد مادة بدورة تدارك في الروزنامة.",
        "footer": "ISET Charguia · السنة الجامعية 2026–2027 · وثائق وهمية لأغراض تعليمية",
        "shortcuts": [
            ("الروزنامة", ":material/event:", "وقتاش الامتحانات؟ أعطيني الروزنامة"),
            ("التدارك", ":material/autorenew:", "وقتاش الراتراپاج؟"),
            ("شهادة طبية", ":material/medical_services:",
             "قداش عندي وقت باش نجيب certificat médical؟"),
            ("الغيابات", ":material/event_busy:", "باش نغيب على امتحان، شنوة نعمل؟"),
            ("التربص", ":material/work:", "قداش من جمعة يدوم الستاج؟"),
        ],
        "weekdays": ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"],
        "months": ["جانفي", "فيفري", "مارس", "أفريل", "ماي", "جوان", "جويلية", "أوت",
                   "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"],
        "months_short": ["جانفي", "فيفري", "مارس", "أفريل", "ماي", "جوان", "جويلية", "أوت",
                         "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"],
    },
}

NAME_RE = r"[A-Za-zÀ-ÿ؀-ۿ' \-]{3,60}"
ID_RE = r"[A-Za-z0-9\-]{3,15}"
EMAIL_RE = r"[^@\s]+@[^@\s]+\.[^@\s]+"


def _lang() -> str:
    return st.session_state.get("lang_pick") or st.session_state.get("lang", "fr")


def t(key: str):
    return T[_lang()][key]


def fmt_date(d: date, weekday: bool = True) -> str:
    tr = T[_lang()]
    text = f"{d.day} {tr['months'][d.month - 1]} {d.year}"
    return f"{tr['weekdays'][d.weekday()]} {text}" if weekday else text


def fmt_short(d: date) -> str:
    return f"{T[_lang()]['weekdays'][d.weekday()]} {d:%d/%m/%Y}"


def esc(value) -> str:
    return html.escape(str(value or ""))


# ---------------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=Noto+Sans+Arabic:wght@400;600;700&display=swap');
:root {
  --navy:#16325c; --blue:#1f5fbf; --blue-soft:#e8f0fb; --teal:#1a9c98;
  --amber:#f2a900; --bg:#f5f7fa; --surface:#ffffff; --text:#1c2733;
  --muted:#5d6b7a; --border:#e1e7ee; --danger:#c62828;
}
html, body, .stApp, .stApp button, .stApp input, .stApp textarea {
  font-family:'Source Sans 3','Noto Sans Arabic',system-ui,sans-serif;
}
/* hide Streamlit chrome but keep the toolbar: it holds the "reopen sidebar" button */
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"] { display:none; }
[data-testid="stExpandSidebarButton"] { background:var(--surface); border:1px solid var(--border);
     border-radius:8px; color:var(--navy); }
header[data-testid="stHeader"], header[data-testid="stHeader"] * { background:transparent; pointer-events:none; }
header[data-testid="stHeader"] button, [data-testid="stSidebarCollapsedControl"] { pointer-events:auto; }
.stApp, [data-testid="stBottom"] > div { background:var(--bg); }
.block-container { padding:1.2rem clamp(1rem,2.5vw,2.5rem) 2rem; max-width:1400px; }

/* header */
.brand { display:flex; align-items:center; gap:.9rem; }
.brand img { height:56px; border-radius:6px; background:#fff; }
.brand .name { font-weight:700; color:var(--navy); font-size:1.05rem; line-height:1.2; }
.brand .name small { display:block; font-weight:400; color:var(--muted); font-size:.82rem; }

/* hero */
.hero { display:flex; justify-content:space-between; align-items:center; gap:1.5rem;
        flex-wrap:wrap; background:var(--navy); border-radius:14px;
        padding:1.4rem 1.8rem; margin:.6rem 0 1.1rem; color:#fff; }
.hero .date { font-size:.8rem; font-weight:600; letter-spacing:.06em;
              text-transform:uppercase; opacity:.75; }
.hero h1 { color:#fff; font-size:1.9rem; font-weight:700; margin:.15rem 0; padding:0; }
.hero p { color:#d6e2f3; margin:0; }
.count { background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.16);
         border-radius:12px; padding:.8rem 1.1rem; min-width:230px; }
.count .lbl { font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; color:#ffd770; }
.count .big { font-size:2rem; font-weight:700; line-height:1.1; color:#fff; }
.count .big small { font-size:.95rem; font-weight:400; opacity:.8; }
.count .sub { font-size:.85rem; color:#d6e2f3; }

/* cards */
.card-title { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:.4rem;
              font-weight:700; color:var(--navy); font-size:1.02rem; margin-bottom:.4rem; }
.tag { white-space:nowrap; font-size:.72rem; font-weight:600; color:var(--teal); background:#e3f5f4;
       border-radius:999px; padding:.1rem .6rem; }
.exam-row { display:flex; gap:.8rem; padding:.65rem 0; border-top:1px solid var(--border); }
.exam-row:first-of-type { border-top:0; }
.exam-row.past { opacity:.5; }
.dbox { align-self:flex-start; flex:0 0 52px; text-align:center; background:var(--blue-soft); border-radius:8px;
        padding:.3rem 0; color:var(--blue); }
.dbox b { display:block; font-size:1.25rem; line-height:1.1; }
.dbox span { font-size:.66rem; font-weight:600; }
.exam-info { flex:1; min-width:0; font-size:.85rem; color:var(--muted); line-height:1.45; }
.exam-info b { color:var(--text); font-size:.95rem; }
.exam-info .dl { color:var(--danger); font-weight:600; }
.badge { align-self:flex-start; white-space:nowrap; font-size:.72rem; font-weight:700;
         border-radius:6px; padding:.12rem .45rem; background:#fff4d6; color:#8a5d00; }
.badge.past { background:#eef1f4; color:var(--muted); }
details.rule { border-top:1px solid var(--border); padding:.55rem 0; font-size:.86rem; }
details.rule summary { cursor:pointer; font-weight:600; color:var(--text); }
details.rule p { color:var(--muted); margin:.4rem 0 0; }
.empty { background:var(--blue-soft); color:var(--navy); border-radius:10px;
         padding:.9rem 1.1rem; font-size:.95rem; }
.letter { background:#fbfcfd; border:1px solid var(--border); border-radius:8px;
          padding:1rem 1.2rem; font-size:.88rem; line-height:1.55; color:var(--text); }

/* buttons */
.stButton, .stButton > div, .stButton [data-testid="stTooltipHoverTarget"] { width:100%; }
.stButton button { width:100%; border-radius:999px; border:1px solid var(--border);
                     background:var(--surface); color:var(--text); font-weight:600; }
.stButton > button:hover { border-color:var(--blue); color:var(--blue); background:var(--blue-soft); }
.stButton > button[kind="primary"] { background:var(--blue); color:#fff; border-color:var(--blue); }
.stButton > button[kind="primary"]:hover { background:var(--navy); color:#fff; }

/* shortcut pills */
[data-testid="stPills"] button { border-radius:999px; background:var(--surface);
     border:1px solid var(--border); font-weight:600; }
[data-testid="stPills"] button:hover { border-color:var(--blue); color:var(--blue); }

/* chat */
[data-testid="stChatMessage"] { background:var(--surface); border:1px solid var(--border);
                                border-radius:12px; padding:.9rem 1rem; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { unicode-bidi:plaintext; }

/* sidebar */
[data-testid="stSidebar"] { background:var(--surface); border-right:1px solid var(--border); }
[data-testid="stSidebarContent"] { padding-top:.5rem; }
[data-testid="stSidebarUserContent"] { padding-top:1rem; }
[data-testid="stSidebar"] .stButton > button { border-radius:8px; text-align:start;
     justify-content:flex-start; font-weight:500; }
.side-h { font-weight:700; color:var(--navy); margin:.6rem 0 .3rem; }
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] { gap:.2rem; }
[data-testid="stSidebar"] .st-key-new_chat { margin-bottom:.5rem; }
[data-testid="stSidebar"] .st-key-new_chat button { width:100%; justify-content:flex-start; }
[data-testid="stSidebar"] .st-key-chat_search { margin-bottom:.3rem; }
[data-testid="stSidebar"] hr { margin:1rem 0; }
.chat-group { font-size:.75rem; font-weight:600; color:var(--muted); margin:.8rem 0 .15rem;
              padding-inline-start:.5rem; }
[data-testid="stSidebar"] .st-key-new_chat button { border:1px solid var(--border);
     background:var(--surface); font-weight:600; }
/* chat rows, Claude-style: one line, subtle highlight, delete on hover */
[data-testid="stSidebar"] [class*="st-key-row"] { gap:0 !important; border-radius:8px;
     flex-wrap:nowrap !important; }
[data-testid="stSidebar"] [class*="st-key-row"]:hover { background:#eef2f7; }
[data-testid="stSidebar"] [class*="st-key-rowcur_"] { background:var(--blue-soft); }
[data-testid="stSidebar"] [class*="st-key-row"] [class*="st-key-c_"] { flex:1 1 auto; min-width:0; }
[data-testid="stSidebar"] [class*="st-key-row"] [class*="st-key-c_"] button {
     border:0; background:transparent; box-shadow:none; min-height:2.1rem;
     padding:.25rem .5rem; font-weight:500; color:var(--text); }
[data-testid="stSidebar"] [class*="st-key-rowcur_"] [class*="st-key-c_"] button {
     color:var(--navy); font-weight:600; }
[data-testid="stSidebar"] [class*="st-key-c_"] button p { white-space:nowrap; overflow:hidden;
     text-overflow:ellipsis; text-align:start; }
[data-testid="stSidebar"] [class*="st-key-c_"] button > div { min-width:0; width:100%;
     justify-content:flex-start; }
[data-testid="stSidebar"] [class*="st-key-del_"], [data-testid="stSidebar"] [class*="st-key-del_"] > div
     { width:auto !important; flex:0 0 auto; }
[data-testid="stSidebar"] [class*="st-key-del_"] button { width:auto; border:0; background:transparent;
     color:var(--muted); padding:0 .4rem; opacity:0; }
[data-testid="stSidebar"] [class*="st-key-row"]:hover [class*="st-key-del_"] button,
[data-testid="stSidebar"] [class*="st-key-rowcur_"] [class*="st-key-del_"] button { opacity:1; }
[data-testid="stSidebar"] [class*="st-key-del_"] button:hover { color:var(--danger); background:transparent; }
@media (hover:none) { [data-testid="stSidebar"] [class*="st-key-del_"] button { opacity:1; } }
.side-muted { color:var(--muted); font-size:.85rem; }
.status { font-size:.8rem; color:var(--muted); display:flex; align-items:center; gap:.4rem;
          margin:.2rem 0; }
.dot { width:8px; height:8px; border-radius:50%; background:var(--teal); flex:0 0 8px; }
.dot.off { background:#c7cfd8; }

.foot { text-align:center; color:var(--muted); font-size:.78rem; margin-top:2rem; }
</style>
"""

RTL_CSS = """
<style>
[data-testid="stMain"], [data-testid="stSidebar"], [role="dialog"] { direction:rtl; }
[data-testid="stMain"] input, [data-testid="stMain"] textarea { text-align:right; }
.letter { direction:ltr; text-align:left; }
</style>
"""


# ---------------------------------------------------------------------------
# SESSION
# ---------------------------------------------------------------------------

def _init_state() -> None:
    st.session_state.setdefault("lang", "fr")
    st.session_state.setdefault("profile", {"name": "", "id": "", "email": ""})
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = uuid.uuid4().hex
    if "agent" not in st.session_state:
        st.session_state.agent = core.Agent()


def _reset_chat(messages=None, chat_id: str | None = None) -> None:
    st.session_state.chat_id = chat_id or uuid.uuid4().hex
    st.session_state.agent = core.Agent(messages)


def _first_name() -> str:
    name = st.session_state.profile.get("name", "").strip()
    return name.split()[0] if name else t("student")


def _initials() -> str:
    parts = st.session_state.profile.get("name", "").split()
    return "".join(p[0] for p in parts[:2]).upper() if parts else ""


def _logo_html() -> str:
    for p in (core.ROOT / "assets" / "isetch.png", core.ROOT / "assets" / "isetch.jpg"):
        if p.exists():
            mime = "png" if p.suffix == ".png" else "jpeg"
            return f'<img alt="ISET" src="data:image/{mime};base64,' \
                   f'{base64.b64encode(p.read_bytes()).decode()}">'
    return ""


# ---------------------------------------------------------------------------
# PIECES
# ---------------------------------------------------------------------------

def _header() -> None:
    left, lang_col, prof_col = st.columns([6, 1.6, 1.8], vertical_alignment="center")
    with left:
        st.markdown(
            f'<div class="brand">{_logo_html()}<div class="name">ISET Charguia'
            f'<small>{esc(t("assistant"))}</small></div></div>',
            unsafe_allow_html=True,
        )
    with lang_col:
        picked = st.segmented_control(
            "Langue", options=list(LANGS), format_func=LANGS.get,
            default=st.session_state.lang, key="lang_pick",
            label_visibility="collapsed",
        )
        if picked:
            st.session_state.lang = picked
    with prof_col:
        label = _initials() + "  " + _first_name() if _initials() else t("profile")
        with st.popover(label, icon=":material/account_circle:", use_container_width=True):
            _profile_form()


def _profile_form() -> None:
    prof = st.session_state.profile
    st.caption(t("profile_help"))
    with st.form("profile_form", border=False):
        name = st.text_input(t("full_name"), value=prof["name"], max_chars=60)
        sid = st.text_input(t("student_id"), value=prof["id"], max_chars=15)
        email = st.text_input(t("email"), value=prof["email"])
        if st.form_submit_button(t("save"), type="primary"):
            if name.strip() and not re.fullmatch(NAME_RE, name.strip()):
                st.error(t("bad_name"))
            elif sid.strip() and not re.fullmatch(ID_RE, sid.strip()):
                st.error(t("bad_id"))
            elif email.strip() and not re.fullmatch(EMAIL_RE, email.strip()):
                st.error(t("bad_email"))
            else:
                st.session_state.profile = {
                    "name": " ".join(name.split()), "id": sid.strip(),
                    "email": email.strip(),
                }
                st.toast(t("saved"), icon=":material/check_circle:")
                st.rerun()


def _chat_group(updated: float) -> str:
    day = datetime.fromtimestamp(updated or 0).date()
    delta = (date.today() - day).days
    if delta <= 0:
        return "today"
    if delta == 1:
        return "yesterday"
    if delta < 7:
        return "week"
    return "older"


def _sidebar() -> None:
    with st.sidebar:
        if st.button(t("new_chat"), icon=":material/edit_square:", key="new_chat"):
            _reset_chat()
            st.rerun()

        chats = core.list_saved_chats()[:60]
        query = ""
        if chats:
            query = st.text_input(t("search_chats"), key="chat_search",
                                  placeholder=t("search_chats"),
                                  label_visibility="collapsed").strip().lower()

        st.markdown(f'<div class="side-h">{esc(t("conversations"))}</div>',
                    unsafe_allow_html=True)
        if not chats:
            st.markdown(f'<div class="side-muted">{esc(t("no_chats"))}</div>',
                        unsafe_allow_html=True)
        shown = [c for c in chats if query in (c.get("title") or "").lower()]
        if chats and not shown:
            st.markdown(f'<div class="side-muted">{esc(t("no_match"))}</div>',
                        unsafe_allow_html=True)

        last_group = None
        for c in shown:
            group = _chat_group(c.get("updated", 0))
            if group != last_group:
                st.markdown(f'<div class="chat-group">{esc(t(group))}</div>',
                            unsafe_allow_html=True)
                last_group = group
            current = c["id"] == st.session_state.chat_id
            row_key = f"{'rowcur' if current else 'row'}_{c['id']}"
            with st.container(key=row_key, horizontal=True, gap="small",
                              vertical_alignment="center"):
                if st.button(c.get("title") or t("new_chat"), key=f"c_{c['id']}",
                             help=c.get("title")):
                    _reset_chat(core.load_chat(c["id"]), c["id"])
                    st.rerun()
                if st.button("", icon=":material/delete:", key=f"del_{c['id']}",
                             type="tertiary", help=t("delete_chat")):
                    core.delete_chat(c["id"])
                    if current:
                        _reset_chat()
                    st.rerun()

        st.divider()
        api_ok = bool(core.GROQ_API_KEY)
        mail_ok = bool(core.SMTP_USER and core.SMTP_PASSWORD and core.MAIL_TO)
        st.markdown(
            f'<div class="status"><span class="dot{"" if api_ok else " off"}"></span>'
            f'{esc(t("api_ok").format(model=core.API_MODEL) if api_ok else t("api_missing"))}</div>'
            f'<div class="status"><span class="dot{"" if mail_ok else " off"}"></span>'
            f'{esc(t("mail_ok") if mail_ok else t("mail_missing"))}</div>',
            unsafe_allow_html=True,
        )


def _hero() -> None:
    nxt = core.next_exam()
    if nxt:
        days = nxt["days_left"]
        big = esc(t("today")) if days == 0 else f'J-{days} <small>· {days} {esc(t("days"))}</small>'
        count = (f'<div class="count"><div class="lbl">{esc(t("next_exam"))}</div>'
                 f'<div class="big">{big}</div>'
                 f'<div class="sub">{esc(nxt["nom"])} · {esc(nxt["examen"])}</div></div>')
    else:
        count = (f'<div class="count"><div class="lbl">{esc(t("next_exam"))}</div>'
                 f'<div class="sub">{esc(t("no_exam"))}</div></div>')
    st.markdown(
        f'<div class="hero"><div><div class="date">{esc(fmt_date(date.today()))}</div>'
        f'<h1>{esc(t("hello").format(name=_first_name()))}</h1>'
        f'<p>{esc(t("subtitle"))}</p></div>{count}</div>',
        unsafe_allow_html=True,
    )


def _planning() -> None:
    with st.container(border=True):
        st.markdown(f'<div class="card-title">{esc(t("planning"))}'
                    f'<span class="tag">{esc(t("source_pdf"))}</span></div>',
                    unsafe_allow_html=True)
        rows = core.exam_schedule()
        if not rows:
            st.markdown(f'<div class="side-muted">{esc(t("cal_missing"))}</div>',
                        unsafe_allow_html=True)
            return
        months = t("months_short")
        out = []
        for r in rows:
            d = r["date"]
            past = r["days_left"] is not None and r["days_left"] < 0
            dbox = (f'<div class="dbox"><b>{d.day:02d}</b><span>{months[d.month - 1]}</span></div>'
                    if d else '<div class="dbox"><b>–</b></div>')
            if r["is_project"]:
                badge = f'<span class="badge past">{esc(t("handin"))}</span>'
            elif past:
                badge = f'<span class="badge past">{esc(t("past"))}</span>'
            elif r["days_left"] == 0:
                badge = f'<span class="badge">{esc(t("today"))}</span>'
            else:
                badge = f'<span class="badge">J-{r["days_left"]}</span>'
            lines = [f'<b>{esc(r["nom"])}</b>']
            if not r["is_project"]:
                lines.append(f'{esc(t("exam"))} : {esc(r["examen"])}')
            else:
                lines.append(esc(r["examen"]))
            lines.append(f'{esc(t("resit"))} : {esc(r["rattrapage"])}')
            if d and r["has_resit"] and not past:
                dl = core.certificate_deadline(d)
                lines.append(f'<span class="dl">{esc(t("deadline").format(d=fmt_short(dl)))}</span>')
            out.append(f'<div class="exam-row{" past" if past else ""}">{dbox}'
                       f'<div class="exam-info">{"<br>".join(lines)}</div>{badge}</div>')
        st.markdown("".join(out), unsafe_allow_html=True)


def _rules() -> None:
    articles = core.load_articles()
    items = []
    for number, title in t("rule_titles").items():
        body = articles.get(number)
        if body:
            body = re.sub(r"^Article\s+\d+\.?\s*", "", body, flags=re.I)
            body = re.split(r"\s*(?:Chapitre \d+|ISET Charguia –)", body)[0]
            items.append(f'<details class="rule"><summary>{esc(title)} · Art. {number}'
                         f'</summary><p>{esc(body)}</p></details>')
    if not items:
        return
    with st.container(border=True):
        st.markdown(f'<div class="card-title">{esc(t("rules"))}</div>' + "".join(items),
                    unsafe_allow_html=True)


def _request_card() -> None:
    with st.container(border=True):
        st.markdown(f'<div class="card-title">{esc(t("request"))}</div>'
                    f'<div class="side-muted" style="margin-bottom:.7rem">'
                    f'{esc(t("request_intro"))}</div>', unsafe_allow_html=True)
        if st.button(t("request_btn"), icon=":material/edit_document:", type="primary",
                     key="open_request"):
            st.dialog(t("request"), width="large")(_request_dialog)()


def _request_dialog() -> None:
    modules = core.resit_modules()
    if not modules:
        st.info(t("no_resit_modules"))
        return
    prof = st.session_state.profile

    key = st.selectbox(t("module"), list(modules), format_func=lambda k: modules[k]["nom"])
    c1, c2 = st.columns(2)
    name = c1.text_input(t("full_name"), value=prof["name"], max_chars=60, key="rq_name")
    sid = c2.text_input(t("student_id"), value=prof["id"], max_chars=15, key="rq_id")
    hosp = st.checkbox(t("hospital"), key="rq_hosp")

    exam_day = core.parse_fr_date(modules[key]["examen"])
    if exam_day:
        st.caption(t("exam_info").format(
            exam=modules[key]["examen"],
            d=fmt_date(core.certificate_deadline(exam_day, hosp))))

    name, sid = " ".join(name.split()), sid.strip()
    if not name or not sid:
        st.info(t("fill_profile"))
        return
    if not re.fullmatch(NAME_RE, name):
        st.error(t("bad_name"))
        return
    if not re.fullmatch(ID_RE, sid):
        st.error(t("bad_id"))
        return

    subject, body = core.build_demande_email(key, name, sid, hospitalisation=hosp)
    st.markdown(f"**{t('preview')}**")
    letter = esc(body).replace("\n", "<br>")
    st.markdown(f'<div class="letter"><b>{esc(t("subject"))} :</b> {esc(subject)}'
                f'<br><br>{letter}</div>', unsafe_allow_html=True)

    st.download_button(
        t("download"), data=f"Objet : {subject}\n\n{body}\n",
        file_name=f"demande_rattrapage_{key.replace(' ', '_')}.txt",
        mime="text/plain", icon=":material/download:",
    )

    if not (core.SMTP_USER and core.SMTP_PASSWORD and core.MAIL_TO):
        st.caption(t("mail_off"))
        return
    ok = st.checkbox(t("confirm").format(to=core.MAIL_TO), key="rq_confirm")
    if st.button(t("send"), icon=":material/send:", type="primary", disabled=not ok,
                 key="rq_send"):
        result = core.send_email(subject, body, cc=prof.get("email", ""))
        if result.startswith("EMAIL_SENT"):
            st.success(t("sent").format(to=core.MAIL_TO))
        else:
            st.error(result.replace("EMAIL_ERROR: ", ""))


def _assistant() -> None:
    agent = st.session_state.agent
    st.markdown(f'<div class="card-title">{esc(t("assistant"))}</div>',
                unsafe_allow_html=True)

    shortcuts = {f"{icon} {label}": question for label, icon, question in t("shortcuts")}

    def _pick() -> None:
        choice = st.session_state.get("chip")
        if choice:
            st.session_state.queued_question = shortcuts.get(choice)
        st.session_state.chip = None

    st.pills("shortcuts", list(shortcuts), key="chip", on_change=_pick,
             label_visibility="collapsed")
    clicked = st.session_state.pop("queued_question", None)

    history = st.container()
    with history:
        if not agent.messages and not clicked:
            st.markdown(f'<div class="empty">{esc(t("empty"))}</div>',
                        unsafe_allow_html=True)
        for m in agent.messages:
            if m.get("role") in ("user", "assistant"):
                with st.chat_message(m["role"], avatar=":material/person:"
                                     if m["role"] == "user" else ":material/school:"):
                    st.markdown(core._format_chat_display(m.get("content", "")))

    if agent.pending_email:
        st.warning(t("pending"), icon=":material/mail:")

    prompt = st.chat_input(t("input")) or clicked
    if prompt:
        with history:
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(prompt)
            with st.chat_message("assistant", avatar=":material/school:"):
                try:
                    with st.spinner(t("thinking")):
                        answer = agent.ask(prompt)
                    st.markdown(core._format_chat_display(answer))
                    if agent.trace:
                        with st.expander(t("steps")):
                            for step in agent.trace:
                                st.write("• " + step)
                except Exception as exc:
                    st.error(t("error"))
                    with st.expander(t("error_detail")):
                        st.code(str(exc))
        core.save_chat(st.session_state.chat_id, agent.messages)


# ---------------------------------------------------------------------------
# PAGE
# ---------------------------------------------------------------------------

def render() -> None:
    st.set_page_config(page_title="ISET Charguia – Assistant", page_icon="🎓",
                       layout="wide", initial_sidebar_state="expanded")
    _init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    if _lang() == "ar":
        st.markdown(RTL_CSS, unsafe_allow_html=True)

    _header()
    _sidebar()
    _hero()

    left, right = st.columns([1.75, 1], gap="large")
    with right:
        _request_card()
        _planning()
        _rules()
    with left:
        with st.container(border=True):
            _assistant()

    st.markdown(f'<div class="foot">{esc(t("footer"))}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    render()
