from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/", StaticFiles(directory="."), name="static")

DEMO_EVENTS = [
    {"time": "09:12", "event": "PERSON_ENTERED", "person": "Person ID 1", "risk": "Low", "detail": "Person entered the main entrance."},
    {"time": "09:18", "event": "LOITERING", "person": "Person ID 3", "risk": "Medium", "detail": "Person stayed near the entrance for 48 seconds."},
    {"time": "10:05", "event": "WATCHLIST_ALERT", "person": "Person ID 3", "risk": "High", "detail": "Repeated loitering visitor detected again."},
]


def demo_answer(question: str):
    q = question.lower().strip()

    if "loiter" in q:
        return """
        <b>Loitering Incident Detected</b><br><br>
        Person ID 3 remained near the Main Entrance for <b>48 seconds</b> at 09:18.<br><br>
        <b>Risk Level:</b> Medium<br>
        <b>AI Recommendation:</b> Review the evidence clip and monitor if the person returns.
        """

    if "highest risk" in q or "suspicious" in q or "risk" in q:
        return """
        <b>Highest Risk Person Identified</b><br><br>
        Person ID 3 currently has the highest risk score.<br><br>
        <b>Reason:</b><br>
        • Repeated loitering behavior<br>
        • Watchlist match history<br>
        • Returned multiple times today<br>
        • Abnormal stay duration near entrance<br><br>
        <b>Risk Level:</b> High
        """

    if "evidence" in q or "clip" in q or "video" in q:
        return """
        <b>Latest Evidence Found</b><br><br>
        Incident: WATCHLIST_ALERT<br>
        Time: 10:05<br>
        Person: Person ID 3<br>
        Camera: Main Entrance<br><br>
        <b>Evidence:</b> Snapshot and incident video clip are ready for review.<br><br>
        <button onclick="openEvidence()">Open Evidence Review</button>
        """

    if "report" in q or "summary" in q:
        return """
        <b>AI Security Report</b><br><br>
        Total events today: 3<br>
        Loitering incidents: 1<br>
        Watchlist alerts: 1<br>
        Highest risk person: Person ID 3<br><br>
        <b>Overall Risk:</b> Medium to High<br>
        <b>Recommended Action:</b> Review evidence and monitor Main Entrance.
        """

    if "entered" in q or "enter" in q:
        return """
        <b>Entry Event Found</b><br><br>
        Person ID 1 entered the Main Entrance at 09:12.<br>
        Risk Level: Low.
        """

    return """
    <b>VisionGuard AI Assistant</b><br><br>
    I can answer demo CCTV questions.<br><br>
    Try asking:<br>
    • Who loitered today?<br>
    • Who has highest risk?<br>
    • Show latest evidence<br>
    • Generate security report
    """


@app.get("/", response_class=HTMLResponse)
def demo_home():
    event_rows = ""

    for e in DEMO_EVENTS:
        risk_class = e["risk"].lower()
        event_rows += f"""
        <tr>
            <td>{e['time']}</td>
            <td>{e['event']}</td>
            <td>{e['person']}</td>
            <td><span class="risk {risk_class}">{e['risk']}</span></td>
            <td>{e['detail']}</td>
        </tr>
        """

    return HTMLResponse(f"""
<!DOCTYPE html>
<html>
<head>
<title>VisionGuard AI Demo</title>
<style>
* {{
    box-sizing:border-box;
}}

html, body {{
    margin:0;
    width:100%;
    overflow-x:hidden;
    background:#070b16;
    color:white;
    font-family:Arial, sans-serif;
}}

.page {{
    padding:24px;
    max-width:100%;
}}

.hero {{
    background:
        linear-gradient(135deg,rgba(124,58,237,.95),rgba(17,24,39,.96)),
        radial-gradient(circle at top right,#ef4444,transparent 35%);
    padding:42px;
    border-radius:26px;
    margin-bottom:20px;
    box-shadow:0 25px 70px rgba(0,0,0,.45);
    position:relative;
    overflow:hidden;
}}

.hero h1 {{
    font-size:52px;
    margin:8px 0;
    letter-spacing:-1px;
}}

.hero .sub {{
    font-size:20px;
    color:#e5e7eb;
}}

.hero .tagline {{
    color:#cbd5e1;
    font-size:15px;
}}

.badge {{
    display:inline-block;
    background:#16a34a;
    padding:8px 13px;
    border-radius:999px;
    font-size:12px;
    font-weight:bold;
}}

.owner {{
    position:absolute;
    right:28px;
    bottom:24px;
    text-align:right;
    color:#e5e7eb;
    font-size:13px;
}}

.kpi-grid {{
    display:grid;
    grid-template-columns:repeat(4, minmax(0, 1fr));
    gap:16px;
    margin-bottom:18px;
}}

.kpi-card {{
    background:#111827;
    border:1px solid #273449;
    border-radius:18px;
    padding:22px;
    box-shadow:0 12px 30px rgba(0,0,0,.28);
}}

.kpi-title {{
    color:#94a3b8;
    font-size:13px;
    margin-bottom:10px;
}}

.kpi-value {{
    font-size:36px;
    font-weight:bold;
    color:#a78bfa;
}}

.grid {{
    display:grid;
    grid-template-columns:minmax(0, 1.08fr) minmax(0, 1fr);
    gap:18px;
}}

.card {{
    background:#111827;
    border:1px solid #273449;
    border-radius:18px;
    padding:20px;
    box-shadow:0 12px 30px rgba(0,0,0,.26);
    max-width:100%;
}}

.camera {{
    height:360px;
    position:relative;
    background:#020617;
    border-radius:16px;
    overflow:hidden;
    border:2px solid #4f46e5;
}}

.video-bg {{
    position:absolute;
    inset:0;
    width:100%;
    height:100%;
    object-fit:cover;
    filter:blur(20px) brightness(.45);
    transform:scale(1.18);
    z-index:0;
}}

.video-main {{
    position:absolute;
    top:0;
    left:50%;
    transform:translateX(-50%);
    height:100%;
    width:auto;
    z-index:1;
}}

.camera:before {{
    content:"";
    position:absolute;
    inset:0;
    background:
        linear-gradient(90deg,rgba(255,255,255,.04) 1px,transparent 1px),
        linear-gradient(rgba(255,255,255,.04) 1px,transparent 1px);
    background-size:40px 40px;
    z-index:2;
    pointer-events:none;
}}

.camera:after {{
    content:"";
    position:absolute;
    inset:0;
    background:repeating-linear-gradient(
        0deg,
        rgba(255,255,255,.04) 0px,
        rgba(255,255,255,.04) 1px,
        transparent 1px,
        transparent 5px
    );
    z-index:2;
    opacity:.35;
    pointer-events:none;
}}

.rec {{
    position:absolute;
    top:14px;
    left:14px;
    color:#ef4444;
    font-weight:bold;
    z-index:5;
    animation:blink 1s infinite;
}}

@keyframes blink {{
    0%,100% {{ opacity:1; }}
    50% {{ opacity:.35; }}
}}

.cam-id {{
    position:absolute;
    top:14px;
    right:14px;
    background:rgba(2,6,23,.78);
    padding:8px 12px;
    border-radius:999px;
    z-index:5;
    font-weight:bold;
}}

.detect-box {{
    position:absolute;
    width:150px;
    height:235px;
    border:3px solid #22c55e;
    border-radius:12px;
    left:45%;
    top:58px;
    z-index:4;
    box-shadow:0 0 20px rgba(34,197,94,.7);
    animation:trackPulse 1.5s infinite;
}}

@keyframes trackPulse {{
    0%,100% {{ box-shadow:0 0 14px rgba(34,197,94,.5); }}
    50% {{ box-shadow:0 0 32px rgba(34,197,94,.95); }}
}}

.detect-label {{
    position:absolute;
    left:45%;
    top:30px;
    background:#22c55e;
    color:#052e16;
    padding:7px 11px;
    border-radius:8px;
    font-weight:bold;
    font-size:12px;
    z-index:5;
}}

.alert-chip {{
    position:absolute;
    bottom:16px;
    right:16px;
    background:#ef4444;
    padding:11px 15px;
    border-radius:12px;
    font-weight:bold;
    z-index:5;
    box-shadow:0 0 22px rgba(239,68,68,.75);
    animation: dangerPulse 1.2s infinite;
}}

@keyframes dangerPulse {{
    0%,100% {{ transform:scale(1); }}
    50% {{ transform:scale(1.05); }}
}}

.meta-box {{
    position:absolute;
    left:16px;
    bottom:16px;
    background:rgba(2,6,23,.82);
    border:1px solid rgba(148,163,184,.45);
    padding:12px;
    border-radius:12px;
    line-height:1.6;
    font-size:12px;
    z-index:5;
}}

button {{
    background:#7c3aed;
    color:white;
    border:0;
    padding:12px 14px;
    border-radius:11px;
    cursor:pointer;
    margin:5px;
    font-weight:bold;
}}

button:hover {{
    background:#6d28d9;
}}

.input-row {{
    display:flex;
    gap:8px;
    margin-top:12px;
}}

input {{
    flex:1;
    padding:13px;
    border-radius:11px;
    border:1px solid #334155;
    background:#020617;
    color:white;
}}

.chat-box {{
    margin-top:14px;
    background:#020617;
    border-radius:14px;
    padding:16px;
    border:1px solid #253047;
    min-height:190px;
    max-height:280px;
    overflow-y:auto;
}}

.message {{
    max-width:82%;
    padding:12px 14px;
    border-radius:14px;
    margin:8px 0;
    line-height:1.55;
    font-size:14px;
}}

.user-msg {{
    background:#7c3aed;
    margin-left:auto;
    border-bottom-right-radius:4px;
}}

.ai-msg {{
    background:#111827;
    border:1px solid #334155;
    margin-right:auto;
    border-bottom-left-radius:4px;
}}

.live-alert {{
    margin-top:18px;
    background:linear-gradient(135deg, rgba(127,29,29,.6), rgba(17,24,39,.96));
    border:1px solid rgba(248,113,113,.7);
    border-radius:18px;
    padding:20px;
    box-shadow:0 0 30px rgba(239,68,68,.28);
    animation:pulseGlow 1.8s infinite;
}}

.alert-top {{
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-bottom:10px;
}}

.alert-badge {{
    background:#ef4444;
    color:white;
    padding:8px 13px;
    border-radius:999px;
    font-size:12px;
    font-weight:bold;
}}

.alert-grid {{
    display:grid;
    grid-template-columns:repeat(4, minmax(0,1fr));
    gap:12px;
    margin-top:12px;
}}

.alert-item {{
    background:rgba(2,6,23,.5);
    border-radius:12px;
    padding:13px;
}}

.alert-label {{
    color:#94a3b8;
    font-size:12px;
    margin-bottom:5px;
}}

.alert-value {{
    font-weight:bold;
}}

.clickable {{
    cursor:pointer;
    border:1px solid rgba(124,58,237,.7);
}}

.clickable:hover {{
    background:rgba(124,58,237,.18);
}}

@keyframes pulseGlow {{
    0% {{ box-shadow:0 0 16px rgba(239,68,68,.18); }}
    50% {{ box-shadow:0 0 36px rgba(239,68,68,.5); }}
    100% {{ box-shadow:0 0 16px rgba(239,68,68,.18); }}
}}

.prediction-box {{
    display:flex;
    gap:30px;
    align-items:center;
}}

.prediction-score {{
    width:155px;
    height:155px;
    min-width:155px;
    border-radius:50%;
    background:radial-gradient(circle,#ef4444,#7f1d1d);
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    font-size:40px;
    font-weight:bold;
    box-shadow:0 0 38px rgba(239,68,68,.6);
}}

.prediction-score small {{
    font-size:14px;
    color:#fecaca;
}}

.risk-line {{
    height:10px;
    background:#1e293b;
    border-radius:999px;
    margin:14px 0;
    overflow:hidden;
}}

.risk-fill {{
    height:100%;
    width:84%;
    background:linear-gradient(90deg,#22c55e,#f97316,#ef4444);
}}

.feature-grid, .evidence-grid {{
    display:grid;
    grid-template-columns:repeat(3, minmax(0,1fr));
    gap:14px;
}}

.feature, .evidence {{
    background:#020617;
    border:1px solid #253047;
    border-radius:14px;
    padding:16px;
    line-height:1.6;
}}

.evidence {{
    border-color:#334155;
    cursor:pointer;
}}

.evidence:hover {{
    border-color:#8b5cf6;
}}

table {{
    width:100%;
    border-collapse:collapse;
}}

th,td {{
    padding:12px;
    border-bottom:1px solid #253047;
    text-align:left;
}}

th {{
    background:#312e81;
}}

.risk {{
    padding:6px 10px;
    border-radius:999px;
    font-weight:bold;
    font-size:12px;
}}

.low {{ background:#064e3b; color:#86efac; }}
.medium {{ background:#7c2d12; color:#fdba74; }}
.high {{ background:#7f1d1d; color:#fca5a5; }}

.footer {{
    text-align:center;
    margin-top:22px;
    padding:28px;
    background:linear-gradient(135deg,#312e81,#111827);
    border-radius:20px;
    border:1px solid #4f46e5;
}}

.contact {{
    color:#cbd5e1;
    line-height:1.8;
}}

.modal {{
    display:none;
    position:fixed;
    inset:0;
    background:rgba(2,6,23,.82);
    z-index:100;
    align-items:center;
    justify-content:center;
    padding:24px;
}}

.modal-content {{
    width:min(1150px, 96vw);
    max-height:92vh;
    overflow:auto;
    background:#0f172a;
    border:1px solid #4f46e5;
    border-radius:20px;
    padding:22px;
    box-shadow:0 30px 100px rgba(0,0,0,.65);
}}

.modal-top {{
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-bottom:16px;
}}

.close {{
    background:#ef4444;
}}

.evidence-layout {{
    display:grid;
    grid-template-columns:1.2fr .9fr;
    gap:16px;
}}

.evidence-video {{
    width:100%;
    max-height:540px;
    background:#020617;
    border-radius:14px;
    border:1px solid #334155;
}}

.ai-summary {{
    background:#020617;
    border:1px solid #334155;
    border-radius:14px;
    padding:16px;
    line-height:1.65;
}}

.timeline {{
    margin-top:14px;
    padding:14px;
    background:#020617;
    border:1px solid #334155;
    border-radius:14px;
}}

.timeline-item {{
    display:flex;
    gap:12px;
    padding:8px 0;
    border-bottom:1px solid #1e293b;
}}

.timeline-item:last-child {{
    border-bottom:0;
}}

.time {{
    color:#a78bfa;
    font-weight:bold;
    width:60px;
}}

@media (max-width: 1000px) {{
    .kpi-grid, .grid, .feature-grid, .evidence-grid, .alert-grid, .evidence-layout {{
        grid-template-columns:1fr;
    }}
    .owner {{
        position:static;
        text-align:left;
        margin-top:20px;
    }}
    .hero h1 {{
        font-size:40px;
    }}
}}
</style>
</head>

<body>
<div class="page">

    <div class="hero">
        <div class="badge">ENTERPRISE DEMO</div>
        <h1>VisionGuard AI</h1>
        <p class="sub">AI-Powered CCTV Surveillance & Investigation Platform</p>
        <p class="tagline">Detect suspicious behavior • Predict threats • Alert security teams • Investigate faster</p>

        <div class="owner">
            Built by <b>Urifatul Jannah</b><br>
            Computer Vision & AI Developer
        </div>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-title">🚨 Threat Level</div><div class="kpi-value">82%</div></div>
        <div class="kpi-card"><div class="kpi-title">👤 People Detected</div><div class="kpi-value">4</div></div>
        <div class="kpi-card"><div class="kpi-title">⚠ Alerts Today</div><div class="kpi-value">12</div></div>
        <div class="kpi-card"><div class="kpi-title">🤖 AI Confidence</div><div class="kpi-value">96%</div></div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>📹 AI CCTV Monitoring</h2>

            <div class="camera">
                <video class="video-bg" src="/demo_clip.mp4" autoplay muted loop playsinline></video>
                <video class="video-main" src="/demo_clip.mp4" autoplay muted loop playsinline></video>

                <div class="rec">● REC</div>
                <div class="cam-id">CAM 01 · Main Entrance</div>

                <div id="detectLabel" class="detect-label">Person ID 3 · HIGH RISK</div>
                <div id="detectBox" class="detect-box"></div>
                <div id="alertChip" class="alert-chip">LOITERING 48s</div>

                <div id="metaBox" class="meta-box">
                    Tracking ID: 3<br>
                    Confidence: 96%<br>
                    Behavior: Loitering<br>
                    Risk Score: 84%
                </div>
            </div>

            <p style="color:#94a3b8;">Demo CCTV video with AI overlay. Simulated public demo using sample footage.</p>
        </div>

        <div class="card">
            <h2>🤖 AI Security Assistant</h2>

            <button onclick="askDemo('Who loitered today?')">Who loitered today?</button>
            <button onclick="askDemo('Who has highest risk?')">Highest risk</button>
            <button onclick="askDemo('Show latest evidence')">Latest evidence</button>
            <button onclick="askDemo('Generate security report')">Security report</button>

            <div class="input-row">
                <input id="q" placeholder="Ask about demo CCTV events..." onkeydown="if(event.key==='Enter') askDemo()">
                <button onclick="askDemo()">Ask</button>
            </div>

            <div id="chat" class="chat-box">
                <div class="message user-msg">Who loitered today?</div>
                <div class="message ai-msg">
                    <b>Loitering Incident Detected</b><br>
                    Person ID 3 remained near the Main Entrance for <b>48 seconds</b> at 09:18.<br>
                    Risk Level: Medium.
                </div>
            </div>
        </div>
    </div>

    <div class="live-alert">
        <div class="alert-top">
            <div>
                <h2 style="margin:0;">🚨 Live Alert</h2>
                <p style="margin:6px 0 0;color:#fecaca;">Watchlist Match Detected at Main Entrance</p>
            </div>
            <div class="alert-badge">HIGH RISK</div>
        </div>

        <div class="alert-grid">
            <div class="alert-item"><div class="alert-label">Person</div><div class="alert-value">Person ID 3</div></div>
            <div class="alert-item"><div class="alert-label">Camera</div><div class="alert-value">Main Entrance</div></div>
            <div class="alert-item"><div class="alert-label">Time</div><div class="alert-value">10:05</div></div>
            <div class="alert-item clickable" onclick="openEvidence()"><div class="alert-label">Action</div><div class="alert-value">Review Evidence</div></div>
        </div>
    </div>

    <div class="card" style="margin-top:18px;">
        <h2>⚠ AI Threat Prediction</h2>
        <div class="prediction-box">
            <div class="prediction-score">84%<small>HIGH RISK</small></div>
            <div>
                <h3 style="margin:0;">Suspicious Behavior Probability</h3>
                <p style="color:#94a3b8;">Person ID 3 shows escalating suspicious activity.</p>
                <div class="risk-line"><div class="risk-fill"></div></div>
                <ul>
                    <li>Repeated loitering detected</li>
                    <li>Returned 3 times today</li>
                    <li>Abnormal stay duration near entrance</li>
                    <li>Risk escalation increased by 18% in the last 2 hours</li>
                </ul>
            </div>
        </div>
    </div>

    <div class="card" style="margin-top:18px;">
        <h2>📁 Evidence Review</h2>
        <div class="evidence-grid">
            <div class="evidence" onclick="openEvidence()">
                <b>Snapshot Evidence</b><br>
                Incident ID: VG-071<br>
                Camera: Main Entrance<br>
                Status: Ready<br><br>
                <button>Open Evidence</button>
            </div>
            <div class="evidence" onclick="openEvidence()">
                <b>Video Clip</b><br>
                Duration: 10 seconds<br>
                Event: Loitering<br>
                Status: Ready<br><br>
                <button>Play Clip</button>
            </div>
            <div class="evidence">
                <b>AI Summary</b><br>
                Person returned multiple times and triggered a high-risk alert.
            </div>
        </div>
    </div>

    <div class="card" style="margin-top:18px;">
        <h2>🧠 AI Security Capabilities</h2>
        <div class="feature-grid">
            <div class="feature"><b>🤖 AI Incident Summary</b><br>Automatically explains suspicious events in natural language.</div>
            <div class="feature"><b>🔎 Natural Language Search</b><br>Ask questions like “Who loitered today?”</div>
            <div class="feature"><b>🧠 Threat Prediction</b><br>Predicts risk escalation from suspicious behavior.</div>
            <div class="feature"><b>📹 Smart Evidence Review</b><br>Links snapshots, video clips, and AI summaries.</div>
            <div class="feature"><b>👤 Multi-Camera Tracking</b><br>Track the same person across multiple cameras.</div>
            <div class="feature"><b>📄 AI Security Report</b><br>Generate incident reports instantly.</div>
        </div>
    </div>

    <div class="card" style="margin-top:18px;">
        <h2>🤖 Try VisionGuard AI</h2>
        <p style="color:#94a3b8;">
            Ask questions like a real security operator.
        </p>
        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:20px;">
            <button onclick="askDemo('Who loitered today?')">👤 Who loitered today?</button>
            <button onclick="askDemo('Show latest evidence')">📹 Show latest evidence</button>
            <button onclick="askDemo('Generate security report')">📄 Generate security report</button>
            <button onclick="askDemo('Why is Person ID 3 high risk?')">⚠ Why is Person ID 3 high risk?</button>
            <button onclick="askDemo('Summarize today incidents')">🤖 Summarize today incidents</button>
            <button onclick="askDemo('Show all high risk events')">🚨 Show all high-risk events</button>
        </div>
    </div>
<div class="footer">

    <h2>🚀 Need an AI CCTV, Computer Vision, or AI Assistant Solution?</h2>

    <p class="contact">

        Built by <b>Urifatul Jannah</b><br>
        Computer Vision & AI Developer

        <br><br>

        📧 <b>Email:</b>
        <a href="mailto:Jannahurifa04@gmail.com" style="color:#9f7aea;text-decoration:none;">
            Jannahurifa04@gmail.com
        </a>

        <br>

        📱 <b>WhatsApp:</b>
        <a href="https://wa.me/6585862675?text=Hi%20Urifatul,%20I%20am%20interested%20in%20your%20AI%20CCTV%20solution."
        target="_blank"
        style="color:#9f7aea;text-decoration:none;">
            +65 8586 2675
        </a>

        <br>

        💼 <b>LinkedIn:</b>
        <a href="https://www.linkedin.com/in/urifa-jannah-185177415"
        target="_blank"
        style="color:#9f7aea;text-decoration:none;">
            linkedin.com/in/urifa-jannah-185177415
        </a>

        <br><br>

        <b>Available for:</b><br>

        ✅ AI CCTV Solutions<br>
        ✅ Computer Vision Projects<br>
        ✅ AI Assistant & RAG Development<br>
        ✅ Freelance Projects<br>
        ✅ Remote Collaboration

    </p>

    <div style="display:flex;justify-content:center;gap:15px;flex-wrap:wrap;margin-top:20px;">

        <a href="mailto:Jannahurifa04@gmail.com">
            <button>📧 Email Me</button>
        </a>

        <a href="https://wa.me/6585862675?text=Hi%20Urifatul,%20I%20am%20interested%20in%20your%20AI%20CCTV%20solution."
        target="_blank">
            <button>💬 WhatsApp</button>
        </a>

        <a href="https://www.linkedin.com/in/urifa-jannah-185177415"
        target="_blank">
            <button>💼 LinkedIn</button>
        </a>

    </div>

</div>

</div> <!-- closes page -->

<div id="evidenceModal" class="modal">
    <div class="modal-content">
        <div class="modal-top">
            <div>
                <h2 style="margin:0;">📁 Evidence Review · Incident VG-071</h2>
                <p style="color:#94a3b8;margin:6px 0 0;">Main Entrance · Watchlist Alert · High Risk</p>
            </div>
            <button class="close" onclick="closeEvidence()">Close</button>
        </div>

        <div class="evidence-layout">
            <div>
                <video id="evidenceVideo" class="evidence-video" src="/demo_clip.mp4" controls muted loop></video>

                <div class="timeline">
                    <b>Incident Timeline</b>
                    <div class="timeline-item"><div class="time">09:12</div><div>Person entered Main Entrance</div></div>
                    <div class="timeline-item"><div class="time">09:18</div><div>Loitering detected for 48 seconds</div></div>
                    <div class="timeline-item"><div class="time">10:05</div><div>Watchlist alert triggered again</div></div>
                </div>
            </div>

            <div class="ai-summary">
                <h3>🤖 AI Incident Summary</h3>
                <p><b>Person:</b> Person ID 3</p>
                <p><b>Risk Level:</b> High</p>
                <p><b>Confidence:</b> 96%</p>
                <p>
                    Person ID 3 was detected near the Main Entrance and triggered a high-risk watchlist alert.
                    The system identified suspicious loitering behavior and repeated appearance history.
                </p>
                <p><b>Recommended Action:</b></p>
                <ul>
                    <li>Review the evidence clip</li>
                    <li>Notify nearby security personnel</li>
                    <li>Monitor future re-entry</li>
                </ul>
            </div>
        </div>
    </div>
</div>

<script>
async function askDemo(text=null){{
    let q = text || document.getElementById("q").value;
    if(!q) return;

    const chat = document.getElementById("chat");

    const userMsg = document.createElement("div");
    userMsg.className = "message user-msg";
    userMsg.innerText = q;
    chat.appendChild(userMsg);

    const aiMsg = document.createElement("div");
    aiMsg.className = "message ai-msg";
    aiMsg.innerHTML = "Analyzing CCTV events...";
    chat.appendChild(aiMsg);

    chat.scrollTop = chat.scrollHeight;
    document.getElementById("q").value = "";

    const res = await fetch("/answer?ask=" + encodeURIComponent(q));
    const data = await res.json();

    setTimeout(() => {{
        aiMsg.innerHTML = data.answer;
        chat.scrollTop = chat.scrollHeight;
    }}, 450);
}}

const detectBox = document.getElementById("detectBox");
const detectLabel = document.getElementById("detectLabel");
const alertChip = document.getElementById("alertChip");
const metaBox = document.getElementById("metaBox");
const video = document.querySelector(".video-main");

detectBox.style.display = "none";
detectLabel.style.display = "none";
alertChip.style.display = "none";
metaBox.style.display = "none";

setInterval(() => {{
    if (!video) return;

    const t = video.currentTime;

    // Person appears twice in this demo video:
    // 1st appearance: 3.5s - 8.8s
    // 2nd appearance: 20.5s - 32.8s
    if ((t >= 3.5 && t <= 8.8) || (t >= 20.5 && t <= 32.8)) {{
        detectBox.style.display = "block";
        detectLabel.style.display = "block";
        metaBox.style.display = "block";
    }} else {{
        detectBox.style.display = "none";
        detectLabel.style.display = "none";
        metaBox.style.display = "none";
    }}

    // Loitering alert appears only during the second longer stay
    if (t >= 27 && t <= 32.8) {{
        alertChip.style.display = "block";
    }} else {{
        alertChip.style.display = "none";
    }}

}}, 200);

function openEvidence(){{
    const modal = document.getElementById("evidenceModal");
    const video = document.getElementById("evidenceVideo");
    modal.style.display = "flex";
    setTimeout(() => video.play(), 150);
}}

function closeEvidence(){{
    const modal = document.getElementById("evidenceModal");
    const video = document.getElementById("evidenceVideo");
    video.pause();
    modal.style.display = "none";
}}
</script>

</body>
</html>
""")


@app.get("/answer")
def answer(ask: str = Query(default="")):
    return {"answer": demo_answer(ask)}
