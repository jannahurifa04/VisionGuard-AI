# 🛡️ VisionGuard AI

## Intelligent AI CCTV Monitoring & Video Analytics Platform

VisionGuard AI is an AI-powered CCTV analytics system designed to transform traditional camera feeds into intelligent, searchable security information.

Instead of only recording video, VisionGuard can detect people, maintain persistent identities, analyze clothing, detect events and configurable objects, save evidence, build person history, and allow users to investigate CCTV activity through a natural-language AI assistant.

---

## 🚀 Current Project Status

VisionGuard AI currently includes working prototypes for:

* ✅ Real-time person detection
* ✅ Person tracking
* ✅ Persistent person identity
* ✅ Face recognition
* ✅ Person Re-Identification
* ✅ Canonical identity mapping
* ✅ Deep-learning clothing-color recognition
* ✅ Human clothing segmentation
* ✅ Loitering detection
* ✅ Person entry/presence/exit events
* ✅ Snapshot evidence
* ✅ Video evidence
* ✅ Person Intelligence profiles
* ✅ Watchlist management
* ✅ Risk assessment
* ✅ AI CCTV dashboard
* ✅ RAG + LLM CCTV Assistant
* ✅ YOLO-World open-vocabulary object detection
* ✅ Object-alert evidence
* ✅ Object-alert SQLite logging

VisionGuard is actively under development.

---

# 📷 Camera Input

The current development system supports an IP camera / smartphone camera feed.

Current prototype input:

```text
Camera / Android IP Webcam
        ↓
HTTP Camera Stream / Snapshot
        ↓
VisionGuard Processing Pipeline
```

Camera frames are processed in real time by the computer-vision pipeline.

---

# 🎯 Person Detection

VisionGuard uses **YOLO11** for real-time person detection.

Main responsibilities:

* Detect people in camera frames
* Generate person bounding boxes
* Provide crops for identity analysis
* Feed detections into tracking
* Support event generation
* Support clothing analysis

Current person detection model:

```text
YOLO11
```

---

# 🏃 Person Tracking

VisionGuard uses **ByteTrack** for short-term person tracking.

ByteTrack provides temporary tracking IDs while a person remains visible.

```text
YOLO11 Detection
       ↓
ByteTrack
       ↓
Temporary Track ID
```

Temporary tracking IDs are not treated as permanent identities.

VisionGuard therefore combines tracking with face recognition, ReID, and persistent memory.

---

# 🧠 Stable Person Identity

One of the main technical challenges in VisionGuard is maintaining the same identity after a person:

* leaves the camera
* returns later
* changes position
* changes viewing angle
* appears under different lighting
* changes clothing

VisionGuard uses multiple identity signals rather than relying only on a tracking ID.

Current identity pipeline:

```text
Person Detection
       ↓
ByteTrack
       ↓
Person Crop
       ↓
┌───────────────────────┐
│ Face Recognition      │
│ Person ReID           │
└───────────────────────┘
       ↓
Identity Fusion
       ↓
Persistent Person ID
       ↓
Canonical ID Mapping
```

---

# 🙂 Face Recognition

VisionGuard uses **InsightFace** for persistent face recognition.

Current configuration:

```text
InsightFace
buffalo_l model pack
GPU inference with ONNX Runtime
```

The `buffalo_l` recognition pipeline provides **ArcFace-based face embedding recognition**.

The face system performs:

* Face detection
* Face quality checking
* Face embedding extraction
* Embedding normalization
* Cosine-similarity matching
* Persistent face galleries
* Multi-embedding person profiles
* Match confidence checking
* Match-margin checking
* Duplicate embedding prevention

Face recognition is treated as the primary long-term identity signal when a clear face is available.

---

# 🧬 Person Re-Identification

VisionGuard also uses **OSNet** through TorchReID for body-based Person Re-Identification.

Current ReID model:

```text
OSNet
osnet_x1_0
```

ReID is useful when:

* the face is not visible
* the person is turned away
* face quality is too low
* short-term identity continuity is needed

VisionGuard maintains multiple body embeddings for a person instead of relying on only one embedding.

---

# 🔗 Canonical Identity Mapping

Real-world recognition systems can sometimes create duplicate galleries for the same real person.

VisionGuard includes a canonical identity layer that can map multiple identity galleries into one permanent person identity.

Example:

```text
Face Gallery A ─┐
                ├──> Canonical Person ID
Face Gallery B ─┘
```

This allows older embeddings to remain useful while preventing duplicate identities from appearing in the application.

---

# 💾 Persistent Identity Memory

VisionGuard stores identity information locally so recognition can survive application restarts.

Persistent memory includes:

* Face embedding galleries
* ReID embedding galleries
* Person profile information
* Canonical identity relationships

Private biometric memory files are excluded from the public GitHub repository.

---

# 👕 Deep-Learning Clothing Analysis

VisionGuard includes a custom deep-learning clothing-color recognition pipeline.

The system does not simply sample random pixels from a person bounding box.

Instead, it first isolates the clothing region.

Pipeline:

```text
Person Detection
       ↓
Person Crop
       ↓
FASHN Human Parsing
       ↓
Upper Clothing Segmentation
       ↓
MobileNetV3
       ↓
Clothing Color
```

---

# 🧍 Human Parsing

VisionGuard uses **FASHN Human Parser** to isolate upper-body clothing.

This reduces interference from:

* skin
* hair
* background
* walls
* furniture
* other objects

The segmented clothing region is passed to the clothing classifier.

---

# 🎨 Clothing Color Classification

VisionGuard uses a custom **MobileNetV3 Small** classifier trained specifically for clothing-color recognition.

Current prototype color classes:

```text
Black
White
Pink
Gray
Yellow
```

The classifier runs using PyTorch / Torchvision.

---

# 📚 Custom Clothing Dataset Pipeline

A complete dataset collection and training pipeline was created for VisionGuard.

The project includes tools for:

### Dataset Capture

```text
capture_clothing_dataset.py
```

The script:

* reads the live camera
* detects the person
* isolates upper clothing
* applies human parsing
* saves clothing crops into labeled folders

### Dataset Preparation

```text
prepare_clothing_dataset.py
```

Used to organize images into training, validation, and test sets.

### Model Training

```text
train_clothing_color.py
```

Used to train the MobileNetV3 clothing-color classifier.

### Live Model Testing

```text
test_clothing_color_live.py
```

Used to test clothing predictions against a live camera feed before integrating the classifier into VisionGuard.

Raw training images and model weights are intentionally excluded from the public repository.

---

# 🌍 YOLO-World Smart Object Monitoring

VisionGuard now includes **YOLO-World open-vocabulary object detection**.

Unlike a traditional detector with only a fixed class list, YOLO-World allows monitored objects to be defined using text prompts.

Example:

```python
OBJECTS_TO_DETECT = [
    "person",
    "backpack",
    "cell phone",
    "bottle",
    "cardboard box",
]
```

This allows VisionGuard to be adapted to different customer environments without training a completely new detector for every object.

---

# 📦 Tested YOLO-World Objects

During prototype testing, VisionGuard successfully detected objects including:

```text
Person
Bottle
Cell Phone
Cardboard Box
```

Testing also showed that specific prompts can perform better than broad prompts.

For example:

```text
"cardboard box"
```

worked better than the more general:

```text
"package"
```

This experiment demonstrates how prompt selection can affect open-vocabulary detection.

---

# 🚨 Smart Object Alerts

VisionGuard includes an independent YOLO-World alert pipeline.

Current pipeline:

```text
Camera
   ↓
YOLO-World
   ↓
Configured Object Detection
   ↓
Confidence Check
   ↓
Alert Cooldown
   ↓
Snapshot Evidence
   ↓
SQLite Database
```

Current monitored alert objects can include:

```text
Backpack
Cell Phone
Cardboard Box
```

---

# 📸 Object Alert Evidence

When a monitored object is detected, VisionGuard can:

* Generate an object alert
* Store the detected object name
* Store detection confidence
* Save timestamp information
* Capture a snapshot
* Store the snapshot path
* Prevent excessive duplicate alerts using cooldown logic

Example:

```text
OBJECT ALERT
Cell Phone detected
Confidence: 0.54
Snapshot saved
Database event saved
```

---

# 🗄️ Object Alert Database

YOLO-World alerts are currently stored in a dedicated SQLite database.

Stored information includes:

```text
ID
Timestamp
Object Name
Confidence
Snapshot Path
```

The next step is integrating these alerts directly into the main VisionGuard dashboard.

---

# 🚶 CCTV Event Engine

VisionGuard contains event logic for analyzing person activity.

Current events include:

```text
PERSON_ENTERED
PERSON_PRESENT
PERSON_LEFT
LOITERING
```

---

# ⏱️ Loitering Detection

VisionGuard tracks how long a person remains in the monitored area.

When the configured duration is exceeded, a:

```text
LOITERING
```

event is generated.

Loitering information can contribute to:

* Person history
* Risk level
* Watchlist decisions
* AI Assistant answers
* Evidence generation

---

# 📹 Evidence Generation

VisionGuard supports automated CCTV evidence creation.

Current evidence capabilities include:

* Person snapshots
* Event snapshots
* Video clips
* YOLO-World object snapshots
* Event timestamps
* Person IDs
* Clothing color
* Camera information

Evidence can later be reviewed from the dashboard.

---

# 👤 Person Intelligence

VisionGuard maintains person-level profiles.

A person profile can contain:

```text
Person ID
Display Name
First Seen
Last Seen
Visit Count
Total Seen
Last Clothing Color
Camera
Risk Level
Average Visit Duration
Longest Visit Duration
Loitering Count
Last Loitering Event
Notes
```

This allows VisionGuard to move beyond simple frame-by-frame detection and build historical knowledge about visitors.

---

# 🛡️ Watchlist System

VisionGuard includes watchlist functionality.

People can be added to a watchlist for monitoring.

The system also contains logic for automatically flagging repeated suspicious activity, such as repeated loitering.

Watchlist data can include:

* Person ID
* Reason
* Risk level
* Status

---

# ⚠️ Risk Assessment

VisionGuard includes rule-based person risk assessment.

Risk levels include:

```text
Low
Medium
High
```

Repeated behavior such as loitering can increase a person's risk status.

---

# 📊 VisionGuard Dashboard

VisionGuard includes a custom web dashboard built with **FastAPI**.

Current dashboard modules include:

* Dashboard
* Live Cameras
* Recent Events
* Evidence
* Person Intelligence
* Watchlist
* AI Assistant
* Analytics / statistics

---

# 📺 AI-Processed Live Feed

The dashboard can display the latest AI-processed camera frame instead of only showing the raw camera feed.

The processed frame can include:

* Person boxes
* Permanent person ID
* Person name
* Identity method
* Clothing color
* Clothing confidence

A runtime status file is also used to communicate the number of people currently visible to the dashboard.

---

# 👥 People Present

VisionGuard maintains a real-time count of people currently visible.

This is separated from historical `PERSON_PRESENT` events so the dashboard does not incorrectly count people who have already left.

---

# 🧠 RAG Memory

VisionGuard includes a Retrieval-Augmented Generation architecture for CCTV investigation.

Technology used:

```text
ChromaDB
Persistent Vector Memory
SQLite Event Database
```

Incident information can be stored and later retrieved as context for the AI Assistant.

---

# 🤖 VisionGuard AI Assistant

VisionGuard includes a natural-language AI Assistant.

Current AI components include:

* Ollama
* Local LLMs
* Retrieval-Augmented Generation
* ChromaDB
* SQLite CCTV data
* Intent parsing

Example questions:

```text
Any loitering today?

What happened last week?

Tell me about Person ID 1 today.

What happened yesterday?

Who is currently present?

Show the latest incident.
```

The assistant combines structured CCTV event information with LLM-generated responses.

---

# 🧩 Intent Parser

VisionGuard includes a custom intent parser for CCTV questions.

The parser can identify information including:

### Event Intent

```text
LOITERING
ENTERED
LEFT
PRESENT
ANY
```

### Time Intent

```text
Today
Yesterday
Days Ago
Last Week
Last Month
Last Year
Latest
All
```

### Person Intent

```text
Person ID
```

This allows natural-language questions to be converted into structured CCTV searches.

---

# 🌐 Conversational CCTV

The long-term goal is for users to interact with CCTV data conversationally rather than manually searching recordings.

Example:

```text
User:
What happened this afternoon?

VisionGuard:
Person ID 11 entered the monitored area,
remained for several minutes and generated
a loitering event.
```

---

# 🗃️ Main Database

VisionGuard uses SQLite for structured CCTV information.

The system currently uses data structures for:

```text
Events
Persons
Watchlist
Object Alerts
```

Event records can include:

```text
Event Time
Event Type
Confidence
Snapshot
Video Clip
Person ID
Clothing Color
Camera ID
Camera Name
```

---

# 🏗️ VisionGuard Architecture

```text
                        CAMERA
                           │
                           ▼
                    FRAME CAPTURE
                           │
                           ▼
                       YOLO11
                   Person Detection
                           │
                           ▼
                      ByteTrack
                           │
                           ▼
                     Person Crop
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       InsightFace                OSNet ReID
   ArcFace-Based Face             Body Identity
       Recognition                  Signal
              │                         │
              └────────────┬────────────┘
                           ▼
                    Identity Fusion
                           │
                           ▼
                   Canonical Person ID
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     Event Engine      Clothing AI     Person Profile
                           │
                           ▼
                  FASHN Human Parsing
                           │
                           ▼
                      MobileNetV3
                           │
                           ▼
                    Clothing Color
          │
          ▼
       SQLite
          │
          ▼
   Evidence System
          │
          ▼
     RAG Memory
          │
          ▼
   ChromaDB + Ollama
          │
          ▼
 VisionGuard AI Assistant
          │
          ▼
    FastAPI Dashboard
```

---

# 🌍 Open-Vocabulary Monitoring Architecture

YOLO-World currently operates as an additional experimental detection layer.

```text
Camera
   │
   ▼
YOLO-World
   │
   ▼
Text-Defined Object Classes
   │
   ▼
Object Detection
   │
   ▼
Confidence Filter
   │
   ▼
Alert Cooldown
   │
   ▼
Snapshot Evidence
   │
   ▼
SQLite Object Alerts
   │
   ▼
Future Dashboard Integration
```

---

# 🛠️ Technology Stack

## Computer Vision

* YOLO11
* YOLO-World
* OpenCV
* ByteTrack

## Face Recognition

* InsightFace
* `buffalo_l`
* ArcFace-based face embeddings
* ONNX Runtime GPU

## Person Re-Identification

* TorchReID
* OSNet `osnet_x1_0`

## Clothing Intelligence

* FASHN Human Parsing
* MobileNetV3 Small
* PyTorch
* Torchvision
* Custom clothing dataset

## AI / Generative AI

* Retrieval-Augmented Generation
* Large Language Models
* Ollama
* ChromaDB

## Backend

* Python
* FastAPI
* SQLite

## Frontend

* HTML
* CSS
* JavaScript

## Data / ML Utilities

* NumPy
* Pillow
* Scikit-learn

## Development

* Git
* GitHub

---

# 🧪 Development & Testing Tools

The repository includes supporting scripts used while developing VisionGuard.

### Clothing AI

```text
capture_clothing_dataset.py
prepare_clothing_dataset.py
train_clothing_color.py
test_clothing_color_live.py
```

### YOLO-World

```text
test_yolo_world.py
yolo_world_alerts.py
```

### Identity

```text
face_memory.py
reid.py
identity_fusion.py
person_profile.py
```

### Main Application

```text
main_reid.py
dashboard.py
```

---

# 🔐 Privacy & Security

VisionGuard handles CCTV and biometric information, so sensitive runtime information is intentionally excluded from the public GitHub repository.

The repository does **not** publish:

```text
Face memory files
Face embeddings
ReID memory
Body embeddings
Raw CCTV images
Clothing training images
Object-alert snapshots
Identity backups
Local runtime files
SQLite production databases
Trained private model weights
```

Relevant exclusions are configured through `.gitignore`.

---

# 📁 Private Files Excluded From GitHub

Examples include:

```text
*.pkl
*.pth
*.db

clothing_color_raw/
clothing_color_dataset/
object_alerts/
identity_backup_*/
runtime/
```

This prevents personal surveillance and biometric information from being accidentally committed to the public repository.

---

# 🔬 Key Engineering Challenges

## Stable Identity

One of the most difficult problems in the project was maintaining the same permanent identity when a person left the camera and returned later.

The solution evolved into a combination of:

```text
Face Recognition
+
OSNet ReID
+
ByteTrack
+
Persistent Galleries
+
Canonical IDs
```

---

## Clothing Color

Traditional HSV-based color detection proved sensitive to:

* Lighting
* Shadows
* Background
* Camera exposure
* Clothing texture

The system was therefore upgraded to:

```text
Human Parsing
+
Deep-Learning Classification
```

using FASHN Human Parsing and MobileNetV3.

---

## Object Detection Flexibility

Traditional YOLO models require predefined training classes.

YOLO-World was added to explore configurable object monitoring using text-defined classes.

---

# 📈 Current Development Roadmap

## ✅ Completed / Working Prototype

* YOLO11 person detection
* ByteTrack tracking
* InsightFace face recognition
* ArcFace-based face embeddings
* OSNet Person ReID
* Persistent face memory
* Persistent ReID memory
* Canonical identity system
* Person Intelligence
* Clothing segmentation
* MobileNetV3 clothing-color classification
* Custom clothing dataset pipeline
* Person event engine
* Loitering detection
* Snapshot evidence
* Video evidence
* Watchlist
* Risk assessment
* FastAPI dashboard
* SQLite event storage
* ChromaDB memory
* RAG
* Ollama LLM integration
* Natural-language CCTV Assistant
* Intent parsing
* YOLO-World prototype
* YOLO-World object alerts
* Object snapshot evidence
* Object-alert SQLite logging

---

# 🔄 Current Work

### YOLO-World Dashboard Integration

Next immediate task:

```text
YOLO-World Detection
        ↓
Object Alert
        ↓
SQLite
        ↓
VisionGuard Dashboard
```

The goal is to display object alerts directly inside the existing VisionGuard UI.

---

# 🎯 Next Major Build

## Multi-Camera Intelligence

The next major VisionGuard capability is planned to support multiple CCTV cameras.

Example:

```text
Camera 1
Person detected
ID 11
      ↓
Person leaves
      ↓
Camera 2
Same person detected
      ↓
VisionGuard restores the same permanent identity
```

Planned functionality:

* Camera 1 + Camera 2
* Shared identity memory
* Cross-camera person ReID
* Cross-camera face recognition
* Camera transition history
* Unified person timeline

---

# 🧠 Future Multimodal AI

A future upgrade will allow the AI Assistant to understand the actual visual evidence, not only database text.

Planned architecture:

```text
Snapshot / Video
        ↓
Vision-Language Model
        ↓
Visual Understanding
        ↓
RAG
        ↓
AI Incident Explanation
```

Possible future questions:

```text
What happened at 3:16 PM?

What was the person carrying?

Describe the incident.

What happened before the loitering alert?
```

---

# 🚀 Future Technology Roadmap

Planned research and development areas include:

* Multi-camera tracking
* Cross-camera identity
* Vision-language models
* Multimodal RAG
* Automatic incident summaries
* Additional clothing colors
* Lighting augmentation for clothing classification
* Configurable monitored-object settings
* RTSP multi-camera ingestion
* NVIDIA DeepStream
* TensorRT optimization
* GPU-accelerated multi-stream analytics
* Cloud deployment
* Enterprise API
* Role-based user access
* Multi-location CCTV
* Mobile dashboard
* Automated incident reports
* Predictive security analytics

---

# 💼 Potential Applications

VisionGuard AI can be adapted for:

* Retail stores
* Offices
* Warehouses
* Factories
* Hotels
* Residential buildings
* Schools
* Hospitals
* Smart buildings
* Security operations

Different customers could configure VisionGuard for different objects.

Example:

```text
Retail:
shopping bag
package
phone

Warehouse:
helmet
safety vest
forklift

Office:
laptop
backpack
phone
```

---

# 🌟 Project Vision

Traditional CCTV mainly:

```text
Records
```

VisionGuard aims to:

```text
SEE
 ↓
DETECT
 ↓
TRACK
 ↓
RECOGNIZE
 ↓
REMEMBER
 ↓
ANALYZE
 ↓
UNDERSTAND
 ↓
EXPLAIN
```

The long-term objective is to create an intelligent CCTV platform capable of understanding people, objects, events, and activity across multiple cameras while allowing users to investigate surveillance information through conversational AI.

---

# 👩‍💻 Developer

**Urifatul Jannah**

Computer Vision • AI Engineering • Machine Learning • Python Development

📧 **Email**
[jannahurifa04@gmail.com](mailto:jannahurifa04@gmail.com)

💼 **LinkedIn**
https://www.linkedin.com/in/urifatul-jannah-185177415/

🐙 **GitHub**
https://github.com/jannahurifa04

🚀 **VisionGuard AI Repository**
https://github.com/jannahurifa04/VisionGuard-AI

---

# 💼 Open to Opportunities

Open to opportunities in:

* Computer Vision
* AI Engineering
* Machine Learning
* Python Development

Especially interested in:

* Remote opportunities
* Contract projects
* International teams
* AI / Computer Vision collaborations

---

# ⭐ Status

**VisionGuard AI — Active Development**

Current focus:

```text
YOLO-World Dashboard Integration
        ↓
Multi-Camera Intelligence
        ↓
Multimodal Vision AI
        ↓
Production-Scale CCTV Analytics
```
