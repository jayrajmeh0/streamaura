# 🧵 Lightweight OTT Streaming Platform

## 📖 Project Overview

This project is a lightweight OTT (Over-the-Top) streaming platform that enables individual users to:

- 🔐 **Sign up and create a personal streaming channel**
- 📡 **Go live directly from their browser** using WebRTC (via [MediaMTX](https://github.com/bluenviron/mediamtx))
- 👀 **Allow viewers to watch the stream in real-time** through a browser after submitting their:
  - First Name
  - Last Name
  - Email
- 📈 **Access viewer statistics** (available to the channel owner)
- 📝 **See a real-time transcript** of the ongoing stream

## 🛠️ Tech Stack

- **Backend:** Node.js
- **Frontend:** React
- **Database:** PostgreSQL
- **Media Server:** MediaMTX (formerly rtsp-simple-server)

---

## 🚀 Features

- Account creation and login for streamers
- Real-time browser-based streaming via WebRTC
- Viewer access via form-based submission
- Viewer analytics and session logs
- Live transcription of ongoing streams

---

## ▶️ Getting Started


#### 📂 Folder Structure

```text
root/
├── backend/
│   └── src/
├── frontend/
│   └── src/
├── docker-compose.yml
├── mediaMTX/
│   └── mediamtx.exe
└── README.md
```

---

### 1. Install MediaMTX

install [MediaMTX](https://github.com/bluenviron/mediamtx/releases) for your OS.

Run `mediaMTX -> mediamtx.exe` to start the media server.

---

### 2. Start Backend & Database (via Docker Compose)

From the root folder:

```bash
docker-compose up -d
```
This will start:

* Node.js backend server
* PostgreSQL database

---

### 3. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Open your browser and visit: [http://localhost:5173](http://localhost:5173)

---

### 4. Streaming Transcript

The platform uses integrated transcription services to capture and display real-time subtitles during the live stream, accessible to the streamer within their dashboard.

---