// LKW-Erkennung & Hofkamera Dashboard Frontend

document.addEventListener("DOMContentLoaded", () => {
  const host = window.location.hostname || "localhost";
  const go2rtcPort = "1984";

  // Elements
  const streamLkw = document.getElementById("stream-lkw");
  const streamKalb = document.getElementById("stream-kalb");
  const fallbackLkw = document.getElementById("fallback-lkw");
  const fallbackKalb = document.getElementById("fallback-kalb");
  const eventsGrid = document.getElementById("events-grid");
  const refreshStatus = document.getElementById("refresh-status");
  const manualRefreshBtn = document.getElementById("manual-refresh-btn");
  const themeToggle = document.getElementById("theme-toggle");
  const systemBadge = document.getElementById("system-badge");

  // Modal Elements
  const modal = document.getElementById("video-modal");
  const modalVideo = document.getElementById("modal-video");
  const modalClose = document.getElementById("modal-close");
  const modalBackdrop = modal.querySelector(".modal-backdrop");
  const modalTitle = document.getElementById("modal-title");
  const modalSubtitle = document.getElementById("modal-subtitle");
  const modalMetaPlate = document.getElementById("modal-meta-plate");
  const modalMetaCompany = document.getElementById("modal-meta-company");
  const modalDownloadBtn = document.getElementById("modal-download-btn");

  let refreshTimer = null;
  let lastRefreshTime = Date.now();

  // 1. Theme Initialisierung (Dark / Light Mode)
  function initTheme() {
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme) {
      document.documentElement.setAttribute("data-theme", savedTheme);
    }
  }

  themeToggle.addEventListener("click", () => {
    const currentTheme = document.documentElement.getAttribute("data-theme");
    const isDark = currentTheme === "dark" || (!currentTheme && window.matchMedia("(prefers-color-scheme: dark)").matches);
    const newTheme = isDark ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", newTheme);
    localStorage.setItem("theme", newTheme);
  });

  // 2. go2rtc Streams & KI-Feed einbinden
  const streamModes = {
    lkw: "mse",
    kalb: "mse",
  };

  const streamKalbAi = document.getElementById("stream-kalb-ai");
  const kalbCowsBadge = document.getElementById("kalb-cows-badge");
  const kalbModelBadge = document.getElementById("kalb-model-badge");

  function updateStreamUrl(name) {
    const iframe = name === "lkw" ? streamLkw : streamKalb;
    const fallback = name === "lkw" ? fallbackLkw : fallbackKalb;
    const mode = streamModes[name] || "mse";

    if (!iframe) return;

    if (name === "kalb" && mode === "ai") {
      // KI-Modus mit Bounding-Boxes
      iframe.style.display = "none";
      iframe.src = "";
      if (streamKalbAi) {
        streamKalbAi.style.display = "block";
        streamKalbAi.src = "/api/cows/live?t=" + Date.now();
      }
      if (fallback) fallback.style.display = "none";
      updateCowStatus();
      return;
    }

    if (name === "kalb") {
      if (streamKalbAi) {
        streamKalbAi.style.display = "none";
        streamKalbAi.src = "";
      }
      iframe.style.display = "block";
    }

    // stream.html mit explizitem Modus aufrufen
    const url = `http://${host}:${go2rtcPort}/stream.html?src=${name}&mode=${mode}`;
    iframe.src = url;

    iframe.onload = () => {
      if (fallback) fallback.style.display = "none";
    };
    // Zusätzlicher Fallback-Timer zur Sicherheit
    setTimeout(() => {
      if (fallback) fallback.style.display = "none";
    }, 1500);
  }

  async function updateCowStatus() {
    try {
      const res = await fetch("/api/cows/status");
      if (!res.ok) return;
      const data = await res.json();
      if (kalbCowsBadge) {
        if (data.cows_detected > 0) {
          kalbCowsBadge.textContent = `🐮 ${data.cows_detected} ${data.cows_detected === 1 ? 'Kuh' : 'Kühe'} erfasst`;
          kalbCowsBadge.style.color = "var(--success, #22c55e)";
          kalbCowsBadge.style.fontWeight = "600";
        } else {
          kalbCowsBadge.textContent = "🐮 Keine Kuh im Fokus";
          kalbCowsBadge.style.color = "var(--text-muted)";
          kalbCowsBadge.style.fontWeight = "normal";
        }
      }
      if (kalbModelBadge) {
        const mName = data.model_path ? data.model_path.split("/").pop().split("\\").pop() : "yolov8n.pt";
        const tag = data.is_custom_model ? "Custom/COLO" : "COCO";
        kalbModelBadge.textContent = `${mName} (${tag})`;
      }
    } catch (e) {
      // Still ignorieren
    }
  }

  function initStreams() {
    updateStreamUrl("lkw");
    updateStreamUrl("kalb");
    updateCowStatus();
    setInterval(updateCowStatus, 5000);

    // Umschalter für Streaming-Modus (MSE / WebRTC / KI)
    document.querySelectorAll(".mode-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const stream = btn.getAttribute("data-stream");
        const mode = btn.getAttribute("data-mode");
        if (stream && mode) {
          streamModes[stream] = mode;
          btn.closest(".stream-modes").querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          updateStreamUrl(stream);
        }
      });
    });
  }

  // 3. Ereignisse laden (/api/events)
  async function loadEvents() {
    try {
      const response = await fetch("/api/events?limit=100");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const events = await response.json();
      renderEvents(events);
      lastRefreshTime = Date.now();
      updateRefreshHint();
    } catch (err) {
      console.warn("Fehler beim Laden der Ereignisse:", err);
      if (eventsGrid.children.length === 0 || eventsGrid.querySelector(".loading-state")) {
        eventsGrid.innerHTML = `
          <div class="empty-state">
            <p>⚠️ Ereignisse konnten nicht geladen werden.</p>
            <button onclick="window.loadEvents()" class="btn btn-secondary" style="margin-top: 1rem;">Erneut versuchen</button>
          </div>
        `;
      }
    }
  }
  window.loadEvents = loadEvents;

  function renderEvents(events) {
    if (!events || events.length === 0) {
      eventsGrid.innerHTML = `
        <div class="empty-state">
          <p>🚚 Noch keine LKW-Ereignisse erfasst.</p>
          <span class="subtitle">Sobald ein LKW die Ladefläche befährt, erscheint er hier automatisch.</span>
        </div>
      `;
      return;
    }

    eventsGrid.innerHTML = "";
    events.forEach((ev) => {
      const card = createEventCard(ev);
      eventsGrid.appendChild(card);
    });
  }

  function createEventCard(ev) {
    const card = document.createElement("div");
    card.className = "event-card";

    // Formatierung Zeit
    const dt = new Date(ev.created_at);
    const dateFormatted = dt.toLocaleDateString("de-DE", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
    });
    const timeFormatted = dt.toLocaleTimeString("de-DE", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

    const durationStr = ev.duration_seconds > 0 ? `${Math.round(ev.duration_seconds)}s` : "";
    const snapshotSrc = ev.snapshot_url || "/static/images/placeholder.jpg";

    card.innerHTML = `
      <div class="thumbnail-wrapper">
        <img src="${snapshotSrc}" alt="LKW Snapshot ${ev.plate || ''}" loading="lazy" onerror="this.src='/static/images/placeholder.jpg'">
        <div class="play-overlay">
          <div class="play-icon">▶</div>
        </div>
        ${durationStr ? `<span class="duration-tag">⏱ ${durationStr}</span>` : ""}
      </div>
      <div class="event-details">
        <div class="event-top-row">
          <span class="plate-badge">${ev.plate || "Unbekannt"}</span>
          <span class="event-time">${timeFormatted} Uhr</span>
        </div>
        ${ev.company ? `<span class="company-badge">${ev.company}</span>` : ""}
        <span class="subtitle">${dateFormatted}</span>
      </div>
    `;

    card.addEventListener("click", () => {
      openModal(ev, dateFormatted, timeFormatted);
    });

    return card;
  }

  // 4. Video Modal Steuerung
  function openModal(ev, dateStr, timeStr) {
    modalTitle.textContent = `LKW-Aufnahme: ${ev.plate || "Unbekannt"}`;
    modalSubtitle.textContent = `${dateStr} um ${timeStr} Uhr (${ev.duration_seconds || 0}s)`;
    modalMetaPlate.textContent = ev.plate || "Unbekannt";

    if (ev.company) {
      modalMetaCompany.textContent = ev.company;
      modalMetaCompany.style.display = "inline-flex";
    } else {
      modalMetaCompany.style.display = "none";
    }

    if (ev.clip_url) {
      modalVideo.src = ev.clip_url;
      modalDownloadBtn.href = ev.clip_url;
      modalDownloadBtn.style.display = "inline-flex";
      modalVideo.play().catch(() => {});
    } else {
      modalVideo.removeAttribute("src");
      modalDownloadBtn.style.display = "none";
    }

    modal.classList.add("active");
    modal.setAttribute("aria-hidden", "false");
  }

  function closeModal() {
    modal.classList.remove("active");
    modal.setAttribute("aria-hidden", "true");
    modalVideo.pause();
    modalVideo.currentTime = 0;
    modalVideo.removeAttribute("src");
  }

  modalClose.addEventListener("click", closeModal);
  modalBackdrop.addEventListener("click", closeModal);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.classList.contains("active")) {
      closeModal();
    }
  });

  // 5. System Status prüfen
  async function checkStatus() {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data = await res.json();
        if (data.event_manager && data.event_manager.active_event) {
          systemBadge.className = "badge badge-success";
          systemBadge.innerHTML = `<span class="indicator-dot" style="background-color: #ef4444;"></span> 🚚 LKW im Bild (${data.event_manager.duration_so_far}s)`;
        } else {
          systemBadge.className = "badge badge-success";
          systemBadge.innerHTML = `<span class="indicator-dot"></span> System aktiv (${data.events_count} LKWs)`;
        }
      }
    } catch (e) {
      systemBadge.className = "badge badge-secondary";
      systemBadge.innerHTML = `<span class="indicator-dot"></span> Offline`;
    }
  }

  function updateRefreshHint() {
    const elapsed = Math.round((Date.now() - lastRefreshTime) / 1000);
    refreshStatus.textContent = `Zuletzt aktualisiert vor ${elapsed}s (Auto-Refresh 15s)`;
  }

  // 6. Polling & Event Listener
  manualRefreshBtn.addEventListener("click", () => {
    loadEvents();
    checkStatus();
  });

  // Alle 15 Sekunden neu laden
  setInterval(() => {
    loadEvents();
    checkStatus();
  }, 15000);

  // Sekundenzähler für Text
  setInterval(updateRefreshHint, 1000);

  // Start
  initTheme();
  initStreams();
  loadEvents();
  checkStatus();
});
