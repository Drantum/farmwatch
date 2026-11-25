// FarmWatch – Modern SPA Dashboard

document.addEventListener("DOMContentLoaded", () => {
  const host = window.location.hostname || "localhost";
  const go2rtcPort = "1984";

  // --- DOM refs ---
  const sidebar = document.getElementById("sidebar");
  const hamburger = document.getElementById("hamburger");
  const pages = document.querySelectorAll(".page");
  const navItems = document.querySelectorAll(".nav-item");

  const streamLkw = document.getElementById("stream-lkw");
  const streamKalb = document.getElementById("stream-kalb");
  const streamKalbAi = document.getElementById("stream-kalb-ai");
  const fallbackLkw = document.getElementById("fallback-lkw");
  const fallbackKalb = document.getElementById("fallback-kalb");
  const cowCountBadge = document.getElementById("cow-count-badge");
  const cowModelBadge = document.getElementById("cow-model-badge");
  const cowStatusBar = document.getElementById("cow-status-bar");

  const eventsContainer = document.getElementById("events-container");
  const searchInput = document.getElementById("search-input");
  const refreshEventsBtn = document.getElementById("refresh-events-btn");
  const systemBadge = document.getElementById("system-badge");
  const systemBadgeText = document.getElementById("system-badge-text");

  const modal = document.getElementById("video-modal");
  const modalVideo = document.getElementById("modal-video");
  const modalClose = document.getElementById("modal-close");
  const modalBackdrop = modal.querySelector(".modal-backdrop");
  const modalTitle = document.getElementById("modal-title");
  const modalSubtitle = document.getElementById("modal-subtitle");
  const modalMetaPlate = document.getElementById("modal-meta-plate");
  const modalMetaCompany = document.getElementById("modal-meta-company");
  const modalDownloadBtn = document.getElementById("modal-download-btn");
  const modalDeleteBtn = document.getElementById("modal-delete-btn");

  let allEvents = [];
  let currentModalEventId = null;

  // =============================
  // 1. Theme
  // =============================
  function initTheme() {
    const saved = localStorage.getItem("theme");
    if (saved) document.documentElement.setAttribute("data-theme", saved);
  }

  function toggleTheme() {
    const cur = document.documentElement.getAttribute("data-theme");
    const isDark = cur === "dark" || (!cur && window.matchMedia("(prefers-color-scheme: dark)").matches);
    const next = isDark ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
  }

  document.getElementById("theme-toggle").addEventListener("click", toggleTheme);
  const tmMobile = document.getElementById("theme-toggle-mobile");
  if (tmMobile) tmMobile.addEventListener("click", toggleTheme);

  // =============================
  // 2. Navigation / Routing
  // =============================
  function navigateTo(page) {
    pages.forEach(p => p.classList.add("hidden"));
    navItems.forEach(n => n.classList.remove("active"));

    const target = document.getElementById(`page-${page}`);
    if (target) {
      target.classList.remove("hidden");
    } else {
      document.getElementById("page-live").classList.remove("hidden");
      page = "live";
    }

    navItems.forEach(n => {
      if (n.dataset.page === page) n.classList.add("active");
    });

    if (page === "aufnahmen") loadEvents();
    if (page === "live") initStreams();

    // Mobile: close sidebar
    sidebar.classList.remove("open");
  }

  navItems.forEach(n => {
    n.addEventListener("click", (e) => {
      e.preventDefault();
      const page = n.dataset.page;
      window.location.hash = page;
    });
  });

  window.addEventListener("hashchange", () => {
    const page = window.location.hash.replace("#", "") || "live";
    navigateTo(page);
  });

  if (hamburger) {
    hamburger.addEventListener("click", () => sidebar.classList.toggle("open"));
  }

  // =============================
  // 3. Streams
  // =============================
  const streamModes = { lkw: "mse", kalb: "mse" };

  function updateStream(name) {
    const iframe = name === "lkw" ? streamLkw : streamKalb;
    const fallback = name === "lkw" ? fallbackLkw : fallbackKalb;
    const mode = streamModes[name] || "mse";

    if (!iframe) return;

    // KI-Modus für Abkalbebox
    if (name === "kalb" && mode === "ai") {
      iframe.style.display = "none";
      iframe.src = "";
      if (streamKalbAi) {
        streamKalbAi.style.display = "block";
        streamKalbAi.src = "/api/cows/live?t=" + Date.now();
        streamKalbAi.onerror = () => {
          // Fallback: snapshot polling
          pollCowSnapshot();
        };
      }
      if (fallback) fallback.style.display = "none";
      if (cowStatusBar) cowStatusBar.style.display = "flex";
      updateCowStatus();
      return;
    }

    // Normal MSE / WebRTC
    if (name === "kalb") {
      if (streamKalbAi) { streamKalbAi.style.display = "none"; streamKalbAi.src = ""; }
      iframe.style.display = "block";
    }

    const url = `http://${host}:${go2rtcPort}/stream.html?src=${name}&mode=${mode}`;
    iframe.src = url;

    iframe.onload = () => { if (fallback) fallback.style.display = "none"; };
    setTimeout(() => { if (fallback) fallback.style.display = "none"; }, 2000);
  }

  let cowSnapshotTimer = null;
  function pollCowSnapshot() {
    if (cowSnapshotTimer) clearInterval(cowSnapshotTimer);
    cowSnapshotTimer = setInterval(() => {
      if (streamModes.kalb !== "ai" || !streamKalbAi) {
        clearInterval(cowSnapshotTimer);
        return;
      }
      streamKalbAi.src = "/api/cows/snapshot?t=" + Date.now();
    }, 250);
  }

  async function updateCowStatus() {
    try {
      const res = await fetch("/api/cows/status");
      if (!res.ok) return;
      const data = await res.json();

      if (cowCountBadge) {
        const n = data.cows_detected || 0;
        if (n > 0) {
          cowCountBadge.textContent = `🐮 ${n} ${n === 1 ? "Kuh" : "Kühe"} erfasst`;
          cowCountBadge.style.color = "var(--success)";
          cowCountBadge.style.fontWeight = "600";
        } else {
          cowCountBadge.textContent = "🐮 Keine Kuh erkannt";
          cowCountBadge.style.color = "";
          cowCountBadge.style.fontWeight = "";
        }
      }
      if (cowModelBadge) {
        const mName = data.model_path ? data.model_path.split("/").pop().split("\\").pop() : "yolov8n.pt";
        const tag = data.is_custom_model ? "Custom" : "COCO";
        cowModelBadge.textContent = `${mName} (${tag})`;
      }
    } catch (_) {}
  }

  function initStreams() {
    updateStream("lkw");
    updateStream("kalb");
    updateCowStatus();
  }

  // Stream mode buttons
  document.querySelectorAll(".mode-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const stream = btn.dataset.stream;
      const mode = btn.dataset.mode;
      if (!stream || !mode) return;

      streamModes[stream] = mode;
      btn.closest(".stream-modes").querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      updateStream(stream);
    });
  });

  // Cow status polling
  setInterval(() => {
    if (streamModes.kalb === "ai") updateCowStatus();
  }, 5000);

  // =============================
  // 4. Events / Aufnahmen
  // =============================
  async function loadEvents() {
    try {
      eventsContainer.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Lade Aufnahmen…</p></div>';
      const res = await fetch("/api/events?limit=500");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      allEvents = await res.json();
      renderEvents(allEvents);
    } catch (err) {
      eventsContainer.innerHTML = '<div class="empty-state"><p>⚠️ Aufnahmen konnten nicht geladen werden.</p></div>';
    }
  }

  function renderEvents(events) {
    const query = (searchInput ? searchInput.value : "").trim().toLowerCase();

    // Filter
    let filtered = events;
    if (query) {
      filtered = events.filter(ev => {
        const plate = (ev.plate || "").toLowerCase();
        const company = (ev.company || "").toLowerCase();
        const date = (ev.created_at || "").toLowerCase();
        return plate.includes(query) || company.includes(query) || date.includes(query);
      });
    }

    if (filtered.length === 0) {
      eventsContainer.innerHTML = `
        <div class="empty-state">
          <p>${query ? "🔍 Keine Treffer für „" + escHtml(query) + """ : "📭 Noch keine Aufnahmen vorhanden"}</p>
        </div>`;
      return;
    }

    // Group by date
    const groups = {};
    for (const ev of filtered) {
      const dateStr = formatDateGroup(ev.created_at);
      if (!groups[dateStr]) groups[dateStr] = [];
      groups[dateStr].push(ev);
    }

    let html = "";
    for (const [dateStr, evList] of Object.entries(groups)) {
      html += `
        <div class="date-section">
          <div class="date-header">
            <h3>📅 ${escHtml(dateStr)}</h3>
            <span class="date-count">${evList.length} Aufnahme${evList.length !== 1 ? "n" : ""}</span>
          </div>
          <div class="events-grid">`;

      for (const ev of evList) {
        const timeStr = formatTime(ev.created_at);
        const dur = formatDuration(ev.duration_seconds);
        const plateHtml = ev.plate ? `<span class="plate-badge">${escHtml(ev.plate)}</span>` : '<span style="color:var(--text-muted); font-size:0.82rem;">Kein Kennzeichen</span>';
        const companyHtml = ev.company ? `<span class="company-badge">${escHtml(ev.company)}</span>` : "";
        const snapUrl = ev.snapshot_url || "";
        const clipUrl = ev.clip_url || "";

        html += `
          <div class="event-card" data-event-id="${ev.id}" data-clip-url="${escAttr(clipUrl)}" data-snap-url="${escAttr(snapUrl)}">
            <div class="thumb-wrap">
              ${snapUrl
                ? `<img src="${escAttr(snapUrl)}" alt="Snapshot" loading="lazy">`
                : `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-muted);">🚛</div>`}
              <div class="play-overlay"><div class="play-circle">▶</div></div>
              <span class="duration-pill">${dur}</span>
            </div>
            <div class="event-info">
              <div class="event-top">
                <span class="event-time">🕐 ${escHtml(timeStr)}</span>
                <div class="event-actions">
                  ${clipUrl ? `<a href="${escAttr(clipUrl)}" download class="event-action-btn" title="Herunterladen" onclick="event.stopPropagation()">⬇</a>` : ""}
                  <button class="event-action-btn delete" data-delete-id="${ev.id}" title="Löschen">🗑</button>
                </div>
              </div>
              <div>${plateHtml}</div>
              ${companyHtml ? `<div>${companyHtml}</div>` : ""}
            </div>
          </div>`;
      }

      html += "</div></div>";
    }

    eventsContainer.innerHTML = html;

    // Attach click handlers
    eventsContainer.querySelectorAll(".event-card").forEach(card => {
      card.addEventListener("click", () => openModal(card));
    });

    eventsContainer.querySelectorAll(".event-action-btn.delete").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const id = parseInt(btn.dataset.deleteId);
        if (id) deleteEvent(id);
      });
    });
  }

  if (searchInput) {
    let debounceTimer;
    searchInput.addEventListener("input", () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => renderEvents(allEvents), 200);
    });
  }

  if (refreshEventsBtn) {
    refreshEventsBtn.addEventListener("click", loadEvents);
  }

  // =============================
  // 5. Modal
  // =============================
  function openModal(card) {
    const clipUrl = card.dataset.clipUrl;
    if (!clipUrl) return;

    const eventId = parseInt(card.dataset.eventId);
    currentModalEventId = eventId;

    const ev = allEvents.find(e => e.id === eventId);

    modalVideo.querySelector("source").src = clipUrl;
    modalVideo.load();

    if (ev) {
      modalTitle.textContent = ev.plate ? `LKW ${ev.plate}` : "LKW-Aufnahme";
      modalSubtitle.textContent = formatDateTime(ev.created_at);
      modalMetaPlate.textContent = ev.plate || "";
      modalMetaPlate.style.display = ev.plate ? "" : "none";
      modalMetaCompany.textContent = ev.company || "";
      modalMetaCompany.style.display = ev.company ? "" : "none";
    }

    modalDownloadBtn.href = clipUrl;
    modal.classList.add("active");
    document.body.style.overflow = "hidden";
  }

  function closeModal() {
    modal.classList.remove("active");
    modalVideo.pause();
    modalVideo.querySelector("source").src = "";
    document.body.style.overflow = "";
    currentModalEventId = null;
  }

  modalClose.addEventListener("click", closeModal);
  modalBackdrop.addEventListener("click", closeModal);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.classList.contains("active")) closeModal();
  });

  if (modalDeleteBtn) {
    modalDeleteBtn.addEventListener("click", async () => {
      if (currentModalEventId) {
        await deleteEvent(currentModalEventId);
        closeModal();
      }
    });
  }

  // =============================
  // 6. Delete Events
  // =============================
  async function deleteEvent(eventId) {
    if (!confirm("Diese Aufnahme wirklich löschen?")) return;
    try {
      const res = await fetch(`/api/events/${eventId}`, { method: "DELETE" });
      if (res.ok) {
        allEvents = allEvents.filter(e => e.id !== eventId);
        renderEvents(allEvents);
      } else {
        alert("Fehler beim Löschen.");
      }
    } catch (err) {
      alert("Netzwerkfehler beim Löschen.");
    }
  }

  // =============================
  // 7. System Status
  // =============================
  async function checkSystemStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) throw new Error();
      const data = await res.json();
      if (systemBadgeText) {
        systemBadgeText.textContent = data.status === "online" ? "Online" : "Offline";
      }
      const dot = document.querySelector("#system-status .status-dot");
      if (dot) {
        dot.classList.toggle("online", data.status === "online");
      }
    } catch (_) {
      if (systemBadgeText) systemBadgeText.textContent = "Offline";
    }
  }

  // =============================
  // 8. Helpers
  // =============================
  function formatDateGroup(isoStr) {
    if (!isoStr) return "Unbekannt";
    const d = new Date(isoStr);
    if (isNaN(d)) return "Unbekannt";
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    if (d.toDateString() === today.toDateString()) return "Heute";
    if (d.toDateString() === yesterday.toDateString()) return "Gestern";

    return d.toLocaleDateString("de-DE", { weekday: "long", day: "numeric", month: "long", year: "numeric" });
  }

  function formatTime(isoStr) {
    if (!isoStr) return "--:--";
    const d = new Date(isoStr);
    if (isNaN(d)) return "--:--";
    return d.toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function formatDateTime(isoStr) {
    if (!isoStr) return "";
    const d = new Date(isoStr);
    if (isNaN(d)) return isoStr;
    return d.toLocaleString("de-DE", { day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" });
  }

  function formatDuration(sec) {
    if (!sec || sec <= 0) return "0s";
    const m = Math.floor(sec / 60);
    const s = Math.round(sec % 60);
    return m > 0 ? `${m}:${String(s).padStart(2, "0")}` : `${s}s`;
  }

  function escHtml(str) {
    const div = document.createElement("div");
    div.textContent = str || "";
    return div.innerHTML;
  }

  function escAttr(str) {
    return (str || "").replace(/"/g, "&quot;").replace(/'/g, "&#39;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // =============================
  // Init
  // =============================
  initTheme();

  const startPage = window.location.hash.replace("#", "") || "live";
  navigateTo(startPage);

  checkSystemStatus();
  setInterval(checkSystemStatus, 30000);
});
