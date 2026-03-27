const storageKeys = {
  progress: "pure-python-preview-progress",
  notes: "pure-python-preview-notes",
  review: "pure-python-preview-review",
};

const state = {
  chapters: [],
  filtered: [],
  currentSlug: null,
  currentDetail: null,
  currentFilePath: null,
  requestToken: 0,
  isLoading: false,
  progress: JSON.parse(localStorage.getItem(storageKeys.progress) || "{}"),
  notes: JSON.parse(localStorage.getItem(storageKeys.notes) || "{}"),
  review: JSON.parse(localStorage.getItem(storageKeys.review) || "{}"),
};

const chapterList = document.getElementById("chapterList");
const searchInput = document.getElementById("searchInput");
const progressLabel = document.getElementById("progressLabel");
const progressFill = document.getElementById("progressFill");
const heroTitle = document.getElementById("heroTitle");
const heroChips = document.getElementById("heroChips");
const lessonTitle = document.getElementById("lessonTitle");
const chapterSummary = document.getElementById("chapterSummary");
const goalList = document.getElementById("goalList");
const assignmentList = document.getElementById("assignmentList");
const lessonContent = document.getElementById("lessonContent");
const resourceLinks = document.getElementById("resourceLinks");
const notesArea = document.getElementById("notesArea");
const noteStatus = document.getElementById("noteStatus");
const fileTabs = document.getElementById("fileTabs");
const fileMeta = document.getElementById("fileMeta");
const codeViewer = document.getElementById("codeViewer");
const markComplete = document.getElementById("markComplete");
const markReview = document.getElementById("markReview");
const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebarToggle");


async function typesetMath() {
  if (!window.MathJax || typeof window.MathJax.typesetPromise !== "function") {
    return;
  }

  if (typeof window.MathJax.typesetClear === "function") {
    window.MathJax.typesetClear([lessonContent]);
  }

  await window.MathJax.typesetPromise([lessonContent]);
}


async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}


function chapterLabel(chapter) {
  return chapter.number === null ? chapter.folder : `Ch. ${chapter.number}`;
}


function findChapter(slug) {
  return state.chapters.find((chapter) => chapter.slug === slug) || null;
}


function setActionState(enabled) {
  markComplete.disabled = !enabled;
  markReview.disabled = !enabled;
}


function saveState(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}


function updateProgress() {
  const completeCount = state.chapters.filter((chapter) => state.progress[chapter.slug]).length;
  const total = state.chapters.length;
  progressLabel.textContent = `${completeCount} / ${total}`;
  progressFill.style.width = total ? `${(completeCount / total) * 100}%` : "0%";
}


function renderList(container, items, fallback) {
  container.innerHTML = "";
  if (!items || !items.length) {
    const li = document.createElement("li");
    li.textContent = fallback;
    container.appendChild(li);
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    container.appendChild(li);
  });
}


function renderHeroChips(detail) {
  heroChips.innerHTML = "";

  const chips = [
    detail.part,
    detail.topic,
    detail.lesson_source_kind === "generated"
      ? "Curriculum-generated"
      : detail.lesson_source_kind === "markdown"
        ? "Markdown lesson"
        : detail.lesson_source_kind === "notebook"
          ? "Notebook lesson"
          : null,
  ].filter(Boolean);

  if (!chips.length) {
    const chip = document.createElement("span");
    chip.className = "hero-chip";
    chip.textContent = "Local curriculum";
    heroChips.appendChild(chip);
    return;
  }

  chips.forEach((value) => {
    const chip = document.createElement("span");
    chip.className = "hero-chip";
    chip.textContent = value;
    heroChips.appendChild(chip);
  });
}


function renderSidebar() {
  chapterList.innerHTML = "";

  state.filtered.forEach((chapter) => {
    const button = document.createElement("button");
    button.className = "chapter-button";
    if (chapter.slug === state.currentSlug) {
      button.classList.add("active");
    }
    if (state.review[chapter.slug]) {
      button.classList.add("review");
    }

    const meta = document.createElement("div");
    meta.className = "chapter-meta";
    meta.innerHTML = `<span>${chapterLabel(chapter)}</span><span>${state.progress[chapter.slug] ? "Done" : "Open"}</span>`;

    const title = document.createElement("span");
    title.className = "chapter-title";
    title.textContent = chapter.title;

    const summary = document.createElement("span");
    summary.className = "chapter-summary";
    summary.textContent = chapter.summary || "Lesson preview available.";

    button.append(meta, title, summary);
    button.addEventListener("click", () => {
      selectChapter(chapter.slug);
      if (window.innerWidth <= 1120) {
        sidebar.classList.remove("open");
      }
    });
    chapterList.appendChild(button);
  });
}


function filterChapters() {
  const query = searchInput.value.trim().toLowerCase();
  if (!query) {
    state.filtered = [...state.chapters];
  } else {
    state.filtered = state.chapters.filter((chapter) => {
      const haystack = [
        chapter.title,
        chapter.summary,
        chapter.folder,
        chapter.part,
        chapter.topic,
        ...(chapter.key_concepts || []),
        ...(chapter.goals || []),
        ...(chapter.assignments || []),
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(query);
    });
  }
  renderSidebar();
}


async function loadFile(path) {
  const requestedPath = path;
  state.currentFilePath = path;
  fileMeta.textContent = `Loading ${path}...`;

  try {
    const response = await fetch(`/api/file?path=${encodeURIComponent(path)}`);
    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }
    const text = await response.text();
    if (state.currentFilePath !== requestedPath) {
      return;
    }
    codeViewer.textContent = text;
    fileMeta.textContent = path;
    [...fileTabs.querySelectorAll(".file-tab")].forEach((button) => {
      button.classList.toggle("active", button.dataset.path === path);
    });
  } catch (error) {
    if (state.currentFilePath !== requestedPath) {
      return;
    }
    codeViewer.textContent = error.message;
    fileMeta.textContent = "File failed to load";
  }
}


function renderResources(detail) {
  resourceLinks.innerHTML = "";
  const seen = new Set();
  const resources = [];

  function pushResource(label, path) {
    if (!path || seen.has(path)) {
      return;
    }
    seen.add(path);
    resources.push({ label, path });
  }

  if (detail.lesson_source_kind === "generated") {
    pushResource("Curriculum outline", detail.curriculum_path);
  } else {
    pushResource(
      detail.lesson_source_kind === "markdown" ? "Primary markdown" : "Primary notebook",
      detail.lesson_source_path
    );
  }

  (detail.markdown_paths || []).forEach((path, index) => {
    pushResource(index === 0 ? "Lesson markdown" : `Markdown ${index + 1}`, path);
  });
  (detail.notebook_paths || []).forEach((path, index) => {
    pushResource(index === 0 ? "Lesson notebook" : `Notebook ${index + 1}`, path);
  });
  (detail.script_paths || []).forEach((path, index) => {
    pushResource(index === 0 ? "Primary script" : `Script ${index + 1}`, path);
  });
  (detail.data_paths || []).forEach((path, index) => {
    pushResource(index === 0 ? "Dataset" : `Dataset ${index + 1}`, path);
  });
  pushResource("Curriculum outline", detail.curriculum_path);

  if (!resources.length) {
    const empty = document.createElement("span");
    empty.className = "resource-empty";
    empty.textContent = "No linked files for this lesson.";
    resourceLinks.appendChild(empty);
    return;
  }

  resources.forEach((resource) => {
    const link = document.createElement("a");
    link.className = "resource-link";
    link.href = `/api/file?path=${encodeURIComponent(resource.path)}`;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = resource.label;
    resourceLinks.appendChild(link);
  });
}


function renderFileTabs(detail) {
  fileTabs.innerHTML = "";
  const paths = detail.script_paths || [];
  if (!paths.length) {
    state.currentFilePath = null;
    fileMeta.textContent = "No script file for this lesson";
    codeViewer.textContent = "This lesson does not include a Python implementation file.";
    return;
  }

  paths.forEach((path) => {
    const button = document.createElement("button");
    button.className = "file-tab";
    button.textContent = path.split("/").at(-1);
    button.dataset.path = path;
    button.addEventListener("click", () => loadFile(path));
    fileTabs.appendChild(button);
  });

  loadFile(paths[0]);
}


function syncNotes() {
  notesArea.value = state.notes[state.currentSlug] || "";
  noteStatus.textContent = "Saved locally";
}


function setLoadingState(slug) {
  const chapter = findChapter(slug) || {};
  state.currentFilePath = null;
  heroTitle.textContent = chapter.title || "Loading chapter";
  lessonTitle.textContent = chapter.title || "Loading chapter";
  chapterSummary.textContent = chapter.summary || "Loading local chapter content...";
  lessonContent.innerHTML = "<p>Loading local chapter content...</p>";
  renderHeroChips(chapter);
  renderList(goalList, chapter.goals || chapter.key_concepts, "Loading chapter goals...");
  renderList(assignmentList, chapter.assignments, "Loading implementation notes...");
  resourceLinks.innerHTML = "";
  fileTabs.innerHTML = "";
  fileMeta.textContent = "Loading files...";
  codeViewer.textContent = "Loading local implementation files...";
}


function showChapterError(slug, error) {
  const chapter = findChapter(slug) || {};
  state.currentFilePath = null;
  heroTitle.textContent = chapter.title || "Chapter failed to load";
  lessonTitle.textContent = chapter.title || "Chapter failed to load";
  chapterSummary.textContent = "The local preview hit an error while loading this chapter.";
  lessonContent.innerHTML = `<p>${error.message}</p>`;
  renderHeroChips(chapter);
  renderList(goalList, chapter.goals || chapter.key_concepts, "No learning goals extracted yet.");
  renderList(assignmentList, chapter.assignments, "No assignments extracted yet.");
  resourceLinks.innerHTML = "";
  fileTabs.innerHTML = "";
  fileMeta.textContent = "No file selected";
  codeViewer.textContent = "The selected chapter failed to load.";
  markComplete.textContent = state.progress[slug] ? "Completed" : "Mark complete";
  markReview.textContent = state.review[slug] ? "Marked for review" : "Review later";
}


async function selectChapter(slug) {
  if (!slug) {
    return;
  }
  const requestToken = ++state.requestToken;
  state.currentSlug = slug;
  state.currentDetail = null;
  state.isLoading = true;
  window.location.hash = slug;
  renderSidebar();
  setActionState(false);
  setLoadingState(slug);

  try {
    const detail = await fetchJson(`/api/chapters/${encodeURIComponent(slug)}`);
    if (requestToken !== state.requestToken) {
      return;
    }
    state.currentDetail = detail;

    heroTitle.textContent = detail.title;
    lessonTitle.textContent = detail.title;
    chapterSummary.textContent = detail.summary || "Lesson preview loaded.";
    lessonContent.innerHTML = detail.html;
    renderHeroChips(detail);
    await typesetMath();

    renderList(goalList, detail.goals || detail.key_concepts, "No learning goals extracted yet.");
    renderList(assignmentList, detail.assignments, "No assignments extracted yet.");
    renderResources(detail);
    renderFileTabs(detail);
    syncNotes();

    markComplete.textContent = state.progress[slug] ? "Completed" : "Mark complete";
    markReview.textContent = state.review[slug] ? "Marked for review" : "Review later";
  } catch (error) {
    if (requestToken !== state.requestToken) {
      return;
    }
    showChapterError(slug, error);
  } finally {
    if (requestToken === state.requestToken) {
      state.isLoading = false;
      setActionState(Boolean(state.currentSlug));
    }
  }
}


function toggleComplete() {
  if (!state.currentSlug) {
    return;
  }
  state.progress[state.currentSlug] = !state.progress[state.currentSlug];
  saveState(storageKeys.progress, state.progress);
  updateProgress();
  renderSidebar();
  markComplete.textContent = state.progress[state.currentSlug] ? "Completed" : "Mark complete";
}


function toggleReview() {
  if (!state.currentSlug) {
    return;
  }
  state.review[state.currentSlug] = !state.review[state.currentSlug];
  saveState(storageKeys.review, state.review);
  renderSidebar();
  markReview.textContent = state.review[state.currentSlug] ? "Marked for review" : "Review later";
}


function initNotes() {
  let noteTimer = null;
  notesArea.addEventListener("input", () => {
    if (!state.currentSlug) {
      return;
    }
    state.notes[state.currentSlug] = notesArea.value;
    noteStatus.textContent = "Saving...";
    window.clearTimeout(noteTimer);
    noteTimer = window.setTimeout(() => {
      saveState(storageKeys.notes, state.notes);
      noteStatus.textContent = "Saved locally";
    }, 150);
  });
}


async function boot() {
  const payload = await fetchJson("/api/chapters");
  state.chapters = payload.chapters || [];
  state.filtered = [...state.chapters];
  updateProgress();
  renderSidebar();
  renderHeroChips({});
  setActionState(false);

  const initialSlug = window.location.hash.replace("#", "") || state.chapters[0]?.slug;
  if (initialSlug) {
    await selectChapter(initialSlug);
  }
}


searchInput.addEventListener("input", filterChapters);
markComplete.addEventListener("click", toggleComplete);
markReview.addEventListener("click", toggleReview);
sidebarToggle.addEventListener("click", () => sidebar.classList.toggle("open"));
window.addEventListener("mathjax-ready", () => {
  typesetMath().catch(() => {});
});
window.addEventListener("hashchange", () => {
  const slug = window.location.hash.replace("#", "");
  if (slug && slug !== state.currentSlug) {
    selectChapter(slug);
  }
});

initNotes();
boot().catch((error) => {
  heroTitle.textContent = "Preview failed to load";
  lessonTitle.textContent = "Preview failed to load";
  chapterSummary.textContent = "The local preview app hit an error while loading chapters.";
  lessonContent.innerHTML = `<p>${error.message}</p>`;
  renderHeroChips({});
  setActionState(false);
});
