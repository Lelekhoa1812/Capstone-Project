const expandedItems = JSON.parse(localStorage.getItem("expandedItems") || "{}");
const renamedLabels = JSON.parse(localStorage.getItem("renamedLabels") || "{}"); // Allow card to change their name (original identified by ts)
let previousKeys = [];
let previousEvents = {}; // Track event status to avoid redundant updates

// ─────────────────────────────────────────
// Refresh event per interval
// ─────────────────────────────────────────
async function fetchEvents() {
    const res = await fetch('/events');
    const data = await res.json();
    renderEvents(data);
}

// ─────────────────────────────────────────
// Update or Create new card
// ─────────────────────────────────────────
function renderEvents(events) {
    const container = document.getElementById('log-container');
    const currentKeys = Object.keys(events).sort();
    const newlyAdded = currentKeys.find(k => !previousKeys.includes(k));
    previousKeys = currentKeys;

    currentKeys.forEach(key => {
        const event = events[key];
        const existing = document.getElementById(`card-${key}`);
        const prevStatus = previousEvents[key]?.status;

        if (!existing) {
            const card = createCard(key, event);
            container.appendChild(card);
            if (key === newlyAdded && event.status === 'done') {
                setTimeout(() => card.scrollIntoView({ behavior: 'smooth', block: 'center' }), 300);
            }
        } else if (event.status !== prevStatus) {
            updateCard(key, event); // Only update if status changed
        }

        previousEvents[key] = { status: event.status }; // Cache latest status
    });
}

// ─────────────────────────────────────────
// Create new card on unmatched key 
// ─────────────────────────────────────────
function createCard(key, event) {
    const readable = renamedLabels[key] || formatTimestamp(key);
    const safeKey = key.replace(/[:.]/g, "-");
    const card = document.createElement('div');
    card.id = `card-${key}`;
    card.className = 'card';

    const removeBtn = document.createElement('button');
    removeBtn.className = 'btn-remove';
    removeBtn.textContent = 'X';
    removeBtn.onclick = () => removeItem(key);

    const tsDiv = document.createElement('div');
    tsDiv.className = 'timestamp';
    tsDiv.innerHTML = `<span class="label-text">${readable}</span>`;

    const editIcon = document.createElement('img');
    editIcon.src = '/static/edit.png';
    editIcon.className = 'icon-edit';
    editIcon.onclick = () => toggleEditMode(tsDiv, key);
    tsDiv.appendChild(editIcon);


    const statusDiv = document.createElement('div');
    statusDiv.className = 'status';

    const actionDiv = document.createElement('div');
    actionDiv.className = 'actions';

    card.appendChild(removeBtn);
    card.appendChild(tsDiv);
    card.appendChild(statusDiv);
    card.appendChild(actionDiv);

    updateCardContent(card, key, event);
    return card;
}

// ─────────────────────────────────────────
// Validate existing card
// ─────────────────────────────────────────
function updateCard(key, event) {
    const card = document.getElementById(`card-${key}`);
    if (card) {
        updateCardContent(card, key, event);
    }
}

// ─────────────────────────────────────────
// Update existing card content
// ─────────────────────────────────────────
function updateCardContent(card, key, event) {
    const statusDiv = card.querySelector('.status');
    const actionDiv = card.querySelector('.actions');
    const safeKey = key.replace(/[:.]/g, "-");

    actionDiv.innerHTML = '';
    if (event.status === 'started') {
        statusDiv.textContent = "Received signal. Data logging started.";
        card.style.backgroundColor = '#780606';
    } else if (event.status === 'processed') {
        statusDiv.textContent = "Data logging finished. Start cleaning process.";
        card.style.backgroundColor = '#2e6930';
    } else if (event.status === 'done') {
        statusDiv.textContent = "Cleaned data saved. Insights is ready.";
        card.style.backgroundColor = '#8a00c2';

        const expandBtn = document.createElement('button');
        expandBtn.className = 'btn-expand';
        expandBtn.textContent = expandedItems[key] ? 'Collapse' : 'Expand';
        expandBtn.onclick = () => toggleExpand(key, expandBtn);

        const expandDiv = document.createElement('div');
        expandDiv.id = `expand-${key}`;
        expandDiv.className = 'expanded-content';
        if (expandedItems[key]) expandDiv.classList.add('show');

        expandDiv.innerHTML = `
            <img src="/plots/heatmap_${safeKey}.png" width="100%">
            <img src="/plots/trend_${safeKey}.png" width="100%">
        `;

        actionDiv.appendChild(expandBtn);
        actionDiv.appendChild(expandDiv);
    }
}

// ─────────────────────────────────────────
// Toggle card expansion
// ─────────────────────────────────────────
function toggleExpand(key, btn) {
    const el = document.getElementById(`expand-${key}`);
    const showing = el.classList.contains('show');
    if (showing) {
        el.classList.remove('show');
        expandedItems[key] = false;
        btn.textContent = 'Expand';
    } else {
        el.classList.add('show');
        expandedItems[key] = true;
        btn.textContent = 'Collapse';
    }
    localStorage.setItem("expandedItems", JSON.stringify(expandedItems));
}

// ─────────────────────────────────────────
// Toggle card edit-view mode
// ─────────────────────────────────────────
function toggleEditMode(container, key) {
    const icon = container.querySelector('.icon-edit');
    if (!container.classList.contains('editing')) {
        const span = container.querySelector('.label-text');
        if (!span) return;
        const input = document.createElement('input');
        input.type = 'text';
        input.value = span.textContent;
        input.className = 'label-input';
        span.replaceWith(input);
        icon.src = '/static/check.png';
        container.classList.add('editing');
    } else {
        const input = container.querySelector('.label-input');
        if (!input) return;
        const newLabel = input.value.trim() || formatTimestamp(key);
        renamedLabels[key] = newLabel;
        localStorage.setItem("renamedLabels", JSON.stringify(renamedLabels));
        const newSpan = document.createElement('span');
        newSpan.className = 'label-text';
        newSpan.textContent = newLabel;
        input.replaceWith(newSpan);
        icon.src = '/static/edit.png';
        container.classList.remove('editing');
    }
}

// ─────────────────────────────────────────
// Remove a card item
// ─────────────────────────────────────────
function removeItem(key) {
    const card = document.getElementById(`card-${key}`);
    if (card) card.remove();
    delete expandedItems[key];
    delete previousEvents[key];
    localStorage.setItem("expandedItems", JSON.stringify(expandedItems));
    fetch(`/events/remove/${key}`, { method: 'DELETE' });
}

// ─────────────────────────────────────────
// Format timestamp as hh:mm dd/mm/yyyy
// ─────────────────────────────────────────
function formatTimestamp(norm_ts) {
    try {
        const parts = norm_ts.split("T");
        if (parts.length !== 2) throw new Error("Invalid format");
        // Extract date and time parts
        const datePart = parts[0]; // e.g., "2025-05-21"
        const timeParts = parts[1].split("-"); // ["hh", "mm", "ss"]
        if (timeParts.length < 3) throw new Error("Incomplete time");
        // Reformat 
        const [year, month, day] = datePart.split("-").map(Number);
        let [hour, minute, second] = timeParts.map(Number);
        hour = (hour - 2 + 24) % 24;        
        // Create Date in local time (note: month is 0-based)
        const dt = new Date(year, month - 1, day, hour, minute, second);
        // Write string
        const timeStr = dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const dateStr = dt.toLocaleDateString('en-AU');
        return `${timeStr} ${dateStr}`;
    } catch (err) {
        console.warn("formatTimestamp fallback:", err.message);
        return norm_ts;
    }
}


// ─────────────────────────────────────────
// Sanitize filenames from timestamp
// ─────────────────────────────────────────
function sanitizeFilename(ts) {
    return ts.replace(/:/g, '-').replace(/ /g, 'T').replace(/\//g, '-');
}

// ─────────────────────────────────────────
fetchEvents();
setInterval(fetchEvents, 1000);
