// WebSocket connections for each client
const connections = {};

// Initialize WebSocket connections on page load
window.addEventListener('DOMContentLoaded', () => {
    // Initialize all 4 clients
    for (let i = 1; i <= 4; i++) {
        initializeClient(i);
    }

    // Update metrics every 2 seconds
    setInterval(updateMetrics, 2000);
    updateMetrics();
});

function initializeClient(clientId) {
    const ws = new WebSocket(`ws://${window.location.host}/ws/client-${clientId}`);

    ws.onopen = () => {
        console.log(`Client ${clientId} connected`);
        updateStatus(clientId, 'ready', 'Ready');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(clientId, data);
    };

    ws.onerror = (error) => {
        console.error(`Client ${clientId} error:`, error);
        updateStatus(clientId, 'error', 'Error');
    };

    ws.onclose = () => {
        console.log(`Client ${clientId} disconnected`);
        updateStatus(clientId, 'error', 'Disconnected');
        // Attempt reconnection after 3 seconds
        setTimeout(() => initializeClient(clientId), 3000);
    };

    connections[clientId] = ws;
}

function handleMessage(clientId, data) {
    const responseArea = document.getElementById(`response-${clientId}`);
    const statusBadge = document.getElementById(`status-${clientId}`);

    switch (data.type) {
        case 'connected':
            updateStatus(clientId, 'ready', 'Ready');
            break;

        case 'status':
            updateStatus(clientId, 'processing', data.message);
            break;

        case 'chunk':
            // Append response chunk
            responseArea.textContent += data.content;
            responseArea.scrollTop = responseArea.scrollHeight;
            break;

        case 'complete':
            // Update status and stats
            updateStatus(clientId, 'complete', 'Complete ✓');

            const timeEl = document.getElementById(`time-${clientId}`);
            const cacheEl = document.getElementById(`cache-${clientId}`);

            timeEl.textContent = `Time: ${data.processing_time}s`;
            cacheEl.textContent = `Cache: ${data.cache_hit ? '✓ HIT' : '✗ MISS'}`;

            // Update cache indicator styling
            if (data.cache_hit) {
                updateStatus(clientId, 'cached', 'Cached ⚡');
                cacheEl.style.color = '#0c5460';
                cacheEl.style.fontWeight = 'bold';
            }

            // Reset to ready after 2 seconds
            setTimeout(() => updateStatus(clientId, 'ready', 'Ready'), 2000);
            break;

        case 'error':
            updateStatus(clientId, 'error', 'Error');
            responseArea.textContent = `Error: ${data.message}`;
            setTimeout(() => updateStatus(clientId, 'ready', 'Ready'), 3000);
            break;
    }
}

function updateStatus(clientId, statusClass, statusText) {
    const badge = document.getElementById(`status-${clientId}`);
    badge.className = `status-badge ${statusClass}`;
    badge.textContent = statusText;
}

function sendQuery(clientId) {
    const input = document.getElementById(`query-${clientId}`);
    const query = input.value.trim();

    if (!query) {
        alert('Please enter a question!');
        return;
    }

    const ws = connections[clientId];
    if (ws && ws.readyState === WebSocket.OPEN) {
        // Clear previous response
        const responseArea = document.getElementById(`response-${clientId}`);
        responseArea.textContent = '';

        // Send query
        ws.send(JSON.stringify({
            type: 'query',
            query: query
        }));

        // Clear cache styling
        const cacheEl = document.getElementById(`cache-${clientId}`);
        cacheEl.style.color = '';
        cacheEl.style.fontWeight = '';
    } else {
        alert(`Client ${clientId} is not connected. Reconnecting...`);
        initializeClient(clientId);
    }
}

// Handle Enter key in input fields
document.addEventListener('keypress', (e) => {
    if (e.target.tagName === 'INPUT' && e.key === 'Enter') {
        const clientId = e.target.id.split('-')[1];
        sendQuery(clientId);
    }
});

async function updateMetrics() {
    try {
        const response = await fetch('/metrics');
        const data = await response.json();

        document.getElementById('total-queries').textContent = data.total_queries;
        document.getElementById('cache-hit-rate').textContent = `${data.cache_hit_rate}%`;
        document.getElementById('avg-time').textContent = `${data.avg_response_time}s`;
        document.getElementById('active-clients').textContent = data.active_clients;
    } catch (error) {
        console.error('Failed to update metrics:', error);
    }
}

// Test Scenarios
function testScenario1() {
    // Cache Performance Test
    console.log('Running Test 1: Cache Performance');

    // Client 1: First query (cache miss)
    document.getElementById('query-1').value = 'What is Node.js?';
    setTimeout(() => sendQuery('1'), 100);

    // Client 2: Different query
    document.getElementById('query-2').value = 'Explain event loop in Node.js';
    setTimeout(() => sendQuery('2'), 500);

    // Client 3: Same as Client 1 (cache hit!)
    document.getElementById('query-3').value = 'What is Node.js?';
    setTimeout(() => sendQuery('3'), 2000);

    // Client 4: Another different query
    document.getElementById('query-4').value = 'What is npm?';
    setTimeout(() => sendQuery('4'), 3000);
}

function testScenario2() {
    // Non-Blocking Test
    console.log('Running Test 2: Non-Blocking Demo');

    // All clients send queries simultaneously
    document.getElementById('query-1').value = 'What is Node.js?';
    document.getElementById('query-2').value = 'Explain callbacks';
    document.getElementById('query-3').value = 'What are promises?';
    document.getElementById('query-4').value = 'Explain async/await';

    // Send all at once (with tiny delays)
    setTimeout(() => sendQuery('1'), 100);
    setTimeout(() => sendQuery('2'), 150);
    setTimeout(() => sendQuery('3'), 200);
    setTimeout(() => sendQuery('4'), 250);
}

function clearAll() {
    // Clear all inputs and responses
    for (let i = 1; i <= 4; i++) {
        document.getElementById(`query-${i}`).value = '';
        document.getElementById(`response-${i}`).textContent = '';
        document.getElementById(`time-${i}`).textContent = 'Time: -';
        document.getElementById(`cache-${i}`).textContent = 'Cache: -';
        updateStatus(i, 'ready', 'Ready');
    }
}
