async function sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();

    // 1) Bail out if empty
    if (!message) return;

    // 2) Clear the input
    input.value = '';

    // 3) Immediately show the user's message (initially without any tag)
    const userDiv = createUserMessageDiv(message);
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.appendChild(userDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // 4) Send to server for moderation + GPT
    const response = await fetch('/chat', {
        method: 'POST',
        credentials: 'include',  // include if you rely on session cookies
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message })
    });

    // 5) Parse server response
    const data = await response.json();

    // 6) Update the user message text with the version from the server (with [uncensored] if needed)
    // Find the text element in the user bubble and update its content.
    const userTextElem = userDiv.querySelector('.message-text');
    userTextElem.textContent = data.user_message;

    // 7) If the message is flagged, update the user bubble (e.g., add flagged details)
    if (data.is_inappropriate) {
        addFlaggedCategories(userDiv, data.flagged_categories);
    }

    // 8) Now display the assistant’s reply
    const aiDiv = createAiMessageDiv(data.ai_response);
    chatMessages.appendChild(aiDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Create a user message bubble
function createUserMessageDiv(message) {
    const userDiv = document.createElement('div');
    userDiv.className = 'message user-message';

    // Bubble container (icon + text)
    const bubbleContainer = document.createElement('div');
    bubbleContainer.className = 'bubble-container';

    const userText = document.createElement('div');
    userText.className = 'message-text';
    userText.textContent = message;

    const userIcon = document.createElement('span');
    userIcon.className = 'profile-icon';
    userIcon.textContent = '👤';

    bubbleContainer.appendChild(userText);
    bubbleContainer.appendChild(userIcon);
    userDiv.appendChild(bubbleContainer);

    return userDiv;
}

// If flagged, append a "Flagged categories" line and style the user bubble
function addFlaggedCategories(userDiv, flaggedCategories) {
    userDiv.classList.add('inappropriate-message');

    const flaggedDiv = document.createElement('div');
    flaggedDiv.className = 'flagged-categories';

    const flaggedText = Object.entries(flaggedCategories)
        .map(([cat, score]) => `${cat} (${score.toFixed(2)})`)
        .join(', ');

    flaggedDiv.textContent = `Flagged categories: ${flaggedText}`;
    userDiv.appendChild(flaggedDiv);
}

// Create an AI message bubble
function createAiMessageDiv(aiText) {
    const aiDiv = document.createElement('div');
    aiDiv.className = 'message ai-message';

    const aiIcon = document.createElement('span');
    aiIcon.className = 'profile-icon';
    aiIcon.textContent = '🤖';

    const aiMessage = document.createElement('div');
    aiMessage.className = 'message-text';
    aiMessage.textContent = aiText;

    aiDiv.appendChild(aiIcon);
    aiDiv.appendChild(aiMessage);
    return aiDiv;
}

// Send message on Enter key
document.getElementById('message-input')
    .addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
