/* global fetch */
(function () {
  'use strict';

  var API_URL = 'http://localhost:8000/ask';

  function buildWidget() {
    // ── Toggle button ──────────────────────────────────────────────────────────
    var toggleBtn = document.createElement('button');
    toggleBtn.id = 'padfoot-toggle';
    toggleBtn.title = 'Ask Documentation Assistant';
    toggleBtn.textContent = '💬';

    // ── Widget shell ───────────────────────────────────────────────────────────
    var widget = document.createElement('div');
    widget.id = 'padfoot-widget';
    widget.innerHTML =
      '<div id="padfoot-header">' +
        '<div class="padfoot-logo">📚</div>' +
        '<div class="padfoot-title">' +
          '<h2>Documentation Assistant</h2>' +
          '<p>Powered by RAG + Gemini</p>' +
        '</div>' +
        '<button id="padfoot-close-btn" title="Close">✕</button>' +
      '</div>' +
      '<div id="padfoot-messages">' +
        '<div id="padfoot-empty">' +
          '<div class="padfoot-empty-icon">💬</div>' +
          '<p>Ask any question about the documentation — software, email, servers, and more.</p>' +
        '</div>' +
      '</div>' +
      '<div id="padfoot-input-area">' +
        '<input id="padfoot-input" type="text"' +
          ' placeholder="e.g. How do I configure Thunderbird?"' +
          ' autocomplete="off" />' +
        '<button id="padfoot-send-btn">Send</button>' +
      '</div>';

    document.body.appendChild(toggleBtn);
    document.body.appendChild(widget);

    // ── Element references ─────────────────────────────────────────────────────
    var messagesEl = document.getElementById('padfoot-messages');
    var inputEl    = document.getElementById('padfoot-input');
    var sendBtn    = document.getElementById('padfoot-send-btn');
    var closeBtn   = document.getElementById('padfoot-close-btn');

    // ── Toggle open / close ────────────────────────────────────────────────────
    toggleBtn.addEventListener('click', function () {
      widget.classList.toggle('padfoot-open');
      if (widget.classList.contains('padfoot-open')) {
        inputEl.focus();
      }
    });

    closeBtn.addEventListener('click', function () {
      widget.classList.remove('padfoot-open');
    });

    // ── Event listeners ────────────────────────────────────────────────────────
    sendBtn.addEventListener('click', sendQuery);
    inputEl.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuery();
      }
    });

    // ── Helpers ────────────────────────────────────────────────────────────────
    function clearEmptyState() {
      var el = document.getElementById('padfoot-empty');
      if (el) el.remove();
    }

    function appendMessage(role, text, sources) {
      sources = sources || [];
      clearEmptyState();

      var wrapper = document.createElement('div');
      wrapper.className = 'padfoot-message padfoot-' + role;

      var bubble = document.createElement('div');
      bubble.className = 'padfoot-bubble';
      bubble.textContent = text;
      wrapper.appendChild(bubble);

      if (role === 'bot' && sources.length > 0) {
        var row = document.createElement('div');
        row.className = 'padfoot-sources';
        sources.forEach(function (src) {
          var filename = src.split('/').pop().split('\\').pop();
          var link = document.createElement('a');
          link.className = 'padfoot-source-pill';
          link.title = filename;
          
          // Use current origin to build the link
          link.href = window.location.origin + '/' + filename;
          link.target = '_blank';
          link.rel = 'noopener noreferrer';
          link.textContent = '📄 ' + filename;
          
          row.appendChild(link);
        });
        wrapper.appendChild(row);
      }

      messagesEl.appendChild(wrapper);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function showTypingIndicator() {
      clearEmptyState();
      var el = document.createElement('div');
      el.className = 'padfoot-typing';
      el.id = 'padfoot-typing-indicator';
      el.innerHTML = '<span></span><span></span><span></span>';
      messagesEl.appendChild(el);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return el;
    }

    function setLoading(loading) {
      inputEl.disabled  = loading;
      sendBtn.disabled  = loading;
      sendBtn.textContent = loading ? '…' : 'Send';
    }

    // ── Core send logic ────────────────────────────────────────────────────────
    function sendQuery() {
      var query = inputEl.value.trim();
      if (!query) return;

      appendMessage('user', query);
      inputEl.value = '';
      setLoading(true);

      var indicator = showTypingIndicator();

      var sourcePage = window.location.pathname.split('/').pop() || 'index.html';

      fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: query,
          source_page: sourcePage
        }),
      })
        .then(function (res) {
          indicator.remove();
          if (!res.ok) {
            return res.json().catch(function () { return {}; }).then(function (err) {
              throw new Error(err.detail || ('Server error ' + res.status));
            });
          }
          return res.json();
        })
        .then(function (data) {
          appendMessage('bot', data.answer, data.sources || []);
        })
        .catch(function (err) {
          indicator.remove();
          clearEmptyState();
          var errEl = document.createElement('div');
          errEl.className = 'padfoot-error';
          errEl.textContent = '⚠️ ' + (err.message || 'Could not reach the backend. Is it running?');
          messagesEl.appendChild(errEl);
          messagesEl.scrollTop = messagesEl.scrollHeight;
        })
        .finally(function () {
          setLoading(false);
          inputEl.focus();
        });
    }
  }

  // ── Wait for DOM before initialising ──────────────────────────────────────────
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', buildWidget);
  } else {
    buildWidget();
  }
})();
