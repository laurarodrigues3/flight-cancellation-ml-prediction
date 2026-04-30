const API_URL = "/api";

function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.glass-card').forEach(card => card.classList.remove('active'));

    event.target.classList.add('active');
    document.getElementById(tabId + '-tab').classList.add('active');
}

// Input Validation Functions
function validateInputs(data) {
    const errors = [];

    // Date format regex (YYYY-MM-DD)
    const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
    // No numbers regex
    const noNumbersRegex = /^[^0-9]*$/;
    // Alphanumeric only (for Tail_Number - letters and numbers, no spaces/special chars)
    const alphanumericRegex = /^[A-Za-z0-9]+$/;
    // Airport codes (letters only, 3 chars typical)
    const airportCodeRegex = /^[A-Za-z]+$/;

    // Check all fields are not empty
    for (const [key, value] of Object.entries(data)) {
        if (value === null || value === undefined || String(value).trim() === '') {
            errors.push(`${key} cannot be empty.`);
        }
    }

    // FlightDate - must be date format YYYY-MM-DD
    if (data.FlightDate && !dateRegex.test(data.FlightDate)) {
        errors.push("FlightDate must follow the format YYYY-MM-DD (e.g., 2022-04-04).");
    }

    // Origin - no numbers allowed
    if (data.Origin && !airportCodeRegex.test(data.Origin)) {
        errors.push("Origin must contain only letters (airport code).");
    }

    // Dest - no numbers allowed
    if (data.Dest && !airportCodeRegex.test(data.Dest)) {
        errors.push("Dest must contain only letters (airport code).");
    }

    // Tail_Number - no spaces or special characters
    if (data.Tail_Number && !alphanumericRegex.test(data.Tail_Number)) {
        errors.push("Tail_Number cannot contain spaces or special characters.");
    }

    // OriginCityName - no numbers
    if (data.OriginCityName && !noNumbersRegex.test(data.OriginCityName)) {
        errors.push("OriginCityName cannot contain numbers.");
    }

    // OriginState - no numbers
    if (data.OriginState && !noNumbersRegex.test(data.OriginState)) {
        errors.push("OriginState cannot contain numbers.");
    }

    // OriginStateName - no numbers
    if (data.OriginStateName && !noNumbersRegex.test(data.OriginStateName)) {
        errors.push("OriginStateName cannot contain numbers.");
    }

    // DestCityName - no numbers
    if (data.DestCityName && !noNumbersRegex.test(data.DestCityName)) {
        errors.push("DestCityName cannot contain numbers.");
    }

    // DestState - no numbers
    if (data.DestState && !noNumbersRegex.test(data.DestState)) {
        errors.push("DestState cannot contain numbers.");
    }

    // DestStateName - no numbers
    if (data.DestStateName && !noNumbersRegex.test(data.DestStateName)) {
        errors.push("DestStateName cannot contain numbers.");
    }

    return errors;
}

// Clear all form fields
function clearFields() {
    const form = document.getElementById('prediction-form');

    // Clear all text and number inputs to blank
    const inputs = form.querySelectorAll('input');
    inputs.forEach(input => {
        input.value = '';
    });

    // Add placeholder option to selects and select it
    const selects = form.querySelectorAll('select');
    selects.forEach(select => {
        // Check if placeholder already exists
        let placeholder = select.querySelector('option[value=""]');
        if (!placeholder) {
            placeholder = document.createElement('option');
            placeholder.value = '';
            placeholder.textContent = '-- Select --';
            placeholder.disabled = true;
            select.insertBefore(placeholder, select.firstChild);
        }
        select.value = '';
    });

    // Hide the result div
    const resultDiv = document.getElementById('prediction-result');
    resultDiv.style.display = 'none';
    resultDiv.innerHTML = '';
    resultDiv.className = '';
}

// Prediction Logic
async function predict() {
    const form = document.getElementById('prediction-form');
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    const resultDiv = document.getElementById('prediction-result');

    // Frontend Validation
    const validationErrors = validateInputs(data);

    // Also validate selects explicitly (disabled options are not in FormData)
    const selects = form.querySelectorAll('select');
    selects.forEach(select => {
        if (select.value === '' || select.value === null) {
            const fieldName = select.name || select.id || 'Field';
            validationErrors.push(`${fieldName} cannot be empty.`);
        }
    });

    if (validationErrors.length > 0) {
        resultDiv.style.display = 'block';
        resultDiv.innerHTML = `<strong>Validation Error:</strong><br>${validationErrors.join('<br>')}`;
        resultDiv.className = 'result-danger';
        return;
    }

    const payload = data;
    const model = document.getElementById('single-model-select').value;

    resultDiv.style.display = 'block';
    resultDiv.innerHTML = '<p>Analyzing flight data...</p>';
    resultDiv.className = '';

    try {
        const response = await fetch(`${API_URL}/predict-single`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data: payload, model: model })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Server Error");
        }

        const result = await response.json();

        resultDiv.innerHTML = `Prediction: ${result.label}`;
        resultDiv.className = result.prediction === 1 ? 'result-danger' : 'result-safe';

    } catch (error) {
        resultDiv.innerText = "Error: " + error.message;
        resultDiv.className = 'result-danger';
    }
}

// File Upload Handling
document.getElementById('file-upload').addEventListener('change', function (e) {
    if (e.target.files[0]) {
        document.getElementById('file-name').innerText = e.target.files[0].name;
    }
});

// Evaluation Logic
async function evaluate_models() {
    const fileInput = document.getElementById('file-upload');
    const model = document.getElementById('eval-model-select').value;
    const resultsDiv = document.getElementById('evaluation-results');

    if (!fileInput.files[0]) {
        alert("Please upload a CSV file first.");
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('model', model);

    resultsDiv.innerHTML = '<p style="text-align:center; margin-top:20px;">Evaluating... This may take a moment.</p>';

    try {
        const response = await fetch(`${API_URL}/evaluate-models?model=${model}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Server Error");
        }

        const data = await response.json();

        let html = '';
        data.results.forEach(res => {
            html += `
                <div style="background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 12px; margin-top: 1rem;">
                    <h3 style="color: var(--primary); margin-bottom: 1rem;">${res.model}</h3>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">${(res.metrics.accuracy * 100).toFixed(1)}%</div>
                            <div class="metric-label">Accuracy</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(res.metrics.f1 * 100).toFixed(1)}%</div>
                            <div class="metric-label">F1 Score</div>
                        </div>
                         <div class="metric-card">
                            <div class="metric-value">${(res.metrics.recall * 100).toFixed(1)}%</div>
                            <div class="metric-label">Recall</div>
                        </div>
                         <div class="metric-card">
                            <div class="metric-value">${(res.metrics.precision * 100).toFixed(1)}%</div>
                            <div class="metric-label">Precision</div>
                        </div>
                    </div>
                </div>
            `;
        });

        resultsDiv.innerHTML = html;

    } catch (error) {
        resultsDiv.innerHTML = `<div style="background: rgba(239, 68, 68, 0.1); border: 1px solid var(--danger); color: var(--danger); padding: 1rem; border-radius: 12px; text-align: center;">
            <strong>Error:</strong> ${error.message}
        </div>`;
    }
}
