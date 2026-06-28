document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('fraudForm');
  const resultBox = document.getElementById('resultBox');
  const submitBtn = document.getElementById('submitBtn');

  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    resultBox.classList.remove('show', 'is-normal', 'is-fraud');
    resultBox.style.display = 'none';
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...';

    try {
      const formData = new FormData(form);
      const response = await fetch('/predict', { method: 'POST', body: formData });
      const data = await response.json();

      submitBtn.disabled = false;
      submitBtn.textContent = 'Analyze Transaction';

      if (!data.success) {
        alert(data.error || 'Prediction failed');
        return;
      }

      resultBox.style.display = 'block';
      resultBox.classList.add('show');
      resultBox.classList.add(data.prediction === 'NORMAL' ? 'is-normal' : 'is-fraud');

      document.getElementById('resultPrediction').textContent = data.prediction;
      document.getElementById('resultProbability').textContent = data.probability + '%';
      document.getElementById('resultRisk').textContent = 'Risk: ' + data.risk_level;
      document.getElementById('resultAction').textContent = data.recommendation;

      const bar = document.getElementById('resultProgress');
      bar.style.width = data.probability + '%';
      bar.className = 'progress-bar ' + (data.probability >= 70 ? 'bg-danger' : data.probability >= 40 ? 'bg-warning' : 'bg-success');

      resultBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } catch (err) {
      submitBtn.disabled = false;
      submitBtn.textContent = 'Analyze Transaction';
      alert('Something went wrong. Please try again.');
    }
  });
});
