const form = document.getElementById('uploadForm');
const input = document.getElementById('imageInput');
const runBtn = document.getElementById('runBtn');
const statusText = document.getElementById('statusText');
const originalImg = document.getElementById('originalImg');
const processedImg = document.getElementById('processedImg');
const cropList = document.getElementById('cropList');
const countBadge = document.getElementById('countBadge');

function setStatus(text, isError = false) {
  statusText.textContent = text;
  statusText.style.color = isError ? '#b00020' : '#596067';
}

function renderCrops(crops) {
  cropList.innerHTML = '';
  countBadge.textContent = String(crops.length);

  if (!crops.length) {
    cropList.innerHTML = '<p>Khong co bien so nao duoc phat hien.</p>';
    return;
  }

  for (const item of crops) {
    const card = document.createElement('article');
    card.className = 'crop-card';
    card.innerHTML = `
      <img src="${item.crop_url}" alt="crop plate" />
      <div class="meta"><strong>Bien so:</strong> ${item.text}</div>
      <div class="meta">Conf: ${item.confidence}</div>
      <div class="meta">Blur: ${item.blur_score}</div>
      <div class="meta">Enhance: ${item.enhanced ? 'Co' : 'Khong'}</div>
    `;
    cropList.appendChild(card);
  }
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!input.files.length) {
    setStatus('Ban chua chon anh.', true);
    return;
  }

  runBtn.disabled = true;
  setStatus('Dang nhan dien...');

  try {
    const body = new FormData();
    body.append('image', input.files[0]);

    const res = await fetch('/api/recognize', { method: 'POST', body });
    const data = await res.json();

    if (!res.ok || !data.ok) {
      throw new Error(data.error || 'Loi khong xac dinh');
    }

    originalImg.src = data.original_url + '?t=' + Date.now();
    processedImg.src = data.processed_url + '?t=' + Date.now();
    renderCrops(data.crops || []);

    setStatus(`Hoan tat. Da phat hien ${data.count} bien so.`);
  } catch (err) {
    setStatus(`Loi: ${err.message}`, true);
  } finally {
    runBtn.disabled = false;
  }
});
