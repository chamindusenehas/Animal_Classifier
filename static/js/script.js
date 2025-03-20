function uploadImage() {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select an image first!');
        return;
    }

    console.log('Uploading file:', file.name, file.size);

    const previewImage = document.getElementById('previewImage');
    previewImage.src = URL.createObjectURL(file);
    previewImage.style.display = 'block';
    previewImage.classList.add('fade-in');

    const formData = new FormData();
    formData.append('image', file);

    // Disable button during upload
    const predictButton = document.getElementById('predictButton');
    predictButton.disabled = true;
    predictButton.textContent = 'Predicting...';

    fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log('Response status:', response.status);
        return response.json();
    })
    .then(data => {
        console.log('Response data:', data);
        
        if (data.success) {
            const predictionText = document.getElementById('predictionText');
            const confidenceText = document.getElementById('confidence');
            predictionText.textContent = data.animal;
            const confidencePercent = data.confidence;
            confidenceText.textContent = `${confidencePercent}%`;
            const result = document.getElementById('result');
            result.classList.add('slide-up');
            result.classList.remove('hidden');
            document.querySelector('.container').style.height = '200vh';
        } else {
            const errorMsg = data.error || 'Unknown error occurred';
            document.getElementById('predictionText').textContent = `Error: ${errorMsg}`;
            document.getElementById('confidence').textContent = '--%';
        }
    })
    .catch(error => {
        console.error('Fetch error:', error);
        document.getElementById('predictionText').textContent = `Error: ${error.message}`;
        document.getElementById('confidence').textContent = '--%';
    })
    .finally(() => {
        predictButton.disabled = false;
        predictButton.textContent = 'Upload & Predict';
    });
}

document.getElementById('imageInput').addEventListener('change', function(e) {
    const previewImage = document.getElementById('previewImage');
    if (e.target.files && e.target.files[0]) {
        previewImage.src = URL.createObjectURL(e.target.files[0]);
        previewImage.style.display = 'block';
        previewImage.classList.add('fade-in');
    }
});