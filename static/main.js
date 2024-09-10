document.addEventListener('DOMContentLoaded', function() {
    console.log('Page loaded and ready.');

    // Add a listener to the file input
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            alert('File selected: ' + fileInput.files[0].name);
        });
    }
});
