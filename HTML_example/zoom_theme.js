// theme-zoom.js

document.addEventListener("DOMContentLoaded", () => {
    // ------------------------
    // Zoom Controls
    // ------------------------
    const zoomControls = document.createElement("div");
    zoomControls.className = "zoom-controls";

    const zoomIn = document.createElement("button");
    zoomIn.id = "zoom-in";
    zoomIn.textContent = "+";

    const zoomOut = document.createElement("button");
    zoomOut.id = "zoom-out";
    zoomOut.textContent = "−";

    zoomControls.appendChild(zoomIn);
    zoomControls.appendChild(zoomOut);
    document.body.appendChild(zoomControls);

    const content = document.getElementById("page-content"); // Only scale this

    // Load saved zoom from localStorage, default 1
    let scale = parseFloat(localStorage.getItem("page-scale")) || 1;
    content.style.transform = `scale(${scale})`;
    content.style.transformOrigin = 'top left';

    // Zoom in
    zoomIn.addEventListener("click", () => {
        scale += 0.1;
        content.style.transform = `scale(${scale})`;
        localStorage.setItem("page-scale", scale.toFixed(2)); // Save zoom
    });

    // Zoom out
    zoomOut.addEventListener("click", () => {
        if (scale > 0.2) {
            scale -= 0.1;
            content.style.transform = `scale(${scale})`;
            localStorage.setItem("page-scale", scale.toFixed(2)); // Save zoom
        }
    });
});

    // ------------------------
    // Theme Switching
    // ------------------------

// theme-switcher.js
function setTheme(themeName) {
    document.body.className = themeName;
    localStorage.setItem('selectedTheme', themeName); // Save the selection
}

// Apply saved theme on page load
document.addEventListener("DOMContentLoaded", () => {
    const savedTheme = localStorage.getItem('selectedTheme') || 'theme-default';
    setTheme(savedTheme);
    
    // Expose setTheme globally for swatches/buttons
    window.setTheme = setTheme;

});

