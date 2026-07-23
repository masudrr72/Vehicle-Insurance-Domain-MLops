document.addEventListener("DOMContentLoaded", () => {

    // Auto-scroll after prediction
    const result = document.getElementById("prediction-result");

    if(result){
        result.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });
    }

    // Show loading text
    const form = document.getElementById("predictionForm");
    const button = document.querySelector(".predict-btn");

    form.addEventListener("submit", () => {
        button.innerHTML =
            '<i class="fa-solid fa-spinner fa-spin"></i> Predicting...';
        button.disabled = true;
    });

});