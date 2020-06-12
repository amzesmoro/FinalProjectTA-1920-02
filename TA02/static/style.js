let mainNav = document.getElementById("js-menu");
let navBarToggle = document.getElementById("js-navbar-toggle");

navBarToggle.addEventListener("click", function() {
    mainNav.classList.toggle("active");
});


let uploadFile = document.getElementById("upload-file");
let uploadBtn = document.getElementById("btn-upload");
let uploadText = document.getElementById("upload-text");

uploadBtn.addEventListener("click", function() {
    uploadFile.click();
});

uploadFile.addEventListener("change", function() {
  if (uploadFile.value) {
    uploadText.innerHTML = uploadFile.value.match(
      /[\/\\]([\w\d\s\.\-\(\)]+)$/
    )[1];
  } else {
    uploadText.innerHTML = "No file chosen, yet. Please upload a file.";
  }
});


let previewData = document.getElementById("preview-data");
let previewDataBtn = document.getElementById("btn-preview-data");

previewDataBtn.addEventListener("click", function() {
  var x = previewData
  if (x.style.display === "block") {
      x.style.display = "none";
  } else {
      x.style.display = "block";
  }
});