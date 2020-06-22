let mainNav = document.getElementById("js-menu");
let navBarToggle = document.getElementById("js-navbar-toggle");

navBarToggle.addEventListener("click", function() {
    mainNav.classList.toggle("active");
});

let uploadFileHw = document.getElementById("upload-file-hw");
let uploadFileSvr = document.getElementById("upload-file-svr");
let uploadBtnHw = document.getElementById("btn-upload-hw");
let uploadBtnSvr = document.getElementById("btn-upload-svr");
let uploadTextNameHw = document.getElementById("upload-textname-hw");
let uploadTextNameSvr = document.getElementById("upload-textname-svr");

uploadBtnHw.addEventListener("click", function() {
    uploadFileHw.click();
});

uploadBtnSvr.addEventListener("click", function() {
  uploadFileSvr.click();
});

uploadFileHw.addEventListener("change", function() {
  if (uploadFileHw.value) {
    uploadTextNameHw.innerHTML = uploadFileHw.value.match(
      /[\/\\]([\w\d\s\.\-\(\)]+)$/
    )[1];
  } else {
    uploadTextNameHw.innerHTML = "No file chosen, yet. Please upload a file.";
  }
});

uploadFileSvr.addEventListener("change", function() {
  if (uploadFileSvr.value) {
    uploadTextNameSvr.innerHTML = uploadFileSvr.value.match(
      /[\/\\]([\w\d\s\.\-\(\)]+)$/
    )[1];
  } else {
    uploadTextNameSvr.innerHTML = "No file chosen, yet. Please upload a file.";
  }
});
