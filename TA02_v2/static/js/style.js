let mainNav = document.getElementById("js-menu");
let navBarToggle = document.getElementById("js-navbar-toggle");

navBarToggle.addEventListener("click", function () {
  mainNav.classList.toggle("active");
});

// Holt Winter
let uploadFileHw = document.getElementById("upload-file-hw");
let uploadBtnHw = document.getElementById("btn-upload-hw");
let uploadTextNameHw = document.getElementById("upload-textname-hw");

uploadBtnHw.addEventListener("click", function () {
  uploadFileHw.click();
});

uploadFileHw.addEventListener("change", function () {
  if (uploadFileHw.value) {
    uploadTextNameHw.innerHTML = uploadFileHw.value.match(
      /[\/\\]([\w\d\s\.\-\(\)]+)$/
    )[1];
  } else {
    uploadTextNameHw.innerHTML = "No file chosen, yet. Please upload a file.";
  }
});

// Univariate SVR
let uploadFileSvrUni = document.getElementById("upload-file-svr-uni");
let uploadBtnSvrUni = document.getElementById("btn-upload-svr-uni");
let uploadTextNameSvrUni = document.getElementById("upload-textname-svr-uni");

uploadBtnSvrUni.addEventListener("click", function () {
  uploadFileSvrUni.click();
});

uploadFileSvrUni.addEventListener("change", function () {
  if (uploadFileSvrUni.value) {
    uploadTextNameSvrUni.innerHTML = uploadFileSvrUni.value.match(
      /[\/\\]([\w\d\s\.\-\(\)]+)$/
    )[1];
  } else {
    uploadTextNameSvrUni.innerHTML = "No file chosen, yet. Please upload a file.";
  }
});

// Multivariate SVR
let uploadFileSvrMulti = document.getElementById("upload-file-svr-multi");
let uploadBtnSvrMulti = document.getElementById("btn-upload-svr-multi");
let uploadTextNameSvrMulti = document.getElementById("upload-textname-svr-multi");

uploadBtnSvrMulti.addEventListener("click", function () {
  uploadFileSvrMulti.click();
});

uploadFileSvrMulti.addEventListener("change", function () {
  if (uploadFileSvrMulti.value) {
    uploadTextNameSvrMulti.innerHTML = uploadFileSvrMulti.value.match(
      /[\/\\]([\w\d\s\.\-\(\)]+)$/
    )[1];
  } else {
    uploadTextNameSvrMulti.innerHTML = "No file chosen, yet. Please upload a file.";
  }
});


// Test
let uploadFileSvr = document.getElementById("upload-file-svr");
let uploadBtnSvr = document.getElementById("btn-upload-svr");
let uploadTextNameSvr = document.getElementById("upload-textname-svr");

uploadBtnSvr.addEventListener("click", function () {
  uploadFileSvr.click();
});


uploadFileSvr.addEventListener("change", function () {
  if (uploadFileSvr.value) {
    uploadTextNameSvr.innerHTML = uploadFileSvr.value.match(
      /[\/\\]([\w\d\s\.\-\(\)]+)$/
    )[1];
  } else {
    uploadTextNameSvr.innerHTML = "No file chosen, yet. Please upload a file.";
  }
});