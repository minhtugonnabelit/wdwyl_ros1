var imgLamp = document.getElementById('yellowlamp');
var imgLamp2 = document.getElementById('bluelamp');

function turnOnYellow() {
    var checkbox = document.getElementById('mylamp');
    if (checkbox.checked == true) {
        imgLamp.setAttribute('src', './static/img/lamp-on.png');
    } else {
        imgLamp.setAttribute('src', './static/img/lamp.png');
    }
}

function turnOnBlue() {
    var checkbox = document.getElementById('mylamp1');
    if (checkbox.checked == true) {
        imgLamp2.setAttribute('src', './static/img/lamp-on_blue.png');
    } else {
        imgLamp2.setAttribute('src', './static/img/lamp.png');
    }
}
