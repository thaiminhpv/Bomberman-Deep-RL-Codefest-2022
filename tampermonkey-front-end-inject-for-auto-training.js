// ==UserScript==
// @name         LOCALHOST TRAIN JS CLUB
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  try to take over the world!
// @author       You
// @match        https://room1.jsclub.me/*
// @match        https://room2.jsclub.me/*
// @match        https://room3.jsclub.me/*
// @icon         https://www.google.com/s2/favicons?sz=64&domain=jsclub.me
// @grant        none
// ==/UserScript==

(function() {
    'use strict';

    if (window.location.href.includes('https://room1.jsclub.me/training/register')) {
        setTimeout(() => document.getElementById('btn-register').click(), 5000)


        setTimeout(function() {
            let url = window.location.href;
            let a = url.split('/')
            let b = a[a.length - 1]

            document.getElementById('map').value='training_map_1'

            console.log(b)
            fetch('http://localhost:5555?game_id=' + b)
            fetch('http://localhost:5001?game_id=' + b)
        }, 1000)


    }

    if (window.location.href.includes('https://room1.jsclub.me/training/stage')) {
        const endgame = () => {document.location.href = 'https://room1.jsclub.me/training/login'}

        document.addEventListener('keypress', (e) => {
            if (e.key === '`') {
                endgame()
            }
        });

        setTimeout(function () {
            endgame()
        }, 1000 * 60 * 5);
    }
})();

