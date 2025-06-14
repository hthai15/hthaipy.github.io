import streamlit as st
import streamlit.components.v1 as components
import random

st.set_page_config(layout="wide")
st.title("🔫 Game Bắn Súng Có Súng!")

if "score" not in st.session_state:
    st.session_state.score = 0

# HTML/JS Game Canvas
game_html = f"""
<canvas id="gameCanvas" width="600" height="400" style="border:1px solid #000;"></canvas>
<p>Điểm: <span id="score">{st.session_state.score}</span></p>

<script>
let canvas = document.getElementById("gameCanvas");
let ctx = canvas.getContext("2d");

let gunImg = new Image();
gunImg.src = "gun.png"; // bạn cần ảnh này trong cùng thư mục hoặc đổi sang URL

let targetImg = new Image();
targetImg.src = "target.png"; // tương tự

let gunX = 250;
let gunY = 340;

let targetX = Math.floor(Math.random() * 540) + 30;
let targetY = 30;

let bulletX = null;
let bulletY = null;

function draw() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Vẽ súng
    ctx.drawImage(gunImg, gunX, gunY, 60, 60);

    // Vẽ mục tiêu
    ctx.drawImage(targetImg, targetX, targetY, 50, 50);

    // Vẽ đạn
    if (bulletY !== null) {{
        ctx.beginPath();
        ctx.arc(bulletX, bulletY, 5, 0, 2 * Math.PI);
        ctx.fillStyle = "black";
        ctx.fill();
        bulletY -= 5;

        // Kiểm tra trúng mục tiêu
        if (
            bulletX >= targetX && bulletX <= targetX + 50 &&
            bulletY >= targetY && bulletY <= targetY + 50
        ) {{
            document.getElementById("score").innerText = parseInt(document.getElementById("score").innerText) + 1;
            bulletY = null;
            targetX = Math.floor(Math.random() * 540) + 30;
            targetY = 30;
        }}

        // Nếu đạn ra khỏi màn
        if (bulletY < 0) {{
            bulletY = null;
        }}
    }}

    requestAnimationFrame(draw);
}}

canvas.addEventListener("mousemove", function(e) {{
    let rect = canvas.getBoundingClientRect();
    gunX = e.clientX - rect.left - 30;
}});

canvas.addEventListener("click", function() {{
    if (bulletY === null) {{
        bulletX = gunX + 30;
        bulletY = gunY;
    }}
}});

gunImg.onload = function() {{
    draw();
}};
</script>
"""

components.html(game_html, height=500)
